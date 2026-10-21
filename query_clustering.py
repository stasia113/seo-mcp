"""
Semantic query clustering for GSC analysis.

Uses sentence-transformers locally to group semantically similar search
queries. Clusters are built from the union of query strings across the
available datasets so that period and YoY reports use the same topic groups.
"""

from __future__ import annotations

from hashlib import sha1
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_SIMILARITY_THRESHOLD = 0.78
DEFAULT_MAX_MEMBER_QUERIES = 10


def _normalize_query(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().lower().split())


def _display_query(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


def _cluster_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Semantic query clustering requires sentence-transformers. "
            "Install it with: pip install sentence-transformers"
        ) from exc

    return SentenceTransformer(model_name)


def _make_agglomerative(distance_threshold: float):
    try:
        return AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=distance_threshold,
        )
    except TypeError:
        # Compatibility with older scikit-learn versions.
        return AgglomerativeClustering(
            n_clusters=None,
            affinity="cosine",
            linkage="average",
            distance_threshold=distance_threshold,
        )


def build_cluster_map(
    query_frames: Iterable[pd.DataFrame],
    query_column: str,
    model_name: str = DEFAULT_MODEL,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> dict[str, dict]:
    """
    Build a stable semantic cluster map from one or more query datasets.

    The input frames may come from the period and YoY exports. Only the query
    text is used to build clusters. Metrics are aggregated later per dataset.
    """

    records: dict[str, str] = {}

    for frame in query_frames:
        if query_column not in frame.columns:
            raise ValueError(
                f"Query column '{query_column}' was not found. "
                f"Available columns: {frame.columns.tolist()}"
            )

        for value in frame[query_column].tolist():
            normalized = _normalize_query(value)
            display = _display_query(value)
            if normalized and normalized not in records:
                records[normalized] = display

    queries = sorted(records.keys())

    if not queries:
        return {}

    if len(queries) == 1:
        only = queries[0]
        cluster_id = _cluster_id([only])
        return {
            only: {
                "cluster_id": cluster_id,
                "label": records[only],
                "members": [records[only]],
            }
        }

    model = _cluster_model(model_name)
    embeddings = model.encode(
        [records[q] for q in queries],
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    distance_threshold = 1.0 - similarity_threshold
    clustering = _make_agglomerative(distance_threshold)
    labels = clustering.fit_predict(np.asarray(embeddings))

    grouped: dict[int, list[str]] = {}
    for query, label in zip(queries, labels):
        grouped.setdefault(int(label), []).append(query)

    result: dict[str, dict] = {}

    for members in grouped.values():
        cluster_id = _cluster_id(members)
        display_members = [records[q] for q in members]
        label = _choose_cluster_label(
            members,
            records,
            embeddings,
            queries,
        )

        info = {
            "cluster_id": cluster_id,
            "label": label,
            "members": sorted(display_members),
        }

        for member in members:
            result[member] = info

    return result


def _cluster_id(members: list[str]) -> str:
    payload = "\n".join(sorted(members)).encode("utf-8")
    return "cluster_" + sha1(payload).hexdigest()[:12]


def _choose_cluster_label(
    members: list[str],
    records: dict[str, str],
    embeddings: np.ndarray,
    all_queries: list[str],
) -> str:
    """Choose a readable representative query, favoring central and concise text."""

    index = {query: i for i, query in enumerate(all_queries)}
    member_indices = [index[q] for q in members]
    member_vectors = embeddings[member_indices]

    centroid = member_vectors.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm:
        centroid = centroid / norm

    similarities = member_vectors @ centroid
    ranked = sorted(
        zip(members, similarities),
        key=lambda item: (-float(item[1]), len(item[0].split()), item[0]),
    )

    # Prefer a concise representative among the strongest central queries.
    top = ranked[: min(5, len(ranked))]
    best = min(
        top,
        key=lambda item: (
            len(item[0].split()),
            len(item[0]),
            -float(item[1]),
            item[0],
        ),
    )

    return records[best[0]]


def aggregate_query_clusters(
    df: pd.DataFrame,
    columns: dict,
    cluster_map: dict[str, dict],
    max_member_queries: int = DEFAULT_MAX_MEMBER_QUERIES,
) -> list[dict]:
    """
    Aggregate current/comparison GSC metrics by semantic cluster.

    CTR is recalculated from summed clicks / summed impressions.
    Position is impressions-weighted. This avoids averaging CTRs or treating
    every query's position as equally representative of the cluster.
    """

    query_column = columns["key_column"]

    work = df.copy()
    work["_normalized_query"] = work[query_column].map(_normalize_query)
    work = work[work["_normalized_query"].isin(cluster_map)].copy()

    numeric_columns = [
        columns["current_clicks"],
        columns["comparison_clicks"],
        columns["current_impressions"],
        columns["comparison_impressions"],
        columns["current_ctr"],
        columns["comparison_ctr"],
        columns["current_position"],
        columns["comparison_position"],
    ]

    for column in numeric_columns:
        if work[column].dtype == "object":
            work[column] = (
                work[column]
                .astype(str)
                .str.replace("%", "", regex=False)
            )
        work[column] = pd.to_numeric(work[column], errors="coerce").fillna(0)

    work["_cluster_id"] = work["_normalized_query"].map(
        lambda q: cluster_map[q]["cluster_id"]
    )

    rows = []

    for cluster_id, group in work.groupby("_cluster_id", sort=False):
        first_query = group["_normalized_query"].iloc[0]
        info = cluster_map[first_query]

        current_clicks = float(group[columns["current_clicks"]].sum())
        comparison_clicks = float(group[columns["comparison_clicks"]].sum())
        current_impressions = float(group[columns["current_impressions"]].sum())
        comparison_impressions = float(group[columns["comparison_impressions"]].sum())

        current_ctr = (
            current_clicks / current_impressions
            if current_impressions > 0 else 0.0
        )
        comparison_ctr = (
            comparison_clicks / comparison_impressions
            if comparison_impressions > 0 else 0.0
        )

        current_position = _weighted_position(
            group,
            columns["current_position"],
            columns["current_impressions"],
        )
        comparison_position = _weighted_position(
            group,
            columns["comparison_position"],
            columns["comparison_impressions"],
        )

        click_change = current_clicks - comparison_clicks
        impression_change = current_impressions - comparison_impressions
        ctr_change = current_ctr - comparison_ctr
        position_change = current_position - comparison_position

        diagnosis = _diagnose(
            click_change,
            impression_change,
            ctr_change,
            position_change,
        )

        member_query_rows = []
        group = group.copy()
        group["_display_query"] = group[query_column].map(_display_query)
        group = group.sort_values(
            columns["current_clicks"], ascending=False
        )

        for _, row in group.head(max_member_queries).iterrows():
            member_query_rows.append({
                "query": row["_display_query"],
                "clicks": _number(row[columns["current_clicks"]]),
                "impressions": _number(row[columns["current_impressions"]]),
            })

        rows.append({
            "cluster_id": cluster_id,
            "cluster": info["label"],
            "query_count": int(len(group)),
            "current": {
                "clicks": _number(current_clicks),
                "impressions": _number(current_impressions),
                "ctr": round(current_ctr, 6),
                "position": round(current_position, 2),
            },
            "previous": {
                "clicks": _number(comparison_clicks),
                "impressions": _number(comparison_impressions),
                "ctr": round(comparison_ctr, 6),
                "position": round(comparison_position, 2),
            },
            "change": {
                "clicks": _number(click_change),
                "impressions": _number(impression_change),
                "ctr": round(ctr_change, 6),
                "position": round(position_change, 2),
            },
            "diagnosis": diagnosis,
            "opportunity_score": round(
                current_impressions / max(current_position, 1),
                2,
            ),
            "member_queries": member_query_rows,
            "member_query_count": len(info["members"]),
        })

    return rows


def _weighted_position(
    group: pd.DataFrame,
    position_column: str,
    weight_column: str,
) -> float:
    positions = pd.to_numeric(group[position_column], errors="coerce").fillna(0)
    weights = pd.to_numeric(group[weight_column], errors="coerce").fillna(0)
    total_weight = float(weights.sum())

    if total_weight > 0:
        return float((positions * weights).sum() / total_weight)

    nonzero = positions[positions > 0]
    return float(nonzero.mean()) if not nonzero.empty else 0.0


def _diagnose(click_change, impression_change, ctr_change, position_change):
    if position_change > 1 and impression_change < 0 and ctr_change < 0:
        return "Ranking, visibility and CTR decline"
    if position_change > 1 and impression_change < 0:
        return "Ranking / visibility decline"
    if position_change > 1 and ctr_change < 0:
        return "Ranking and CTR decline"
    if impression_change < 0 and abs(position_change) <= 1:
        return "Impression decline with stable ranking"
    if ctr_change < 0 and impression_change >= 0 and abs(position_change) <= 1:
        return "CTR decline"
    if impression_change > 0 and click_change < 0:
        return "Visibility growth without click growth"
    return "Mixed / requires review"


def _number(value):
    value = float(value)
    if value.is_integer():
        return int(value)
    return round(value, 2)
