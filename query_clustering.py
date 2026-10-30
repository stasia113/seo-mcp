"""
Semantic query clustering for large GSC datasets.

Uses Sentence Transformers with the ONNX backend and AgglomerativeClustering
on a bounded set of high-value representative queries. This configuration
uses the same low-memory settings as the current remote server. All remaining queries
are embedded in small batches and assigned to the nearest representative
cluster. This preserves the preferred model and agglomerative clustering
quality while avoiding the O(n^2) memory requirement of running agglomerative
clustering across tens of thousands of queries.

Public interface preserved for analysis.py:
- build_cluster_map()
- aggregate_query_clusters()
"""

from __future__ import annotations

import gzip
import json
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SIMILARITY_THRESHOLD = 0.78
DEFAULT_EMBED_BATCH_SIZE = 16
DEFAULT_ASSIGN_BATCH_SIZE = 64
DEFAULT_MAX_ANCHORS = 300
DEFAULT_MAX_MEMBER_QUERIES = 10
CACHE_VERSION = "v7-scalable-anchor-agglomerative-lowmem"

CACHE_DIR = Path(__file__).parent / "cache" / "query_clustering"


def _normalize_query(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().lower().split())


def _display_query(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


def _normalize_model_name(model_name: str) -> str:
    if model_name == "all-MiniLM-L6-v2":
        return DEFAULT_MODEL
    return model_name


@lru_cache(maxsize=2)
def _cluster_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Semantic query clustering requires sentence-transformers. "
            "Install it with ONNX support."
        ) from exc

    return SentenceTransformer(
        _normalize_model_name(model_name),
        backend="onnx",
    )


def _cache_key(
    queries: list[str],
    model_name: str,
    similarity_threshold: float,
    max_anchors: int,
) -> str:
    payload = json.dumps(
        {
            "version": CACHE_VERSION,
            "queries": queries,
            "model": _normalize_model_name(model_name),
            "threshold": similarity_threshold,
            "max_anchors": max_anchors,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    return sha256(payload).hexdigest()[:24]


def _cache_paths(cache_key: str) -> tuple[Path, Path]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return (
        CACHE_DIR / f"{cache_key}.clusters.json.gz",
        CACHE_DIR / f"{cache_key}.anchors.npy",
    )


def _load_cached_clusters(path: Path) -> dict[str, dict] | None:
    if not path.exists():
        return None

    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)

        if payload.get("version") != CACHE_VERSION:
            return None

        assignments = payload.get("assignments", {})
        definitions = payload.get("clusters", {})

        info_by_id: dict[str, dict] = {}
        for cluster_id, definition in definitions.items():
            info_by_id[cluster_id] = {
                "cluster_id": cluster_id,
                "label": definition.get("label", ""),
                "members": definition.get("members", []),
            }

        result: dict[str, dict] = {}
        for query, cluster_id in assignments.items():
            info = info_by_id.get(cluster_id)
            if info is not None:
                result[query] = info

        return result
    except (OSError, ValueError, TypeError, EOFError):
        return None


def _save_cached_clusters(
    path: Path,
    clusters: dict[str, dict],
    queries: list[str],
    model_name: str,
    similarity_threshold: float,
    max_anchors: int,
) -> None:
    definitions: dict[str, dict] = {}
    assignments: dict[str, str] = {}

    for query, info in clusters.items():
        cluster_id = info["cluster_id"]
        assignments[query] = cluster_id

        if cluster_id not in definitions:
            definitions[cluster_id] = {
                "label": info.get("label", ""),
                "members": info.get("members", []),
            }

    payload = {
        "version": CACHE_VERSION,
        "model": _normalize_model_name(model_name),
        "similarity_threshold": similarity_threshold,
        "max_anchors": max_anchors,
        "query_count": len(queries),
        "assignments": assignments,
        "clusters": definitions,
    }

    temporary_path = path.with_name(path.name + ".tmp")

    with gzip.open(
        temporary_path,
        "wt",
        encoding="utf-8",
        compresslevel=6,
    ) as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    temporary_path.replace(path)


def _load_or_create_embeddings(
    queries: list[str],
    model_name: str,
    cache_key: str,
    cache_name: str,
) -> np.ndarray:
    _, embedding_path = _cache_paths(cache_key)
    embedding_path = embedding_path.with_name(cache_name)

    if embedding_path.exists():
        try:
            embeddings = np.load(
                embedding_path,
                allow_pickle=False,
            )
            if (
                embeddings.ndim == 2
                and embeddings.shape[0] == len(queries)
            ):
                return np.asarray(embeddings, dtype=np.float32)
        except (OSError, ValueError):
            pass

    model = _cluster_model(_normalize_model_name(model_name))

    embeddings = model.encode(
        queries,
        batch_size=DEFAULT_EMBED_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2 or embeddings.shape[0] != len(queries):
        raise ValueError(
            "Unexpected embedding shape: "
            f"{embeddings.shape}; expected ({len(queries)}, dimensions)."
        )

    np.save(
        embedding_path,
        embeddings,
        allow_pickle=False,
    )

    return embeddings


def _frame_query_scores(
    frame: pd.DataFrame,
    query_column: str,
) -> dict[str, float]:
    """Return an importance score based on clicks and impressions when present."""

    scores: dict[str, float] = {}

    metric_columns = []
    for column in frame.columns:
        name = str(column).lower()
        if "impression" in name or "click" in name:
            metric_columns.append(column)

    for _, row in frame[[query_column] + metric_columns].iterrows():
        query = _normalize_query(row[query_column])
        if not query:
            continue

        score = 0.0
        for column in metric_columns:
            value = pd.to_numeric(row[column], errors="coerce")
            if pd.isna(value):
                continue
            name = str(column).lower()
            if "impression" in name:
                score += float(value)
            elif "click" in name:
                score += float(value) * 10.0

        scores[query] = scores.get(query, 0.0) + score

    return scores


def _choose_anchors(
    queries: list[str],
    importance: dict[str, float],
    max_anchors: int,
) -> list[str]:
    if len(queries) <= max_anchors:
        return queries

    ranked = sorted(
        queries,
        key=lambda query: (
            -importance.get(query, 0.0),
            len(query.split()),
            len(query),
            query,
        ),
    )

    return ranked[:max_anchors]


def _choose_cluster_label(
    members: list[int],
    queries: list[str],
    embeddings: np.ndarray,
) -> str:
    if not members:
        return ""

    member_vectors = embeddings[members]
    centroid = member_vectors.mean(axis=0)
    norm = float(np.linalg.norm(centroid))

    if norm > 0:
        centroid = centroid / norm

    similarities = member_vectors @ centroid
    ranked = sorted(
        zip(members, similarities.tolist()),
        key=lambda item: (
            -float(item[1]),
            len(queries[item[0]].split()),
            len(queries[item[0]]),
            queries[item[0]],
        ),
    )

    top = ranked[: min(5, len(ranked))]
    best_index = min(
        top,
        key=lambda item: (
            len(queries[item[0]].split()),
            len(queries[item[0]]),
            -float(item[1]),
            queries[item[0]],
        ),
    )[0]

    return queries[best_index]


def _cluster_id(member_queries: list[str]) -> str:
    payload = "\n".join(sorted(member_queries)).encode("utf-8")
    return "cluster_" + sha256(payload).hexdigest()[:12]


def _build_communities(
    embeddings: np.ndarray,
    similarity_threshold: float,
) -> list[list[int]]:
    """Run the preferred average-linkage AgglomerativeClustering on anchors only."""

    try:
        from sklearn.cluster import AgglomerativeClustering
    except ImportError as exc:
        raise ImportError(
            "Semantic query clustering requires scikit-learn. "
            "Install it with: python -m pip install scikit-learn"
        ) from exc

    if len(embeddings) == 0:
        return []

    if len(embeddings) == 1:
        return [[0]]

    distance_threshold = 1.0 - float(similarity_threshold)

    try:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=distance_threshold,
        )
    except TypeError:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            affinity="cosine",
            linkage="average",
            distance_threshold=distance_threshold,
        )

    labels = clustering.fit_predict(embeddings)

    communities: dict[int, list[int]] = {}
    for index, label in enumerate(labels.tolist()):
        communities.setdefault(int(label), []).append(index)

    return list(communities.values())


def _assign_queries_to_anchor_clusters(
    queries: list[str],
    anchors: list[str],
    anchor_embeddings: np.ndarray,
    anchor_cluster_ids: np.ndarray,
    anchor_cluster_info: dict[str, dict],
    model_name: str,
    similarity_threshold: float,
) -> dict[str, dict]:
    """Embed non-anchor queries in batches and assign them to nearest anchors."""

    model = _cluster_model(_normalize_model_name(model_name))
    result: dict[str, dict] = {}

    anchor_matrix = np.asarray(anchor_embeddings, dtype=np.float32)
    anchor_cluster_ids = np.asarray(anchor_cluster_ids)

    anchor_set = set(anchors)

    for start in range(0, len(queries), DEFAULT_ASSIGN_BATCH_SIZE):
        batch_queries = queries[start:start + DEFAULT_ASSIGN_BATCH_SIZE]
        pending = [query for query in batch_queries if query not in anchor_set]

        if not pending:
            continue

        batch_embeddings = model.encode(
            pending,
            batch_size=DEFAULT_EMBED_BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        batch_embeddings = np.asarray(batch_embeddings, dtype=np.float32)

        similarities = batch_embeddings @ anchor_matrix.T
        best_indices = np.argmax(similarities, axis=1)
        best_scores = similarities[
            np.arange(len(pending)),
            best_indices,
        ]

        for query, best_index, score in zip(
            pending,
            best_indices.tolist(),
            best_scores.tolist(),
        ):
            if float(score) >= similarity_threshold:
                cluster_id = str(anchor_cluster_ids[best_index])
                result[query] = anchor_cluster_info[cluster_id]
            else:
                cluster_id = _cluster_id([query])
                result[query] = {
                    "cluster_id": cluster_id,
                    "label": query,
                    "members": [query],
                }

    return result


def build_cluster_map(
    query_frames: Iterable[pd.DataFrame],
    query_column: str,
    model_name: str = DEFAULT_MODEL,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> dict[str, dict]:
    """
    Build a scalable semantic cluster map.

    All queries remain eligible for aggregation. Only a bounded set of
    high-value representative queries is passed to AgglomerativeClustering.
    Remaining queries are embedded in batches and assigned to their nearest
    representative cluster when similarity meets the threshold.
    """

    records: dict[str, str] = {}
    importance: dict[str, float] = {}

    for frame in query_frames:
        if query_column not in frame.columns:
            raise ValueError(
                f"Query column '{query_column}' was not found. "
                f"Available columns: {frame.columns.tolist()}"
            )

        frame_scores = _frame_query_scores(
            frame,
            query_column,
        )
        importance.update({
            query: importance.get(query, 0.0) + score
            for query, score in frame_scores.items()
        })

        for value in frame[query_column].tolist():
            normalized = _normalize_query(value)
            display = _display_query(value)
            if normalized and normalized not in records:
                records[normalized] = display

    queries = sorted(records.keys())

    if not queries:
        return {}

    model_name = _normalize_model_name(model_name)
    max_anchors = DEFAULT_MAX_ANCHORS

    cache_key = _cache_key(
        queries,
        model_name,
        similarity_threshold,
        max_anchors,
    )
    cluster_path, _ = _cache_paths(cache_key)

    cached = _load_cached_clusters(cluster_path)
    if cached is not None:
        return cached

    anchors = _choose_anchors(
        queries,
        importance,
        max_anchors,
    )

    print(
        f"Semantic clustering: {len(queries):,} queries; "
        f"using {len(anchors):,} anchors with AgglomerativeClustering."
    )

    anchor_cache_key = sha256(
        "\n".join(anchors).encode("utf-8")
    ).hexdigest()[:24]

    anchor_embeddings = _load_or_create_embeddings(
        anchors,
        model_name,
        anchor_cache_key,
        f"{anchor_cache_key}.anchors.npy",
    )

    communities = _build_communities(
        anchor_embeddings,
        similarity_threshold,
    )

    anchor_cluster_ids = np.empty(
        len(anchors),
        dtype=object,
    )
    anchor_cluster_info: dict[str, dict] = {}

    for members in communities:
        member_queries = [anchors[index] for index in members]
        cluster_id = _cluster_id(member_queries)
        label = _choose_cluster_label(
            members,
            anchors,
            anchor_embeddings,
        )

        info = {
            "cluster_id": cluster_id,
            "label": label,
            "members": sorted(
                records[query]
                for query in member_queries
            ),
        }
        anchor_cluster_info[cluster_id] = info

        for index in members:
            anchor_cluster_ids[index] = cluster_id

    result: dict[str, dict] = {}

    for index, query in enumerate(anchors):
        result[query] = anchor_cluster_info[
            str(anchor_cluster_ids[index])
        ]

    assigned = _assign_queries_to_anchor_clusters(
        queries,
        anchors,
        anchor_embeddings,
        anchor_cluster_ids,
        anchor_cluster_info,
        model_name,
        similarity_threshold,
    )

    for query, info in assigned.items():
        result[query] = info

    # Rebuild member lists after assignment so every query is represented in
    # the cluster definition used by aggregation/reporting.
    definitions: dict[str, dict] = {}
    for query, info in result.items():
        cluster_id = info["cluster_id"]
        definition = definitions.setdefault(
            cluster_id,
            {
                "cluster_id": cluster_id,
                "label": info.get("label", query),
                "members": [],
            },
        )
        definition["members"].append(records[query])

    for query, info in result.items():
        definition = definitions[info["cluster_id"]]
        info["members"] = sorted(definition["members"])

    _save_cached_clusters(
        cluster_path,
        result,
        queries,
        model_name,
        similarity_threshold,
        max_anchors,
    )

    return result


def aggregate_query_clusters(
    df: pd.DataFrame,
    columns: dict,
    cluster_map: dict[str, dict],
    max_member_queries: int = DEFAULT_MAX_MEMBER_QUERIES,
) -> list[dict]:
    """
    Aggregate current/comparison GSC metrics by semantic cluster.

    CTR is recalculated from summed clicks / summed impressions.
    Position is impressions-weighted.
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
        work[column] = pd.to_numeric(
            work[column],
            errors="coerce",
        ).fillna(0)

    work["_cluster_id"] = work["_normalized_query"].map(
        lambda query: cluster_map[query]["cluster_id"]
    )

    rows = []

    for cluster_id, group in work.groupby("_cluster_id", sort=False):
        first_query = group["_normalized_query"].iloc[0]
        info = cluster_map[first_query]

        current_clicks = float(
            group[columns["current_clicks"]].sum()
        )
        comparison_clicks = float(
            group[columns["comparison_clicks"]].sum()
        )
        current_impressions = float(
            group[columns["current_impressions"]].sum()
        )
        comparison_impressions = float(
            group[columns["comparison_impressions"]].sum()
        )

        current_ctr = (
            current_clicks / current_impressions * 100
            if current_impressions > 0 else 0.0
        )
        comparison_ctr = (
            comparison_clicks / comparison_impressions * 100
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

        diagnosis = (
            _diagnose(
                click_change,
                impression_change,
                ctr_change,
                position_change,
            )
            if click_change < 0
            else "No click loss"
        )

        member_query_rows = []
        group = group.copy()
        group["_display_query"] = group[query_column].map(_display_query)
        group = group.sort_values(
            columns["current_clicks"],
            ascending=False,
        )

        for _, row in group.head(max_member_queries).iterrows():
            member_query_rows.append({
                "query": row["_display_query"],
                "clicks": _number(row[columns["current_clicks"]]),
                "impressions": _number(
                    row[columns["current_impressions"]]
                ),
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
    positions = pd.to_numeric(
        group[position_column],
        errors="coerce",
    ).fillna(0)
    weights = pd.to_numeric(
        group[weight_column],
        errors="coerce",
    ).fillna(0)
    total_weight = float(weights.sum())

    if total_weight > 0:
        return float(
            (positions * weights).sum() / total_weight
        )

    nonzero = positions[positions > 0]
    return float(nonzero.mean()) if not nonzero.empty else 0.0


def _diagnose(
    click_change,
    impression_change,
    ctr_change,
    position_change,
):
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
