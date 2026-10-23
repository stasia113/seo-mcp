"""
Fast semantic query clustering for GSC analysis.

This module is designed for the SEO MCP server. It uses Sentence Transformers
for embeddings and Sentence Transformers' fast community detection instead of
sklearn AgglomerativeClustering.

The expensive embedding and clustering results are cached on disk. The cache
key includes the query universe, model, and similarity threshold, so period and
YoY datasets can share one cluster map when they contain the same query universe.
"""

from __future__ import annotations

import json
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_SIMILARITY_THRESHOLD = 0.78
DEFAULT_MIN_COMMUNITY_SIZE = 2
DEFAULT_EMBED_BATCH_SIZE = 64
DEFAULT_COMMUNITY_BATCH_SIZE = 1024
CACHE_VERSION = "v2-fast-community"

CACHE_DIR = Path(__file__).parent / "cache" / "query_clustering"


def _normalize_query(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().lower().split())


def _display_query(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


@lru_cache(maxsize=2)
def _cluster_model(model_name: str):
    """Load the embedding model once per server process."""

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Semantic query clustering requires sentence-transformers. "
            "Install it with: pip install sentence-transformers"
        ) from exc

    return SentenceTransformer(model_name)


def _cache_key(
    queries: list[str],
    model_name: str,
    similarity_threshold: float,
) -> str:
    payload = json.dumps(
        {
            "version": CACHE_VERSION,
            "queries": queries,
            "model": model_name,
            "threshold": similarity_threshold,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    return sha256(payload).hexdigest()[:24]


def _cache_paths(cache_key: str) -> tuple[Path, Path]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return (
        CACHE_DIR / f"{cache_key}.clusters.json",
        CACHE_DIR / f"{cache_key}.embeddings.npy",
    )


def _load_cached_clusters(path: Path) -> dict[str, dict] | None:
    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != CACHE_VERSION:
            return None
        return payload.get("clusters")
    except (OSError, ValueError, TypeError):
        return None


def _save_cached_clusters(
    path: Path,
    clusters: dict[str, dict],
    queries: list[str],
    model_name: str,
    similarity_threshold: float,
) -> None:
    payload = {
        "version": CACHE_VERSION,
        "model": model_name,
        "similarity_threshold": similarity_threshold,
        "query_count": len(queries),
        "clusters": clusters,
    }

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_or_create_embeddings(
    queries: list[str],
    model_name: str,
    cache_key: str,
) -> np.ndarray:
    _, embedding_path = _cache_paths(cache_key)

    if embedding_path.exists():
        try:
            embeddings = np.load(embedding_path, allow_pickle=False)
            if embeddings.shape[0] == len(queries):
                return embeddings
        except (OSError, ValueError):
            pass

    model = _cluster_model(model_name)

    embeddings = model.encode(
        [queries[i] for i in range(len(queries))],
        batch_size=DEFAULT_EMBED_BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)
    np.save(embedding_path, embeddings, allow_pickle=False)
    return embeddings


def _choose_cluster_label(
    members: list[int],
    queries: list[str],
) -> str:
    """Choose a readable label from a community's central/concise queries."""

    if not members:
        return ""

    # community_detection returns its central point first. Prefer that point,
    # unless a clearly shorter query is available among the first few members.
    candidates = [queries[index] for index in members[:5]]
    central = candidates[0]

    shortest = min(
        candidates,
        key=lambda value: (
            len(value.split()),
            len(value),
            value,
        ),
    )

    # Avoid replacing a useful central label with an ultra-short generic term.
    if len(shortest.split()) >= 2 and len(shortest) <= len(central) * 0.8:
        return shortest

    return central


def _cluster_id(member_queries: list[str]) -> str:
    payload = "\n".join(sorted(member_queries)).encode("utf-8")
    return "cluster_" + sha256(payload).hexdigest()[:12]


def _build_communities(
    embeddings: np.ndarray,
    queries: list[str],
    similarity_threshold: float,
) -> list[list[int]]:
    """Run Sentence Transformers' fast community detection."""

    try:
        from sentence_transformers.util import community_detection
    except ImportError as exc:
        raise ImportError(
            "The installed sentence-transformers version does not expose "
            "community_detection. Upgrade sentence-transformers."
        ) from exc

    communities = community_detection(
        embeddings,
        threshold=similarity_threshold,
        min_community_size=DEFAULT_MIN_COMMUNITY_SIZE,
        batch_size=DEFAULT_COMMUNITY_BATCH_SIZE,
        show_progress_bar=False,
    )

    # Communities can overlap. Assign a query to the first community that
    # contains it so every query receives exactly one cluster.
    assigned: set[int] = set()
    unique_communities: list[list[int]] = []

    for community in communities:
        members = [int(index) for index in community if int(index) not in assigned]
        if not members:
            continue

        assigned.update(members)
        unique_communities.append(members)

    # Queries not placed in a multi-query community remain singleton clusters.
    for index in range(len(queries)):
        if index not in assigned:
            unique_communities.append([index])

    return unique_communities


def build_cluster_map(
    query_frames: Iterable[pd.DataFrame],
    query_column: str,
    model_name: str = DEFAULT_MODEL,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> dict[str, dict]:
    """
    Build a semantic cluster map from one or more query datasets.

    The function preserves the original public interface used by analysis.py.
    The query universe is deduplicated and sorted before caching, which makes
    the result independent of workbook row order.
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

    cache_key = _cache_key(
        queries,
        model_name,
        similarity_threshold,
    )
    cluster_path, _ = _cache_paths(cache_key)

    cached = _load_cached_clusters(cluster_path)
    if cached is not None:
        return cached

    embeddings = _load_or_create_embeddings(
        queries,
        model_name,
        cache_key,
    )

    communities = _build_communities(
        embeddings,
        queries,
        similarity_threshold,
    )

    result: dict[str, dict] = {}

    for members in communities:
        member_queries = [queries[index] for index in members]
        cluster_id = _cluster_id(member_queries)
        label = _choose_cluster_label(members, queries)
        display_members = [records[query] for query in member_queries]

        info = {
            "cluster_id": cluster_id,
            "label": label,
            "members": sorted(display_members),
        }

        for query in member_queries:
            result[query] = info

    _save_cached_clusters(
        cluster_path,
        result,
        queries,
        model_name,
        similarity_threshold,
    )

    return result


def aggregate_query_clusters(
    df: pd.DataFrame,
    columns: dict,
    cluster_map: dict[str, dict],
    max_member_queries: int = 10,
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