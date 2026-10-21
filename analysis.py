import pandas as pd
from collections import Counter
from urllib.parse import urlparse

from gsc import (
    load_sheet,
    detect_gsc_columns,
    get_project_config,
    get_dataset_config,
    get_excel_file,
    get_commercial_exclusions,
    get_commercial_inclusions,
    get_commercial_excluded_contains,
)

from query_clustering import (
    build_cluster_map,
    aggregate_query_clusters,
)

def analyze_sheet_overview(
    sheet_name: str,
    min_impressions: int = 100,
    project: str = "fdp",
    dataset: str = "period"
) -> dict:
    """
    Create a high-level SEO overview for
    Queries or Pages.
    """

    columns = detect_gsc_columns(
        sheet_name,
        project=project,
        dataset=dataset
    )

    df = load_sheet(
        sheet_name,
        project=project,
        dataset=dataset
    ).copy()

    selected = [
        columns["key_column"],
        columns["current_clicks"],
        columns["comparison_clicks"],
        columns["current_impressions"],
        columns["comparison_impressions"],
        columns["current_ctr"],
        columns["comparison_ctr"],
        columns["current_position"],
        columns["comparison_position"]
    ]

    result = df[selected].copy()

    # Convert metrics to numbers
    for column in selected[1:]:

        if result[column].dtype == "object":
            result[column] = (
                result[column]
                .astype(str)
                .str.replace(
                    "%",
                    "",
                    regex=False
                )
            )

        result[column] = (
            pd.to_numeric(
                result[column],
                errors="coerce"
            )
        )

    # Calculate changes

    result["click_change"] = (
        result[columns["current_clicks"]]
        -
        result[columns["comparison_clicks"]]
    )

    result["impression_change"] = (
        result[columns["current_impressions"]]
        -
        result[columns["comparison_impressions"]]
    )

    result["ctr_change"] = (
        result[columns["current_ctr"]]
        -
        result[columns["comparison_ctr"]]
    )

    result["position_change"] = (
        result[columns["current_position"]]
        -
        result[columns["comparison_position"]]
    )

    # Totals

    current_clicks = (
        result[columns["current_clicks"]]
        .sum()
    )

    previous_clicks = (
        result[columns["comparison_clicks"]]
        .sum()
    )

    current_impressions = (
        result[columns["current_impressions"]]
        .sum()
    )

    previous_impressions = (
        result[columns["comparison_impressions"]]
        .sum()
    )

    return {
        "sheet": sheet_name,

        "rows_analyzed": len(result),

        "clicks": {
            "current": int(current_clicks),
            "previous": int(previous_clicks),
            "change": int(
                current_clicks
                -
                previous_clicks
            )
        },

        "impressions": {
            "current": int(current_impressions),
            "previous": int(previous_impressions),
            "change": int(
                current_impressions
                -
                previous_impressions
            )
        },

        "rows_losing_clicks": int(
            (
                result["click_change"]
                < 0
            ).sum()
        ),

        "rows_gaining_clicks": int(
            (
                result["click_change"]
                > 0
            ).sum()
        ),

        "ranking_declines": int(
            (
                result["position_change"]
                > 1
            ).sum()
        ),

        "ranking_improvements": int(
            (
                result["position_change"]
                < -1
            ).sum()
        )
    }

def find_click_losses(
    sheet_name: str,
    limit: int = 25,
    project: str = "fdp",
    dataset: str = "period"
) -> list[dict]:
    """
    Find the biggest click losses in Queries or Pages.
    """

    columns = detect_gsc_columns(
        sheet_name,
        project=project,
        dataset=dataset
    )

    df = load_sheet(
        sheet_name,
        project=project,
        dataset=dataset
    ).copy()

    selected = [
        columns["key_column"],
        columns["current_clicks"],
        columns["comparison_clicks"],
        columns["current_impressions"],
        columns["comparison_impressions"],
        columns["current_ctr"],
        columns["comparison_ctr"],
        columns["current_position"],
        columns["comparison_position"]
    ]

    result = df[selected].copy()


    # Convert metrics
    for column in selected[1:]:

        if result[column].dtype == "object":

            result[column] = (
                result[column]
                .astype(str)
                .str.replace(
                    "%",
                    "",
                    regex=False
                )
            )

        result[column] = pd.to_numeric(
            result[column],
            errors="coerce"
        )


    # Calculate changes

    result["click_change"] = (
        result[columns["current_clicks"]]
        -
        result[columns["comparison_clicks"]]
    )

    result["impression_change"] = (
        result[columns["current_impressions"]]
        -
        result[columns["comparison_impressions"]]
    )

    result["ctr_change"] = (
        result[columns["current_ctr"]]
        -
        result[columns["comparison_ctr"]]
    )

    result["position_change"] = (
        result[columns["current_position"]]
        -
        result[columns["comparison_position"]]
    )


    # Keep only losses

    result = result[
        result["click_change"] < 0
    ]


    # Sort largest losses first

    result = result.sort_values(
        "click_change",
        ascending=True
    )


    return result.head(
        limit
    ).to_dict(
        orient="records"
    )

def find_gsc_opportunities(
    sheet_name: str,
    min_impressions: int = 100,
    min_position: float = 4,
    max_position: float = 20,
    limit: int = 25,
    project: str = "fdp",
    dataset: str = "period"
) -> list[dict]:
    """
    Find SEO opportunities using the current GSC period.

    The workbook must contain comparison data:
    current period vs previous period.

    Opportunities are rows with:
    - at least min_impressions
    - average position between min_position and max_position

    The output also includes comparison metrics and changes.
    """

    columns = detect_gsc_columns(
        sheet_name,
        project=project,
        dataset=dataset
    )

    df = load_sheet(
        sheet_name,
        project=project,
        dataset=dataset
    ).copy()

    selected = [
        columns["key_column"],

        columns["current_clicks"],
        columns["comparison_clicks"],

        columns["current_impressions"],
        columns["comparison_impressions"],

        columns["current_ctr"],
        columns["comparison_ctr"],

        columns["current_position"],
        columns["comparison_position"]
    ]

    result = df[selected].copy()

    # Convert metrics to numbers

    for column in selected[1:]:

        if result[column].dtype == "object":

            result[column] = (
                result[column]
                .astype(str)
                .str.replace(
                    "%",
                    "",
                    regex=False
                )
            )

        result[column] = pd.to_numeric(
            result[column],
            errors="coerce"
        )

    # Calculate changes

    result["click_change"] = (
        result[columns["current_clicks"]]
        -
        result[columns["comparison_clicks"]]
    )

    result["impression_change"] = (
        result[columns["current_impressions"]]
        -
        result[columns["comparison_impressions"]]
    )

    result["ctr_change"] = (
        result[columns["current_ctr"]]
        -
        result[columns["comparison_ctr"]]
    )

    result["position_change"] = (
        result[columns["current_position"]]
        -
        result[columns["comparison_position"]]
    )

    # Filter opportunities

    result = result[
        (
            result[
                columns["current_impressions"]
            ]
            >= min_impressions
        )
        &
        (
            result[
                columns["current_position"]
            ]
            >= min_position
        )
        &
        (
            result[
                columns["current_position"]
            ]
            <= max_position
        )
    ].copy()

    # Opportunity score
    #
    # This is only a sorting aid:
    # more impressions + better ranking = higher score.

    result["opportunity_score"] = (
        result[
            columns["current_impressions"]
        ]
        /
        result[
            columns["current_position"]
        ].clip(lower=1)
    )

    result = result.sort_values(
        "opportunity_score",
        ascending=False
    )

    return result.head(
        limit
    ).to_dict(
        orient="records"
    )

def diagnose_performance_losses(
    sheet_name: str,
    limit: int = 25,
    project: str = "fdp",
    dataset: str = "period"
) -> list[dict]:
    """
    Diagnose the largest click losses using current
    vs previous GSC comparison data.

    Positive position change means rankings became worse.
    Negative position change means rankings improved.
    """

    columns = detect_gsc_columns(
        sheet_name,
        project=project,
        dataset=dataset
    )

    df = load_sheet(
        sheet_name,
        project=project,
        dataset=dataset
    ).copy()

    selected = [
        columns["key_column"],

        columns["current_clicks"],
        columns["comparison_clicks"],

        columns["current_impressions"],
        columns["comparison_impressions"],

        columns["current_ctr"],
        columns["comparison_ctr"],

        columns["current_position"],
        columns["comparison_position"]
    ]

    result = df[selected].copy()

    # Convert metrics to numbers

    for column in selected[1:]:

        if result[column].dtype == "object":

            result[column] = (
                result[column]
                .astype(str)
                .str.replace(
                    "%",
                    "",
                    regex=False
                )
            )

        result[column] = pd.to_numeric(
            result[column],
            errors="coerce"
        )

    # Calculate changes

    result["click_change"] = (
        result[columns["current_clicks"]]
        -
        result[columns["comparison_clicks"]]
    )

    result["impression_change"] = (
        result[columns["current_impressions"]]
        -
        result[columns["comparison_impressions"]]
    )

    result["ctr_change"] = (
        result[columns["current_ctr"]]
        -
        result[columns["comparison_ctr"]]
    )

    result["position_change"] = (
        result[columns["current_position"]]
        -
        result[columns["comparison_position"]]
    )

    # Keep only rows that lost clicks

    result = result[
        result["click_change"] < 0
    ].copy()

    def diagnose(row):

        click_change = row["click_change"]
        impression_change = row["impression_change"]
        ctr_change = row["ctr_change"]
        position_change = row["position_change"]

        if pd.isna(click_change):
            return "Insufficient data"

        # Ranking, visibility and CTR all declined

        if (
            position_change > 1
            and impression_change < 0
            and ctr_change < 0
        ):
            return (
                "Ranking, visibility and CTR decline"
            )

        # Ranking became materially worse
        # and impressions also declined

        if (
            position_change > 1
            and impression_change < 0
        ):
            return (
                "Ranking / visibility decline"
            )

        # Ranking became materially worse
        # and CTR also declined

        if (
            position_change > 1
            and ctr_change < 0
        ):
            return (
                "Ranking and CTR decline"
            )

        # Impressions declined while ranking
        # remained relatively stable

        if (
            impression_change < 0
            and abs(position_change) <= 1
        ):
            return (
                "Impression decline with stable ranking"
            )

        # CTR declined while impressions and
        # rankings remained relatively stable

        if (
            ctr_change < 0
            and impression_change >= 0
            and abs(position_change) <= 1
        ):
            return "CTR decline"

        # Visibility increased but clicks did not

        if (
            impression_change > 0
            and click_change < 0
        ):
            return (
                "Visibility growth without click growth"
            )

        return "Mixed / requires review"

    result["diagnosis"] = result.apply(
        diagnose,
        axis=1
    )

    # Largest click losses first

    result = result.sort_values(
        "click_change",
        ascending=True
    )

    return result.head(
        limit
    ).to_dict(
        orient="records"
    )

def diagnose_commercial_page_losses(
    limit: int = 25,
    project: str = "fdp",
    dataset: str = "period"
) -> list[dict]:
    """
    Diagnose click losses for commercial pages only.

    Excludes pages under:
    /blog/
    /resources/
    /case-study/
    /images/
    /videos/
    /symptoms/
    /recent-projects/
    /glossary/
    """

    sheet_name = "Pages"

    columns = detect_gsc_columns(
        sheet_name,
        project=project,
        dataset=dataset
    )

    df = load_sheet(
        sheet_name,
        project=project,
        dataset=dataset
    ).copy()

    page_column = columns["key_column"]

    excluded_sections = set(get_commercial_exclusions(project))
    include_prefixes = [
        prefix.lower()
        for prefix in get_commercial_inclusions(project)
    ]
    excluded_contains = [
        value.lower()
        for value in get_commercial_excluded_contains(project)
    ]

    def is_commercial_page(value):
        if pd.isna(value):
            return False

        url = str(value).strip()

        if not url:
            return False

        # Explicit literal exclusions such as URL fragments.
        url_lower = url.lower()
        if any(
            token in url_lower
            for token in excluded_contains
        ):
            return False

        path = urlparse(url).path.lower()

        # Some projects use an explicit commercial-page allowlist.
        # When present, a page must be nested under one of these prefixes.
        if include_prefixes:
            return any(
                path.startswith(prefix)
                for prefix in include_prefixes
            )

        path_segments = {
            segment
            for segment in path.split("/")
            if segment
        }

        return not bool(
            path_segments.intersection(
                excluded_sections
            )
        )

    # Keep commercial pages only

    df = df[
        df[page_column].apply(
            is_commercial_page
        )
    ].copy()

    selected = [
        columns["key_column"],

        columns["current_clicks"],
        columns["comparison_clicks"],

        columns["current_impressions"],
        columns["comparison_impressions"],

        columns["current_ctr"],
        columns["comparison_ctr"],

        columns["current_position"],
        columns["comparison_position"]
    ]

    result = df[selected].copy()

    # Convert metrics to numbers

    for column in selected[1:]:

        if result[column].dtype == "object":

            result[column] = (
                result[column]
                .astype(str)
                .str.replace(
                    "%",
                    "",
                    regex=False
                )
            )

        result[column] = pd.to_numeric(
            result[column],
            errors="coerce"
        )

    # Calculate changes

    result["click_change"] = (
        result[columns["current_clicks"]]
        -
        result[columns["comparison_clicks"]]
    )

    result["impression_change"] = (
        result[columns["current_impressions"]]
        -
        result[columns["comparison_impressions"]]
    )

    result["ctr_change"] = (
        result[columns["current_ctr"]]
        -
        result[columns["comparison_ctr"]]
    )

    result["position_change"] = (
        result[columns["current_position"]]
        -
        result[columns["comparison_position"]]
    )

    # Keep only pages that lost clicks

    result = result[
        result["click_change"] < 0
    ].copy()

    def diagnose(row):

        impression_change = row["impression_change"]
        ctr_change = row["ctr_change"]
        position_change = row["position_change"]

        if (
            position_change > 1
            and impression_change < 0
            and ctr_change < 0
        ):
            return (
                "Ranking, visibility and CTR decline"
            )

        if (
            position_change > 1
            and impression_change < 0
        ):
            return (
                "Ranking / visibility decline"
            )

        if (
            position_change > 1
            and ctr_change < 0
        ):
            return (
                "Ranking and CTR decline"
            )

        if (
            impression_change < 0
            and abs(position_change) <= 1
        ):
            return (
                "Impression decline with stable ranking"
            )

        if (
            ctr_change < 0
            and impression_change >= 0
            and abs(position_change) <= 1
        ):
            return "CTR decline"

        if (
            impression_change > 0
            and row["click_change"] < 0
        ):
            return (
                "Visibility growth without click growth"
            )

        return "Mixed / requires review"

    result["diagnosis"] = result.apply(
        diagnose,
        axis=1
    )

    # Largest click losses first

    result = result.sort_values(
        "click_change",
        ascending=True
    )

    return result.head(
        limit
    ).to_dict(
        orient="records"
    )


def _clean_number(value, decimals=2):
    """
    Convert pandas/numpy numeric values into
    clean Python numbers for MCP output.
    """

    if pd.isna(value):
        return None

    value = float(value)

    if decimals == 0:
        return int(round(value))

    return round(value, decimals)


def _compact_row(
    row: dict,
    columns: dict,
    include_diagnosis: bool = False,
    include_score: bool = False
) -> dict:
    """
    Convert a raw analysis row into a compact,
    LLM-friendly structure.
    """

    result = {
        "target": row.get(
            columns["key_column"]
        ),

        "current": {
            "clicks": _clean_number(
                row.get(columns["current_clicks"]),
                0
            ),
            "impressions": _clean_number(
                row.get(columns["current_impressions"]),
                0
            ),
            "ctr": _clean_number(
                row.get(columns["current_ctr"]),
                4
            ),
            "position": _clean_number(
                row.get(columns["current_position"]),
                2
            )
        },

        "previous": {
            "clicks": _clean_number(
                row.get(columns["comparison_clicks"]),
                0
            ),
            "impressions": _clean_number(
                row.get(columns["comparison_impressions"]),
                0
            ),
            "ctr": _clean_number(
                row.get(columns["comparison_ctr"]),
                4
            ),
            "position": _clean_number(
                row.get(columns["comparison_position"]),
                2
            )
        },

        "change": {
            "clicks": _clean_number(
                row.get("click_change"),
                0
            ),
            "impressions": _clean_number(
                row.get("impression_change"),
                0
            ),
            "ctr": _clean_number(
                row.get("ctr_change"),
                4
            ),
            "position": _clean_number(
                row.get("position_change"),
                2
            )
        }
    }

    if include_diagnosis:
        result["diagnosis"] = row.get(
            "diagnosis"
        )

    if include_score:
        result["opportunity_score"] = _clean_number(
            row.get("opportunity_score"),
            2
        )

    return result


def build_semantic_query_clusters(
    project: str = "fdp",
    dataset: str = "period",
    similarity_threshold: float = 0.78,
    model_name: str = "all-MiniLM-L6-v2",
) -> list[dict]:
    """
    Build semantic query clusters using the union of available period and YoY
    query lists. The same cluster map is therefore reused for both comparisons.
    """

    frames = []

    for dataset_name in ("period", "yoy"):
        try:
            frame = load_sheet(
                "Queries",
                project=project,
                dataset=dataset_name,
            )
            frames.append(frame)
        except FileNotFoundError:
            continue

    if not frames:
        return []

    columns = detect_gsc_columns(
        "Queries",
        project=project,
        dataset=dataset,
    )

    cluster_map = build_cluster_map(
        frames,
        columns["key_column"],
        model_name=model_name,
        similarity_threshold=similarity_threshold,
    )

    current_df = load_sheet(
        "Queries",
        project=project,
        dataset=dataset,
    ).copy()

    clusters = aggregate_query_clusters(
        current_df,
        columns,
        cluster_map,
    )

    return sorted(
        clusters,
        key=lambda row: row["current"]["impressions"],
        reverse=True,
    )

def seo_summary(
    min_impressions: int = 100,
    opportunity_limit: int = 10,
    loss_limit: int = 10,
    project: str = "fdp",
    dataset: str = "period",
    cluster_queries: bool = True,
    cluster_similarity_threshold: float = 0.78,
    cluster_model: str = "all-MiniLM-L6-v2",
    cluster_limit: int = 50,
) -> dict:
    """
    Create a compact consolidated SEO analysis
    from the GSC comparison export.

    Analyzes:

    - Queries
    - All Pages
    - Commercial Pages

    The workbook must contain current-period
    and previous-period GSC metrics.
    """

    # =========================================================
    # QUERY ANALYSIS
    # =========================================================

    query_columns = detect_gsc_columns(
        "Queries",
        project=project,
        dataset=dataset
    )

    query_overview = analyze_sheet_overview(
        "Queries",
        project=project,
        dataset=dataset
    )

    query_all_losses = diagnose_performance_losses(
        "Queries",
        limit=1_000_000,
        project=project,
        dataset=dataset
    )

    query_losses = [
        _compact_row(
            row,
            query_columns,
            include_diagnosis=True
        )
        for row in query_all_losses[:loss_limit]
    ]

    query_diagnosis_counts = dict(
        Counter(
            row["diagnosis"]
            for row in query_all_losses
        )
    )

    query_opportunity_rows = find_gsc_opportunities(
        "Queries",
        project=project,
        dataset=dataset,
        min_impressions=min_impressions,
        min_position=4,
        max_position=20,
        limit=opportunity_limit
    )

    query_opportunities = [
        _compact_row(
            row,
            query_columns,
            include_score=True
        )
        for row in query_opportunity_rows
    ]


    # =========================================================
    # PAGE ANALYSIS
    # =========================================================

    page_columns = detect_gsc_columns(
        "Pages",
        project=project,
        dataset=dataset
    )

    page_overview = analyze_sheet_overview(
        "Pages",
        project=project,
        dataset=dataset
    )

    page_all_losses = diagnose_performance_losses(
        "Pages",
        limit=1_000_000,
        project=project,
        dataset=dataset
    )

    page_losses = [
        _compact_row(
            row,
            page_columns,
            include_diagnosis=True
        )
        for row in page_all_losses[:loss_limit]
    ]

    page_diagnosis_counts = dict(
        Counter(
            row["diagnosis"]
            for row in page_all_losses
        )
    )

    page_opportunity_rows = find_gsc_opportunities(
        "Pages",
        project=project,
        dataset=dataset,
        min_impressions=min_impressions,
        min_position=4,
        max_position=20,
        limit=opportunity_limit
    )

    page_opportunities = [
        _compact_row(
            row,
            page_columns,
            include_score=True
        )
        for row in page_opportunity_rows
    ]


    # =========================================================
    # COMMERCIAL PAGE ANALYSIS
    # =========================================================

    commercial_all_losses = (
        diagnose_commercial_page_losses(
            limit=1_000_000,
            project=project,
            dataset=dataset
        )
    )

    commercial_losses = [
        _compact_row(
            row,
            page_columns,
            include_diagnosis=True
        )
        for row in commercial_all_losses[:loss_limit]
    ]

    commercial_diagnosis_counts = dict(
        Counter(
            row["diagnosis"]
            for row in commercial_all_losses
        )
    )


    # =========================================================
    # SEMANTIC QUERY CLUSTERS
    # =========================================================

    semantic_query_clusters = []

    if cluster_queries:
        semantic_query_clusters = build_semantic_query_clusters(
            project=project,
            dataset=dataset,
            similarity_threshold=cluster_similarity_threshold,
            model_name=cluster_model,
        )

    semantic_cluster_diagnosis_counts = dict(
        Counter(
            row["diagnosis"]
            for row in semantic_query_clusters
            if row.get("diagnosis")
        )
    )

    semantic_cluster_losses = [
        row for row in semantic_query_clusters
        if row["change"]["clicks"] < 0
    ]

    semantic_cluster_opportunities = [
        row for row in semantic_query_clusters
        if row["current"]["impressions"] >= min_impressions
        and 4 <= row["current"]["position"] <= 20
    ]

    semantic_cluster_losses.sort(
        key=lambda row: row.get("change", {}).get("clicks", 0)
    )

    semantic_cluster_opportunities.sort(
        key=lambda row: row.get("opportunity_score", 0),
        reverse=True,
    )

    semantic_cluster_loss_top = semantic_cluster_losses[:loss_limit]
    semantic_cluster_opportunity_top = semantic_cluster_opportunities[:opportunity_limit]
    semantic_cluster_top = semantic_query_clusters[:cluster_limit]
    semantic_cluster_ranking_decline_count = sum(
        1
        for row in semantic_query_clusters
        if row.get("change", {}).get("position", 0) > 1
    )

    # =========================================================
    # EXECUTIVE FINDINGS
    # =========================================================

    findings = []
    comparison_label = get_dataset_config(project, dataset)["label"]
    comparison_reference = (
        "the previous 3 months"
        if dataset == "period"
        else "the same period last year"
    )

    query_click_change = (
        query_overview["clicks"]["change"]
    )

    page_click_change = (
        page_overview["clicks"]["change"]
    )


    if query_click_change < 0:

        findings.append(
            f"Query clicks declined by "
            f"{abs(query_click_change):,} "
            f"compared with {comparison_reference}."
        )

    elif query_click_change > 0:

        findings.append(
            f"Query clicks increased by "
            f"{query_click_change:,} "
            f"compared with {comparison_reference}."
        )


    if page_click_change < 0:

        findings.append(
            f"Page clicks declined by "
            f"{abs(page_click_change):,} "
            f"compared with {comparison_reference}."
        )

    elif page_click_change > 0:

        findings.append(
            f"Page clicks increased by "
            f"{page_click_change:,} "
            f"compared with {comparison_reference}."
        )


    if query_diagnosis_counts:

        dominant_query_diagnosis = max(
            query_diagnosis_counts,
            key=query_diagnosis_counts.get
        )

        findings.append(
            "The most frequent query-loss diagnosis "
            f"is '{dominant_query_diagnosis}'."
        )


    if page_diagnosis_counts:

        dominant_page_diagnosis = max(
            page_diagnosis_counts,
            key=page_diagnosis_counts.get
        )

        findings.append(
            "The most frequent page-loss diagnosis "
            f"is '{dominant_page_diagnosis}'."
        )


    findings.append(
        f"{len(commercial_all_losses)} commercial pages "
        "lost clicks compared with the previous period."
    )


    # =========================================================
    # COMPACT OUTPUT
    # =========================================================

    return {

        "analysis": {
            "source": get_excel_file(project, dataset).name,
            "project": project,
            "brand": get_project_config(project)["brand"],
            "dataset": dataset,
            "comparison": get_dataset_config(project, dataset)["label"]
        },


        "executive_findings": findings,


        "queries": {

            "overview": query_overview,

            "diagnosis_counts": (
                query_diagnosis_counts
            ),

            "top_click_losses": query_losses,

            "top_opportunities": (
                query_opportunities
            ),

            "semantic_clusters_enabled": cluster_queries,
            "semantic_cluster_count": len(semantic_query_clusters),
            "semantic_cluster_loss_count": len(semantic_cluster_losses),
            "semantic_cluster_diagnosis_counts": semantic_cluster_diagnosis_counts,
            "semantic_cluster_opportunity_count": len(semantic_cluster_opportunities),
            "semantic_cluster_ranking_decline_count": semantic_cluster_ranking_decline_count,
            "semantic_cluster_losses": semantic_cluster_loss_top,
            "semantic_cluster_opportunities": semantic_cluster_opportunity_top,
            "semantic_clusters": semantic_cluster_top,
        },


        "pages": {

            "overview": page_overview,

            "diagnosis_counts": (
                page_diagnosis_counts
            ),

            "top_click_losses": page_losses,

            "top_opportunities": (
                page_opportunities
            )
        },


        "commercial_pages": {

            "excluded_sections": [
                f"/{section}/"
                for section in get_commercial_exclusions(project)
            ],

            "included_prefixes": get_commercial_inclusions(project),

            "excluded_contains": get_commercial_excluded_contains(project),

            "loss_count": len(
                commercial_all_losses
            ),

            "diagnosis_counts": (
                commercial_diagnosis_counts
            ),

            "top_click_losses": (
                commercial_losses
            )
        }
    }