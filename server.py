from typing import Literal

from mcp.server.mcpserver import MCPServer


mcp = MCPServer("SEO MCP")


# ---------------------------------------------------------------------------
# Inspector dropdown choices
# ---------------------------------------------------------------------------

SingleProjectChoice = Literal[
    "FDP Mold Remediation",
    "O2 Mold Testing",
    "USA Clean Master",
]

SummaryProjectChoice = Literal[
    "FDP Mold Remediation",
    "O2 Mold Testing",
    "USA Clean Master",
    "All projects",
]

DatasetChoice = Literal[
    "Last 3 months vs previous 3 months",
    "Last 3 months vs same period last year",
]

SummaryDatasetChoice = Literal[
    "Last 3 months vs previous 3 months",
    "Last 3 months vs same period last year",
    "Both comparisons",
]

SheetChoice = Literal[
    "Queries",
    "Pages",
]


PROJECT_MAP = {
    "FDP Mold Remediation": "fdp",
    "O2 Mold Testing": "o2",
    "USA Clean Master": "usa_clean_master",
    "All projects": "all",
}


DATASET_MAP = {
    "Last 3 months vs previous 3 months": "period",
    "Last 3 months vs same period last year": "yoy",
    "Both comparisons": "both",
}


# ---------------------------------------------------------------------------
# Detailed analysis tools
#
# Heavy modules are imported inside the tool functions rather than when the
# MCP server starts. This keeps Inspector initialization fast.
# ---------------------------------------------------------------------------

@mcp.tool()
def click_losses(
    sheet_name: SheetChoice,
    project: SingleProjectChoice = "FDP Mold Remediation",
    dataset: DatasetChoice = "Last 3 months vs previous 3 months",
    limit: int = 25,
) -> list[dict]:
    """Find the largest click losses for Queries or Pages."""

    from analysis import find_click_losses

    return find_click_losses(
        sheet_name,
        limit=limit,
        project=PROJECT_MAP[project],
        dataset=DATASET_MAP[dataset],
    )


@mcp.tool()
def gsc_opportunities(
    sheet_name: SheetChoice,
    project: SingleProjectChoice = "FDP Mold Remediation",
    dataset: DatasetChoice = "Last 3 months vs previous 3 months",
    min_impressions: int = 100,
    min_position: float = 4,
    max_position: float = 20,
    limit: int = 25,
) -> list[dict]:
    """Find SEO opportunities in comparison-based GSC data."""

    from analysis import find_gsc_opportunities

    return find_gsc_opportunities(
        sheet_name,
        min_impressions=min_impressions,
        min_position=min_position,
        max_position=max_position,
        limit=limit,
        project=PROJECT_MAP[project],
        dataset=DATASET_MAP[dataset],
    )


@mcp.tool()
def performance_loss_diagnosis(
    sheet_name: SheetChoice,
    project: SingleProjectChoice = "FDP Mold Remediation",
    dataset: DatasetChoice = "Last 3 months vs previous 3 months",
    limit: int = 25,
) -> list[dict]:
    """Diagnose the largest click losses for Queries or Pages."""

    from analysis import diagnose_performance_losses

    return diagnose_performance_losses(
        sheet_name,
        limit=limit,
        project=PROJECT_MAP[project],
        dataset=DATASET_MAP[dataset],
    )


@mcp.tool()
def commercial_page_performance_loss_diagnosis(
    project: SingleProjectChoice = "FDP Mold Remediation",
    dataset: DatasetChoice = "Last 3 months vs previous 3 months",
    limit: int = 25,
) -> list[dict]:
    """Diagnose click losses among commercial pages only."""

    from analysis import diagnose_commercial_page_losses

    return diagnose_commercial_page_losses(
        limit=limit,
        project=PROJECT_MAP[project],
        dataset=DATASET_MAP[dataset],
    )


# ---------------------------------------------------------------------------
# Main SEO report tool
# ---------------------------------------------------------------------------

@mcp.tool()
def seo_summary_tool(
    project: SummaryProjectChoice = "FDP Mold Remediation",
    dataset: SummaryDatasetChoice = "Both comparisons",
    min_impressions: int = 100,
    opportunity_limit: int = 10,
    loss_limit: int = 10,
    cluster_queries: bool = True,
    cluster_similarity_threshold: float = 0.78,
    cluster_model: str = "all-MiniLM-L6-v2",
    cluster_limit: int = 50,
) -> dict:
    """
    Analyze GSC data and generate the SEO HTML report.

    Project and dataset are presented as dropdowns in MCP Inspector.

    Dataset options:
    - Last 3 months vs previous 3 months
    - Last 3 months vs same period last year
    - Both comparisons

    The HTML report uses semantic query clusters for query-related tables.
    """

    project_key = PROJECT_MAP[project]
    dataset_key = DATASET_MAP[dataset]

    if project_key == "all":
        project_names = (
            "fdp",
            "o2",
            "usa_clean_master",
        )

        results = {}
        missing = {}

        for project_name in project_names:
            try:
                results[project_name] = _run_project(
                    project_name,
                    dataset_key,
                    min_impressions,
                    opportunity_limit,
                    loss_limit,
                    cluster_queries,
                    cluster_similarity_threshold,
                    cluster_model,
                    cluster_limit,
                )
            except FileNotFoundError as exc:
                missing[project_name] = str(exc)

        return {
            "project": "all",
            "dataset": dataset_key,
            "reports": results,
            "missing": missing,
        }

    return _run_project(
        project_key,
        dataset_key,
        min_impressions,
        opportunity_limit,
        loss_limit,
        cluster_queries,
        cluster_similarity_threshold,
        cluster_model,
        cluster_limit,
    )


def _run_project(
    project: str,
    dataset: str,
    min_impressions: int,
    opportunity_limit: int,
    loss_limit: int,
    cluster_queries: bool,
    cluster_similarity_threshold: float,
    cluster_model: str,
    cluster_limit: int,
) -> dict:
    """Run one project for one or both comparison datasets."""

    # Keep GSC/pandas imports out of MCP startup.
    from gsc import get_excel_file, get_project_config
    from html_report import (
        build_html_report,
        build_combined_html_report,
    )

    period_path = get_excel_file(project, "period")
    yoy_path = get_excel_file(project, "yoy")

    if dataset == "period":
        if not period_path.exists():
            raise FileNotFoundError(
                f"Period workbook not found: {period_path}"
            )

        summary = _run_summary(
            project,
            "period",
            min_impressions,
            opportunity_limit,
            loss_limit,
            cluster_queries,
            cluster_similarity_threshold,
            cluster_model,
            cluster_limit,
        )

        report_path = build_html_report(summary)
        summary["report"] = {
            "html_file": report_path.name,
            "html_path": str(report_path),
        }
        return summary

    if dataset == "yoy":
        if not yoy_path.exists():
            raise FileNotFoundError(
                f"YoY workbook not found: {yoy_path}"
            )

        summary = _run_summary(
            project,
            "yoy",
            min_impressions,
            opportunity_limit,
            loss_limit,
            cluster_queries,
            cluster_similarity_threshold,
            cluster_model,
            cluster_limit,
        )

        report_path = build_html_report(summary)
        summary["report"] = {
            "html_file": report_path.name,
            "html_path": str(report_path),
        }
        return summary

    if dataset == "both":
        if not period_path.exists() or not yoy_path.exists():
            missing = []
            if not period_path.exists():
                missing.append(f"period: {period_path}")
            if not yoy_path.exists():
                missing.append(f"yoy: {yoy_path}")

            raise FileNotFoundError(
                f"Both datasets are required for project='{project}'. "
                + "; ".join(missing)
            )

        period_summary = _run_summary(
            project,
            "period",
            min_impressions,
            opportunity_limit,
            loss_limit,
            cluster_queries,
            cluster_similarity_threshold,
            cluster_model,
            cluster_limit,
        )

        yoy_summary = _run_summary(
            project,
            "yoy",
            min_impressions,
            opportunity_limit,
            loss_limit,
            cluster_queries,
            cluster_similarity_threshold,
            cluster_model,
            cluster_limit,
        )

        report_path = build_combined_html_report(
            period_summary,
            yoy_summary,
        )

        return {
            "project": project,
            "brand": get_project_config(project)["brand"],
            "dataset": "both",
            "period": period_summary,
            "yoy": yoy_summary,
            "report": {
                "html_file": report_path.name,
                "html_path": str(report_path),
            },
        }

    raise ValueError(
        "dataset must be 'period', 'yoy', or 'both'."
    )


def _run_summary(
    project: str,
    dataset: str,
    min_impressions: int,
    opportunity_limit: int,
    loss_limit: int,
    cluster_queries: bool,
    cluster_similarity_threshold: float,
    cluster_model: str,
    cluster_limit: int,
) -> dict:
    """Run the existing analysis layer for one dataset."""

    from analysis import seo_summary

    return seo_summary(
        project=project,
        dataset=dataset,
        min_impressions=min_impressions,
        opportunity_limit=opportunity_limit,
        loss_limit=loss_limit,
        cluster_queries=cluster_queries,
        cluster_similarity_threshold=cluster_similarity_threshold,
        cluster_model=cluster_model,
        cluster_limit=cluster_limit,
    )


if __name__ == "__main__":
    mcp.run()
