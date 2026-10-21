from pathlib import Path
import pandas as pd

from mcp.server.mcpserver import MCPServer

from gsc import (
    get_excel_file,
    get_project_config,
    get_dataset_config,
    list_projects,
    load_sheet,
)

from analysis import (
    analyze_sheet_overview,
    find_click_losses,
    find_gsc_opportunities,
    diagnose_performance_losses,
    diagnose_commercial_page_losses,
    seo_summary,
)

from html_report import build_html_report, build_combined_html_report


mcp = MCPServer("SEO MCP")


@mcp.tool()
def projects() -> dict:
    """Show configured SEO projects and their GSC datasets."""
    return list_projects()


@mcp.tool()
def list_sheets(
    project: str = "fdp",
    dataset: str = "period"
) -> dict:
    """Show available sheets in a project's GSC workbook."""

    file_path = get_excel_file(project, dataset)

    if not file_path.exists():
        raise FileNotFoundError(
            f"GSC workbook not found: {file_path}"
        )

    excel = pd.ExcelFile(file_path)

    return {
        "project": project,
        "dataset": dataset,
        "brand": get_project_config(project)["brand"],
        "file": file_path.name,
        "sheets": excel.sheet_names,
    }


@mcp.tool()
def preview_sheet(
    sheet_name: str,
    project: str = "fdp",
    dataset: str = "period",
    rows: int = 5
) -> list[dict]:
    """Preview rows from a project's Queries or Pages sheet."""

    df = load_sheet(
        sheet_name,
        project=project,
        dataset=dataset
    )

    return df.head(rows).to_dict(
        orient="records"
    )


@mcp.tool()
def seo_overview(
    sheet_name: str,
    project: str = "fdp",
    dataset: str = "period"
) -> dict:
    """Return a high-level GSC overview for Queries or Pages."""

    return analyze_sheet_overview(
        sheet_name,
        project=project,
        dataset=dataset
    )


@mcp.tool()
def click_losses(
    sheet_name: str,
    project: str = "fdp",
    dataset: str = "period",
    limit: int = 25
) -> list[dict]:
    """Find the largest click losses."""

    return find_click_losses(
        sheet_name,
        limit=limit,
        project=project,
        dataset=dataset
    )


@mcp.tool()
def gsc_opportunities(
    sheet_name: str,
    project: str = "fdp",
    dataset: str = "period",
    min_impressions: int = 100,
    min_position: float = 4,
    max_position: float = 20,
    limit: int = 25
) -> list[dict]:
    """Find SEO opportunities in comparison-based GSC data."""

    return find_gsc_opportunities(
        sheet_name,
        min_impressions=min_impressions,
        min_position=min_position,
        max_position=max_position,
        limit=limit,
        project=project,
        dataset=dataset
    )


@mcp.tool()
def performance_loss_diagnosis(
    sheet_name: str,
    project: str = "fdp",
    dataset: str = "period",
    limit: int = 25
) -> list[dict]:
    """Diagnose the largest click losses."""

    return diagnose_performance_losses(
        sheet_name,
        limit=limit,
        project=project,
        dataset=dataset
    )


@mcp.tool()
def commercial_page_performance_loss_diagnosis(
    project: str = "fdp",
    dataset: str = "period",
    limit: int = 25
) -> list[dict]:
    """Diagnose click losses among commercial pages only."""

    return diagnose_commercial_page_losses(
        limit=limit,
        project=project,
        dataset=dataset
    )


@mcp.tool()
def seo_summary_tool(
    project: str = "fdp",
    dataset: str = "both",
    min_impressions: int = 100,
    opportunity_limit: int = 10,
    loss_limit: int = 10,
    cluster_queries: bool = True,
    cluster_similarity_threshold: float = 0.78,
    cluster_model: str = "all-MiniLM-L6-v2",
    cluster_limit: int = 50,
) -> dict:
    """
    Analyze one or both GSC datasets for a project and generate an HTML report.

    dataset='period' = last 3 months vs previous 3 months.
    dataset='yoy' = last 3 months vs same period last year.
    dataset='both' = both comparisons plus a cross-comparison report.

    Use project='all' to generate reports for every configured project
    whose requested workbook(s) exist.
    """

    project = project.lower().strip()
    dataset = dataset.lower().strip()

    if dataset not in {"period", "yoy", "both"}:
        raise ValueError("dataset must be 'period', 'yoy', or 'both'.")

    def run_one(project_name: str, dataset_name: str) -> dict:
        summary = seo_summary(
            project=project_name,
            dataset=dataset_name,
            min_impressions=min_impressions,
            opportunity_limit=opportunity_limit,
            loss_limit=loss_limit,
            cluster_queries=cluster_queries,
            cluster_similarity_threshold=cluster_similarity_threshold,
            cluster_model=cluster_model,
            cluster_limit=cluster_limit,
        )
        report_path = build_html_report(summary)
        summary["report"] = {
            "html_file": report_path.name,
            "html_path": str(report_path),
        }
        return summary

    def run_project(project_name: str) -> dict:
        period_path = get_excel_file(project_name, "period")
        yoy_path = get_excel_file(project_name, "yoy")

        if dataset == "period":
            if not period_path.exists():
                raise FileNotFoundError(f"Period workbook not found: {period_path}")
            return run_one(project_name, "period")

        if dataset == "yoy":
            if not yoy_path.exists():
                raise FileNotFoundError(f"YoY workbook not found: {yoy_path}")
            return run_one(project_name, "yoy")

        if not period_path.exists() or not yoy_path.exists():
            missing = []
            if not period_path.exists():
                missing.append(f"period: {period_path}")
            if not yoy_path.exists():
                missing.append(f"yoy: {yoy_path}")
            raise FileNotFoundError(
                f"Both datasets are required for project='{project_name}'. "
                + "; ".join(missing)
            )

        period_summary = seo_summary(
            project=project_name,
            dataset="period",
            min_impressions=min_impressions,
            opportunity_limit=opportunity_limit,
            loss_limit=loss_limit,
            cluster_queries=cluster_queries,
            cluster_similarity_threshold=cluster_similarity_threshold,
            cluster_model=cluster_model,
            cluster_limit=cluster_limit,
        )

        yoy_summary = seo_summary(
            project=project_name,
            dataset="yoy",
            min_impressions=min_impressions,
            opportunity_limit=opportunity_limit,
            loss_limit=loss_limit,
            cluster_queries=cluster_queries,
            cluster_similarity_threshold=cluster_similarity_threshold,
            cluster_model=cluster_model,
            cluster_limit=cluster_limit,
        )

        report_path = build_combined_html_report(
            period_summary,
            yoy_summary
        )

        return {
            "project": project_name,
            "brand": get_project_config(project_name)["brand"],
            "dataset": "both",
            "period": period_summary,
            "yoy": yoy_summary,
            "report": {
                "html_file": report_path.name,
                "html_path": str(report_path),
            },
        }

    if project == "all":
        results = {}
        missing = {}

        for project_name in ("fdp", "o2", "usa_clean_master"):
            try:
                results[project_name] = run_project(project_name)
            except FileNotFoundError as exc:
                missing[project_name] = str(exc)

        return {
            "project": "all",
            "dataset": dataset,
            "reports": results,
            "missing": missing,
        }

    return run_project(project)


if __name__ == "__main__":
    mcp.run()
