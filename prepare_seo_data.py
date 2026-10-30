from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from dateutil.relativedelta import relativedelta

from gsc_api import (
    GSC_PROPERTIES,
    PROJECT_LABELS,
    query_project_pages,
    query_project_queries,
)


PROJECTS = ["fdp", "o2", "usa_clean_master"]
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BASE_DIR = Path(__file__).resolve().parent
PREPARED_DIR = BASE_DIR / "prepared"
API_SOURCE_DIR = PREPARED_DIR / "gsc_api_sources"

# GSC commonly has a reporting delay. Keep the current period ending a few
# days before today so the comparison is based on a complete reporting window.
DEFAULT_REPORTING_DELAY_DAYS = 3
DEFAULT_ROW_LIMIT = 25_000


# In-memory API data used by the existing analysis.py. This lets us migrate
# the data source without rewriting the established analysis/report layer.
_API_FRAMES: dict[tuple[str, str, str], pd.DataFrame] = {}
_API_COLUMNS: dict[tuple[str, str], dict[str, str]] = {}
_API_SOURCE_PATHS: dict[tuple[str, str], Path] = {}


def _period_ranges(delay_days: int = DEFAULT_REPORTING_DELAY_DAYS) -> dict[str, dict[str, str]]:
    end = date.today() - timedelta(days=delay_days)
    start = end - relativedelta(months=3) + timedelta(days=1)

    previous_end = start - timedelta(days=1)
    previous_start = previous_end - relativedelta(months=3) + timedelta(days=1)

    yoy_start = start - relativedelta(years=1)
    yoy_end = end - relativedelta(years=1)

    return {
        "period": {
            "current_start": start.isoformat(),
            "current_end": end.isoformat(),
            "comparison_start": previous_start.isoformat(),
            "comparison_end": previous_end.isoformat(),
            "label": "Last 3 months vs previous 3 months",
        },
        "yoy": {
            "current_start": start.isoformat(),
            "current_end": end.isoformat(),
            "comparison_start": yoy_start.isoformat(),
            "comparison_end": yoy_end.isoformat(),
            "label": "Last 3 months vs same period last year",
        },
    }


def _merge_metrics(
    current_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    dimension: str,
    key_column: str,
) -> pd.DataFrame:
    current = pd.DataFrame(current_rows)
    comparison = pd.DataFrame(comparison_rows)

    current_columns = [dimension, "clicks", "impressions", "ctr", "position"]
    comparison_columns = [dimension, "clicks", "impressions", "ctr", "position"]

    if current.empty:
        current = pd.DataFrame(columns=current_columns)
    else:
        current = current[current_columns].copy()

    if comparison.empty:
        comparison = pd.DataFrame(columns=comparison_columns)
    else:
        comparison = comparison[comparison_columns].copy()

    current = current.rename(
        columns={
            dimension: key_column,
            "clicks": "Current Clicks",
            "impressions": "Current Impressions",
            "ctr": "Current CTR",
            "position": "Current Position",
        }
    )

    comparison = comparison.rename(
        columns={
            dimension: key_column,
            "clicks": "Comparison Clicks",
            "impressions": "Comparison Impressions",
            "ctr": "Comparison CTR",
            "position": "Comparison Position",
        }
    )

    merged = current.merge(
        comparison,
        on=key_column,
        how="outer",
    )

    numeric = [
        "Current Clicks",
        "Comparison Clicks",
        "Current Impressions",
        "Comparison Impressions",
        "Current CTR",
        "Comparison CTR",
        "Current Position",
        "Comparison Position",
    ]

    for column in numeric:
        merged[column] = pd.to_numeric(
            merged[column],
            errors="coerce",
        ).fillna(0)

    # Search Console API CTR is a ratio. The existing analysis/report layer
    # expects percentage points, so 0.003 becomes 0.3.
    merged["Current CTR"] *= 100
    merged["Comparison CTR"] *= 100

    return merged


def _fetch_dataset(
    project: str,
    dataset: str,
    ranges: dict[str, dict[str, str]],
    row_limit: int,
) -> dict[str, pd.DataFrame]:
    spec = ranges[dataset]

    print(
        f"  GSC {dataset}: "
        f"{spec['current_start']} to {spec['current_end']} "
        f"vs {spec['comparison_start']} to {spec['comparison_end']}"
    )

    print("    Fetching Queries...")
    current_queries = query_project_queries(
        project,
        spec["current_start"],
        spec["current_end"],
        row_limit=row_limit,
    )
    comparison_queries = query_project_queries(
        project,
        spec["comparison_start"],
        spec["comparison_end"],
        row_limit=row_limit,
    )

    print("    Fetching Pages...")
    current_pages = query_project_pages(
        project,
        spec["current_start"],
        spec["current_end"],
        row_limit=row_limit,
    )
    comparison_pages = query_project_pages(
        project,
        spec["comparison_start"],
        spec["comparison_end"],
        row_limit=row_limit,
    )

    queries = _merge_metrics(
        current_queries,
        comparison_queries,
        dimension="query",
        key_column="Top queries",
    )

    pages = _merge_metrics(
        current_pages,
        comparison_pages,
        dimension="page",
        key_column="Top pages",
    )

    print(
        f"    Rows: Queries={len(queries):,}, Pages={len(pages):,}"
    )

    return {
        "Queries": queries,
        "Pages": pages,
    }


def _install_api_source_for_analysis(
    project: str,
    datasets: dict[str, dict[str, pd.DataFrame]],
    ranges: dict[str, dict[str, str]],
) -> None:
    """Patch the existing analysis layer to read live API DataFrames."""

    import analysis

    for dataset, sheets in datasets.items():
        for sheet_name, frame in sheets.items():
            _API_FRAMES[(project, dataset, sheet_name)] = frame

        _API_COLUMNS[(project, dataset)] = {
            "key_column_queries": "Top queries",
            "key_column_pages": "Top pages",
        }

        source_dir = API_SOURCE_DIR / project
        source_dir.mkdir(parents=True, exist_ok=True)
        source_path = source_dir / f"{dataset}.gsc-api"
        source_path.write_text(
            json.dumps(
                {
                    "project": project,
                    "property": GSC_PROPERTIES[project],
                    "dataset": dataset,
                    "comparison": ranges[dataset],
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        _API_SOURCE_PATHS[(project, dataset)] = source_path

    original_load_sheet = analysis.load_sheet
    original_detect = analysis.detect_gsc_columns
    original_get_excel = getattr(analysis, "get_excel_file", None)

    def api_load_sheet(
        sheet_name: str,
        project: str = "fdp",
        dataset: str = "period",
    ) -> pd.DataFrame:
        key = (project, dataset, sheet_name)
        if key not in _API_FRAMES:
            raise FileNotFoundError(
                f"No live GSC API data loaded for {project}/{dataset}/{sheet_name}."
            )
        return _API_FRAMES[key].copy()

    def api_detect_gsc_columns(
        sheet_name: str,
        project: str = "fdp",
        dataset: str = "period",
    ) -> dict[str, str]:
        if sheet_name == "Queries":
            key_column = "Top queries"
        elif sheet_name == "Pages":
            key_column = "Top pages"
        else:
            raise ValueError(
                f"Unsupported sheet: {sheet_name}. Expected Queries or Pages."
            )

        return {
            "key_column": key_column,
            "current_clicks": "Current Clicks",
            "comparison_clicks": "Comparison Clicks",
            "current_impressions": "Current Impressions",
            "comparison_impressions": "Comparison Impressions",
            "current_ctr": "Current CTR",
            "comparison_ctr": "Comparison CTR",
            "current_position": "Current Position",
            "comparison_position": "Comparison Position",
        }

    def api_get_excel_file(project: str, dataset: str):
        # The existing analysis/report layer uses this primarily as a source
        # label. Return a harmless API marker path instead of an XLSX path.
        return _API_SOURCE_PATHS[(project, dataset)]

    analysis.load_sheet = api_load_sheet
    analysis.detect_gsc_columns = api_detect_gsc_columns

    if original_get_excel is not None:
        analysis.get_excel_file = api_get_excel_file

    # Return a restore callback for callers that want to undo the patch.
    def restore() -> None:
        analysis.load_sheet = original_load_sheet
        analysis.detect_gsc_columns = original_detect
        if original_get_excel is not None:
            analysis.get_excel_file = original_get_excel

    return restore


def _data_fingerprint(
    project: str,
    datasets: dict[str, dict[str, pd.DataFrame]],
    ranges: dict[str, dict[str, str]],
) -> str:
    payload: list[dict[str, Any]] = []

    for dataset in ("period", "yoy"):
        for sheet_name in ("Queries", "Pages"):
            frame = datasets[dataset][sheet_name]
            payload.append(
                {
                    "dataset": dataset,
                    "sheet": sheet_name,
                    "rows": len(frame),
                    "columns": list(frame.columns),
                    "hash": hashlib.sha256(
                        pd.util.hash_pandas_object(
                            frame,
                            index=True,
                        ).values.tobytes()
                    ).hexdigest(),
                }
            )

    payload.append({"project": project, "ranges": ranges})

    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def prepare_project(
    project: str,
    force: bool = False,
    min_impressions: int = 100,
    opportunity_limit: int = 10,
    loss_limit: int = 10,
    cluster_similarity_threshold: float = 0.78,
    cluster_limit: int = 50,
    row_limit: int = DEFAULT_ROW_LIMIT,
    reporting_delay_days: int = DEFAULT_REPORTING_DELAY_DAYS,
) -> dict:
    import analysis
    from html_report import build_combined_html_report

    brand = PROJECT_LABELS[project]
    ranges = _period_ranges(reporting_delay_days)
    project_dir = PREPARED_DIR / project
    project_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = project_dir / "manifest.json"

    print(f"Preparing {brand} from Google Search Console API...")

    datasets = {
        "period": _fetch_dataset(
            project,
            "period",
            ranges,
            row_limit,
        ),
        "yoy": _fetch_dataset(
            project,
            "yoy",
            ranges,
            row_limit,
        ),
    }

    fingerprint = _data_fingerprint(
        project,
        datasets,
        ranges,
    )

    if not force and manifest_path.exists():
        try:
            previous = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            previous = None

        if (
            previous
            and previous.get("source") == "google_search_console_api"
            and previous.get("data_fingerprint") == fingerprint
            and previous.get("cluster_similarity_threshold") == cluster_similarity_threshold
            and previous.get("cluster_model") == DEFAULT_MODEL
            and previous.get("cluster_limit") == cluster_limit
        ):
            report = project_dir / "report.html"
            if report.exists():
                print(
                    f"{brand}: API data unchanged, using prepared report."
                )
                return {
                    "project": project,
                    "brand": brand,
                    "status": "cached",
                    "source": "google_search_console_api",
                    "report": str(report),
                }

    # Load both datasets into the existing analysis layer before calling
    # seo_summary. This also lets the existing semantic clustering code build
    # one cluster map from the union of period + YoY queries.
    restore = _install_api_source_for_analysis(
        project,
        datasets,
        ranges,
    )

    try:
        print("  Building recent-period analysis...")
        period_summary = analysis.seo_summary(
            project=project,
            dataset="period",
            min_impressions=min_impressions,
            opportunity_limit=opportunity_limit,
            loss_limit=loss_limit,
            cluster_queries=True,
            cluster_similarity_threshold=cluster_similarity_threshold,
            cluster_model=DEFAULT_MODEL,
            cluster_limit=cluster_limit,
        )

        print("  Building YoY analysis...")
        yoy_summary = analysis.seo_summary(
            project=project,
            dataset="yoy",
            min_impressions=min_impressions,
            opportunity_limit=opportunity_limit,
            loss_limit=loss_limit,
            cluster_queries=True,
            cluster_similarity_threshold=cluster_similarity_threshold,
            cluster_model=DEFAULT_MODEL,
            cluster_limit=cluster_limit,
        )

        period_summary.setdefault("analysis", {})["source"] = (
            "Google Search Console API"
        )
        yoy_summary.setdefault("analysis", {})["source"] = (
            "Google Search Console API"
        )

        print("  Building combined HTML report...")
        combined_report = build_combined_html_report(
            period_summary,
            yoy_summary,
        )

        prepared_report = project_dir / "report.html"
        prepared_report.write_text(
            combined_report.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

        period_json = project_dir / "period.json"
        yoy_json = project_dir / "yoy.json"

        _save_json(period_json, period_summary)
        _save_json(yoy_json, yoy_summary)

        manifest = {
            "project": project,
            "brand": brand,
            "source": "google_search_console_api",
            "property": GSC_PROPERTIES[project],
            "prepared_at": datetime.now().isoformat(timespec="seconds"),
            "data_fingerprint": fingerprint,
            "date_ranges": ranges,
            "reporting_delay_days": reporting_delay_days,
            "cluster_model": DEFAULT_MODEL,
            "cluster_similarity_threshold": cluster_similarity_threshold,
            "cluster_limit": cluster_limit,
            "row_limit": row_limit,
            "report": str(prepared_report),
            "period_json": str(period_json),
            "yoy_json": str(yoy_json),
        }

        _save_json(manifest_path, manifest)

        print(f"  Done: {prepared_report}")

        return {
            "project": project,
            "brand": brand,
            "status": "prepared",
            "source": "google_search_console_api",
            "report": str(prepared_report),
            "period_json": str(period_json),
            "yoy_json": str(yoy_json),
            "date_ranges": ranges,
        }

    finally:
        restore()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare SEO MCP analysis from live Google Search Console data."
        )
    )

    parser.add_argument(
        "--project",
        choices=PROJECTS + ["all"],
        default="fdp",
    )
    parser.add_argument(
        "--force",
        action="store_true",
    )
    parser.add_argument(
        "--min-impressions",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--opportunity-limit",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--loss-limit",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--cluster-threshold",
        type=float,
        default=0.78,
    )
    parser.add_argument(
        "--cluster-limit",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--row-limit",
        type=int,
        default=DEFAULT_ROW_LIMIT,
        help="Maximum rows per GSC query/page request.",
    )
    parser.add_argument(
        "--reporting-delay-days",
        type=int,
        default=DEFAULT_REPORTING_DELAY_DAYS,
        help="Days to leave between today and the current GSC period end.",
    )

    args = parser.parse_args()

    PREPARED_DIR.mkdir(parents=True, exist_ok=True)

    projects = (
        PROJECTS
        if args.project == "all"
        else [args.project]
    )

    results = []

    for project in projects:
        try:
            results.append(
                prepare_project(
                    project=project,
                    force=args.force,
                    min_impressions=args.min_impressions,
                    opportunity_limit=args.opportunity_limit,
                    loss_limit=args.loss_limit,
                    cluster_similarity_threshold=args.cluster_threshold,
                    cluster_limit=args.cluster_limit,
                    row_limit=args.row_limit,
                    reporting_delay_days=args.reporting_delay_days,
                )
            )
        except Exception as exc:
            print(
                f"ERROR: {project}: "
                f"{type(exc).__name__}: {exc}"
            )
            results.append(
                {
                    "project": project,
                    "status": "error",
                    "error": str(exc),
                }
            )

    _save_json(
        PREPARED_DIR / "last_run.json",
        {
            "run_at": datetime.now().isoformat(timespec="seconds"),
            "source": "google_search_console_api",
            "results": results,
        },
    )

    print("\nPreparation finished.")


if __name__ == "__main__":
    main()
