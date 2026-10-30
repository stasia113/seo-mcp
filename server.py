from typing import Literal
from pathlib import Path
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

from mcp.server.mcpserver import MCPServer


mcp = MCPServer("SEO MCP")


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

RefreshProjectChoice = Literal[
    "FDP Mold Remediation",
    "O2 Mold Testing",
    "USA Clean Master",
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

BASE_DIR = Path(__file__).parent
PREPARED_DIR = BASE_DIR / "prepared"
REFRESH_DIR = BASE_DIR / "refresh"
PREPARE_SCRIPT = BASE_DIR / "prepare_seo_data.py"

DEFAULT_CLUSTER_LIMIT = 10


def _prepared_path(project: str, dataset: str) -> Path:
    return PREPARED_DIR / project / f"{dataset}.json"


def _load_prepared(project: str, dataset: str) -> dict:
    path = _prepared_path(project, dataset)
    if not path.exists():
        raise FileNotFoundError(
            f"Prepared analysis not found: {path}. "
            f"Run refresh_gsc_data for {project}."
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Prepared analysis is not valid JSON: {path}") from exc


def _load_dataset(project: str, dataset: str) -> dict:
    return _load_prepared(project, dataset)


def _report_path(project: str) -> Path:
    return PREPARED_DIR / project / "report.html"


def _summary_result(project: str, dataset: str) -> dict:
    if dataset == "period":
        summary = _load_dataset(project, "period")
        report = _report_path(project)
        summary["report"] = {
            "html_file": report.name,
            "html_path": str(report),
            "prepared": report.exists(),
            "last_modified": _file_time(report),
        }
        return summary

    if dataset == "yoy":
        summary = _load_dataset(project, "yoy")
        report = _report_path(project)
        summary["report"] = {
            "html_file": report.name,
            "html_path": str(report),
            "prepared": report.exists(),
            "last_modified": _file_time(report),
        }
        return summary

    if dataset == "both":
        period = _load_dataset(project, "period")
        yoy = _load_dataset(project, "yoy")
        report = _report_path(project)
        return {
            "project": project,
            "brand": period.get("analysis", {}).get("brand", project),
            "dataset": "both",
            "period": period,
            "yoy": yoy,
            "report": {
                "html_file": report.name,
                "html_path": str(report),
                "prepared": report.exists(),
                "last_modified": _file_time(report),
            },
        }

    raise ValueError("dataset must be 'period', 'yoy', or 'both'.")


def _file_time(path: Path) -> str | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _get_table(project: str, dataset: str, section: str, table: str, limit: int) -> list[dict]:
    summary = _load_dataset(project, dataset)
    rows = summary.get(section, {}).get(table, [])
    return rows[:limit]


def _state_path() -> Path:
    REFRESH_DIR.mkdir(parents=True, exist_ok=True)
    return REFRESH_DIR / "status.json"


def _log_path() -> Path:
    REFRESH_DIR.mkdir(parents=True, exist_ok=True)
    return REFRESH_DIR / "prepare.log"


def _read_state() -> dict:
    path = _state_path()
    if not path.exists():
        return {
            "status": "idle",
            "project": None,
            "pid": None,
            "started_at": None,
            "finished_at": None,
            "return_code": None,
            "log_file": str(_log_path()),
        }
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "status": "unknown",
            "project": None,
            "pid": None,
            "started_at": None,
            "finished_at": None,
            "return_code": None,
            "log_file": str(_log_path()),
        }


def _write_state(state: dict) -> None:
    path = _state_path()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(path)


def _process_is_running(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _start_refresh(project: str, force: bool = False) -> dict:
    if not PREPARE_SCRIPT.exists():
        raise FileNotFoundError(f"Preparation script not found: {PREPARE_SCRIPT}")

    current = _read_state()
    if current.get("status") == "running" and _process_is_running(current.get("pid")):
        return {
            "status": "already_running",
            "project": current.get("project"),
            "pid": current.get("pid"),
            "started_at": current.get("started_at"),
            "log_file": current.get("log_file"),
            "message": "A GSC refresh is already running. Use gsc_refresh_status to monitor it.",
        }

    REFRESH_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _log_path()
    command = [
        sys.executable,
        str(PREPARE_SCRIPT),
        "--project",
        project,
        "--cluster-limit",
        str(DEFAULT_CLUSTER_LIMIT),
    ]
    if force:
        command.append("--force")

    log_handle = open(log_path, "w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            cwd=str(BASE_DIR),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except Exception:
        log_handle.close()
        raise
    log_handle.close()

    started_at = datetime.now(timezone.utc).isoformat()
    state = {
        "status": "running",
        "project": project,
        "pid": process.pid,
        "started_at": started_at,
        "finished_at": None,
        "return_code": None,
        "log_file": str(log_path),
        "command": command,
        "cluster_limit": DEFAULT_CLUSTER_LIMIT,
        "force": force,
    }
    _write_state(state)
    return {
        "status": "started",
        "project": project,
        "pid": process.pid,
        "started_at": started_at,
        "log_file": str(log_path),
        "cluster_limit": DEFAULT_CLUSTER_LIMIT,
        "force": force,
        "message": "GSC refresh started in the background. The MCP request is not waiting for preparation to finish. Use gsc_refresh_status to monitor it.",
    }


@mcp.tool()
def click_losses(
    sheet_name: SheetChoice,
    project: SingleProjectChoice = "FDP Mold Remediation",
    dataset: DatasetChoice = "Last 3 months vs previous 3 months",
    limit: int = 25,
) -> list[dict]:
    """Return prepared click-loss results for Queries or Pages."""
    project_key = PROJECT_MAP[project]
    dataset_key = DATASET_MAP[dataset]
    section = "queries" if sheet_name == "Queries" else "pages"
    return _get_table(project_key, dataset_key, section, "top_click_losses", limit)


@mcp.tool()
def gsc_opportunities(
    sheet_name: SheetChoice,
    project: SingleProjectChoice = "FDP Mold Remediation",
    dataset: DatasetChoice = "Last 3 months vs previous 3 months",
    limit: int = 25,
) -> list[dict]:
    """Return prepared SEO opportunities for Queries or Pages."""
    project_key = PROJECT_MAP[project]
    dataset_key = DATASET_MAP[dataset]
    section = "queries" if sheet_name == "Queries" else "pages"
    return _get_table(project_key, dataset_key, section, "top_opportunities", limit)


@mcp.tool()
def performance_loss_diagnosis(
    sheet_name: SheetChoice,
    project: SingleProjectChoice = "FDP Mold Remediation",
    dataset: DatasetChoice = "Last 3 months vs previous 3 months",
    limit: int = 25,
) -> list[dict]:
    """Return prepared performance-loss diagnoses for Queries or Pages."""
    project_key = PROJECT_MAP[project]
    dataset_key = DATASET_MAP[dataset]
    section = "queries" if sheet_name == "Queries" else "pages"
    return _get_table(project_key, dataset_key, section, "top_click_losses", limit)


@mcp.tool()
def commercial_page_performance_loss_diagnosis(
    project: SingleProjectChoice = "FDP Mold Remediation",
    dataset: DatasetChoice = "Last 3 months vs previous 3 months",
    limit: int = 25,
) -> list[dict]:
    """Return prepared click-loss diagnoses for commercial pages."""
    project_key = PROJECT_MAP[project]
    dataset_key = DATASET_MAP[dataset]
    return _get_table(project_key, dataset_key, "commercial_pages", "top_click_losses", limit)


@mcp.tool()
def refresh_gsc_data(
    project: RefreshProjectChoice = "FDP Mold Remediation",
    force: bool = False,
) -> dict:
    """Start a background refresh from Google Search Console."""
    return _start_refresh(PROJECT_MAP[project], force=force)


@mcp.tool()
def gsc_refresh_status() -> dict:
    """Return the status of the background GSC preparation job."""
    state = _read_state()

    if state.get("status") == "running":
        # The preparation script writes its final log line before the Python
        # process exits. On macOS, checking only the PID can briefly report
        # "running" even though the report has already been generated.
        log_path = Path(state.get("log_file", ""))
        log_text = ""
        if log_path.exists():
            try:
                log_text = log_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                log_text = ""

        if "ERROR:" in log_text:
            state["status"] = "failed"
            state["finished_at"] = datetime.now(timezone.utc).isoformat()
            state["return_code"] = 1
            _write_state(state)
        elif "Preparation finished." in log_text:
            state["status"] = "completed"
            state["finished_at"] = datetime.now(timezone.utc).isoformat()
            state["return_code"] = 0
            _write_state(state)
        elif _process_is_running(state.get("pid")):
            return {**state, "message": "GSC refresh is still running."}
        else:
            project = state.get("project")
            report = _report_path(project) if project else None
            period = _prepared_path(project, "period") if project else None
            yoy = _prepared_path(project, "yoy") if project else None
            success = bool(
                report
                and period
                and yoy
                and report.exists()
                and period.exists()
                and yoy.exists()
            )
            state["status"] = "completed" if success else "failed"
            state["finished_at"] = datetime.now(timezone.utc).isoformat()
            state["return_code"] = 0 if success else None
            _write_state(state)

    project = state.get("project")
    report = _report_path(project) if project else None
    return {
        **state,
        "report": str(report) if report else None,
        "report_exists": bool(report and report.exists()),
        "report_last_modified": _file_time(report) if report else None,
        "message": (
            "GSC refresh completed."
            if state.get("status") == "completed"
            else "GSC refresh failed."
            if state.get("status") == "failed"
            else "GSC refresh is still running."
        ),
    }


@mcp.tool()
def seo_summary_tool(
    project: SummaryProjectChoice = "FDP Mold Remediation",
    dataset: SummaryDatasetChoice = "Both comparisons",
) -> dict:
    """Return prepared SEO analysis and the already-generated HTML report."""
    project_key = PROJECT_MAP[project]
    dataset_key = DATASET_MAP[dataset]
    if project_key == "all":
        results = {}
        missing = {}
        for project_name in ("fdp", "o2", "usa_clean_master"):
            try:
                results[project_name] = _summary_result(project_name, dataset_key)
            except FileNotFoundError as exc:
                missing[project_name] = str(exc)
        return {"project": "all", "dataset": dataset_key, "reports": results, "missing": missing}
    return _summary_result(project_key, dataset_key)


if __name__ == "__main__":
    mcp.run()
