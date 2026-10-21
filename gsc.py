from pathlib import Path
import pandas as pd


DATA_DIR = Path(__file__).parent / "data"


COMMON_COMMERCIAL_EXCLUSIONS = [
    "blog",
    "resources",
    "case-study",
    "images",
    "videos",
    "symptoms",
    "recent-projects",
    "glossary",
]


PROJECTS = {
    "fdp": {
        "brand": "FDP Mold Remediation",
        "commercial_excluded_sections": COMMON_COMMERCIAL_EXCLUSIONS,
        "datasets": {
            "period": {
                "file": "data/fdp/FDP-GSC.xlsx",
                "label": "Last 3 months vs previous 3 months",
            },
            "yoy": {
                "file": "data/fdp/FDP-GSC-YoY.xlsx",
                "label": "Last 3 months vs same period last year",
            },
        },
    },
    "o2": {
        "brand": "O2 Mold Testing",
        "commercial_excluded_sections": COMMON_COMMERCIAL_EXCLUSIONS + ["expert-advice"],
        "datasets": {
            "period": {
                "file": "data/o2/O2-GSC.xlsx",
                "label": "Last 3 months vs previous 3 months",
            },
            "yoy": {
                "file": "data/o2/O2-GSC-YoY.xlsx",
                "label": "Last 3 months vs same period last year",
            },
        },
    },
    "usa_clean_master": {
        "brand": "USA Clean Master",
        # For USA Clean Master, the commercial-page analysis is an allowlist,
        # not a general exclusion filter. Only URLs under these paths qualify.
        "commercial_include_prefixes": [
            "/carpetcleaning/",
            "/rugcleaning/",
            "/upholsterycleaning/",
            "/mattress-cleaning.aspx",
        ],
        "commercial_excluded_contains": ["#"],
        "commercial_excluded_sections": COMMON_COMMERCIAL_EXCLUSIONS + ["knowledge-base"],
        "datasets": {
            "period": {
                "file": "data/usa-clean-master/USA-Clean-Master-GSC.xlsx",
                "label": "Last 3 months vs previous 3 months",
            },
            "yoy": {
                "file": "data/usa-clean-master/USA-Clean-Master-GSC-YoY.xlsx",
                "label": "Last 3 months vs same period last year",
            },
        },
    },
}


def get_project_config(project: str = "fdp") -> dict:
    project = project.lower().strip()

    if project not in PROJECTS:
        available = ", ".join(PROJECTS.keys())
        raise ValueError(
            f"Unknown project '{project}'. Available projects: {available}"
        )

    return PROJECTS[project]



def get_commercial_exclusions(project: str = "fdp") -> list[str]:
    """Return URL path segments excluded from the commercial-page view."""
    config = get_project_config(project)
    return list(config.get("commercial_excluded_sections", COMMON_COMMERCIAL_EXCLUSIONS))


def get_commercial_inclusions(project: str = "fdp") -> list[str]:
    """Return URL path prefixes that qualify for the commercial-page view."""
    config = get_project_config(project)
    return list(config.get("commercial_include_prefixes", []))


def get_commercial_excluded_contains(project: str = "fdp") -> list[str]:
    """Return literal strings that disqualify a URL from the commercial-page view."""
    config = get_project_config(project)
    return list(config.get("commercial_excluded_contains", []))

def get_dataset_config(
    project: str = "fdp",
    dataset: str = "period"
) -> dict:
    config = get_project_config(project)
    dataset = dataset.lower().strip()

    if dataset not in config["datasets"]:
        raise ValueError(
            f"Unknown dataset '{dataset}' for project '{project}'. "
            "Use 'period' or 'yoy'."
        )

    return config["datasets"][dataset]


def get_excel_file(
    project: str = "fdp",
    dataset: str = "period"
) -> Path:
    config = get_dataset_config(project, dataset)
    configured = Path(__file__).parent / config["file"]

    # Backward-compatible fallback for the existing FDP setup.
    if (
        project.lower() == "fdp"
        and dataset.lower() == "period"
        and not configured.exists()
    ):
        legacy = DATA_DIR / "FDP-GSC.xlsx"
        if legacy.exists():
            return legacy

    return configured


def list_projects() -> dict:
    return {
        project: {
            "brand": config["brand"],
            "datasets": {
                name: {
                    "file": dataset_config["file"],
                    "label": dataset_config["label"],
                    "exists": get_excel_file(project, name).exists(),
                    "commercial_excluded_sections": [
                        f"/{section}/"
                        for section in get_commercial_exclusions(project)
                    ],
                    "commercial_include_prefixes": get_commercial_inclusions(project),
                    "commercial_excluded_contains": get_commercial_excluded_contains(project),
                }
                for name, dataset_config in config["datasets"].items()
            },
        }
        for project, config in PROJECTS.items()
    }


def load_sheet(
    sheet_name: str,
    project: str = "fdp",
    dataset: str = "period"
):
    file_path = get_excel_file(project, dataset)

    if not file_path.exists():
        raise FileNotFoundError(
            f"GSC workbook not found for project='{project}', "
            f"dataset='{dataset}': {file_path}"
        )

    return pd.read_excel(
        file_path,
        sheet_name=sheet_name
    )


def detect_gsc_columns(
    sheet_name: str,
    project: str = "fdp",
    dataset: str = "period"
):
    """
    Detect the current and comparison GSC columns.

    The first period is always the current Last 3 months data.
    The comparison is either the previous period or the same period last year,
    depending on the selected dataset.
    """

    df = load_sheet(
        sheet_name,
        project=project,
        dataset=dataset
    )

    columns = df.columns.tolist()
    lower = {column: str(column).lower() for column in columns}

    if sheet_name.lower() == "queries":
        key_candidates = [
            column for column in columns
            if "query" in lower[column]
            or "top queries" in lower[column]
        ]
    elif sheet_name.lower() == "pages":
        key_candidates = [
            column for column in columns
            if "page" in lower[column]
            or "top pages" in lower[column]
        ]
    else:
        raise ValueError(
            "Only Queries and Pages sheets are supported."
        )

    if not key_candidates:
        raise ValueError(
            f"Could not find the main column in {sheet_name}. "
            f"Available columns: {columns}"
        )

    metric_groups = {
        "clicks": [],
        "impressions": [],
        "ctr": [],
        "position": [],
    }

    for column in columns:
        name = lower[column]

        if "click" in name:
            metric_groups["clicks"].append(column)
        elif "impression" in name:
            metric_groups["impressions"].append(column)
        elif "ctr" in name:
            metric_groups["ctr"].append(column)
        elif "position" in name:
            metric_groups["position"].append(column)

    comparison_terms = [
        "previous",
        "same period last year",
        "previous period",
        "comparison",
    ]

    current_terms = [
        "last 3 months",
        "last three months",
        "current",
    ]

    detected = {}

    for metric, candidates in metric_groups.items():
        if len(candidates) < 2:
            raise ValueError(
                f"Could not detect current and comparison {metric} columns "
                f"for project='{project}', dataset='{dataset}'. "
                f"Found: {candidates}. Available columns: {columns}"
            )

        current = [
            column for column in candidates
            if any(term in lower[column] for term in current_terms)
        ]

        comparison = [
            column for column in candidates
            if any(term in lower[column] for term in comparison_terms)
        ]

        if current and comparison:
            detected[metric] = [current[0], comparison[0]]
        else:
            # GSC exports normally place current before comparison.
            detected[metric] = candidates[:2]

    return {
        "key_column": key_candidates[0],
        "current_clicks": detected["clicks"][0],
        "comparison_clicks": detected["clicks"][1],
        "current_impressions": detected["impressions"][0],
        "comparison_impressions": detected["impressions"][1],
        "current_ctr": detected["ctr"][0],
        "comparison_ctr": detected["ctr"][1],
        "current_position": detected["position"][0],
        "comparison_position": detected["position"][1],
    }
