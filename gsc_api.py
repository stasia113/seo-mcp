"""
Google Search Console API access for the SEO MCP.

One OAuth client can be used for all configured GSC properties as long as the
Google account authorized during OAuth has access to those properties.

Examples:
    python gsc_api.py list-sites
    python gsc_api.py test-project --project fdp
    python gsc_api.py test-project --project fdp --dimension page
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


BASE_DIR = Path(__file__).resolve().parent
CREDENTIALS_FILE = BASE_DIR / "credentials.json"
TOKEN_FILE = BASE_DIR / "token.json"

SCOPES = [
    "https://www.googleapis.com/auth/webmasters.readonly"
]

API_SERVICE = "searchconsole"
API_VERSION = "v1"

GSC_PROPERTIES: dict[str, str] = {
    "fdp": "https://www.fdpmoldremediation.com/",
    "o2": "https://www.o2moldtesting.com/",
    "usa_clean_master": "https://www.usacleanmaster.com/",
}

PROJECT_LABELS = {
    "fdp": "FDP Mold Remediation",
    "o2": "O2 Mold Testing",
    "usa_clean_master": "USA Clean Master",
}


def authenticate_gsc():
    """Authenticate once and return an authorized Search Console service."""
    credentials: Credentials | None = None

    if TOKEN_FILE.exists():
        try:
            credentials = Credentials.from_authorized_user_file(
                str(TOKEN_FILE),
                SCOPES,
            )
        except Exception:
            credentials = None

    if credentials and credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
        TOKEN_FILE.write_text(
            credentials.to_json(),
            encoding="utf-8",
        )

    if not credentials or not credentials.valid:
        if not CREDENTIALS_FILE.exists():
            raise FileNotFoundError(
                f"Missing {CREDENTIALS_FILE.name}. "
                "Save the Google Cloud OAuth Desktop App credentials "
                "in the SEO MCP project folder."
            )

        flow = InstalledAppFlow.from_client_secrets_file(
            str(CREDENTIALS_FILE),
            SCOPES,
        )

        credentials = flow.run_local_server(
            port=0,
            access_type="offline",
            prompt="consent",
        )

        TOKEN_FILE.write_text(
            credentials.to_json(),
            encoding="utf-8",
        )

    return build(
        API_SERVICE,
        API_VERSION,
        credentials=credentials,
        cache_discovery=False,
    )


def list_gsc_sites() -> list[dict[str, str]]:
    """Return Search Console properties available to the authorized account."""
    service = authenticate_gsc()
    response = service.sites().list().execute()
    entries = response.get("siteEntry", [])

    return [
        {
            "site_url": entry.get("siteUrl", ""),
            "permission_level": entry.get("permissionLevel", ""),
        }
        for entry in entries
    ]


def get_project_property(project: str) -> str:
    """Return the configured Search Console property for a project."""
    project = project.strip().lower()

    if project not in GSC_PROPERTIES:
        raise ValueError(
            f"Unknown project '{project}'. "
            f"Expected one of: {', '.join(GSC_PROPERTIES)}"
        )

    property_url = GSC_PROPERTIES[project].strip()

    if not property_url:
        raise ValueError(
            f"No GSC property is configured for {PROJECT_LABELS[project]}."
        )

    return property_url


def query_gsc(
    project: str,
    start_date: str,
    end_date: str,
    dimensions: Iterable[str] = ("query",),
    row_limit: int = 25_000,
    search_type: str = "web",
    data_state: str = "final",
    country: str | None = None,
    device: str | None = None,
) -> list[dict[str, Any]]:
    """Query Search Console Search Analytics with pagination."""
    if row_limit < 1:
        raise ValueError("row_limit must be at least 1")

    row_limit = min(row_limit, 25_000)
    property_url = get_project_property(project)
    service = authenticate_gsc()
    dimensions = tuple(dimensions)

    allowed_dimensions = {
        "query",
        "page",
        "country",
        "device",
        "searchAppearance",
        "date",
    }

    invalid = [
        dimension
        for dimension in dimensions
        if dimension not in allowed_dimensions
    ]

    if invalid:
        raise ValueError(f"Unsupported GSC dimension(s): {invalid}")

    filters = []

    if country:
        filters.append({
            "dimension": "country",
            "operator": "equals",
            "expression": country,
        })

    if device:
        filters.append({
            "dimension": "device",
            "operator": "equals",
            "expression": device,
        })

    request_body: dict[str, Any] = {
        "startDate": start_date,
        "endDate": end_date,
        "dimensions": list(dimensions),
        "searchType": search_type,
        "dataState": data_state,
        "rowLimit": row_limit,
    }

    if filters:
        request_body["dimensionFilterGroups"] = [{
            "groupType": "and",
            "filters": filters,
        }]

    rows: list[dict[str, Any]] = []
    start_row = 0

    while True:
        request_body["startRow"] = start_row

        response = (
            service.searchanalytics()
            .query(
                siteUrl=property_url,
                body=request_body,
            )
            .execute()
        )

        api_rows = response.get("rows", [])
        if not api_rows:
            break

        for row in api_rows:
            keys = row.get("keys", [])
            item: dict[str, Any] = {
                "clicks": float(row.get("clicks", 0)),
                "impressions": float(row.get("impressions", 0)),
                "ctr": float(row.get("ctr", 0)),
                "position": float(row.get("position", 0)),
            }

            for index, dimension in enumerate(dimensions):
                item[dimension] = (
                    keys[index] if index < len(keys) else ""
                )

            rows.append(item)

        if len(api_rows) < row_limit:
            break

        start_row += row_limit

    return rows


def query_project_queries(
    project: str,
    start_date: str,
    end_date: str,
    row_limit: int = 25_000,
) -> list[dict[str, Any]]:
    return query_gsc(
        project=project,
        start_date=start_date,
        end_date=end_date,
        dimensions=("query",),
        row_limit=row_limit,
    )


def query_project_pages(
    project: str,
    start_date: str,
    end_date: str,
    row_limit: int = 25_000,
) -> list[dict[str, Any]]:
    return query_gsc(
        project=project,
        start_date=start_date,
        end_date=end_date,
        dimensions=("page",),
        row_limit=row_limit,
    )


def _print_test_results(
    project: str,
    start_date: str,
    end_date: str,
    dimension: str,
    display_rows: int,
    api_row_limit: int,
) -> None:
    try:
        if dimension == "query":
            data = query_project_queries(
                project=project,
                start_date=start_date,
                end_date=end_date,
                row_limit=api_row_limit,
            )
        else:
            data = query_project_pages(
                project=project,
                start_date=start_date,
                end_date=end_date,
                row_limit=api_row_limit,
            )
    except HttpError as exc:
        print("Google Search Console API error:")
        print(exc)
        raise SystemExit(1)
    except Exception as exc:
        print(f"GSC query error: {exc}")
        raise SystemExit(1)

    print(f"Project: {PROJECT_LABELS[project]}")
    print(f"Property: {GSC_PROPERTIES[project]}")
    print(f"Dimension: {dimension}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Rows returned: {len(data)}")
    print(f"Showing first {min(display_rows, len(data))} rows:\n")

    for index, item in enumerate(data[:display_rows], start=1):
        key = item.get(dimension, "")
        print(
            f"{index}. {key} | "
            f"clicks={item['clicks']:.0f} | "
            f"impressions={item['impressions']:.0f} | "
            f"ctr={item['ctr'] * 100:.2f}% | "
            f"position={item['position']:.2f}"
        )


def _test_project(args) -> None:
    project = args.project.strip().lower()

    if project not in GSC_PROPERTIES:
        raise SystemExit(
            "Unknown project. Choose: fdp, o2, usa_clean_master"
        )

    if args.start_date and args.end_date:
        start_date = args.start_date
        end_date = args.end_date
    elif args.start_date or args.end_date:
        raise SystemExit(
            "Provide both --start-date and --end-date, or neither."
        )
    else:
        end = date.today() - timedelta(days=3)
        start = end - timedelta(days=args.days - 1)
        start_date = start.isoformat()
        end_date = end.isoformat()

    _print_test_results(
        project=project,
        start_date=start_date,
        end_date=end_date,
        dimension=args.dimension,
        display_rows=args.rows,
        api_row_limit=args.api_row_limit,
    )


def _print_sites() -> None:
    try:
        sites = list_gsc_sites()
    except HttpError as exc:
        print("Google Search Console API error:")
        print(exc)
        raise SystemExit(1)
    except Exception as exc:
        print(f"GSC connection error: {exc}")
        raise SystemExit(1)

    if not sites:
        print("No Search Console properties were returned.")
        return

    print("Search Console properties available to this Google account:\n")

    for site in sites:
        print(
            f"{site['site_url']}"
            f"    [{site['permission_level']}]"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Google Search Console API helper for SEO MCP."
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    subparsers.add_parser(
        "list-sites",
        help="List GSC properties accessible to the authorized account.",
    )

    test_parser = subparsers.add_parser(
        "test-project",
        help="Query a small sample of live GSC data for a project.",
    )
    test_parser.add_argument(
        "--project",
        required=True,
        choices=["fdp", "o2", "usa_clean_master"],
    )
    test_parser.add_argument(
        "--dimension",
        choices=["query", "page"],
        default="query",
    )
    test_parser.add_argument(
        "--rows",
        type=int,
        default=10,
        help="Number of returned rows to display. Default: 10.",
    )
    test_parser.add_argument(
        "--api-row-limit",
        type=int,
        default=25_000,
        help="GSC rows requested per API page. Default: 25000.",
    )
    test_parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days for the default recent test window.",
    )
    test_parser.add_argument(
        "--start-date",
        default=None,
        help="Explicit YYYY-MM-DD start date.",
    )
    test_parser.add_argument(
        "--end-date",
        default=None,
        help="Explicit YYYY-MM-DD end date.",
    )

    args = parser.parse_args()

    if args.command == "list-sites":
        _print_sites()
    elif args.command == "test-project":
        _test_project(args)


if __name__ == "__main__":
    main()
