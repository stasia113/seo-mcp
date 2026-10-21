"""Read-only Google Search Console client for a local SEO MCP server.

First authorize interactively via ``python connect_gsc.py``.
MCP tool calls then reuse the cached token without opening a browser.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Any

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

ROOT = Path(__file__).resolve().parent
CREDENTIALS_FILE = ROOT / "credentials.json"
TOKEN_FILE = ROOT / ".gsc_token.json"
SCOPES = ["https://www.googleapis.com/auth/webmasters.readonly"]


def _save_token(credentials: Credentials) -> None:
    TOKEN_FILE.write_text(credentials.to_json(), encoding="utf-8")
    if os.name != "nt":
        TOKEN_FILE.chmod(0o600)


def authorize() -> Credentials:
    """Authorize this computer via your browser and save a local OAuth token."""
    credentials = None

    if TOKEN_FILE.is_file():
        credentials = Credentials.from_authorized_user_file(
            str(TOKEN_FILE), SCOPES
        )

    if credentials and credentials.valid:
        return credentials

    if credentials and credentials.expired and credentials.refresh_token:
        try:
            credentials.refresh(Request())
            _save_token(credentials)
            return credentials
        except RefreshError:
            # For example, an external OAuth app still in Testing may have
            # issued a refresh token that expired after seven days.
            credentials = None

    if not CREDENTIALS_FILE.is_file():
        raise FileNotFoundError(
            "Missing credentials.json. Download a Desktop OAuth client JSON "
            "from Google Cloud and place it next to gsc_api.py."
        )

    flow = InstalledAppFlow.from_client_secrets_file(
        str(CREDENTIALS_FILE), scopes=SCOPES
    )
    credentials = flow.run_local_server(
        host="localhost",
        port=0,
        open_browser=True,
        access_type="offline",
        prompt="consent",
    )
    _save_token(credentials)
    return credentials


def _saved_credentials() -> Credentials:
    """Return valid saved credentials; never start an interactive sign-in."""
    if not TOKEN_FILE.is_file():
        raise RuntimeError(
            "Google Search Console is not authorized on this computer. "
            "Run 'python connect_gsc.py' in VS Code first."
        )

    credentials = Credentials.from_authorized_user_file(
        str(TOKEN_FILE), SCOPES
    )

    if credentials.valid:
        return credentials

    if credentials.expired and credentials.refresh_token:
        try:
            credentials.refresh(Request())
        except RefreshError as exc:
            raise RuntimeError(
                "The Google OAuth token could not be refreshed. "
                "Run 'python connect_gsc.py' to reconnect this computer."
            ) from exc
        _save_token(credentials)
        return credentials

    raise RuntimeError(
        "Saved Google credentials are no longer valid. "
        "Run 'python connect_gsc.py' to authorize again."
    )


def _service():
    return build(
        "searchconsole", "v1", credentials=_saved_credentials(),
        cache_discovery=False
    )


def list_gsc_sites() -> list[dict[str, str]]:
    """List exact GSC property IDs and permission levels for this account."""
    response = _service().sites().list().execute()
    return [
        {
            "siteUrl": entry.get("siteUrl", ""),
            "permissionLevel": entry.get("permissionLevel", ""),
        }
        for entry in response.get("siteEntry", [])
    ]


def query_gsc(
    site_url: str,
    start_date: str,
    end_date: str,
    dimension: str = "query",
    limit: int = 25,
) -> dict[str, Any]:
    """Read top search queries or pages for an exact Search Console property.

    CTR is returned as a 0-1 fraction by Google's API. No conversion is
    performed in this connection test; downstream reports must format it.
    """
    if dimension not in {"query", "page"}:
        raise ValueError("dimension must be 'query' or 'page'.")
    if not 1 <= limit <= 1000:
        raise ValueError("limit must be between 1 and 1000 for this test tool.")

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    if start > end:
        raise ValueError("start_date must be on or before end_date.")
    if not site_url.strip():
        raise ValueError("site_url must be the exact property URL from list_gsc_sites.")

    response = _service().searchanalytics().query(
        siteUrl=site_url,
        body={
            "startDate": start_date,
            "endDate": end_date,
            "dimensions": [dimension],
            "type": "web",
            "dataState": "final",
            "rowLimit": limit,
            "startRow": 0,
        },
    ).execute()

    rows = [
        {
            dimension: row.get("keys", [""])[0],
            "clicks": row.get("clicks", 0),
            "impressions": row.get("impressions", 0),
            "ctr_fraction": row.get("ctr", 0),
            "position": row.get("position", 0),
        }
        for row in response.get("rows", [])
    ]

    return {
        "siteUrl": site_url,
        "start_date": start_date,
        "end_date": end_date,
        "dimension": dimension,
        "row_count": len(rows),
        "rows": rows,
    }
