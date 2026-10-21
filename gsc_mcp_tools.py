"""Register read-only Google Search Console tools on the existing MCPServer."""

from gsc_api import list_gsc_sites as api_list_sites
from gsc_api import query_gsc as api_query_gsc


def register_gsc_tools(mcp) -> None:
    @mcp.tool()
    def list_gsc_sites() -> list[dict]:
        """List GSC properties accessible to the locally authorized Google account."""
        return api_list_sites()

    @mcp.tool()
    def query_gsc(
        site_url: str,
        start_date: str,
        end_date: str,
        dimension: str = "query",
        limit: int = 25,
    ) -> dict:
        """Fetch read-only Search Console query/page metrics for explicit dates.

        Use site_url exactly as shown by list_gsc_sites. CTR is returned
        as a fraction (0-1), not in percentage points.
        """
        return api_query_gsc(
            site_url=site_url,
            start_date=start_date,
            end_date=end_date,
            dimension=dimension,
            limit=limit,
        )
