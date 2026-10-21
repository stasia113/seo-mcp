
from __future__ import annotations

import html
from pathlib import Path
from typing import Any


def _esc(value: Any) -> str:
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def _fmt_int(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{int(round(float(value))):,}"
    except (TypeError, ValueError):
        return "-"


def _fmt_float(value: Any, decimals: int = 2) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "-"


def _fmt_pct(value: Any, decimals: int = 2) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):,.{decimals}f}%"
    except (TypeError, ValueError):
        return "-"


def _change_pct(current: Any, previous: Any) -> float | None:
    try:
        current = float(current)
        previous = float(previous)
    except (TypeError, ValueError):
        return None

    if previous == 0:
        return None

    return ((current - previous) / previous) * 100


def _change_text(value: Any, decimals: int = 1, suffix: str = "") -> str:
    if value is None:
        return "-"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "-"

    sign = "+" if value > 0 else ""
    return f"{sign}{value:,.{decimals}f}{suffix}"


def _delta_class(value: Any, inverse: bool = False) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""

    if value == 0:
        return ""

    positive = value > 0
    if inverse:
        positive = not positive

    return "positive" if positive else "negative"


def _kpi(label: str, value: str, delta: str, delta_class: str = "") -> str:
    return f"""
<div class="kpi">
  <div class="kpi-label">{_esc(label)}</div>
  <div class="kpi-value">{_esc(value)}</div>
  <div class="kpi-delta {delta_class}">{_esc(delta)}</div>
</div>
"""


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return '<p class="note">No rows available for this section.</p>'

    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)

    body_parts = []
    for row in rows:
        cells = []
        for index, value in enumerate(row):
            cells.append(
                f"<td>{_esc(value)}</td>"
            )
        body_parts.append("<tr>" + "".join(cells) + "</tr>")

    return f"""
<div class="table-wrap">
<table>
<thead><tr>{head}</tr></thead>
<tbody>
{''.join(body_parts)}
</tbody>
</table>
</div>
"""


def _compact_change_row(item: dict[str, Any]) -> list[str]:
    change = item.get("change", {})
    current = item.get("current", {})
    previous = item.get("previous", {})

    return [
        item.get("target", ""),
        _fmt_int(current.get("clicks")),
        _fmt_int(previous.get("clicks")),
        _change_text(change.get("clicks"), 0),
        _fmt_int(current.get("impressions")),
        _fmt_int(previous.get("impressions")),
        _change_text(change.get("impressions"), 0),
    ]


def _loss_rows(items: list[dict[str, Any]]) -> list[list[str]]:
    rows = []

    for item in items:
        current = item.get("current", {})
        previous = item.get("previous", {})
        change = item.get("change", {})

        rows.append([
            item.get("target", ""),
            _fmt_int(current.get("clicks")),
            _fmt_int(previous.get("clicks")),
            _change_text(change.get("clicks"), 0),
            _fmt_int(current.get("impressions")),
            _fmt_int(previous.get("impressions")),
            _change_text(change.get("impressions"), 0),
            _fmt_pct(current.get("ctr"), 2),
            _fmt_pct(previous.get("ctr"), 2),
            _change_text(change.get("ctr"), 2, " pp"),
            _fmt_float(current.get("position"), 2),
            _fmt_float(previous.get("position"), 2),
            _change_text(change.get("position"), 2),
            item.get("diagnosis", ""),
        ])

    return rows



def _semantic_cluster_table(items: list[dict[str, Any]]) -> str:
    if not items:
        return '<p class="note">No semantic query clusters are available.</p>'

    headers = [
        "Cluster",
        "Queries",
        "Current clicks",
        "Current impressions",
        "Current CTR",
        "Current position",
        "Click change",
        "Impression change",
        "Position change",
        "Diagnosis",
        "Member queries",
    ]

    rows = []
    for item in items:
        current = item.get("current", {})
        change = item.get("change", {})
        members = item.get("member_queries", [])

        member_html = "<details><summary>Show queries</summary><ul>"
        for member in members:
            member_html += (
                "<li>"
                + _esc(member.get("query", ""))
                + " - "
                + _fmt_int(member.get("clicks"))
                + " clicks, "
                + _fmt_int(member.get("impressions"))
                + " impressions</li>"
            )
        member_html += "</ul></details>"

        rows.append([
            item.get("cluster", ""),
            _fmt_int(item.get("query_count")),
            _fmt_int(current.get("clicks")),
            _fmt_int(current.get("impressions")),
            _fmt_pct(current.get("ctr"), 2),
            _fmt_float(current.get("position"), 2),
            _change_text(change.get("clicks"), 0),
            _change_text(change.get("impressions"), 0),
            _change_text(change.get("position"), 2),
            item.get("diagnosis", ""),
            member_html,
        ])

    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = []
    for row in rows:
        cells = []
        for index, value in enumerate(row):
            if index == len(row) - 1:
                cells.append(f"<td>{value}</td>")
            else:
                cells.append(f"<td>{_esc(value)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")

    return f"""
<div class="table-wrap">
<table>
<thead><tr>{head}</tr></thead>
<tbody>{''.join(body)}</tbody>
</table>
</div>
"""

def _opportunity_rows(items: list[dict[str, Any]]) -> list[list[str]]:
    rows = []

    for item in items:
        current = item.get("current", {})
        change = item.get("change", {})

        rows.append([
            item.get("target", ""),
            _fmt_int(current.get("impressions")),
            _fmt_int(current.get("clicks")),
            _fmt_pct(current.get("ctr"), 2),
            _fmt_float(current.get("position"), 2),
            _change_text(change.get("clicks"), 0),
            _change_text(change.get("position"), 2),
            _fmt_float(item.get("opportunity_score"), 1),
        ])

    return rows


def _svg_bar_chart(
    data: dict[str, int],
    title: str,
    width: int = 1000,
    row_height: int = 44
) -> str:
    """
    Create a self-contained SVG horizontal bar chart.
    No external charting library is required.
    """

    if not data:
        return (
            '<div class="chart-empty">No data available for this chart.</div>'
        )

    items = list(data.items())
    max_value = max(max(int(v), 0) for _, v in items) or 1
    height = 70 + len(items) * row_height

    left = 320
    right = 80
    chart_width = width - left - right

    parts = [
        (
            f'<svg class="svg-chart" viewBox="0 0 {width} {height}" '
            'xmlns="http://www.w3.org/2000/svg" role="img" '
            f'aria-label="{_esc(title)}">'
        ),
        (
            f'<text x="0" y="25" class="svg-title">'
            f'{_esc(title)}</text>'
        ),
    ]

    y = 58

    for label, value in items:
        value = max(int(value), 0)
        bar_width = int((value / max_value) * chart_width)

        parts.append(
            f'<text x="0" y="{y + 15}" class="svg-label">'
            f'{_esc(label)}</text>'
        )
        parts.append(
            f'<rect x="{left}" y="{y}" width="{chart_width}" '
            f'height="24" rx="5" class="svg-track"/>'
        )
        parts.append(
            f'<rect x="{left}" y="{y}" width="{bar_width}" '
            f'height="24" rx="5" class="svg-bar"/>'
        )
        parts.append(
            f'<text x="{left + chart_width + 12}" y="{y + 17}" '
            f'class="svg-value">{_fmt_int(value)}</text>'
        )

        y += row_height

    parts.append("</svg>")
    return "\n".join(parts)


def _comparison_svg(summary: dict[str, Any]) -> str:
    query = summary["queries"]["overview"]
    pages = summary["pages"]["overview"]

    labels = ["Semantic topics", "Pages"]
    current = [
        query["clicks"]["current"],
        pages["clicks"]["current"],
    ]
    previous = [
        query["clicks"]["previous"],
        pages["clicks"]["previous"],
    ]

    width = 1000
    height = 310
    left = 170
    right = 40
    top = 70
    bottom = 55
    plot_width = width - left - right
    plot_height = height - top - bottom

    max_value = max(
        max(current, default=0),
        max(previous, default=0),
        1
    )

    group_width = plot_width / len(labels)
    bar_width = 70

    comparison_label = summary["analysis"].get(
        "comparison",
        "comparison period"
    )

    parts = [
        f'<svg class="svg-chart" viewBox="0 0 {width} {height}" '
        'xmlns="http://www.w3.org/2000/svg" role="img" '
        f'aria-label="Current vs comparison clicks: {_esc(comparison_label)}">',
        '<text x="0" y="25" class="svg-title">Clicks: current vs comparison</text>',
        '<rect x="720" y="10" width="14" height="14" class="svg-current"/>',
        '<text x="742" y="22" class="svg-legend">Current</text>',
        '<rect x="805" y="10" width="14" height="14" class="svg-previous"/>',
        '<text x="827" y="22" class="svg-legend">Previous</text>',
    ]

    for i, label in enumerate(labels):
        x_center = left + group_width * (i + 0.5)

        current_height = (current[i] / max_value) * plot_height
        previous_height = (previous[i] / max_value) * plot_height

        current_x = x_center - bar_width - 5
        previous_x = x_center + 5

        current_y = top + plot_height - current_height
        previous_y = top + plot_height - previous_height

        parts.append(
            f'<rect x="{current_x:.1f}" y="{current_y:.1f}" '
            f'width="{bar_width}" height="{current_height:.1f}" '
            f'class="svg-current"/>'
        )
        parts.append(
            f'<rect x="{previous_x:.1f}" y="{previous_y:.1f}" '
            f'width="{bar_width}" height="{previous_height:.1f}" '
            f'class="svg-previous"/>'
        )

        parts.append(
            f'<text x="{current_x + bar_width / 2:.1f}" '
            f'y="{max(current_y - 8, 48):.1f}" text-anchor="middle" '
            f'class="svg-value">{_fmt_int(current[i])}</text>'
        )
        parts.append(
            f'<text x="{previous_x + bar_width / 2:.1f}" '
            f'y="{max(previous_y - 8, 48):.1f}" text-anchor="middle" '
            f'class="svg-value">{_fmt_int(previous[i])}</text>'
        )

        parts.append(
            f'<text x="{x_center:.1f}" y="{height - 18}" '
            f'text-anchor="middle" class="svg-label">{_esc(label)}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def _executive_findings(summary: dict[str, Any]) -> list[str]:
    query = summary["queries"]["overview"]
    query_analysis = summary["queries"]
    pages = summary["pages"]["overview"]
    commercial = summary["commercial_pages"]

    findings = []

    query_clicks = query["clicks"]
    page_clicks = pages["clicks"]
    page_impressions = pages["impressions"]

    query_click_pct = _change_pct(
        query_clicks["current"],
        query_clicks["previous"]
    )
    page_click_pct = _change_pct(
        page_clicks["current"],
        page_clicks["previous"]
    )

    findings.append(
        f"Clicks attributed to search queries changed from "
        f"{_fmt_int(query_clicks['previous'])} to "
        f"{_fmt_int(query_clicks['current'])}, "
        f"{_change_text(query_clicks['change'], 0)}"
        + (
            f" ({_change_text(query_click_pct, 1, '%')})."
            if query_click_pct is not None
            else "."
        )
    )

    findings.append(
        f"Page-level clicks changed from "
        f"{_fmt_int(page_clicks['previous'])} to "
        f"{_fmt_int(page_clicks['current'])}, "
        f"{_change_text(page_clicks['change'], 0)}"
        + (
            f" ({_change_text(page_click_pct, 1, '%')})."
            if page_click_pct is not None
            else "."
        )
    )

    findings.append(
        f"Page-level impressions changed by "
        f"{_change_text(page_impressions['change'], 0)}, "
        f"from {_fmt_int(page_impressions['previous'])} to "
        f"{_fmt_int(page_impressions['current'])}."
    )

    findings.append(
        f"{_fmt_int(query_analysis.get('semantic_cluster_loss_count', 0))} "
        f"semantic query clusters and "
        f"{_fmt_int(pages['rows_losing_clicks'])} page rows lost clicks."
    )

    findings.append(
        f"{_fmt_int(pages['ranking_declines'])} page rows show a "
        f"ranking decline greater than 1 position, while "
        f"{_fmt_int(pages['ranking_improvements'])} show an improvement "
        "greater than 1 position."
    )

    findings.append(
        f"{_fmt_int(commercial['loss_count'])} commercial pages "
        "lost clicks in the comparison."
    )

    return findings


def _action_plan(summary: dict[str, Any]) -> list[str]:
    query_counts = summary["queries"].get(
        "semantic_cluster_diagnosis_counts",
        {}
    )
    page_counts = summary["pages"]["diagnosis_counts"]
    commercial = summary["commercial_pages"]

    actions = []

    if page_counts.get("Ranking, visibility and CTR decline", 0):
        actions.append(
            "Review the highest-loss pages showing simultaneous ranking, "
            "visibility, and CTR decline. These combine several negative "
            "signals and should be examined at the page level first."
        )

    if page_counts.get("Ranking / visibility decline", 0):
        actions.append(
            "Review pages with ranking and visibility declines for changes "
            "in search demand, page relevance, internal competition, and "
            "SERP composition before making content changes."
        )

    if commercial.get("loss_count", 0):
        actions.append(
            "Review the largest commercial-page losses separately from "
            "informational content. The commercial-page filter excludes "
            f"{', '.join(commercial['excluded_sections'])}."
        )

    if summary["pages"]["top_opportunities"]:
        actions.append(
            "Work through the highest-impression ranking opportunities "
            "in positions 4-20. Check the page, query intent, title, "
            "snippet, internal links, and competing results before changing "
            "the page."
        )

    if query_counts.get("CTR decline", 0):
        actions.append(
            "Review semantic query clusters with CTR decline where ranking "
            "and impressions remained relatively stable. These are candidates "
            "for SERP snippet and intent review rather than assuming a "
            "ranking problem."
        )

    actions.append(
        "If the analysis needs to establish which URL ranks for a specific "
        "query, whether pages are competing, or whether changes differ by "
        "country or device, export query-by-page and the required GSC "
        "dimensions. The current Queries and Pages sheets are analyzed "
        "separately."
    )

    return actions[:8]


def _commercial_scope_description(summary: dict[str, Any]) -> str:
    commercial = summary["commercial_pages"]
    included = commercial.get("included_prefixes", [])
    excluded_contains = commercial.get("excluded_contains", [])
    excluded_sections = commercial.get("excluded_sections", [])

    parts = []

    if included:
        parts.append(
            "Only URLs nested under "
            + ", ".join(included)
            + " are included."
        )
    else:
        parts.append(
            "Commercial pages are identified by excluding "
            + ", ".join(excluded_sections)
            + "."
        )

    if excluded_contains:
        parts.append(
            "URLs containing "
            + ", ".join(excluded_contains)
            + " are excluded."
        )

    return " ".join(parts)


def _methodology(summary: dict[str, Any]) -> str:
    return (
        "This report is generated from the Queries and Pages sheets in "
        f"{summary['analysis']['source']}. The comparison is "
        f"{summary['analysis']['comparison']}. Query and page metrics are "
        "analyzed separately because the workbook does not provide a "
        "query-by-page cross-tab in the current tool. "
        f"{_commercial_scope_description(summary)} "
        "Semantic query clusters are built from the available query lists and "
        "use aggregated clicks and impressions with impressions-weighted "
        "position. Average position changes are interpreted as movement in "
        "the exported GSC rows, not as a fixed rank for the site."
    )


def build_html_report(summary: dict[str, Any]) -> Path:
    """
    Build a standalone HTML SEO report from seo_summary() output.

    The report uses self-contained CSS and inline SVG charts, so no
    external assets or charting libraries are required to view it.
    """

    analysis_meta = summary.get("analysis", {})
    source = str(
        analysis_meta.get(
            "source",
            "GSC export"
        )
    )
    brand = str(
        analysis_meta.get(
            "brand",
            "SEO"
        )
    )

    stem = Path(source).stem or "GSC"
    report_dir = Path(__file__).parent / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        report_dir /
        f"{stem}-SEO-Summary.html"
    )

    query = summary["queries"]
    pages = summary["pages"]
    commercial = summary["commercial_pages"]

    findings = _executive_findings(summary)
    actions = _action_plan(summary)

    query_diag = query.get(
        "semantic_cluster_diagnosis_counts",
        {}
    )
    page_diag = pages["diagnosis_counts"]

    query_diag_chart = _svg_bar_chart(
        query_diag,
        "Semantic query cluster loss diagnosis distribution"
    )

    page_diag_chart = _svg_bar_chart(
        page_diag,
        "Page loss diagnosis distribution"
    )

    comparison_chart = _comparison_svg(summary)

    page_click_change_pct = _change_pct(
        pages["overview"]["clicks"]["current"],
        pages["overview"]["clicks"]["previous"]
    )

    page_impression_change_pct = _change_pct(
        pages["overview"]["impressions"]["current"],
        pages["overview"]["impressions"]["previous"]
    )

    commercial_loss_count = commercial["loss_count"]

    kpis = "".join([
        _kpi(
            "Page clicks",
            _fmt_int(pages["overview"]["clicks"]["current"]),
            (
                f"{_change_text(page_click_change_pct, 1, '%')} "
                "vs comparison"
                if page_click_change_pct is not None
                else "Change unavailable"
            ),
            _delta_class(page_click_change_pct)
        ),
        _kpi(
            "Page impressions",
            _fmt_int(pages["overview"]["impressions"]["current"]),
            (
                f"{_change_text(page_impression_change_pct, 1, '%')} "
                "vs comparison"
                if page_impression_change_pct is not None
                else "Change unavailable"
            ),
            _delta_class(page_impression_change_pct)
        ),
        _kpi(
            "Semantic clusters losing clicks",
            _fmt_int(query.get("semantic_cluster_loss_count", 0)),
            (
                f"{_fmt_int(query.get('semantic_cluster_count', 0))} "
                "clusters analyzed"
            ),
            ""
        ),
        _kpi(
            "Commercial pages losing clicks",
            _fmt_int(commercial_loss_count),
            "Filtered from page export",
            "negative" if commercial_loss_count else ""
        ),
    ])

    semantic_cluster_table = _semantic_cluster_table(
        query.get("semantic_clusters", [])
    )

    semantic_cluster_loss_table = _semantic_cluster_table(
        query.get("semantic_cluster_losses", [])
    )

    semantic_cluster_opportunity_table = _semantic_cluster_table(
        query.get("semantic_cluster_opportunities", [])
    )

    page_loss_table = _table(
        [
            "Page",
            "Current clicks",
            "Comparison clicks",
            "Click change",
            "Current impressions",
            "Comparison impressions",
            "Impression change",
            "Current CTR",
            "Comparison CTR",
            "CTR change",
            "Current position",
            "Comparison position",
            "Position change",
            "Diagnosis",
        ],
        _loss_rows(pages["top_click_losses"])
    )

    commercial_loss_table = _table(
        [
            "Commercial page",
            "Current clicks",
            "Comparison clicks",
            "Click change",
            "Current impressions",
            "Comparison impressions",
            "Impression change",
            "Current CTR",
            "Comparison CTR",
            "CTR change",
            "Current position",
            "Comparison position",
            "Position change",
            "Diagnosis",
        ],
        _loss_rows(commercial["top_click_losses"])
    )

    page_opportunity_table = _table(
        [
            "Page",
            "Current impressions",
            "Current clicks",
            "Current CTR",
            "Current position",
            "Click change",
            "Position change",
            "Opportunity score",
        ],
        _opportunity_rows(pages["top_opportunities"])
    )

    diagnosis_notes = []
    for label, count in sorted(
        page_diag.items(),
        key=lambda item: item[1],
        reverse=True
    )[:5]:
        diagnosis_notes.append(
            f"{_esc(label)}: {_fmt_int(count)} rows"
        )

    diagnosis_note_html = (
        "<ul>" +
        "".join(
            f"<li>{item}</li>"
            for item in diagnosis_notes
        ) +
        "</ul>"
        if diagnosis_notes
        else '<p class="note">No diagnosed losses were returned.</p>'
    )

    finding_html = "".join(
        f"""
<div class="finding">
  <div class="num">{index}</div>
  <div><p>{_esc(text)}</p></div>
</div>
"""
        for index, text in enumerate(findings, start=1)
    )

    action_html = "".join(
        f"<li>{_esc(action)}</li>"
        for action in actions
    )

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>{_esc(brand)} Organic Search Performance Summary</title>
<style>
:root {{
  color-scheme: light dark;
  --bg: #f5f6f7;
  --card: #ffffff;
  --text: #202124;
  --muted: #646970;
  --border: #d9dde2;
  --soft: #eef1f4;
  --positive: #246a3b;
  --negative: #a12626;
  --accent: #315f7d;
  --chart: #315f7d;
  --chart-soft: #9aa8b3;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #151719;
    --card: #202326;
    --text: #f0f1f2;
    --muted: #b6bac0;
    --border: #3b4046;
    --soft: #292e33;
    --positive: #8bd3a2;
    --negative: #ffaaaa;
    --accent: #9bc5df;
    --chart: #9bc5df;
    --chart-soft: #68737c;
  }}
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font-family: Arial, Helvetica, sans-serif;
  line-height: 1.55;
  background: var(--bg);
  color: var(--text);
}}
main {{
  width: min(1140px, calc(100% - 32px));
  margin: auto;
  padding: 38px 0 70px;
}}
h1 {{
  font-size: clamp(31px, 5vw, 48px);
  line-height: 1.08;
  margin: 0 0 10px;
}}
h2 {{ font-size: 26px; margin: 0 0 15px; }}
h3 {{ font-size: 19px; margin: 20px 0 8px; }}
p {{ margin: 0 0 14px; }}
.subtitle {{ color: var(--muted); margin-bottom: 28px; }}
.section {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 25px;
  margin: 18px 0;
}}
.kpis {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin-top: 20px;
}}
.kpi {{
  background: var(--soft);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px;
}}
.kpi-label {{ color: var(--muted); font-size: 13px; }}
.kpi-value {{ font-size: 27px; font-weight: 700; }}
.kpi-delta {{ font-size: 13px; }}
.positive {{ color: var(--positive); }}
.negative {{ color: var(--negative); }}
.callout {{
  background: var(--soft);
  border-left: 4px solid var(--accent);
  padding: 14px 16px;
  margin: 16px 0;
}}
table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
th, td {{
  padding: 10px 9px;
  border-bottom: 1px solid var(--border);
  text-align: right;
  white-space: nowrap;
}}
th:first-child, td:first-child {{
  text-align: left;
  white-space: normal;
  min-width: 220px;
}}
th {{
  color: var(--muted);
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: .03em;
}}
.table-wrap {{ overflow-x: auto; }}
.svg-chart {{
  width: 100%;
  display: block;
  background: white;
  border-radius: 8px;
  margin: 14px 0 4px;
}}
.svg-title {{
  font: 700 17px Arial, Helvetica, sans-serif;
  fill: #202124;
}}
.svg-label {{
  font: 13px Arial, Helvetica, sans-serif;
  fill: #646970;
}}
.svg-value {{
  font: 700 13px Arial, Helvetica, sans-serif;
  fill: #202124;
}}
.svg-legend {{
  font: 12px Arial, Helvetica, sans-serif;
  fill: #646970;
}}
.svg-track {{ fill: #eef1f4; }}
.svg-bar {{ fill: #315f7d; }}
.svg-current {{ fill: #315f7d; }}
.svg-previous {{ fill: #9aa8b3; }}
.chart-empty {{
  padding: 30px;
  background: var(--soft);
  border-radius: 8px;
  color: var(--muted);
}}
.finding {{
  display: grid;
  grid-template-columns: 42px 1fr;
  gap: 12px;
  margin: 18px 0;
}}
.num {{
  width: 36px;
  height: 36px;
  display: grid;
  place-items: center;
  border-radius: 50%;
  background: var(--accent);
  color: white;
  font-weight: 700;
}}
ul, ol {{ padding-left: 23px; }}
li {{ margin: 8px 0; }}
.note {{ color: var(--muted); font-size: 13px; }}
details summary {{
  cursor: pointer;
  color: var(--accent);
}}
details ul {{
  margin: 8px 0 0;
  padding-left: 18px;
}}
.meta-grid {{
  display: grid;
  grid-template-columns: 170px 1fr;
  gap: 8px 18px;
  font-size: 14px;
}}
.meta-label {{ color: var(--muted); }}
@media (max-width: 760px) {{
  .kpis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .section {{ padding: 18px; }}
  .meta-grid {{ grid-template-columns: 1fr; gap: 2px; }}
}}
@media print {{
  body {{ background: white; }}
  main {{ width: 100%; padding: 0; }}
  .section {{ break-inside: avoid; }}
}}
</style>
</head>
<body>
<main>

<h1>{_esc(brand)} Organic Search Performance Summary</h1>
<p class="subtitle">{_esc(summary['analysis']['comparison'])}</p>

<section class="section">
<h2>Report scope</h2>
<div class="meta-grid">
<div class="meta-label">Source</div><div>{_esc(source)}</div>
<div class="meta-label">Semantic query clusters</div><div>{_fmt_int(query.get('semantic_cluster_count', 0))}</div>
<div class="meta-label">Page rows</div><div>{_fmt_int(pages['overview']['rows_analyzed'])}</div>
<div class="meta-label">Commercial-page filter</div><div>Applied to the Pages sheet</div>
</div>
</section>

<section class="section">
<h2>Executive Summary</h2>
<div class="kpis">
{kpis}
</div>

<div class="finding-list">
{finding_html}
</div>
</section>

<section class="section">
<h2>Site-wide comparison</h2>
<p>
Search demand is summarized through semantic query clusters, while Pages remains
a URL-level analysis. The comparison is useful for identifying whether losses are
visible across search topics, pages, or both.
</p>
{comparison_chart}

{_table(
    [
        "Dataset",
        "Current clicks",
        "Comparison clicks",
        "Click change",
        "Current impressions",
        "Comparison impressions",
        "Impression change",
        "Rows losing clicks",
        "Ranking declines >1"
    ],
    [
        [
            "Semantic query clusters",
            _fmt_int(query["overview"]["clicks"]["current"]),
            _fmt_int(query["overview"]["clicks"]["previous"]),
            _change_text(query["overview"]["clicks"]["change"], 0),
            _fmt_int(query["overview"]["impressions"]["current"]),
            _fmt_int(query["overview"]["impressions"]["previous"]),
            _change_text(query["overview"]["impressions"]["change"], 0),
            _fmt_int(query.get("semantic_cluster_loss_count", 0)),
            _fmt_int(query.get("semantic_cluster_ranking_decline_count", 0)),
        ],
        [
            "Pages",
            _fmt_int(pages["overview"]["clicks"]["current"]),
            _fmt_int(pages["overview"]["clicks"]["previous"]),
            _change_text(pages["overview"]["clicks"]["change"], 0),
            _fmt_int(pages["overview"]["impressions"]["current"]),
            _fmt_int(pages["overview"]["impressions"]["previous"]),
            _change_text(pages["overview"]["impressions"]["change"], 0),
            _fmt_int(pages["overview"]["rows_losing_clicks"]),
            _fmt_int(pages["overview"]["ranking_declines"]),
        ],
    ]
)}
</section>

<section class="section">
<h2>Semantic query cluster performance</h2>
<p>
Queries are grouped by semantic similarity and analyzed as topic clusters rather
than as separate query rows. Clicks and impressions are summed within each cluster.
CTR is recalculated from the aggregated clicks and impressions, while average
position is impressions-weighted. The same cluster map is built from the union of
the available period and YoY query lists so the topic groups remain consistent
across comparisons.
</p>

<h3>Top semantic query clusters</h3>
{semantic_cluster_table}

<h3>Largest semantic cluster losses</h3>
{semantic_cluster_loss_table}

<h3>Semantic cluster loss diagnosis</h3>
{query_diag_chart}

<h3>Top semantic cluster opportunities</h3>
{semantic_cluster_opportunity_table}
</section>

<section class="section">
<h2>Page performance</h2>
<p>
These are the largest page-level click losses in the Pages export. The table
shows clicks, impressions, CTR, and average position for the current and comparison
click loss can be interpreted in the context of visibility and ranking changes.
Page-level losses are sorted by absolute click decline.
</p>
{page_loss_table}

<h3>Page loss diagnosis</h3>
{page_diag_chart}

<div class="callout">
<strong>Diagnosis distribution:</strong>
{diagnosis_note_html}
</div>
</section>

<section class="section">
<h2>Commercial page performance</h2>
<p>
{_esc(_commercial_scope_description(summary))} It shows clicks, impressions,
CTR, and average position for both periods so commercial-page losses can be
interpreted by visibility, ranking, and click-through changes rather than clicks alone.
</p>

{commercial_loss_table}
</section>

<section class="section">
<h2>SEO opportunities</h2>
<p>
Opportunities are semantic query clusters or pages with at least the configured
impression threshold and an average position between 4 and 20. The opportunity score is only a sorting
aid based on impressions divided by current position.
</p>

<h3>Top semantic query cluster opportunities</h3>
{semantic_cluster_opportunity_table}

<h3>Top page opportunities</h3>
{page_opportunity_table}
</section>

<section class="section">
<h2>Prioritized recovery plan</h2>
<ol>
{action_html}
</ol>
</section>

<section class="section">
<h2>What the current data can and cannot establish</h2>
<ul>
<li>The workbook supports separate semantic query-cluster and page-level comparison and loss diagnosis.</li>
<li>The commercial-page view can isolate likely business pages using URL path exclusions.</li>
<li>The current tool does not establish which exact URL ranked for a specific query when Queries and Pages are supplied separately.</li>
<li>The current tool does not establish whether a loss came from cannibalization, country mix, device mix, or a specific SERP feature without the corresponding GSC dimensions.</li>
<li>Average position is a GSC metric across the exported impressions and should not be treated as a fixed ranking for every search.</li>
</ul>
</section>

<section class="section">
<h2>Methodology and caveats</h2>
<p>{_esc(_methodology(summary))}</p>
<p class="note">
Source: Google Search Console workbook supplied for the analysis.
</p>
</section>

</main>
</body>
</html>
"""

    output_path.write_text(
        html_doc,
        encoding="utf-8"
    )

    return output_path



def _extract_main_sections(document: str) -> str:
    """Extract the report sections after the standalone title block."""
    start = document.find('<section class="section">')
    end = document.rfind('</main>')
    if start == -1 or end == -1 or end <= start:
        return document
    return document[start:end]


def _comparison_table_for_both(period_summary: dict[str, Any], yoy_summary: dict[str, Any]) -> str:
    """Create a compact comparison of period and YoY movement."""

    def row(label: str, section: str, metric: str) -> list[str]:
        period_overview = period_summary[section]["overview"][metric]
        yoy_overview = yoy_summary[section]["overview"][metric]

        period_pct = _change_pct(
            period_overview["current"],
            period_overview["previous"]
        )
        yoy_pct = _change_pct(
            yoy_overview["current"],
            yoy_overview["previous"]
        )

        return [
            label,
            _change_text(period_overview["change"], 0),
            _change_text(period_pct, 1, "%"),
            _change_text(yoy_overview["change"], 0),
            _change_text(yoy_pct, 1, "%"),
        ]

    rows = [
        row("Query clicks", "queries", "clicks"),
        row("Query impressions", "queries", "impressions"),
        row("Page clicks", "pages", "clicks"),
        row("Page impressions", "pages", "impressions"),
    ]

    return _table(
        [
            "Metric",
            "Recent change",
            "Recent %",
            "YoY change",
            "YoY %",
        ],
        rows
    )


def _cross_diagnosis_table(period_summary: dict[str, Any], yoy_summary: dict[str, Any]) -> str:
    labels = sorted(
        set(period_summary["pages"]["diagnosis_counts"]) |
        set(yoy_summary["pages"]["diagnosis_counts"])
    )

    rows = [
        [
            label,
            _fmt_int(period_summary["pages"]["diagnosis_counts"].get(label, 0)),
            _fmt_int(yoy_summary["pages"]["diagnosis_counts"].get(label, 0)),
        ]
        for label in labels
    ]

    return _table(
        ["Page-loss diagnosis", "Recent period", "YoY"],
        rows
    )


def build_combined_html_report(
    period_summary: dict[str, Any],
    yoy_summary: dict[str, Any]
) -> Path:
    """
    Build one standalone HTML report containing both comparison datasets.

    The report includes a cross-comparison followed by the full recent-period
    and YoY analyses. Temporary single-dataset reports are removed after their
    HTML bodies and shared styles are extracted.
    """

    brand = str(
        period_summary.get("analysis", {}).get(
            "brand", "SEO"
        )
    )

    period_path = build_html_report(period_summary)
    yoy_path = build_html_report(yoy_summary)

    try:
        period_document = period_path.read_text(encoding="utf-8")
        yoy_document = yoy_path.read_text(encoding="utf-8")
    finally:
        # The combined report is the requested output. Remove the temporary
        # standalone files generated above.
        try:
            period_path.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            yoy_path.unlink(missing_ok=True)
        except OSError:
            pass

    style_start = period_document.find("<style>")
    style_end = period_document.find("</style>", style_start)
    if style_start == -1 or style_end == -1:
        style_block = ""
    else:
        style_block = period_document[style_start:style_end + len("</style>")]

    period_sections = _extract_main_sections(period_document)
    yoy_sections = _extract_main_sections(yoy_document)

    period_label = period_summary["analysis"]["comparison"]
    yoy_label = yoy_summary["analysis"]["comparison"]

    cross_table = _comparison_table_for_both(
        period_summary,
        yoy_summary
    )

    cross_diagnosis = _cross_diagnosis_table(
        period_summary,
        yoy_summary
    )

    period_click_pct = _change_pct(
        period_summary["pages"]["overview"]["clicks"]["current"],
        period_summary["pages"]["overview"]["clicks"]["previous"]
    )
    yoy_click_pct = _change_pct(
        yoy_summary["pages"]["overview"]["clicks"]["current"],
        yoy_summary["pages"]["overview"]["clicks"]["previous"]
    )

    report_dir = Path(__file__).parent / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    output_path = report_dir / f"{brand.replace(' ', '-')}-GSC-SEO-Summary.html"

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>{_esc(brand)} GSC SEO Summary</title>
{style_block}
</head>
<body>
<main>
<h1>{_esc(brand)} GSC SEO Summary</h1>
<p class="subtitle">Recent-period and year-over-year analysis</p>

<section class="section">
<h2>Cross-comparison</h2>
<p>
The recent-period comparison shows what changed against the immediately preceding
three months. The YoY comparison shows how the same current three-month period
compares with the corresponding period last year. The two baselines are kept separate.
</p>
<div class="kpis">
{_kpi(
    "Recent page clicks",
    _fmt_int(period_summary["pages"]["overview"]["clicks"]["current"]),
    _change_text(period_click_pct, 1, "%") + " vs previous 3 months",
    _delta_class(period_click_pct)
)}
{_kpi(
    "YoY page clicks",
    _fmt_int(yoy_summary["pages"]["overview"]["clicks"]["current"]),
    _change_text(yoy_click_pct, 1, "%") + " vs same period last year",
    _delta_class(yoy_click_pct)
)}
{_kpi(
    "Recent commercial losses",
    _fmt_int(period_summary["commercial_pages"]["loss_count"]),
    "commercial pages lost clicks",
    "negative" if period_summary["commercial_pages"]["loss_count"] else ""
)}
{_kpi(
    "YoY commercial losses",
    _fmt_int(yoy_summary["commercial_pages"]["loss_count"]),
    "commercial pages lost clicks",
    "negative" if yoy_summary["commercial_pages"]["loss_count"] else ""
)}
</div>

<h3>Movement by metric</h3>
{cross_table}

<h3>Page-loss diagnosis</h3>
{cross_diagnosis}
</section>

<section class="section">
<h2>Recent period</h2>
<p class="subtitle">{_esc(period_label)}</p>
{period_sections}
</section>

<section class="section">
<h2>Year over year</h2>
<p class="subtitle">{_esc(yoy_label)}</p>
{yoy_sections}
</section>

<section class="section">
<h2>Comparison methodology</h2>
<p>
The report deliberately keeps the two GSC comparisons independent. A recent-period
loss is not treated as a YoY loss, and a YoY loss is not treated as evidence of a
recent deterioration. The cross-comparison is used to identify whether a pattern is
recent, persistent, or different between the two baselines.
</p>
</section>

</main>
</body>
</html>
"""

    output_path.write_text(html_doc, encoding="utf-8")
    return output_path
