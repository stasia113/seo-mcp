SEO MCP - multi-project, dual-dataset, semantic-cluster report version

This version supports three projects and two GSC comparison datasets per project.

Projects and workbooks:

FDP Mold Remediation
  period: data/fdp/FDP-GSC.xlsx
  yoy:    data/fdp/FDP-GSC-YoY.xlsx

O2 Mold Testing
  period: data/o2/O2-GSC.xlsx
  yoy:    data/o2/O2-GSC-YoY.xlsx

USA Clean Master
  period: data/usa-clean-master/USA-Clean-Master-GSC.xlsx
  yoy:    data/usa-clean-master/USA-Clean-Master-GSC-YoY.xlsx

Dataset definitions:
  period = Last 3 months vs previous 3 months
  yoy    = Last 3 months vs same period last year
  both   = both datasets plus a cross-comparison report

Commercial-page exclusions:
  FDP: /blog/, /resources/, /case-study/, /images/, /videos/, /symptoms/, /recent-projects/, /glossary/
  O2:  all FDP exclusions plus /expert-advice/
  UCM: all FDP exclusions plus /knowledge-base/

Main MCP tool:
  seo_summary_tool

Examples:
  project=fdp, dataset=both
  project=o2, dataset=both
  project=usa_clean_master, dataset=both
  project=all, dataset=both

When dataset=both for a project, one combined HTML report is generated containing:
  - Cross-comparison of recent period and YoY
  - Recent-period analysis
  - YoY analysis
  - Semantic query cluster performance
  - Page performance
  - Commercial-page performance
  - SEO opportunities
  - Diagnosis and recovery sections

Query reporting:
  - The HTML report does not show separate raw query-performance tables.
  - Query-related losses, opportunities, diagnosis, and performance tables use
    semantic query clusters.
  - Each cluster shows its aggregated clicks, impressions, CTR, weighted position,
    changes, diagnosis, and expandable member queries.
  - The same cluster map is built from the union of the period and YoY query lists,
    so the topic groups remain consistent across comparisons.
  - Raw query-level losses/opportunities remain available in the MCP analysis output
    for auditability, but are not displayed as separate query tables in the HTML report.

Reports are written to the reports/ directory.

USA Clean Master commercial-page scope:
  Included URL prefixes:
    /carpetcleaning/
    /rugcleaning/
    /upholsterycleaning/
    /mattress-cleaning.aspx
  Excluded literal URL content:
    #
  For USA Clean Master, the allowlist above takes precedence over the general
  commercial exclusions. A page must match one of the included prefixes and
  must not contain '#'.

Semantic query clustering:
- Install requirements-clustering.txt.
- seo_summary_tool enables clustering by default.
- Default similarity threshold: 0.78.
- Default embedding model: all-MiniLM-L6-v2.
- Period and YoY query lists are combined to build one stable cluster map.
- Clicks and impressions are summed within each cluster.
- CTR is recalculated from aggregated clicks and impressions.
- Position is impressions-weighted.
