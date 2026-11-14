"""``rq1`` stage: derive corners, summarise, write derived tables and figures (Task 15)."""

from __future__ import annotations

from setpiece.config import Config
from setpiece.coverage import team_sample_table
from setpiece.io import derived_dir, read_table, write_manifest
from setpiece.rq1.corners import build_rq1_corners
from setpiece.rq1.figures import figure_title, plot_attribute_bars, plot_endpoints
from setpiece.rq1.summaries import summarise_rq1


def run_rq1(cfg: Config) -> dict:
    """Run the RQ1 stage.

    Reads the ``events`` table, derives ``rq1_corners``, builds team
    summaries and sample labels, and writes:
      - ``derived/rq1_corners.parquet``
      - ``derived/rq1_summary.parquet``
      - ``derived/team_samples.parquet``
      - figures in ``reports/figures/rq1/``

    Returns a counts dict for the manifest.
    """
    events = read_table("events", cfg)

    # Build rq1_corners.
    rq1 = build_rq1_corners(events, cfg)
    dd = derived_dir(cfg)
    dd.mkdir(parents=True, exist_ok=True)
    rq1.to_parquet(dd / "rq1_corners.parquet", index=False)

    # Build team sample table.
    samples = team_sample_table(events, cfg.slices)
    samples.to_parquet(dd / "team_samples.parquet", index=False)

    # Build summaries.
    summary = summarise_rq1(rq1, samples, cfg)
    summary.to_parquet(dd / "rq1_summary.parquet", index=False)

    # Figures: one bar chart per (team, attribute) for teams with corners.
    fig_dir = cfg.reports_dir / "figures" / "rq1"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for (comp_id, seas_id, team_id), grp in rq1.groupby(
        ["competition_id", "season_id", "team_id"]
    ):
        team_name = grp["team_name"].iloc[0] if "team_name" in grp.columns else str(team_id)
        # Find slice name.
        slice_name = f"{comp_id}/{seas_id}"
        for s in cfg.slices:
            if s.competition_id == comp_id and s.season_id == seas_id:
                slice_name = s.name
                break

        n = len(grp)
        dates = grp["match_date"].dropna().sort_values()
        date_min = str(dates.iloc[0]) if len(dates) > 0 else "?"
        date_max = str(dates.iloc[-1]) if len(dates) > 0 else "?"
        title = figure_title(team_name, slice_name, n, date_min, date_max)

        # Endpoint scatter.
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 8))
        plot_endpoints(grp, ax, title)
        fig.savefig(fig_dir / f"{comp_id}_{seas_id}_{team_id}_endpoints.svg")
        plt.close(fig)

        # Bar charts per attribute (use summary rows for this team).
        team_summary = summary[
            (summary["competition_id"] == comp_id)
            & (summary["season_id"] == seas_id)
            & (summary["team_id"] == team_id)
        ]
        for attr in ["technique", "coarse_zone"]:
            attr_data = team_summary[team_summary["attribute"] == attr]
            if not attr_data.empty:
                bar_fig = plot_attribute_bars(attr_data, attr, f"{team_name} — {attr}")
                bar_fig.savefig(fig_dir / f"{comp_id}_{seas_id}_{team_id}_{attr}.svg")
                plt.close(bar_fig)

    counts = {
        "rq1_corners": len(rq1),
        "teams": int(samples["team_id"].nunique()) if not samples.empty else 0,
        "summary_rows": len(summary),
    }

    write_manifest("rq1", cfg, counts)
    return counts
