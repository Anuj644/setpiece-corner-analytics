"""Draw observed players and their camera footprint on a half-pitch."""

from pathlib import Path

from matplotlib.patches import Polygon, Rectangle

from setpiece.config import Config, load_config
from setpiece.plots.pitch import draw_half_pitch
from setpiece.rq3.features import snapshot_players, visible_polygon, zones

_ATTACK = "#B13F27"
_DEFENCE = "#2458A6"


def plot_snapshot(row, corner_360_row, ax, cfg: Config | None = None, *, source: bool = False):
    """Draw the source snapshot in the left-corner frame, with the four zones marked."""
    if cfg is None:
        cfg = load_config(Path(__file__).resolve().parents[3] / "configs" / "default.yaml")
    side = "left" if source else row["corner_side"]
    polygon = visible_polygon(corner_360_row["visible_area"], side)
    xy, attack, outfield = snapshot_players(corner_360_row["players"], side)
    draw_half_pitch(ax, cfg)
    ax.add_patch(Polygon(polygon, facecolor="#AEC7C0", edgecolor="#56847A", alpha=0.35, zorder=0))
    if not source:
        for x0, x1, y0, y1 in zones(cfg).values():
            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    fill=False,
                    edgecolor="#3F6048",
                    linestyle="--",
                    linewidth=0.9,
                )
            )
    for mask, color, team in ((attack, _ATTACK, "Attack"), (~attack, _DEFENCE, "Defence")):
        ax.scatter(*xy[mask & outfield].T, color=color, s=24, label=team, zorder=3)
    for mask, color in ((attack, _ATTACK), (~attack, _DEFENCE)):
        keeper = mask & ~outfield
        if keeper.any():
            ax.scatter(
                *xy[keeper].T,
                s=45,
                marker="s",
                color=color,
                edgecolors="black",
                linewidths=1.0,
                label="Keeper"
                if "Keeper" not in ax.get_legend_handles_labels()[1]
                else "_nolegend_",
                zorder=4,
            )
    ax.legend(loc="lower left", fontsize=8)
