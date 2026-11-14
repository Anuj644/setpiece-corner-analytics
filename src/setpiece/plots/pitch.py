"""Half-pitch drawing (Task 15) in StatsBomb coordinates: x 0-120, y 0-80, attacking goal at x = 120.

The penalty box and six-yard box come from the config (``box``, ``six_yard_x_min`` and the central
width band), so a figure always shows the zones the numbers were computed with.
"""

from __future__ import annotations

from matplotlib.patches import Rectangle

from setpiece.config import Config

# Facts of the StatsBomb coordinate system, not method parameters.
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
HALFWAY_X = PITCH_LENGTH / 2
GOAL_Y = (36.0, 44.0)  # the goal mouth on the goal line


def draw_half_pitch(ax, cfg: Config) -> None:
    """Draw the attacking half (x 60-120, y 0-80) on ``ax``: outline, penalty box, six-yard box, goal.

    Both boxes extend from their configured start line to the goal line at x = 120. The y axis is
    inverted, so y = 0 is at the top.
    """
    box = cfg.p("box")
    six_x = cfg.p("six_yard_x_min")
    six_lo, six_hi = cfg.p("zone_width_bands")["central"]  # the six-yard box is as wide as "central"

    ax.set_xlim(HALFWAY_X, PITCH_LENGTH)
    ax.set_ylim(PITCH_WIDTH, 0)
    ax.set_aspect("equal")

    ax.add_patch(
        Rectangle(
            (HALFWAY_X, 0), PITCH_LENGTH - HALFWAY_X, PITCH_WIDTH,
            fill=False, edgecolor="black", linewidth=1.5,
        )
    )  # fmt: skip
    ax.add_patch(
        Rectangle(
            (box["x_min"], box["y_min"]), PITCH_LENGTH - box["x_min"], box["y_max"] - box["y_min"],
            fill=False, edgecolor="black", linewidth=1.2,
        )
    )  # fmt: skip
    ax.add_patch(
        Rectangle(
            (six_x, six_lo), PITCH_LENGTH - six_x, six_hi - six_lo,
            fill=False, edgecolor="black", linewidth=1.0,
        )
    )  # fmt: skip
    ax.plot([PITCH_LENGTH, PITCH_LENGTH], GOAL_Y, color="black", linewidth=3)

    ax.set_xticks([])
    ax.set_yticks([])
