"""Half-pitch drawing utilities (Task 15).

Uses the StatsBomb coordinate system: x 0–120, y 0–80.
The half-pitch covers the attacking half: x 60–120, y 0–80.
"""

from __future__ import annotations


def draw_half_pitch(ax) -> None:
    """Draw the attacking half of a football pitch on ``ax``.

    x: 60–120, y: 0–80. Includes the penalty box, six-yard box, and goal line.
    """
    ax.set_xlim(60, 120)
    ax.set_ylim(0, 80)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    # Pitch boundary (half only).
    ax.plot([60, 60], [0, 80], color="black", linewidth=1.5)  # halfway line
    ax.plot([120, 120], [0, 80], color="black", linewidth=1.5)  # goal line
    ax.plot([60, 120], [0, 0], color="black", linewidth=1.5)  # bottom touchline
    ax.plot([60, 120], [80, 80], color="black", linewidth=1.5)  # top touchline

    # Penalty box: x 102–120, y 18–62.
    ax.plot([102, 102], [18, 62], color="black", linewidth=1.2)
    ax.plot([60, 102], [18, 18], color="black", linewidth=1.2)
    ax.plot([60, 102], [62, 62], color="black", linewidth=1.2)

    # Six-yard box: x 114–120, y 30–50.
    ax.plot([114, 114], [30, 50], color="black", linewidth=1.0)
    ax.plot([60, 114], [30, 30], color="black", linewidth=1.0)
    ax.plot([60, 114], [50, 50], color="black", linewidth=1.0)

    # Goal posts.
    ax.plot([120, 120], [36, 44], color="black", linewidth=2.5)

    ax.set_xticks([])
    ax.set_yticks([])
