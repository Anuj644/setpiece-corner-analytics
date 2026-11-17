"""Cluster profile and snapshot figures for central-box positioning."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from setpiece.config import Config
from setpiece.plots.snapshot import plot_snapshot
from setpiece.rq3.features import FEATURES, ZONE_NAMES

_ATTACK = "#B13F27"
_DEFENCE = "#2458A6"


def save_snapshot(row, raw, cfg: Config, path: Path, *, show_source: bool = False) -> None:
    fig, axes = plt.subplots(1, 2 if show_source else 1, figsize=(9 if show_source else 5, 6))
    if show_source:
        native, normalized = axes
        plot_snapshot(row, raw, native, cfg, source=True)
        native.set_title("Source coordinates")
    else:
        normalized = axes
    plot_snapshot(row, raw, normalized, cfg)
    normalized.set_title("Left-corner reference")
    fig.suptitle(f"Central-box positioning | {row['team_name']} | Match {row['match_id']}")
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def save_cluster_profiles(centroids: pd.DataFrame, path: Path) -> None:
    """Show the mean number of attackers and defenders in the four configured zones."""
    k = len(centroids)
    fig, axes = plt.subplots(1, k, figsize=(3.2 * k, 3.7), sharey=True, squeeze=False)
    x = np.arange(len(ZONE_NAMES))
    width = 0.34
    near_far = ["Near\nsix-yard", "Near\ndeep", "Far\nsix-yard", "Far\ndeep"]
    for ax, (_, center) in zip(axes[0], centroids.iterrows()):
        counts = center[FEATURES].to_numpy(dtype=float)
        ax.bar(x - width / 2, counts[:4], width, color=_ATTACK, label="Attack")
        ax.bar(x + width / 2, counts[4:], width, color=_DEFENCE, label="Defence")
        status = "" if center["supported"] else " (descriptive)"
        ax.set_title(f"Group {center['label']}{status}")
        ax.set_xticks(x, near_far)
    axes[0, 0].set_ylabel("Mean players in area")
    axes[0, -1].legend()
    fig.suptitle("Central-box positioning")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
