"""Tests for the forecast reliability figure (Task 20)."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from setpiece.rq1.forecast.figures import plot_reliability


@pytest.fixture
def outcomes_and_predictions():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300).astype(float)
    return y, {
        "model": rng.uniform(0.05, 0.95, 300),
        "league_side": np.full(300, 0.6),
        "team_side_shrunk": rng.uniform(0.3, 0.8, 300),
    }


def test_one_line_per_method_plus_the_diagonal(outcomes_and_predictions):
    y, preds = outcomes_and_predictions
    fig = plot_reliability(y, preds, "t", chosen="league_side")
    ax = fig.axes[0]
    labels = [line.get_label() for line in ax.get_lines()]
    assert labels == ["perfect calibration", "model", "league_side (chosen)", "team_side_shrunk"]
    plt.close(fig)


def test_the_chosen_method_is_drawn_thicker_and_the_others_are_not(outcomes_and_predictions):
    y, preds = outcomes_and_predictions
    fig = plot_reliability(y, preds, "t", chosen="model")
    widths = {line.get_label(): line.get_linewidth() for line in fig.axes[0].get_lines()}
    assert widths["model (chosen)"] > widths["league_side"] == widths["team_side_shrunk"]
    plt.close(fig)


def test_every_method_is_drawn_from_its_own_predictions(outcomes_and_predictions):
    from setpiece.rq1.forecast.evaluate import reliability_table

    y, preds = outcomes_and_predictions
    fig = plot_reliability(y, preds, "t", chosen="model")
    lines = {line.get_label().split(" ")[0]: line for line in fig.axes[0].get_lines()}
    for name, p in preds.items():
        rt = reliability_table(y, p)
        rt = rt[rt["n"] > 0]
        assert list(lines[name].get_xdata()) == pytest.approx(list(rt["mean_p"]))
        assert list(lines[name].get_ydata()) == pytest.approx(list(rt["frac_pos"]))
    plt.close(fig)


def test_only_the_chosen_method_gets_bin_counts_and_all_are_listed_in_the_legend(
    outcomes_and_predictions,
):
    y, preds = outcomes_and_predictions
    fig = plot_reliability(y, preds, "title text", chosen="team_side_shrunk")
    ax = fig.axes[0]
    counts = [int(a.get_text()) for a in ax.texts]
    assert sum(counts) == len(y)  # the bins of the chosen method hold every test corner
    legend = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend == ["perfect calibration", "model", "league_side", "team_side_shrunk (chosen)"]
    assert ax.get_title() == "title text"
    plt.close(fig)


def test_without_a_chosen_method_nothing_is_marked(outcomes_and_predictions):
    y, preds = outcomes_and_predictions
    fig = plot_reliability(y, preds, "t")
    assert not any("(chosen)" in line.get_label() for line in fig.axes[0].get_lines())
    assert not fig.axes[0].texts
    plt.close(fig)
