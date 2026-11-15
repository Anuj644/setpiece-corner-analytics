"""Tests for RQ1 figures (Task 15)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from setpiece.rq1.figures import figure_title, plot_attribute_bars


def test_title_has_n_slice_dates():
    t = figure_title("PSG", "Ligue 1 2022/23", 170, "2022-08-06", "2023-06-03")
    assert "n=170" in t and "Ligue 1 2022/23" in t and "2022-08-06" in t


def test_bars_figure_saves_svg(tmp_path, summary_one_team):
    fig = plot_attribute_bars(summary_one_team, "technique", "t")
    fig.savefig(tmp_path / "a.svg")
    assert (tmp_path / "a.svg").stat().st_size > 0


def whiskers(fig):
    """The (y, ci_lo, ci_hi) of every whisker drawn on the first axes."""
    (lines,) = fig.axes[0].collections
    return sorted((float(s[0][1]), float(s[0][0]), float(s[1][0])) for s in lines.get_segments())


def test_every_bar_has_a_whisker_at_its_stored_interval(summary_one_team):
    fig = plot_attribute_bars(summary_one_team, "technique", "t")
    # bars run from the smallest share up: Outswinging 25 [0, 50] first, Inswinging 75 [50, 100]
    assert whiskers(fig) == [(0.0, 0.0, 50.0), (1.0, 50.0, 100.0)]
    assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == ["Outswinging", "Inswinging"]
    assert [p.get_width() for p in fig.axes[0].patches] == [25.0, 75.0]
    plt.close(fig)


def test_an_interval_that_does_not_contain_the_estimate_is_drawn_as_computed():
    """A percentile interval can lie to one side of the point estimate; ``xerr`` would refuse it."""
    summary = pd.DataFrame(
        {
            "attribute": ["technique"],
            "category": ["Inswinging"],
            "n": [1],
            "denominator": [12],
            "pct": [8.3],
            "ci_lo": [10.0],
            "ci_hi": [40.0],
        }
    )
    fig = plot_attribute_bars(summary, "technique", "t")
    assert whiskers(fig) == [(0.0, 10.0, 40.0)]
    plt.close(fig)


def test_bars_are_labelled_with_their_counts_and_the_interval_is_explained(summary_one_team):
    fig = plot_attribute_bars(summary_one_team, "technique", "t")
    texts = [t.get_text().strip() for t in fig.axes[0].texts]
    assert texts == ["2/8", "6/8"]
    assert "95% match-bootstrap interval" in fig.axes[0].get_xlabel()
    plt.close(fig)


def test_a_team_season_without_corners_draws_an_empty_chart_without_failing():
    none_row = pd.DataFrame(
        {
            "attribute": ["technique"],
            "category": ["none"],
            "n": [0],
            "denominator": [0],
            "pct": [np.nan],
            "ci_lo": [np.nan],
            "ci_hi": [np.nan],
        }
    )
    fig = plot_attribute_bars(none_row, "technique", "t")
    assert fig.axes[0].get_title() == "t"
    plt.close(fig)


@pytest.mark.parametrize("attribute", ["technique", "corner_side"])
def test_an_attribute_missing_from_the_summary_gives_an_empty_chart(summary_one_team, attribute):
    fig = plot_attribute_bars(summary_one_team[summary_one_team.attribute == "zzz"], attribute, "t")
    assert not fig.axes[0].patches
    plt.close(fig)
