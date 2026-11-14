"""Tests for RQ1 figures (Task 15)."""


from setpiece.rq1.figures import figure_title, plot_attribute_bars


def test_title_has_n_slice_dates():
    t = figure_title("PSG", "Ligue 1 2022/23", 170, "2022-08-06", "2023-06-03")
    assert "n=170" in t and "Ligue 1 2022/23" in t and "2022-08-06" in t


def test_bars_figure_saves_svg(tmp_path, summary_one_team):
    fig = plot_attribute_bars(summary_one_team, "technique", "t")
    fig.savefig(tmp_path / "a.svg")
    assert (tmp_path / "a.svg").stat().st_size > 0
