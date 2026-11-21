"""Tests for the RQ2 summaries, sensitivity flags, delivery rates, figures and stage (Task 26)."""

import dataclasses
import json
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from rq2_helpers import outcome, outcomes_df, rq1_corner, rq1_df

from setpiece.io import read_table, write_table
from setpiece.rq2 import figures
from setpiece.rq2.chains import RQ2_COLUMNS
from setpiece.rq2.figures import plot_end_reasons, plot_shots_per_corner
from setpiece.rq2.summaries import (
    DELIVERY_COLUMNS,
    DELIVERY_DTYPES,
    END_REASON_COLUMNS,
    END_REASON_DTYPES,
    MEASURES,
    SUMMARY_COLUMNS,
    SUMMARY_DTYPES,
    end_reason_table,
    rates_by_delivery,
    sensitivity_flags,
    summarise_rq2,
)
from setpiece.stages import rq2 as rq2_stage
from setpiece.stages.rq2 import run_rq2
from setpiece.stats import match_bootstrap_ratio


def with_min_matches(cfg, n):
    """``cfg`` with an interval from ``n`` matches with an eligible corner on."""
    param = cfg.params["bootstrap_min_matches"]
    params = {**cfg.params, "bootstrap_min_matches": dataclasses.replace(param, value=n)}
    return dataclasses.replace(cfg, params=params)


@pytest.fixture
def cfg2(cfg):
    """The default config but for an interval from 2 matches on, so tiny fixtures have intervals."""
    return with_min_matches(cfg, 2)


def chains_in_matches(n, team=1):
    """Two complete chains of ``team`` in each of ``n`` matches (every third match has a shot)."""
    return outcomes_df([
        outcome(f"{team}-{m}-{i}", 100 + m, team, shot_count=int(m % 3 == 0 and i == 0))
        for m in range(n) for i in range(2)
    ])  # fmt: skip


# --- the tests of the plan ----------------------------------------------------------------


def test_denominator_excludes_incomplete(outcomes_with_incomplete, samples, cfg):
    s = summarise_rq2(outcomes_with_incomplete, samples, cfg).query("measure == 'shots_per_corner' and team_id == 1").iloc[0]
    assert s.denominator == 4 and s.n_incomplete == 1 and s.n_corners == 5


def test_numerator_le_denominator_for_proportions(outcomes_with_incomplete, samples, cfg):
    s = summarise_rq2(outcomes_with_incomplete, samples, cfg).query("measure == 'pct_corners_with_shot'")
    assert (s.numerator <= s.denominator).all()


def test_sensitivity_flag_half_width():
    main = pd.DataFrame({"team_id": [1], "measure": ["shots_per_corner"], "value": [0.30], "ci_lo": [0.20], "ci_hi": [0.40]})
    alt = {"cap10": main.assign(value=0.36), "cap20": main.assign(value=0.41), "nocap": main.assign(value=0.33)}
    assert bool(sensitivity_flags(main, alt).sensitive.iloc[0]) is True


def test_zero_denominator_rows(samples, cfg):
    s = summarise_rq2(pd.DataFrame(columns=RQ2_COLUMNS), samples, cfg)
    assert (s.denominator == 0).all() and s.value.isna().all()


# --- summarise_rq2 ------------------------------------------------------------------------


@pytest.fixture
def summary(outcomes_with_incomplete, samples, cfg2):
    return summarise_rq2(outcomes_with_incomplete, samples, cfg2)


def row(summary, team, measure):
    return summary.query("team_id == @team and measure == @measure").iloc[0]


def test_one_row_per_team_season_and_measure(summary):
    assert MEASURES == [
        "shots_per_corner", "pct_corners_with_shot", "goals_per_corner", "losses_per_corner",
        "defending_restarts_per_corner"]  # fmt: skip
    assert list(summary.columns) == SUMMARY_COLUMNS
    assert {c: str(t) for c, t in summary.dtypes.items()} == SUMMARY_DTYPES
    assert len(summary) == 15
    assert list(summary.team_id[::5]) == [1, 2, 3]  # samples order; the measures in plan order
    assert list(summary.measure[:5]) == MEASURES
    first = summary.iloc[0]
    assert (first.competition_id, first.season_id, first.team_name) == (9, 281, "Team A")
    assert (first.matches_in_sample, first.sample_label) == (4, "incomplete_season (4 of 34)")


def test_numerators_and_denominators_of_every_measure(summary):
    team1 = {m: (int(row(summary, 1, m).numerator), int(row(summary, 1, m).denominator)) for m in MEASURES}
    assert team1 == {
        "shots_per_corner": (3, 4), "pct_corners_with_shot": (3, 4), "goals_per_corner": (1, 4),
        "losses_per_corner": (1, 4), "defending_restarts_per_corner": (2, 4)}  # fmt: skip
    team2 = {m: (int(row(summary, 2, m).numerator), int(row(summary, 2, m).denominator)) for m in MEASURES}
    assert team2 == {
        "shots_per_corner": (1, 3), "pct_corners_with_shot": (1, 3), "goals_per_corner": (0, 3),
        "losses_per_corner": (0, 3), "defending_restarts_per_corner": (0, 3)}  # fmt: skip
    assert row(summary, 1, "shots_per_corner").value == 0.75
    assert row(summary, 2, "shots_per_corner").value == pytest.approx(1 / 3)


def test_defending_restarts_count_stoppage_endings_too(summary):
    """Chain 1-2 ends at the restart itself, chain 1-3 at a stoppage before its throw-in."""
    assert row(summary, 1, "defending_restarts_per_corner").numerator == 2
    assert row(summary, 2, "defending_restarts_per_corner").numerator == 0  # attacking new corner


def test_incomplete_chains_are_in_n_corners_only(summary):
    for team, n_corners, n_incomplete in [(1, 5, 1), (2, 3, 0), (3, 0, 0)]:
        for m in MEASURES:
            r = row(summary, team, m)
            assert (r.n_corners, r.n_incomplete) == (n_corners, n_incomplete)
    assert row(summary, 1, "shots_per_corner").numerator == 3  # not 6: chain 1-5's shots do not count


def test_interval_resamples_all_the_team_seasons_matches(summary, cfg):
    """Matches 101, 102 and 103 have corners; the fourth match of team 1 has none: a (0, 0) entry."""
    r = row(summary, 1, "shots_per_corner")
    num, den = np.array([1, 2, 0, 0]), np.array([2, 2, 0, 0])
    assert (r.ci_lo, r.ci_hi) == match_bootstrap_ratio(num, den, cfg.p("bootstrap_reps"), cfg.seed)
    assert r.ci_lo < r.value < r.ci_hi
    assert r.n_matches == 2  # matches with a complete chain


def test_matches_without_a_corner_are_part_of_the_resample(samples, cfg):
    """The interval resamples all matches_in_sample matches: the idle ones are (0, 0) draws."""
    shots = [(0, 0), (1, 0), (1, 1), (2, 0), (0, 1), (3, 0)]  # shots of the two chains of each match
    out = outcomes_df([outcome(f"{m}-{i}", m, 1, shot_count=n) for m, pair in enumerate(shots) for i, n in enumerate(pair)])
    intervals = {}
    for played in (6, 12):
        s = samples.assign(matches_in_sample=pd.array([played, 3, 3], dtype="Int64"))
        r = row(summarise_rq2(out, s, cfg), 1, "shots_per_corner")
        num, den = np.zeros(played), np.zeros(played)
        num[:6], den[:6] = [sum(pair) for pair in shots], 2
        assert (r.ci_lo, r.ci_hi) == match_bootstrap_ratio(num, den, cfg.p("bootstrap_reps"), cfg.seed)
        assert r.n_matches == 6  # the first match has no shot, but two complete chains: den > 0
        intervals[played] = (r.ci_lo, r.ci_hi)
    assert intervals[6] != intervals[12]


def test_interval_is_nan_below_the_minimum_of_informative_matches(summary):
    for m in MEASURES:
        one = row(summary, 2, m)  # all three corners in one match
        assert one.n_matches == 1 and pd.isna(one.ci_lo) and pd.isna(one.ci_hi)
        assert one.denominator == 3 and not pd.isna(one.value)
        none = row(summary, 3, m)
        assert none.n_matches == 0 and pd.isna(none.ci_lo) and pd.isna(none.ci_hi)
    assert not row(summary, 1, "shots_per_corner")[["ci_lo", "ci_hi"]].isna().any()


def test_the_interval_minimum_is_a_config_param(cfg):
    param = cfg.params["bootstrap_min_matches"]
    assert param.value == 5 and param.status == "starting_choice"


def test_an_interval_needs_the_configured_number_of_informative_matches(samples, cfg):
    def shots_row(n_matches, config):
        s = samples.assign(matches_in_sample=pd.array([n_matches, 3, 3], dtype="Int64"))
        return row(summarise_rq2(chains_in_matches(n_matches), s, config), 1, "shots_per_corner")

    four, five = shots_row(4, cfg), shots_row(5, cfg)
    assert (four.n_matches, five.n_matches) == (4, 5)
    assert pd.isna(four.ci_lo) and pd.isna(four.ci_hi) and not pd.isna(four.value)
    assert not pd.isna(five.ci_lo) and not pd.isna(five.ci_hi)
    assert not pd.isna(shots_row(4, with_min_matches(cfg, 4)).ci_lo)  # the threshold is the param
    assert pd.isna(shots_row(5, with_min_matches(cfg, 6)).ci_lo)


def test_the_minimum_counts_matches_with_a_complete_chain_not_matches_in_sample(samples, cfg):
    out = chains_in_matches(5)
    out.loc[out.match_id.isin([100, 101]), "outcome_status"] = "incomplete"  # 3 informative left
    s = samples.assign(matches_in_sample=pd.array([9, 3, 3], dtype="Int64"))
    r = row(summarise_rq2(out, s, cfg), 1, "shots_per_corner")
    assert r.n_matches == 3 and pd.isna(r.ci_lo)


def test_match_with_only_incomplete_chains_is_not_informative(samples, cfg2):
    out = outcomes_df([
        outcome("a", 1, 1, shot_count=1),
        outcome("b", 2, 1, end_reason="timeline_end", outcome_status="incomplete"),
    ])  # fmt: skip
    r = row(summarise_rq2(out, samples, cfg2), 1, "shots_per_corner")
    assert (r.n_corners, r.denominator, r.n_matches) == (2, 1, 1)
    assert pd.isna(r.ci_lo)


def test_zero_denominator_row_has_nothing_to_report(samples, cfg):
    s = summarise_rq2(pd.DataFrame(columns=RQ2_COLUMNS), samples, cfg)
    assert len(s) == 15 and list(s.columns) == SUMMARY_COLUMNS
    assert {c: str(t) for c, t in s.dtypes.items()} == SUMMARY_DTYPES
    assert (s.n_corners == 0).all() and (s.numerator == 0).all() and (s.n_matches == 0).all()
    assert s.ci_lo.isna().all() and s.ci_hi.isna().all()


def test_typed_empty_outcomes_and_no_samples(samples, cfg):
    empty = outcomes_df([])
    assert (summarise_rq2(empty, samples, cfg).denominator == 0).all()
    none = summarise_rq2(empty, samples.iloc[:0], cfg)
    assert len(none) == 0 and {c: str(t) for c, t in none.dtypes.items()} == SUMMARY_DTYPES


def test_only_samples_of_the_configured_slices_get_rows(outcomes_with_incomplete, samples, cfg):
    elsewhere = samples.iloc[:1].assign(competition_id=99, season_id=1, team_id=7)
    stray = outcomes_df([outcome("x", 900, 7, competition_id=99, season_id=1, shot_count=5)])
    both = pd.concat([samples, elsewhere], ignore_index=True)
    out = summarise_rq2(pd.concat([outcomes_with_incomplete, stray], ignore_index=True), both, cfg)
    pd.testing.assert_frame_equal(out, summarise_rq2(outcomes_with_incomplete, samples, cfg))


def test_a_team_season_missing_from_the_samples_raises(outcomes_with_incomplete, samples, cfg):
    with pytest.raises(ValueError, match=r"team_samples.*9/281/2|9/281/2.*team_samples"):
        summarise_rq2(outcomes_with_incomplete, samples[samples.team_id != 2], cfg)


def test_a_repeated_team_season_in_the_samples_raises(outcomes_with_incomplete, samples, cfg):
    with pytest.raises(ValueError, match="more than once"):
        summarise_rq2(outcomes_with_incomplete, pd.concat([samples, samples.iloc[:1]]), cfg)


def test_more_matches_with_corners_than_matches_in_sample_raises(outcomes_with_incomplete, samples, cfg):
    short = samples.assign(matches_in_sample=pd.array([2, 3, 3], dtype="Int64"))
    with pytest.raises(ValueError, match=r"matches_in_sample.*9/281/1|9/281/1.*matches_in_sample"):
        summarise_rq2(outcomes_with_incomplete, short, cfg)


@pytest.mark.parametrize("column", ["shot_count", "goal", "lost_possession"])
def test_null_outcome_in_a_complete_chain_raises_and_is_never_filled(outcomes_with_incomplete, samples, cfg, column):
    bad = outcomes_with_incomplete.copy()
    bad.loc[0, column] = pd.NA
    with pytest.raises(ValueError, match=column):
        summarise_rq2(bad, samples, cfg)


def test_null_outcome_in_an_incomplete_chain_is_not_used(outcomes_with_incomplete, samples, cfg):
    bad = outcomes_with_incomplete.copy()
    bad.loc[4, ["shot_count", "goal", "lost_possession"]] = pd.NA  # chain 1-5, incomplete
    pd.testing.assert_frame_equal(
        summarise_rq2(bad, samples, cfg), summarise_rq2(outcomes_with_incomplete, samples, cfg))


def test_null_ids_raise(outcomes_with_incomplete, samples, cfg):
    for column in ["match_id", "team_id", "competition_id", "season_id"]:
        bad = outcomes_with_incomplete.copy()
        bad.loc[0, column] = pd.NA
        with pytest.raises(ValueError, match=column):
            summarise_rq2(bad, samples, cfg)


@pytest.mark.parametrize("function", ["summary", "delivery", "end_reasons"])
def test_null_outcome_status_raises_instead_of_counting_as_incomplete(
    function, outcomes_with_incomplete, samples, rq1_for_outcomes, cfg2
):
    bad = outcomes_with_incomplete.copy()
    bad.loc[0, "outcome_status"] = pd.NA
    calls = {
        "summary": lambda: summarise_rq2(bad, samples, cfg2),
        "delivery": lambda: rates_by_delivery(bad, rq1_for_outcomes, cfg2),
        "end_reasons": lambda: end_reason_table(bad, samples, cfg2),
    }
    with pytest.raises(ValueError, match="outcome_status"):
        calls[function]()


def test_summarise_does_not_modify_its_inputs_and_is_repeatable(outcomes_with_incomplete, samples, cfg):
    before = outcomes_with_incomplete.copy(), samples.copy()
    first = summarise_rq2(outcomes_with_incomplete, samples, cfg)
    pd.testing.assert_frame_equal(outcomes_with_incomplete, before[0])
    pd.testing.assert_frame_equal(samples, before[1])
    pd.testing.assert_frame_equal(summarise_rq2(outcomes_with_incomplete, samples, cfg), first)


def test_row_order_of_the_outcomes_does_not_change_the_intervals(outcomes_with_incomplete, samples, cfg):
    shuffled = outcomes_with_incomplete.sample(frac=1, random_state=4)
    pd.testing.assert_frame_equal(
        summarise_rq2(shuffled, samples, cfg), summarise_rq2(outcomes_with_incomplete, samples, cfg))


def test_bootstrap_uses_the_configured_reps_and_seed(samples, cfg):
    shots = [(0, 0), (1, 0), (1, 1), (2, 0), (0, 1), (3, 0)]
    out = outcomes_df([outcome(f"{m}-{i}", m, 1, shot_count=n) for m, pair in enumerate(shots) for i, n in enumerate(pair)])
    s = samples.assign(matches_in_sample=pd.array([6, 3, 3], dtype="Int64"))

    def interval(config):
        r = row(summarise_rq2(out, s, config), 1, "shots_per_corner")
        return r.ci_lo, r.ci_hi

    assert interval(cfg) != interval(dataclasses.replace(cfg, seed=cfg.seed + 1))
    few = dataclasses.replace(cfg, params={**cfg.params, "bootstrap_reps": dataclasses.replace(cfg.params["bootstrap_reps"], value=50)})
    assert interval(few) == match_bootstrap_ratio(np.array([0, 1, 2, 2, 1, 3.0]), np.full(6, 2.0), 50, cfg.seed)


# --- end_reason_table ---------------------------------------------------------------------


def test_end_reason_table_counts_every_chain_with_nulls_kept(outcomes_with_incomplete, samples, cfg2):
    t = end_reason_table(outcomes_with_incomplete, samples, cfg2)
    assert list(t.columns) == END_REASON_COLUMNS
    assert {c: str(d) for c, d in t.dtypes.items()} == END_REASON_DTYPES
    assert t.n.sum() == 8 and len(t) == 8
    team1 = t[t.team_id == 1].set_index("end_reason")
    assert list(team1.index) == ["goal", "possession_change", "restart", "stoppage", "timeline_end"]
    assert (team1.loc["restart", "restart_type"], team1.loc["restart", "restart_team"]) == ("Goal Kick", "defending")
    assert (team1.loc["stoppage", "restart_type"], team1.loc["stoppage", "restart_team"]) == ("Throw-in", "defending")
    assert team1.loc[["goal", "possession_change", "timeline_end"], ["restart_type", "restart_team"]].isna().all().all()
    assert team1.loc["timeline_end", "outcome_status"] == "incomplete"  # incomplete chains are kept
    assert (t[t.team_id == 2].restart_team.dropna() == "attacking").all()


def test_end_reason_rows_carry_their_denominator_as_n_corners(outcomes_with_incomplete, samples, cfg2):
    """Every published rate has its denominator; none of its chains is left out, incomplete ones
    being rows of their own, so the excluded count is zero and needs no column."""
    t = end_reason_table(outcomes_with_incomplete, samples, cfg2)
    assert str(t.n_corners.dtype) == "Int64"
    assert t.groupby("team_id").n.sum().to_dict() == {1: 5, 2: 3}
    for team, n_corners in [(1, 5), (2, 3)]:  # team 1's five chains include the incomplete one
        rows = t[t.team_id == team]
        assert (rows.n_corners == n_corners).all() and rows.n.sum() == n_corners
        assert (rows.share == rows.n / rows.n_corners).all()
    assert t[t.outcome_status == "incomplete"].n.sum() == 1  # counted, as a row of its own


def test_end_reason_table_splits_by_restart_type_and_team(samples, cfg2):
    out = outcomes_df([
        outcome("a", 1, 1, restart_type="Goal Kick", restart_team="defending"),
        outcome("b", 1, 1, restart_type="Goal Kick", restart_team="defending"),
        outcome("c", 1, 1, restart_type="Throw-in", restart_team="attacking"),
        outcome("d", 2, 1, restart_type="Throw-in", restart_team="defending"),
    ])  # fmt: skip
    t = end_reason_table(out, samples, cfg2)
    assert [(r.restart_type, r.restart_team, r.n) for r in t.itertuples()] == [
        ("Goal Kick", "defending", 2), ("Throw-in", "attacking", 1), ("Throw-in", "defending", 1)]


def test_end_reason_table_empty_is_typed(samples, cfg):
    for empty in (outcomes_df([]), pd.DataFrame(columns=RQ2_COLUMNS)):
        t = end_reason_table(empty, samples, cfg)
        assert len(t) == 0 and list(t.columns) == END_REASON_COLUMNS
        assert {c: str(d) for c, d in t.dtypes.items()} == END_REASON_DTYPES


def end_reason_fixture():
    """Team 1 in three matches (four in the sample): five chains, one of them incomplete."""
    return outcomes_df([
        outcome("a", 1, 1, restart_type="Goal Kick", restart_team="defending"),
        outcome("b", 1, 1, end_reason="goal", shot_count=1, goal=True),
        outcome("c", 2, 1, restart_type="Goal Kick", restart_team="defending"),
        outcome("d", 2, 1, end_reason="timeline_end", outcome_status="incomplete"),
        outcome("e", 3, 1, end_reason="goal", shot_count=1, goal=True),
    ])  # fmt: skip


def padding_fixture():
    """Team 1 in six matches, 21 chains: 7 goals, 13 goal-kick restarts and 1 incomplete chain.

    Per match (101..106), as (goals, restarts, incomplete): (2, 2, 0), (1, 3, 1), (0, 3, 0),
    (3, 1, 0), (0, 2, 0), (1, 2, 0). The counts differ from match to match, so the interval of a
    row is not a single value, and a team that played more matches than it has chains in is one
    whose interval depends on the (0, 0) matches it is padded with.
    """
    shapes = {101: "GGRR", 102: "GRRRP", 103: "RRR", 104: "GGGR", 105: "RR", 106: "GRR"}
    kinds = {
        "G": {"end_reason": "goal", "shot_count": 1, "goal": True},
        "R": {"restart_type": "Goal Kick", "restart_team": "defending"},
        "P": {"end_reason": "timeline_end", "outcome_status": "incomplete"},
    }
    return outcomes_df([
        outcome(f"{m}-{i}", m, 1, **kinds[kind])
        for m, shape in shapes.items() for i, kind in enumerate(shape)
    ])  # fmt: skip


def test_end_reason_rows_have_a_share_and_a_padded_match_bootstrap_interval(samples, cfg):
    sample = samples.assign(matches_in_sample=pd.array([10, 3, 3], dtype="Int64"))  # 4 idle matches
    t = end_reason_table(padding_fixture(), sample, cfg)
    assert list(t.columns[-5:]) == ["n", "n_corners", "share", "ci_lo", "ci_hi"]
    assert [str(t[c].dtype) for c in ("n", "n_corners", "share", "ci_lo", "ci_hi")] == (
        ["Int64", "Int64"] + ["Float64"] * 3
    )
    assert t.end_reason.tolist() == ["goal", "restart", "timeline_end"]
    # the shares: the chains of the row over all 21 chains of the team, the incomplete one included
    assert t.n.tolist() == [7, 13, 1] and t.n_corners.tolist() == [21, 21, 21]
    assert t.share.tolist() == [7 / 21, 13 / 21, 1 / 21]
    assert t.outcome_status.tolist() == ["complete", "complete", "incomplete"]
    assert t.groupby("team_id").share.sum().iloc[0] == pytest.approx(1.0)

    # per match (101..106): all chains of the team are the denominator, the chains of the row's
    # key the numerator; the sample has four more matches, each a (0, 0) entry of the resample
    den = np.array([4, 5, 3, 4, 2, 3.0])
    numerators = {
        "goal": np.array([2, 1, 0, 3, 0, 1.0]),
        "restart": np.array([2, 3, 3, 1, 2, 2.0]),
        "timeline_end": np.array([0, 1, 0, 0, 0, 0.0]),
    }
    reps, seed = cfg.p("bootstrap_reps"), cfg.seed
    padding_matters = []
    for r in t.itertuples():
        num = numerators[r.end_reason]
        padded = match_bootstrap_ratio(np.r_[num, np.zeros(4)], np.r_[den, np.zeros(4)], reps, seed)
        assert (r.ci_lo, r.ci_hi) == padded
        padding_matters.append(padded != match_bootstrap_ratio(num, den, reps, seed))
    assert any(padding_matters)  # so a table that left the idle matches out would fail above


def test_end_reason_intervals_follow_the_informative_match_minimum(samples, cfg):
    def goals(n_matches):
        out = chains_in_matches(n_matches).assign(end_reason="goal")
        s = samples.assign(matches_in_sample=pd.array([n_matches, 3, 3], dtype="Int64"))
        return end_reason_table(out, s, cfg).iloc[0]

    assert pd.isna(goals(4).ci_lo) and pd.isna(goals(4).ci_hi) and goals(4).share == 1.0
    assert not pd.isna(goals(5).ci_lo)


def test_end_reason_table_raises_for_team_seasons_it_cannot_pad(samples, cfg2):
    out = end_reason_fixture()
    with pytest.raises(ValueError, match="team_samples"):
        end_reason_table(out, samples[samples.team_id != 1], cfg2)
    with pytest.raises(ValueError, match="matches_in_sample"):
        end_reason_table(out, samples.assign(matches_in_sample=pd.array([2, 3, 3], dtype="Int64")), cfg2)


# --- sensitivity_flags --------------------------------------------------------------------


def main_frame(value=0.30, lo=0.20, hi=0.40, **extra):
    return pd.DataFrame({"team_id": [1], "measure": ["shots_per_corner"], "value": [value],
                         "ci_lo": [lo], "ci_hi": [hi], **extra})  # fmt: skip


def variants(main, cap10, cap20, nocap):
    return {"cap10": main.assign(value=cap10), "cap20": main.assign(value=cap20), "nocap": main.assign(value=nocap)}


def test_flag_columns_keep_one_value_per_variant():
    main = main_frame()
    flags = sensitivity_flags(main, variants(main, 0.36, 0.41, 0.33))
    assert list(flags.columns) == [
        "team_id", "measure", "value_main", "ci_lo", "ci_hi", "value_cap10", "value_cap20",
        "value_nocap", "sensitive"]  # fmt: skip
    r = flags.iloc[0]
    assert (r.value_main, r.value_cap10, r.value_cap20, r.value_nocap) == (0.30, 0.36, 0.41, 0.33)
    assert str(flags.sensitive.dtype) == "boolean" and str(flags.value_cap10.dtype) == "Float64"


def test_flag_is_false_when_every_variant_is_within_half_the_interval():
    main = main_frame()
    flags = sensitivity_flags(main, variants(main, 0.36, 0.25, 0.39))
    assert flags.sensitive.tolist() == [False]


def test_flag_boundary_is_strict():
    main = main_frame(value=0.5, lo=0.375, hi=0.625)  # half-width 0.125, exactly representable
    assert sensitivity_flags(main, variants(main, 0.625, 0.375, 0.5)).sensitive.tolist() == [False]
    assert sensitivity_flags(main, variants(main, 0.625, 0.375, 0.6251)).sensitive.tolist() == [True]


def test_flag_is_true_for_a_move_in_either_direction():
    main = main_frame()
    assert sensitivity_flags(main, variants(main, 0.1, 0.3, 0.3)).sensitive.tolist() == [True]


def test_flag_is_unknown_when_the_main_interval_is_nan():
    main = main_frame(lo=np.nan, hi=np.nan)
    flags = sensitivity_flags(main, variants(main, 0.9, 0.9, 0.9))
    assert flags.sensitive.isna().tolist() == [True] and str(flags.sensitive.dtype) == "boolean"


def test_flag_is_unknown_when_the_main_interval_has_zero_width():
    main = main_frame(value=0.0, lo=0.0, hi=0.0)  # e.g. 0 shots in every match: nothing to compare to
    flags = sensitivity_flags(main, variants(main, 0.2, 0.0, 0.0))
    assert flags.sensitive.isna().tolist() == [True]
    same = main_frame(0.3, 0.3, 0.3)
    assert sensitivity_flags(same, variants(same, 0.3, 0.3, 0.9)).sensitive.isna().tolist() == [True]


def test_flags_carry_the_counts_and_n_matches_of_the_main_row(summary):
    flags = sensitivity_flags(summary, {"cap10": summary, "nocap": summary})
    assert list(flags.columns) == [
        "competition_id", "season_id", "team_id", "measure", "team_name", "matches_in_sample",
        "sample_label", "numerator", "denominator", "n_incomplete", "n_matches", "value_main",
        "ci_lo", "ci_hi", "value_cap10", "numerator_cap10", "denominator_cap10",
        "n_incomplete_cap10", "value_nocap", "numerator_nocap", "denominator_nocap",
        "n_incomplete_nocap", "sensitive"]  # fmt: skip
    assert flags.numerator.tolist() == [3, 1, 0] and flags.denominator.tolist() == [4, 3, 0]
    assert flags.n_incomplete.tolist() == [1, 0, 0] and flags.n_matches.tolist() == [2, 1, 0]
    for column in ("numerator", "denominator", "n_incomplete", "n_matches"):
        assert str(flags[column].dtype) == "Int64"
    assert flags.sensitive.tolist() == [False, pd.NA, pd.NA]  # no move; one match; no value


def test_flags_carry_the_numerator_denominator_and_excluded_count_of_every_variant(summary):
    """A variant's value is a rate, so it comes with the counts of that variant: the cap changes
    which chains are complete, hence the denominator and the excluded count, not only the shots."""
    cap10 = summary.assign(
        numerator=summary.numerator + 1, denominator=summary.denominator + 2,
        n_incomplete=summary.n_incomplete + 3)  # fmt: skip
    nocap = summary.assign(
        numerator=summary.numerator + 10, denominator=summary.denominator + 20,
        n_incomplete=summary.n_incomplete + 30)  # fmt: skip
    flags = sensitivity_flags(summary, {"cap10": cap10, "nocap": nocap})
    assert flags.numerator.tolist() == [3, 1, 0]  # the main variant's, unchanged
    assert flags.numerator_cap10.tolist() == [4, 2, 1]
    assert flags.denominator_cap10.tolist() == [6, 5, 2]
    assert flags.n_incomplete_cap10.tolist() == [4, 3, 3]
    assert flags.numerator_nocap.tolist() == [13, 11, 10]
    assert flags.denominator_nocap.tolist() == [24, 23, 20]
    assert flags.n_incomplete_nocap.tolist() == [31, 30, 30]
    counts = [f"{c}_{v}" for v in ("cap10", "nocap") for c in ("numerator", "denominator", "n_incomplete")]
    assert {c: str(flags[c].dtype) for c in counts} == dict.fromkeys(counts, "Int64")
    # a free-text key (the stage's keys come from the configured caps) is no exception
    odd = sensitivity_flags(summary, {"cap7.5": cap10})
    assert odd["numerator_cap7.5"].tolist() == [4, 2, 1]
    assert str(odd["n_incomplete_cap7.5"].dtype) == "Int64"


def test_flags_carry_a_variants_counts_only_when_that_frame_has_them(summary):
    bare = summary[["competition_id", "season_id", "team_id", "measure", "value", "ci_lo", "ci_hi"]]
    flags = sensitivity_flags(summary, {"cap10": summary, "nocap": bare})
    assert "denominator_cap10" in flags.columns
    assert [c for c in flags.columns if c.endswith("_nocap")] == ["value_nocap"]
    bare_main = sensitivity_flags(bare, {"cap10": summary})  # the main's own counts: none to carry
    assert "numerator" not in bare_main.columns and "n_incomplete" not in bare_main.columns
    assert bare_main.numerator_cap10.tolist() == [3, 1, 0]


def test_flag_is_unknown_when_a_variant_has_no_value_unless_another_variant_moved_enough():
    main = main_frame()
    assert sensitivity_flags(main, variants(main, np.nan, 0.31, 0.32)).sensitive.isna().tolist() == [True]
    assert sensitivity_flags(main, variants(main, np.nan, 0.5, 0.32)).sensitive.tolist() == [True]


def test_flags_are_for_shots_per_corner_only_and_keyed_by_team_season():
    main = pd.DataFrame({
        "competition_id": [9, 9, 9], "season_id": [281, 281, 281], "team_id": [1, 1, 2],
        "measure": ["shots_per_corner", "goals_per_corner", "shots_per_corner"],
        "value": [0.3, 0.1, 0.4], "ci_lo": [0.2, 0.0, 0.3], "ci_hi": [0.4, 0.2, 0.5],
        "team_name": ["A", "A", "B"], "matches_in_sample": [3, 3, 3], "sample_label": ["s", "s", "s"],
    })  # fmt: skip
    alt = {
        "cap10": main.assign(value=[0.9, 0.1, 0.41]),  # team 1 shots moves 0.6; its goals row is not read
        "nocap": main.assign(value=[0.3, 0.1, 0.4]),
    }
    flags = sensitivity_flags(main, alt)
    assert list(flags.columns) == [
        "competition_id", "season_id", "team_id", "measure", "team_name", "matches_in_sample",
        "sample_label", "value_main", "ci_lo", "ci_hi", "value_cap10", "value_nocap", "sensitive"]  # fmt: skip
    assert flags.team_id.tolist() == [1, 2] and set(flags.measure) == {"shots_per_corner"}
    assert flags.sensitive.tolist() == [True, False]
    assert flags.value_cap10.tolist() == [0.9, 0.41]


def test_variant_keys_are_free_text_and_must_exist():
    main = main_frame()
    flags = sensitivity_flags(main, {"cap7.5": main.assign(value=0.31)})
    assert "value_cap7.5" in flags.columns and flags.sensitive.tolist() == [False]
    with pytest.raises(ValueError, match="alternative"):
        sensitivity_flags(main, {})


def test_flags_never_look_up_a_duplicate_row():
    main = main_frame()
    with pytest.raises(ValueError):  # pandas MergeError is a ValueError
        sensitivity_flags(main, {"cap10": pd.concat([main, main])})


# --- rates_by_delivery --------------------------------------------------------------------


def delivery_row(by_delivery, attribute, category, measure="shots_per_corner"):
    return by_delivery.query("attribute == @attribute and category == @category and measure == @measure").iloc[0]


@pytest.fixture
def by_delivery(outcomes_with_incomplete, rq1_for_outcomes, cfg2):
    return rates_by_delivery(outcomes_with_incomplete, rq1_for_outcomes, cfg2)


def test_delivery_rows_are_slice_attribute_category_and_measure(by_delivery):
    assert list(by_delivery.columns) == DELIVERY_COLUMNS
    assert {c: str(t) for c, t in by_delivery.dtypes.items()} == DELIVERY_DTYPES
    assert len(by_delivery) == 18  # (3 techniques + 3 zones + 3 proxy values) x 2 shot measures
    assert set(by_delivery.measure) == {"shots_per_corner", "pct_corners_with_shot"}
    assert set(by_delivery.attribute) == {"technique", "coarse_zone", "short_corner_proxy"}
    assert (by_delivery.competition_id == 9).all() and (by_delivery.season_id == 281).all()
    assert (by_delivery.slice_name == "1. Bundesliga 2023/24").all()  # the stray slice has no chains
    first = by_delivery.iloc[0]
    assert (first.attribute, first.category, first.measure) == ("technique", "Inswinging", "shots_per_corner")


def test_delivery_categories_are_text_with_unknown_kept(by_delivery):
    cats = lambda a: sorted(by_delivery[by_delivery.attribute == a].category.unique())
    assert cats("technique") == ["Inswinging", "Outswinging", "Unknown"]
    assert cats("coarse_zone") == ["Unknown", "central", "near"]  # a null zone
    assert cats("short_corner_proxy") == ["False", "True", "Unknown"]  # a proxy that is NA


def test_delivery_denominators_are_complete_chains_of_that_category(by_delivery):
    ins = delivery_row(by_delivery, "technique", "Inswinging")
    assert (ins.numerator, ins.denominator, ins.n_corners, ins.n_incomplete) == (3, 3, 4, 1)
    assert ins.value == 1.0 and ins.n_matches == 3
    out = delivery_row(by_delivery, "technique", "Outswinging", "pct_corners_with_shot")
    assert (out.numerator, out.denominator, out.n_corners, out.value) == (1, 2, 2, 0.5)
    unknown = delivery_row(by_delivery, "technique", "Unknown")
    assert (unknown.numerator, unknown.denominator, unknown.value) == (0, 2, 0.0)
    for attribute in ("technique", "coarse_zone", "short_corner_proxy"):
        total = by_delivery[(by_delivery.attribute == attribute) & (by_delivery.measure == "shots_per_corner")]
        assert total.denominator.sum() == 7 and total.n_corners.sum() == 8  # every chain once
    false = delivery_row(by_delivery, "short_corner_proxy", "False")
    na = delivery_row(by_delivery, "short_corner_proxy", "Unknown")
    assert (false.denominator, false.numerator, na.denominator, na.numerator) == (3, 3, 2, 1)


def test_delivery_interval_resamples_the_slices_matches(by_delivery, cfg):
    """Matches with a corner in slice 9/281, both teams: 101, 102, 103 and 201."""
    r = delivery_row(by_delivery, "technique", "Outswinging")
    lo, hi = match_bootstrap_ratio(np.array([0, 1, 0, 0]), np.array([0, 1, 0, 1]), cfg.p("bootstrap_reps"), cfg.seed)
    assert (r.ci_lo, r.ci_hi) == (lo, hi) and r.n_matches == 2
    one = delivery_row(by_delivery, "technique", "Unknown", "pct_corners_with_shot")
    assert one.n_matches == 2 and not pd.isna(one.ci_lo)


def test_delivery_interval_is_nan_with_fewer_than_bootstrap_min_matches_informative_matches(cfg):
    out = outcomes_df([outcome("a", 1, 1, shot_count=1), outcome("b", 1, 2, shot_count=0),
                       outcome("c", 2, 1, shot_count=0)])  # fmt: skip
    rq1 = rq1_df([rq1_corner("a", "Inswinging", "near", False), rq1_corner("b", "Inswinging", "near", False),
                  rq1_corner("c", "Outswinging", "far", False)])  # fmt: skip
    r = delivery_row(rates_by_delivery(out, rq1, cfg), "technique", "Outswinging")
    assert r.denominator == 1 and r.n_matches == 1 and pd.isna(r.ci_lo) and pd.isna(r.ci_hi)
    assert not pd.isna(r.value)


def test_delivery_interval_needs_the_configured_number_of_informative_matches(cfg):
    def inswinging(n_matches):
        out = chains_in_matches(n_matches)
        rq1 = rq1_df([rq1_corner(e, "Inswinging", "near", False) for e in out.corner_event_id])
        return delivery_row(rates_by_delivery(out, rq1, cfg), "technique", "Inswinging")

    four, five = inswinging(4), inswinging(5)
    assert four.n_matches == 4 and pd.isna(four.ci_lo) and pd.isna(four.ci_hi)
    assert five.n_matches == 5 and not pd.isna(five.ci_lo)


def test_delivery_slices_come_in_config_order_with_their_names(cfg):
    out = outcomes_df([
        outcome("a", 1, 1, competition_id=9, season_id=281),
        outcome("b", 2, 2, competition_id=43, season_id=106),
    ])  # fmt: skip
    rq1 = rq1_df([rq1_corner("a", "Inswinging", "near", False), rq1_corner("b", "Inswinging", "near", False)])
    t = rates_by_delivery(out, rq1, cfg)
    assert list(dict.fromkeys(zip(t.competition_id, t.season_id, t.slice_name))) == [
        (43, 106, "FIFA World Cup 2022"), (9, 281, "1. Bundesliga 2023/24")]


def test_a_corner_without_an_rq1_row_raises(outcomes_with_incomplete, rq1_for_outcomes, cfg):
    stale = rq1_for_outcomes[rq1_for_outcomes.event_id != "2-3"]
    with pytest.raises(ValueError, match=r"rq1_corners.*2-3|2-3.*rq1_corners"):
        rates_by_delivery(outcomes_with_incomplete, stale, cfg)


def test_a_duplicate_rq1_corner_raises(outcomes_with_incomplete, rq1_for_outcomes, cfg):
    with pytest.raises(ValueError):  # the one-to-one join
        rates_by_delivery(outcomes_with_incomplete, pd.concat([rq1_for_outcomes, rq1_for_outcomes.iloc[:1]]), cfg)


def test_delivery_nulls_in_complete_chains_raise(outcomes_with_incomplete, rq1_for_outcomes, cfg):
    bad = outcomes_with_incomplete.copy()
    bad.loc[1, "shot_count"] = pd.NA
    with pytest.raises(ValueError, match="shot_count"):
        rates_by_delivery(bad, rq1_for_outcomes, cfg)


def test_delivery_empty_outcomes_gives_a_typed_empty_table(rq1_for_outcomes, cfg):
    for empty in (outcomes_df([]), pd.DataFrame(columns=RQ2_COLUMNS)):
        t = rates_by_delivery(empty, rq1_for_outcomes, cfg)
        assert len(t) == 0 and list(t.columns) == DELIVERY_COLUMNS
        assert {c: str(d) for c, d in t.dtypes.items()} == DELIVERY_DTYPES


def test_delivery_does_not_modify_its_inputs(outcomes_with_incomplete, rq1_for_outcomes, cfg):
    before = outcomes_with_incomplete.copy(), rq1_for_outcomes.copy()
    rates_by_delivery(outcomes_with_incomplete, rq1_for_outcomes, cfg)
    pd.testing.assert_frame_equal(outcomes_with_incomplete, before[0])
    pd.testing.assert_frame_equal(rq1_for_outcomes, before[1])


# --- figures ------------------------------------------------------------------------------


def only_complete(outcomes):
    return outcomes[outcomes.outcome_status == "complete"]


def test_shots_figure_has_the_title_and_one_row_per_team_with_a_value(summary):
    fig = plot_shots_per_corner(summary[summary.measure == "shots_per_corner"], "T: n=7 corners")
    try:
        ax = fig.axes[0]
        assert ax.get_title().startswith("T: n=7 corners")
        # team 3 has no value; the lowest value is at the bottom; chain 1-5 is incomplete
        assert [t.get_text() for t in ax.get_yticklabels()] == [
            "Team B (3 eligible of 3 corners, 0 incomplete)",
            "Team A (4 eligible of 5 corners, 1 incomplete)"]  # fmt: skip
    finally:
        figures.plt.close(fig)


def test_shots_figure_labels_are_just_n_when_nothing_is_incomplete(
    outcomes_with_incomplete, samples, cfg2
):
    done = summarise_rq2(only_complete(outcomes_with_incomplete), samples, cfg2)
    fig = plot_shots_per_corner(done[done.measure == "shots_per_corner"], "t")
    try:
        assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == ["Team B (n=3)", "Team A (n=4)"]
    finally:
        figures.plt.close(fig)


def test_shots_figure_draws_whiskers_that_exclude_the_estimate_and_nan_rows(summary):
    shots = summary[summary.measure == "shots_per_corner"].copy()
    shots.loc[shots.team_id == 1, ["ci_lo", "ci_hi"]] = [0.8, 0.9]  # a percentile interval can miss 0.75
    fig = plot_shots_per_corner(shots, "t")
    try:
        ax = fig.axes[0]
        # team 2 (value 1/3, no interval) is the bottom row, team 1 (value 0.75) the top one
        segments = ax.collections[0].get_segments()
        assert len(segments) == 1  # one whisker: team 2's interval is NaN
        np.testing.assert_allclose(segments[0], [[0.8, 1], [0.9, 1]])
        filled, hollow = ax.lines
        assert (list(filled.get_xdata()), list(filled.get_ydata())) == ([0.75], [1])
        assert list(hollow.get_xdata()) == pytest.approx([1 / 3]) and list(hollow.get_ydata()) == [0]
        assert hollow.get_markerfacecolor() == "none"
        assert [x.get_text() for x in ax.get_legend().get_texts()] == ["no interval (too few matches)"]
    finally:
        figures.plt.close(fig)
    nothing = plot_shots_per_corner(shots.assign(value=np.nan, ci_lo=np.nan, ci_hi=np.nan), "t")
    try:
        assert [x.get_text() for x in nothing.axes[0].texts] == ["no eligible corners"]
        assert nothing.axes[0].get_legend() is None and not nothing.axes[0].collections
    finally:
        figures.plt.close(nothing)


def test_end_reason_figure_has_the_title_and_a_legend_with_fixed_colours(
    outcomes_with_incomplete, summary, samples, cfg2
):
    t = end_reason_table(outcomes_with_incomplete, samples, cfg2)
    names = summary[["competition_id", "season_id", "team_id", "team_name"]].drop_duplicates()
    data = t.merge(names, on=["competition_id", "season_id", "team_id"])
    fig = plot_end_reasons(data, "T: n=8 corners")
    try:
        ax = fig.axes[0]
        assert ax.get_title().startswith("T: n=8 corners")
        assert [x.get_text() for x in ax.get_yticklabels()] == [
            "Team A (4 eligible of 5 corners, 1 incomplete)",
            "Team B (3 eligible of 3 corners, 0 incomplete)"]  # fmt: skip
        assert {x.get_text() for x in ax.get_legend().get_texts()} == {r.replace("_", " ") for r in t.end_reason}
    finally:
        figures.plt.close(fig)
    fig_again = plot_end_reasons(data.iloc[:2], "t")
    figures.plt.close(fig_again)  # a subset of the reasons keeps its colours and still draws


def test_end_reason_figure_labels_are_just_n_when_nothing_is_incomplete(
    outcomes_with_incomplete, summary, samples, cfg2
):
    t = end_reason_table(only_complete(outcomes_with_incomplete), samples, cfg2)
    names = summary[["competition_id", "season_id", "team_id", "team_name"]].drop_duplicates()
    fig = plot_end_reasons(t.merge(names, on=["competition_id", "season_id", "team_id"]), "t")
    try:
        assert [x.get_text() for x in fig.axes[0].get_yticklabels()] == ["Team A (n=4)", "Team B (n=3)"]
    finally:
        figures.plt.close(fig)


def test_figures_without_data_say_so_and_draw_no_legend(samples, cfg):
    empty = end_reason_table(outcomes_df([]), samples, cfg).assign(team_name=pd.array([], dtype="string"))
    fig = plot_end_reasons(empty, "T: n=0 corners")
    try:
        assert fig.axes[0].get_title() == "T: n=0 corners" and fig.axes[0].get_legend() is None
        assert [x.get_text() for x in fig.axes[0].texts] == ["no corners"]
    finally:
        figures.plt.close(fig)


# --- the rq2 stage ------------------------------------------------------------------------


def derived(cfg, name):
    return read_table(f"derived/{name}", cfg)


def test_stage_writes_the_four_typed_summary_tables(rq2_cfg):
    run_rq2(rq2_cfg)
    summary = derived(rq2_cfg, "rq2_summary")
    assert list(summary.columns) == SUMMARY_COLUMNS and len(summary) == 20  # 4 teams x 5 measures
    assert {c: str(t) for c, t in summary.dtypes.items()} == SUMMARY_DTYPES
    assert sorted(summary.team_id.unique()) == [10, 20, 30, 40]
    team20 = summary[summary.team_id == 20]  # its only corner runs off the end of the match
    assert (team20.denominator == 0).all() and (team20.n_incomplete == 1).all() and team20.value.isna().all()
    delivery = derived(rq2_cfg, "rq2_by_delivery")
    assert list(delivery.columns) == DELIVERY_COLUMNS and len(delivery) == 24
    assert {c: str(t) for c, t in delivery.dtypes.items()} == DELIVERY_DTYPES
    ends = derived(rq2_cfg, "rq2_end_reasons")
    assert list(ends.columns) == END_REASON_COLUMNS and ends.n.sum() == 4
    assert {c: str(t) for c, t in ends.dtypes.items()} == END_REASON_DTYPES
    assert (ends.n_corners == 1).all()  # one chain per team: the denominator of every share
    sens = derived(rq2_cfg, "rq2_sensitivity")
    assert len(sens) == 4 and set(sens.measure) == {"shots_per_corner"}
    assert list(sens.columns) == [
        "competition_id", "season_id", "team_id", "measure", "team_name", "matches_in_sample",
        "sample_label", "numerator", "denominator", "n_incomplete", "n_matches", "value_main",
        "ci_lo", "ci_hi", "value_cap10", "numerator_cap10", "denominator_cap10",
        "n_incomplete_cap10", "value_cap20", "numerator_cap20", "denominator_cap20",
        "n_incomplete_cap20", "value_nocap", "numerator_nocap", "denominator_nocap",
        "n_incomplete_nocap", "sensitive"]  # fmt: skip
    assert str(sens.sensitive.dtype) == "boolean" and sens.sensitive.isna().all()  # one match each
    assert (sens.n_matches == sens.denominator).all() and sens.denominator.tolist() == [1, 1, 1, 0]
    assert all(str(sens[c].dtype) == "Int64" for c in sens.columns if "numerator" in c
               or "denominator" in c or "n_incomplete" in c)  # fmt: skip
    assert (ends.share == 1.0).all() and ends.ci_lo.isna().all()  # one chain per team, one match


def test_stage_summary_matches_summarise_rq2_of_the_written_tables(rq2_cfg, cfg):
    run_rq2(rq2_cfg)
    expected = summarise_rq2(derived(rq2_cfg, "rq2_outcomes"), derived(rq2_cfg, "team_samples"), rq2_cfg)
    pd.testing.assert_frame_equal(derived(rq2_cfg, "rq2_summary"), expected)
    flags = derived(rq2_cfg, "rq2_sensitivity")
    alt = {name: summarise_rq2(derived(rq2_cfg, f"rq2_outcomes_{name}"), derived(rq2_cfg, "team_samples"), rq2_cfg)
           for name in ("cap10", "cap20", "nocap")}
    pd.testing.assert_frame_equal(flags, sensitivity_flags(expected, alt))


def test_stage_sensitivity_columns_follow_the_configured_variants(rq2_cfg):
    sensitivity = dataclasses.replace(rq2_cfg.params["cap_sensitivity"], value=[7.5, None])
    run_rq2(dataclasses.replace(rq2_cfg, params={**rq2_cfg.params, "cap_sensitivity": sensitivity}))
    flags = derived(rq2_cfg, "rq2_sensitivity")
    assert [c for c in flags.columns if c.startswith("value_")] == ["value_main", "value_cap7.5", "value_nocap"]
    for count in ("numerator", "denominator", "n_incomplete"):  # the counts follow the same keys
        assert [c for c in flags.columns if c.startswith(f"{count}_")] == [f"{count}_cap7.5", f"{count}_nocap"]


def test_stage_sensitivity_carries_the_counts_of_each_variants_own_summary(rq2_cfg):
    """Each variant's numerator, denominator and excluded count are those of its own outcome
    table, not the main one's (the keys are the stage's, from ``_variants``)."""
    run_rq2(rq2_cfg)
    flags = derived(rq2_cfg, "rq2_sensitivity")
    samples = derived(rq2_cfg, "team_samples")
    variants = rq2_stage._variants(rq2_cfg)
    assert len(variants) == 4
    for table in variants:
        key = "main" if table == rq2_stage.MAIN_TABLE else table.removeprefix(f"{rq2_stage.MAIN_TABLE}_")
        suffix = "" if key == "main" else f"_{key}"
        own = summarise_rq2(derived(rq2_cfg, table), samples, rq2_cfg).query("measure == 'shots_per_corner'")
        for count in ("numerator", "denominator", "n_incomplete"):
            assert not flags[f"{count}{suffix}"].isna().any()
            assert flags[f"{count}{suffix}"].tolist() == own[count].tolist()


def test_stage_handles_a_team_without_corners(rq2_cfg):
    samples = derived(rq2_cfg, "team_samples")
    extra = samples.iloc[:1].assign(team_id=99, team_name="Idle FC", matches_in_sample=2)
    write_table(pd.concat([samples, extra], ignore_index=True), "derived/team_samples", rq2_cfg, None)
    run_rq2(rq2_cfg)
    idle = derived(rq2_cfg, "rq2_summary").query("team_id == 99")
    assert len(idle) == 5 and (idle.n_corners == 0).all() and idle.value.isna().all()
    assert len(derived(rq2_cfg, "rq2_sensitivity")) == 5
    assert sorted(derived(rq2_cfg, "rq2_end_reasons").team_id.unique()) == [10, 20, 30, 40]  # no row for 99


def snapshot(cfg):
    """Every file under the processed and reports folders: its modification time and bytes."""
    roots = (cfg.processed_dir, cfg.reports_dir)
    files = [p for root in roots if root.exists() for p in sorted(root.rglob("*")) if p.is_file()]
    return {str(p): (p.stat().st_mtime_ns, p.read_bytes()) for p in files}


@pytest.mark.parametrize("missing", ["rq1_corners", "team_samples"])
def test_stage_without_rq1_tables_fails_fast_and_writes_nothing(rq2_cfg, missing):
    (rq2_cfg.processed_dir / "derived" / f"{missing}.parquet").unlink()
    before = snapshot(rq2_cfg)
    with pytest.raises(FileNotFoundError, match=rf"{missing}.*run rq1 first"):
        run_rq2(rq2_cfg)
    assert snapshot(rq2_cfg) == before  # no outcome table, no summary, no figure, no manifest
    assert not list((rq2_cfg.processed_dir / "derived").glob("rq2_*"))
    assert not (rq2_cfg.processed_dir / "manifests" / "rq2.json").exists()
    assert not (rq2_cfg.reports_dir / "figures" / "rq2").exists()


def test_stage_reads_the_rq1_tables_before_it_builds_a_chain(rq2_cfg, monkeypatch):
    (rq2_cfg.processed_dir / "derived" / "rq1_corners.parquet").unlink()
    built = []
    monkeypatch.setattr(rq2_stage, "build_rq2_outcomes", lambda *a: built.append(a))
    with pytest.raises(FileNotFoundError, match="run rq1 first"):
        run_rq2(rq2_cfg)
    assert built == []


BREAKS = {
    "rq1_corners missing": (
        lambda cfg: (cfg.processed_dir / "derived" / "rq1_corners.parquet").unlink(),
        FileNotFoundError, "run rq1 first"),
    "a team missing from team_samples": (
        lambda cfg: write_table(derived(cfg, "team_samples").query("team_id != 20"),
                                "derived/team_samples", cfg, None),
        ValueError, "team_samples"),
    "a corner missing from rq1_corners": (
        lambda cfg: write_table(derived(cfg, "rq1_corners").query("event_id != 'm2-c2'"),
                                "derived/rq1_corners", cfg, None),
        ValueError, "m2-c2"),
}  # fmt: skip


@pytest.mark.parametrize("case", list(BREAKS))
def test_stage_failure_leaves_the_previous_run_untouched(rq2_cfg, case):
    """The outcomes, the summaries, the figures and the manifest are written only when every
    table has been computed, so a data-dependent failure leaves every file as it was (an error
    while writing can still leave new tables next to the old manifest)."""
    run_rq2(rq2_cfg)
    breaker, error, match = BREAKS[case]
    breaker(rq2_cfg)
    before = snapshot(rq2_cfg)
    assert any(path.endswith("rq2.json") for path in before)
    with pytest.raises(error, match=match):
        run_rq2(rq2_cfg)
    assert snapshot(rq2_cfg) == before


def test_stage_manifest_has_the_summary_row_counts(rq2_cfg):
    counts = run_rq2(rq2_cfg)
    assert list(counts) == ["rq2_outcomes", "rq2_outcomes_cap10", "rq2_outcomes_cap20", "rq2_outcomes_nocap"]
    manifest = json.loads((rq2_cfg.processed_dir / "manifests" / "rq2.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["summary_rows"] == {
        "rq2_summary": 20, "rq2_sensitivity": 4, "rq2_by_delivery": 24, "rq2_end_reasons": 4}
    assert manifest["counts"]["slice_corners"] == {"43/106": 2, "9/281": 2}


def test_stage_writes_two_figures_per_slice_and_clears_the_folder_first(rq2_cfg):
    fig_dir = rq2_cfg.reports_dir / "figures" / "rq2"
    fig_dir.mkdir(parents=True)
    (fig_dir / "stale.svg").write_text("<svg/>", encoding="utf-8")
    run_rq2(rq2_cfg)
    assert sorted(p.name for p in fig_dir.iterdir()) == [
        "43_106_end_reasons.svg", "43_106_shots_per_corner.svg",
        "9_281_end_reasons.svg", "9_281_shots_per_corner.svg"]  # fmt: skip
    assert all(p.stat().st_size > 0 for p in fig_dir.iterdir())
    run_rq2(rq2_cfg)  # a rerun rewrites the same files
    assert len(list(fig_dir.iterdir())) == 4


def test_stage_figure_folder_is_cleared_without_ignoring_errors(rq2_cfg, monkeypatch):
    calls = []

    def spy(path, *args, **kwargs):
        calls.append((path, args, kwargs))
        return shutil.rmtree(path, *args, **kwargs)

    monkeypatch.setattr(rq2_stage, "shutil", SimpleNamespace(rmtree=spy))
    run_rq2(rq2_cfg)  # the folder does not exist yet: that alone is fine
    assert calls == [(rq2_cfg.reports_dir / "figures" / "rq2", (), {})]

    def locked(path, *args, **kwargs):
        raise PermissionError(f"{path} is locked")

    monkeypatch.setattr(rq2_stage, "shutil", SimpleNamespace(rmtree=locked))
    with pytest.raises(PermissionError):  # a figure that cannot be removed stops the stage
        run_rq2(rq2_cfg)


def test_stage_figure_titles_carry_n_slice_and_date_range(rq2_cfg, monkeypatch):
    titles = {}
    for name in ("plot_shots_per_corner", "plot_end_reasons"):
        real = getattr(figures, name)

        def spy(data, title, _real=real, _name=name):
            titles.setdefault(_name, []).append(title)
            return _real(data, title)

        monkeypatch.setattr(figures, name, spy)
    run_rq2(rq2_cfg)
    # the World Cup slice has an incomplete chain (team 20's corner runs off the end of the match):
    # its titles say how many corners are eligible of how many; the Bundesliga slice has none
    for name in ("plot_shots_per_corner", "plot_end_reasons"):
        assert any("n=1 eligible of 2 corners (1 incomplete" in t and "FIFA World Cup 2022" in t
                   and "2022-11-21" in t for t in titles[name])  # fmt: skip
        assert any("n=2 corners" in t and "1. Bundesliga 2023/24" in t and "2023-08-18" in t
                   and "incomplete" not in t for t in titles[name])  # fmt: skip
    assert len(titles["plot_shots_per_corner"]) == len(titles["plot_end_reasons"]) == 2


def test_stage_with_stale_rq1_corners_raises_rather_than_filling_unknown(rq2_cfg):
    corners = derived(rq2_cfg, "rq1_corners")
    write_table(corners[corners.event_id != "m2-c2"], "derived/rq1_corners", rq2_cfg, None)
    with pytest.raises(ValueError, match="m2-c2"):
        run_rq2(rq2_cfg)
