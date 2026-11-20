"""Smoke tests for the half-pitch drawing (Task 15)."""

import dataclasses

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from setpiece.plots.pitch import PITCH_LENGTH, PITCH_WIDTH, draw_half_pitch


def drawn_rectangles(cfg):
    fig, ax = plt.subplots()
    draw_half_pitch(ax, cfg)
    rects = [p.get_bbox() for p in ax.patches]
    return fig, ax, rects


def test_boxes_open_toward_the_goal_line_not_the_halfway_line(cfg, tmp_path):
    """Regression: the boxes were drawn at x 60-102 / 60-114, the wrong side of their start line."""
    fig, _ax, rects = drawn_rectangles(cfg)
    box, six = cfg.p("box"), cfg.p("six_yard_x_min")
    spans = {(round(b.x0, 6), round(b.x1, 6), round(b.y0, 6), round(b.y1, 6)) for b in rects}
    assert (box["x_min"], PITCH_LENGTH, box["y_min"], box["y_max"]) in spans
    lo, hi = cfg.p("zone_width_bands")["central"]
    assert (six, PITCH_LENGTH, lo, hi) in spans
    for b in rects:  # nothing outside the half pitch
        assert PITCH_LENGTH / 2 <= b.x0 and b.x1 <= PITCH_LENGTH
        assert 0 <= b.y0 and b.y1 <= PITCH_WIDTH
    plt.close(fig)


def test_half_pitch_renders_to_svg(cfg, tmp_path):
    fig, ax, _ = drawn_rectangles(cfg)
    assert ax.get_xlim() == (PITCH_LENGTH / 2, PITCH_LENGTH)
    fig.savefig(tmp_path / "pitch.svg")
    plt.close(fig)
    assert (tmp_path / "pitch.svg").stat().st_size > 0


def test_boxes_follow_the_config_parameters(cfg):
    other = cfg.params["six_yard_x_min"]
    changed = type(other)(value=110, status=other.status, note=other.note)
    cfg2 = dataclasses.replace(cfg, params={**cfg.params, "six_yard_x_min": changed})
    fig, _, rects = drawn_rectangles(cfg2)
    assert any(round(b.x0, 6) == pytest.approx(110) for b in rects)
    plt.close(fig)
