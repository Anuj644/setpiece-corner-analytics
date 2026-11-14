"""Data tests for rq1_corners against pinned audit references (Task 12, Step 5)."""

import pytest

from setpiece.io import read_table
from setpiece.rq1.corners import build_rq1_corners

AUDIT = [(9, 281), (11, 90), (7, 235)]


@pytest.mark.data
@pytest.mark.parametrize(
    "slices,inside,outside",
    [(AUDIT, 647, 318), ([(43, 106)], 455, 114)],
)
def test_box_counts(cfg, slices, inside, outside):
    r = build_rq1_corners(read_table("events", cfg, slices), cfg)
    assert (int(r.ends_in_box.sum()), int((~r.ends_in_box.astype(bool)).sum())) == (
        inside,
        outside,
    )
    assert r.corner_side.notna().all() and r.start_y.between(2, 78, inclusive="neither").sum() == 0
