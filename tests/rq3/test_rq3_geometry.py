import numpy as np
import pytest

from setpiece.rq3.geometry import (
    mirror_xy,
    points_in_polygon,
    polygon_area,
    polygon_from_flat,
    zone_grid,
)


def test_polygon_includes_edges_vertices_and_excludes_concave_notch_in_both_windings():
    polygon = polygon_from_flat([0, 0, 4, 0, 4, 4, 2, 2, 0, 4])
    assert polygon_area(polygon) == polygon_area(polygon[::-1]) == 12
    points = [[1, 1], [0, 2], [2, 2], [2, 4], [-0.01, 2], [3, 3]]
    expected = [True, True, True, False, False, True]
    for ring in (polygon, polygon[::-1], np.vstack([polygon, polygon[0]])):
        assert points_in_polygon(points, ring).tolist() == expected
    assert len(points_in_polygon([], polygon)) == 0


@pytest.mark.parametrize("flat", [[], [0, 0, 1, 1, 1], [0, 0, 1, 1, 2, np.nan], [[0, 0]]])
def test_bad_flat_coordinates_are_rejected(flat):
    with pytest.raises(ValueError):
        polygon_from_flat(flat)


def test_right_corner_mirrors_width_only_without_mutating_original():
    xy = np.array([[120, 80], [116.5, 42.5], [106, 20]])
    np.testing.assert_allclose(mirror_xy(xy, "right"), [[120, 0], [116.5, 37.5], [106, 60]])
    np.testing.assert_allclose(mirror_xy(xy, "left"), xy)
    np.testing.assert_allclose(mirror_xy(mirror_xy(xy, "right"), "right"), xy)
    np.testing.assert_allclose(xy[:, 1], [80, 42.5, 20])


def test_grid_keeps_all_edges_when_the_step_does_not_divide_bounds():
    grid = zone_grid(102, 107, 30, 33, 2)
    assert sorted(set(grid[:, 0])) == [102, 104, 106, 107]
    assert sorted(set(grid[:, 1])) == [30, 32, 33]
    assert len(grid) == 12
    with pytest.raises(ValueError, match="positive"):
        zone_grid(102, 107, 30, 33, 0)
