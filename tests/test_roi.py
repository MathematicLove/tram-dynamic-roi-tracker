"""Unit tests for the pure geometry helpers in algorithm/roi.py.

They run on synthetic arrays only: no model weights, video or camera needed.
"""
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("cv2")
pytest.importorskip("ultralytics")

from algorithm import roi  # noqa: E402


def trapezoid_rows(h=200, w=400, y_top=40, w_top=20, w_bot=200, cx=200):
    """Rail bed whose width grows linearly from w_top (y_top) to w_bot (h-1)."""
    ys = np.arange(y_top, h)
    widths = np.linspace(w_top, w_bot, len(ys))
    ls = (cx - widths / 2).astype(int)
    rs = (cx + widths / 2).astype(int)
    return ys, ls, rs, w


def trapezoid_mask(h=200, w=400, **kw):
    ys, ls, rs, _ = trapezoid_rows(h=h, w=w, **kw)
    mask = np.zeros((h, w), dtype=np.uint8)
    for y, l, r in zip(ys, ls, rs):
        mask[y, l : r + 1] = 255
    return mask


class TestFilters:
    def test_medfilt1d_removes_spike(self):
        a = np.array([5, 5, 5, 100, 5, 5, 5, 5, 5, 5])
        out = roi._medfilt1d(a, 5)
        assert out[3] == 5
        assert len(out) == len(a)

    def test_medfilt1d_short_input_is_copied(self):
        a = np.array([1, 2, 3])
        out = roi._medfilt1d(a, 7)
        assert np.array_equal(out, a)
        assert out is not a

    def test_medfilt1d_even_kernel_is_accepted(self):
        out = roi._medfilt1d(np.arange(20), 6)
        assert len(out) == 20

    def test_smooth1d_keeps_constant_signal(self):
        a = np.full(50, 7.0)
        assert np.allclose(roi._smooth1d(a, 11), 7.0)

    def test_smooth1d_reduces_noise(self):
        rng = np.random.default_rng(0)
        a = 100 + rng.normal(0, 5, 200)
        assert np.std(roi._smooth1d(a, 21)) < np.std(a)

    def test_smooth1d_short_input_is_copied(self):
        a = np.array([1.0, 2.0, 3.0])
        assert np.array_equal(roi._smooth1d(a, 21), a)


class TestRowExtent:
    def test_single_pixel(self):
        assert roi._row_extent(np.array([42])) == (42, 42)

    def test_contiguous_run(self):
        assert roi._row_extent(np.arange(10, 31)) == (10, 30)

    def test_small_gap_is_bridged(self):
        cols = np.concatenate([np.arange(10, 20), np.arange(25, 40)])
        assert roi._row_extent(cols) == (10, 39)

    def test_stray_speck_is_ignored(self):
        cols = np.concatenate([np.arange(100, 150), [390]])
        assert roi._row_extent(cols) == (100, 149)


class TestRailRows:
    def test_empty_mask(self):
        ys, ls, rs = roi.rail_rows(np.zeros((50, 50), dtype=np.uint8))
        assert len(ys) == len(ls) == len(rs) == 0

    def test_trapezoid_widens_toward_bottom(self):
        ys, ls, rs = roi.rail_rows(trapezoid_mask())
        assert len(ys) > 100
        widths = rs - ls
        assert widths[-1] > widths[0]
        assert np.all(np.diff(ys) > 0)

    def test_rows_wider_than_max_fraction_are_dropped(self):
        mask = np.zeros((20, 100), dtype=np.uint8)
        mask[:, :] = 255
        ys, _, _ = roi.rail_rows(mask)
        assert len(ys) == 0

    def test_rows_with_too_few_pixels_are_dropped(self):
        mask = np.zeros((20, 100), dtype=np.uint8)
        mask[:, 10:12] = 255
        ys, _, _ = roi.rail_rows(mask, min_px=3)
        assert len(ys) == 0


class TestZoneBounds:
    def test_empty(self):
        e = np.array([], dtype=int)
        assert roi.zone_bounds(e, e, e, 100) == {}

    def test_zones_are_nested_and_clipped(self):
        ys, ls, rs, w = trapezoid_rows()
        zones = roi.zone_bounds(ys, ls, rs, w)
        assert set(zones) == {"yellow", "orange", "red"}
        for left, right in zones.values():
            assert left.min() >= 0 and right.max() <= w - 1
        # Larger real-world offset -> wider zone.
        assert np.all(zones["yellow"][0] <= zones["orange"][0])
        assert np.all(zones["orange"][0] <= zones["red"][0])
        assert np.all(zones["yellow"][1] >= zones["orange"][1])
        assert np.all(zones["orange"][1] >= zones["red"][1])


class TestFitWidthLine:
    def test_recovers_linear_width(self):
        ys = np.arange(50, 150)
        widths = 2.0 * ys + 10
        ls = np.zeros_like(ys)
        rs = widths.astype(int)
        a, b = roi._fit_width_line(ys, ls, rs)
        assert a == pytest.approx(2.0, abs=0.05)
        assert b == pytest.approx(10.0, abs=2.0)

    def test_too_few_rows_returns_none(self):
        ys = np.arange(3)
        assert roi._fit_width_line(ys, ys * 0, ys + 10) is None

    def test_non_growing_width_returns_none(self):
        ys = np.arange(50, 100)
        ls = np.zeros_like(ys)
        rs = np.full_like(ys, 100)
        assert roi._fit_width_line(ys, ls, rs) is None


class TestDistanceModel:
    def test_row_and_distance_are_inverse(self):
        model = roi._DistanceModel(a=2.0, b=-20.0, f_px=1000.0)
        for y in (50.0, 120.0, 300.0):
            assert model.row_at_dist(model.dist_mm(y)) == pytest.approx(y)

    def test_closer_rows_are_nearer(self):
        model = roi._DistanceModel(a=2.0, b=-20.0, f_px=1000.0)
        assert model.dist_mm(300.0) < model.dist_mm(100.0)

    def test_build_returns_none_for_bad_fit(self):
        ys = np.arange(3)
        assert roi.build_distance_model(ys, ys * 0, ys + 10, 1000.0) is None


class TestDistanceAndGrid:
    def test_estimate_distance_zero_at_or_below_nearest_row(self):
        ys, ls, rs, _ = trapezoid_rows()
        assert roi.estimate_distance_m(int(ys.max()), ys, ls, rs, 1000.0) == 0.0
        assert roi.estimate_distance_m(int(ys.max()) + 20, ys, ls, rs, 1000.0) == 0.0

    def test_estimate_distance_grows_toward_horizon(self):
        ys, ls, rs, _ = trapezoid_rows()
        near = roi.estimate_distance_m(int(ys.max()) - 20, ys, ls, rs, 1000.0)
        far = roi.estimate_distance_m(int(ys.min()) + 5, ys, ls, rs, 1000.0)
        assert 0.0 < near < far

    def test_estimate_distance_needs_two_rows(self):
        one = np.array([10])
        assert roi.estimate_distance_m(5, one, one, one + 10, 1000.0) is None

    def test_grid_positions_empty_for_short_input(self):
        one = np.array([10])
        assert roi.grid_positions(one, one, one + 10) == []

    def test_grid_positions_are_ordered_and_inside_rail(self):
        ys, ls, rs, _ = trapezoid_rows()
        marks = roi.grid_positions(ys, ls, rs, f_px=1000.0)
        for row, dist in marks:
            # Marks stop once yy <= y_top - 1, so rounding can land 1px above.
            assert ys.min() - 1 <= row < ys.max()
            assert dist > 0
        rows = [m[0] for m in marks]
        dists = [m[1] for m in marks]
        assert rows == sorted(rows, reverse=True)
        assert dists == sorted(dists)


class TestGroups:
    def test_find_groups_drops_small_components(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:40, 10:40] = 255
        mask[80:82, 80:82] = 255
        groups = roi.find_groups(mask, min_area=100)
        assert len(groups) == 1

    def test_find_groups_falls_back_to_full_mask(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:7, 5:7] = 255
        groups = roi.find_groups(mask, min_area=1000)
        assert len(groups) == 1 and groups[0] is mask

    def test_select_group_prefers_continuity_with_previous_center(self):
        h = 100
        left = np.zeros((h, 200), dtype=np.uint8)
        left[60:, 20:40] = 255
        right = np.zeros((h, 200), dtype=np.uint8)
        right[60:, 150:170] = 255
        chosen = roi._select_rail_group([left, right], h, prev_center=155.0)
        assert chosen is right
