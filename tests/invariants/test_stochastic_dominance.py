from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from nns import fsd, fsd_uni, lpm, ssd, ssd_uni, tsd, tsd_uni


def _r_sd_reference(x: NDArray[np.float64], y: NDArray[np.float64], degree: int) -> int:
    """R's NNS.FSD/NNS.SSD/NNS.TSD decision rule, transcribed literally."""
    grid = np.sort(np.concatenate((x, y)))
    if degree == 1:
        # LPM.ratio(0, grid, sample) == LPM(0, grid, sample) == ECDF
        curve_x = np.asarray(lpm(0, grid, x), dtype=np.float64)
        curve_y = np.asarray(lpm(0, grid, y), dtype=np.float64)
    else:
        curve_x = np.asarray(lpm(degree - 1, grid, x), dtype=np.float64)
        curve_y = np.asarray(lpm(degree - 1, grid, y), dtype=np.float64)

    mean_ok_xy = degree == 1 or float(np.mean(x)) >= float(np.mean(y))
    mean_ok_yx = degree == 1 or float(np.mean(y)) >= float(np.mean(x))
    curves_identical = np.array_equal(curve_x, curve_y)

    if not np.any(curve_x > curve_y) and x.min() >= y.min() and mean_ok_xy and not curves_identical:
        return 1
    if not np.any(curve_y > curve_x) and y.min() >= x.min() and mean_ok_yx and not curves_identical:
        return -1
    return 0


def test_sd_antisymmetry() -> None:
    x = np.array([0.0, 0.1, 0.2, 0.3])
    y = np.array([-0.1, 0.0, 0.1, 0.2])

    assert fsd(x, y) == -fsd(y, x)
    assert ssd(x, y) == -ssd(y, x)
    assert tsd(x, y) == -tsd(y, x)


def test_fsd_implies_ssd_implies_tsd() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([0.0, 1.0, 2.0, 3.0])

    assert fsd(x, y) == 1
    assert ssd(x, y) == 1
    assert tsd(x, y) == 1


def test_self_does_not_dominate() -> None:
    x = np.array([-1.0, 0.0, 1.0, 2.0])

    assert fsd(x, x) == 0
    assert ssd(x, x) == 0
    assert tsd(x, x) == 0


def test_unequal_length_shifted_samples_dominate() -> None:
    x = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    y = np.array([1.0, 2.0, 3.0])

    assert fsd(x, y) == 1
    assert ssd(x, y) == 1
    assert tsd(x, y) == 1
    assert fsd(y, x) == -1

    assert fsd_uni(x, y) == 1
    assert fsd_uni(x, y, "continuous") == 1
    assert ssd_uni(x, y) == 1
    assert tsd_uni(x, y) == 1
    assert fsd_uni(y, x) == 0
    assert ssd_uni(y, x) == 0
    assert tsd_uni(y, x) == 0


def test_unequal_length_antisymmetry() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(1.0, 1.0, 37)
    y = rng.normal(0.0, 1.0, 61)

    assert fsd(x, y) == -fsd(y, x)
    assert ssd(x, y) == -ssd(y, x)
    assert tsd(x, y) == -tsd(y, x)


def test_mtcars_transmission_groups_match_r() -> None:
    # mtcars mpg split by transmission: R's NNS.FSD returns "X FSD Y" for
    # (manual, auto) despite the samples having different lengths (13 vs 19).
    auto_mpg = np.array(
        [21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 17.8, 16.4, 17.3,
         15.2, 10.4, 10.4, 14.7, 21.5, 15.5, 15.2, 13.3, 19.2]
    )
    manual_mpg = np.array(
        [21.0, 21.0, 22.8, 32.4, 30.4, 33.9, 27.3, 26.0, 30.4, 15.8, 19.7, 15.0, 21.4]
    )

    assert fsd(manual_mpg, auto_mpg) == 1
    assert fsd(auto_mpg, manual_mpg) == -1
    assert fsd_uni(manual_mpg, auto_mpg) == 1


@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("seed", range(8))
def test_unequal_length_matches_r_decision_rule(degree: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    sizes = rng.integers(3, 60, size=2)
    shift = rng.uniform(-0.5, 0.5)
    x = rng.normal(shift, 1.0, int(sizes[0]))
    y = rng.normal(0.0, rng.uniform(0.5, 1.5), int(sizes[1]))

    function = {1: fsd, 2: ssd, 3: tsd}[degree]
    assert function(x, y) == _r_sd_reference(x, y, degree)


@pytest.mark.parametrize("seed", range(6))
def test_fsd_uni_continuous_matches_dense_ratio_definition(seed: int) -> None:
    # Reference: LPM1/(LPM1 + UPM1) evaluated densely on the merged grid.
    rng = np.random.default_rng(seed)
    sizes = rng.integers(2, 40, size=2)
    x = np.round(rng.normal(0.3, 1.0, int(sizes[0])), 1)
    y = np.round(rng.normal(0.0, 1.0, int(sizes[1])), 1)

    def dominates(a: NDArray[np.float64], b: NDArray[np.float64]) -> int:
        if np.array_equal(np.sort(a), np.sort(b)) or a.min() < b.min():
            return 0
        grid = np.sort(np.concatenate((a, b)))
        curves = []
        for sample in (a, b):
            lower = np.mean(np.maximum(grid[:, None] - sample, 0.0), axis=1)
            upper = np.mean(np.maximum(sample - grid[:, None], 0.0), axis=1)
            total = lower + upper
            curves.append(np.divide(lower, total, out=np.zeros_like(lower), where=total != 0))
        if np.allclose(curves[0], curves[1], rtol=0.0, atol=1e-12):
            return 0
        return int(not np.any(curves[0] > curves[1] + 1e-12))

    assert fsd_uni(x, y, "continuous") == dominates(x, y)
    assert fsd_uni(y, x, "continuous") == dominates(y, x)


def test_merged_thresholds_equal_union_and_searchsorted() -> None:
    from nns.stochastic_dominance import _merged_thresholds

    rng = np.random.default_rng(31)
    for n, m, digits in ((1, 1, 0), (7, 3, 0), (50, 80, 1), (400, 400, 6), (300, 120, 15)):
        x = np.sort(np.round(rng.normal(size=n), digits))
        y = np.sort(np.round(rng.normal(size=m), digits))
        thresholds, x_counts, y_counts = _merged_thresholds(x, y)
        expected = np.union1d(x, y)
        np.testing.assert_array_equal(thresholds, expected)
        np.testing.assert_array_equal(x_counts, np.searchsorted(x, expected, side="right"))
        np.testing.assert_array_equal(y_counts, np.searchsorted(y, expected, side="right"))
