from __future__ import annotations

import subprocess
from collections.abc import Callable

import numpy as np
import pytest
from _r import nns
from _tolerances import EXACT
from conftest import EdgeCase
from numpy.typing import NDArray

from nns import lpm, lpm_ratio, upm, upm_ratio

DEGREES = [0.0, 0.5, 1.0, 2.0, 3.0]
SIZES = [10, 100, 1000]


@pytest.mark.parity
@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("size", SIZES)
def test_lpm_matches_r_scalar_targets(
    rng: np.random.Generator,
    degree: float,
    size: int,
) -> None:
    x = rng.normal(size=size)

    for target in _scalar_targets(x):
        expected = nns("LPM", degree, target, x.tolist())
        assert isinstance(expected, np.ndarray)
        assert np.allclose(lpm(degree, target, x), expected.item(), atol=EXACT)


@pytest.mark.parity
@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("size", SIZES)
def test_upm_matches_r_scalar_targets(
    rng: np.random.Generator,
    degree: float,
    size: int,
) -> None:
    x = rng.normal(size=size)

    for target in _scalar_targets(x):
        expected = nns("UPM", degree, target, x.tolist())
        assert isinstance(expected, np.ndarray)
        assert np.allclose(upm(degree, target, x), expected.item(), atol=EXACT)


@pytest.mark.parity
@pytest.mark.parametrize("degree", DEGREES)
def test_lpm_matches_r_vector_target(rng: np.random.Generator, degree: float) -> None:
    x = rng.normal(size=100)
    target = np.linspace(x.min(), x.max(), 20)

    expected = nns("LPM", degree, target.tolist(), x.tolist())
    assert isinstance(expected, np.ndarray)
    np.testing.assert_allclose(lpm(degree, target, x), expected, atol=EXACT)


@pytest.mark.parity
@pytest.mark.parametrize("degree", DEGREES)
def test_upm_matches_r_vector_target(rng: np.random.Generator, degree: float) -> None:
    x = rng.normal(size=100)
    target = np.linspace(x.min(), x.max(), 20)

    expected = nns("UPM", degree, target.tolist(), x.tolist())
    assert isinstance(expected, np.ndarray)
    np.testing.assert_allclose(upm(degree, target, x), expected, atol=EXACT)


@pytest.mark.parity
@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("size", SIZES)
def test_lpm_ratio_matches_r_scalar_targets(
    rng: np.random.Generator,
    degree: float,
    size: int,
) -> None:
    x = rng.normal(size=size)

    for target in _scalar_targets(x):
        expected = nns("LPM.ratio", degree, target, x.tolist())
        assert isinstance(expected, np.ndarray)
        assert np.allclose(lpm_ratio(degree, target, x), expected.item(), atol=EXACT)


@pytest.mark.parity
@pytest.mark.parametrize("degree", DEGREES)
def test_lpm_ratio_matches_r_vector_target(rng: np.random.Generator, degree: float) -> None:
    x = rng.normal(size=100)
    target = np.linspace(x.min(), x.max(), 20)

    expected = nns("LPM.ratio", degree, target.tolist(), x.tolist())
    assert isinstance(expected, np.ndarray)
    np.testing.assert_allclose(lpm_ratio(degree, target, x), expected, atol=EXACT)


@pytest.mark.parity
@pytest.mark.parametrize("degree", [0.0, 1.0, 1.5, 2.0])
@pytest.mark.parametrize(
    "target",
    [2.0, [2.0, 1.0, 5.0, 0.5, 4.0]],
)
def test_excess_ret_moments_match_r(degree: float, target: float | list[float]) -> None:
    # R LPM/UPM(excess_ret = TRUE): element-wise targets; degree-0 LPM counts ties
    # (patched R), so LPM0 + UPM0 = 1.
    x = [1.0, 2.0, 3.0, 0.5, 4.0]
    for label, function in (("LPM", lpm), ("UPM", upm)):
        expected = nns(label, degree, target, x, True)
        assert isinstance(expected, np.ndarray)
        actual = function(degree, np.asarray(target), np.asarray(x), excess_ret=True)
        assert actual == pytest.approx(expected.item(), abs=EXACT)
    if degree == 0.0:
        total = lpm(0, np.asarray(target), np.asarray(x), excess_ret=True) + upm(
            0, np.asarray(target), np.asarray(x), excess_ret=True
        )
        assert total == pytest.approx(1.0, abs=EXACT)


def test_excess_ret_rejects_mismatched_target_length() -> None:
    with pytest.raises(ValueError, match="length 1 or same length"):
        lpm(1, np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]), excess_ret=True)


@pytest.mark.parity
@pytest.mark.parametrize("degree", [0.0, 0.5, 1.0, 2.0])
@pytest.mark.parametrize(
    "target, x",
    [(3.0, [3.0, 3.0, 3.0, 3.0]), ([1.0, 3.0, 4.0], [3.0, 3.0]), (2.0, [2.0])],
)
def test_ratios_at_a_point_mass_match_r(
    degree: float, target: float | list[float], x: list[float]
) -> None:
    # Patched R: the 0/0 case at degree > 0 returns the degree-0 ratio.
    values = np.asarray(x)
    for label, function in (("LPM.ratio", lpm_ratio), ("UPM.ratio", upm_ratio)):
        expected = nns(label, degree, target, x)
        assert isinstance(expected, np.ndarray)
        np.testing.assert_allclose(
            np.asarray(function(degree, np.asarray(target), values)).reshape(-1),
            expected.reshape(-1),
            atol=EXACT,
        )


@pytest.mark.parity
def test_ratio_overflow_stays_nan_like_r() -> None:
    # Patched R fills only the constant-sample 0/0; Inf/Inf from overflow stays NaN.
    x = [1e200, -1e200]
    for label, function in (("LPM.ratio", lpm_ratio), ("UPM.ratio", upm_ratio)):
        assert np.isnan(np.asarray(nns(label, 3.0, 0.0, x))).all()
        with np.errstate(over="ignore"):
            assert np.isnan(function(3.0, 0.0, np.asarray(x)))


@pytest.mark.parity
@pytest.mark.parametrize("function_name, function", [("LPM", lpm), ("UPM", upm)])
def test_edge_cases_match_r_category(
    edge_case: EdgeCase,
    function_name: str,
    function: Callable[
        [float, float, NDArray[np.float64] | NDArray[np.int64]],
        float | NDArray[np.float64],
    ],
) -> None:
    degree = 1.0
    target = 0.0

    if edge_case.values.size == 0:
        with pytest.raises(ValueError):
            function(degree, target, edge_case.values)
        expected = nns(function_name, degree, target, edge_case.values.tolist())
        if isinstance(expected, np.ndarray):
            assert np.isnan(expected)
            return
        assert np.isscalar(expected)
        assert np.isnan(expected)
        return

    if not np.all(np.isfinite(edge_case.values)):
        # R's LPM/UPM stop with "`variable` must be finite." for NA, NaN and Inf.
        with pytest.raises(ValueError, match="finite"):
            function(degree, target, edge_case.values)
        return

    try:
        expected = nns(function_name, degree, target, edge_case.values.tolist())
    except subprocess.CalledProcessError:
        result = function(degree, target, edge_case.values)
        if edge_case.name == "contains-nan":
            assert np.isnan(result)
            return
        assert np.isscalar(result)
        return

    assert isinstance(expected, np.ndarray)
    result = function(degree, target, edge_case.values)
    assert np.allclose(result, expected.item(), atol=EXACT, equal_nan=True)


def _scalar_targets(x: NDArray[np.float64]) -> list[float]:
    return [0.0, float(x.mean()), float(x.min()), float(x.max()), 0.01]
