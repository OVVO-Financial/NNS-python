from __future__ import annotations

import os
import subprocess
from collections.abc import Callable

import numpy as np
import pytest
from _r import nns
from _tolerances import EXACT
from conftest import EdgeCase
from numpy.typing import NDArray

from pynns import lpm, lpm_ratio, upm

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

    if not np.all(np.isfinite(edge_case.values)) and (
        os.environ.get("PYNNS_OFFLINE") == "1" or os.environ.get("PYNNS_R_CACHE_ONLY") == "1"
    ):
        result = function(degree, target, edge_case.values)
        if edge_case.name == "contains-nan":
            assert np.isnan(result)
            return
        assert np.isscalar(result)
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
