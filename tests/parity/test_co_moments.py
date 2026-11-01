from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from _r import nns
from _tolerances import EXACT
from conftest import EdgeCase
from numpy.typing import NDArray

from nns import co_lpm, co_upm, d_lpm, d_upm

DEGREES = [0.0, 0.5, 1.0, 2.0, 3.0]
RHO_VALUES = [-0.7, 0.0, 0.7]
SIZES = [10, 100, 1000]


@pytest.mark.parity
@pytest.mark.parametrize(
    "function_name,function",
    [
        ("Co.LPM", co_lpm),
        ("Co.UPM", co_upm),
    ],
)
@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("rho", RHO_VALUES)
@pytest.mark.parametrize("size", SIZES)
def test_co_moments_match_r(
    rng: np.random.Generator,
    function_name: str,
    function: Callable[
        [
            float,
            NDArray[np.float64],
            NDArray[np.float64],
            float | NDArray[np.float64],
            float | NDArray[np.float64],
        ],
        float | NDArray[np.float64],
    ],
    degree: float,
    rho: float,
    size: int,
) -> None:
    x, y = _xy(rng, size, rho)

    for target_x, target_y in _targets(x, y):
        expected = nns(
            function_name,
            degree,
            x.tolist(),
            y.tolist(),
            _to_r(target_x),
            _to_r(target_y),
        )
        assert isinstance(expected, np.ndarray)
        result = function(degree, x, y, target_x, target_y)
        np.testing.assert_allclose(result, expected, atol=EXACT)


@pytest.mark.parity
@pytest.mark.parametrize(
    "function_name,function",
    [
        ("D.LPM", d_lpm),
        ("D.UPM", d_upm),
    ],
)
@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("rho", RHO_VALUES)
@pytest.mark.parametrize("size", SIZES)
def test_divergent_moments_match_r(
    rng: np.random.Generator,
    function_name: str,
    function: Callable[
        [
            float,
            float,
            NDArray[np.float64],
            NDArray[np.float64],
            float | NDArray[np.float64],
            float | NDArray[np.float64],
        ],
        float | NDArray[np.float64],
    ],
    degree: float,
    rho: float,
    size: int,
) -> None:
    x, y = _xy(rng, size, rho)

    for target_x, target_y in _targets(x, y):
        expected = nns(
            function_name,
            degree,
            degree,
            x.tolist(),
            y.tolist(),
            _to_r(target_x),
            _to_r(target_y),
        )
        assert isinstance(expected, np.ndarray)
        result = function(degree, degree, x, y, target_x, target_y)
        np.testing.assert_allclose(result, expected, atol=EXACT)


@pytest.mark.parity
@pytest.mark.parametrize("function", [co_lpm, co_upm])
def test_co_moments_raise_for_mismatched_lengths(
    edge_case: EdgeCase,
    function: Callable[
        [float, NDArray[np.float64], NDArray[np.float64], float, float],
        float | NDArray[np.float64],
    ],
) -> None:
    x = edge_case.values.astype(np.float64)
    y = np.append(x, 1.0)

    with pytest.raises(ValueError):
        function(1.0, x, y, 0.0, 0.0)


@pytest.mark.parity
@pytest.mark.parametrize("function", [d_lpm, d_upm])
def test_divergent_moments_raise_for_mismatched_lengths(
    edge_case: EdgeCase,
    function: Callable[
        [float, float, NDArray[np.float64], NDArray[np.float64], float, float],
        float | NDArray[np.float64],
    ],
) -> None:
    x = edge_case.values.astype(np.float64)
    y = np.append(x, 1.0)

    with pytest.raises(ValueError):
        function(1.0, 1.0, x, y, 0.0, 0.0)


def _xy(
    rng: np.random.Generator,
    size: int,
    rho: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    # Build the correlated pair from independent standard normals with an explicit
    # Cholesky factor. multivariate_normal factorises the covariance with LAPACK,
    # whose last bits differ across NumPy/BLAS builds, which changed the inputs and
    # therefore the R cache keys from one environment to the next.
    z = rng.standard_normal((size, 2))
    x = z[:, 0]
    y = rho * z[:, 0] + np.sqrt(1.0 - rho * rho) * z[:, 1]
    return x, y


def _targets(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> list[tuple[float | NDArray[np.float64], float | NDArray[np.float64]]]:
    return [
        (0.0, 0.0),
        (float(x.mean()), float(y.mean())),
        (float(x[0]), float(y[0])),
        (np.linspace(x.min(), x.max(), 5), np.linspace(y.min(), y.max(), 5)),
    ]


def _to_r(value: float | NDArray[np.float64]) -> float | list[float]:
    if isinstance(value, np.ndarray):
        return [float(item) for item in value.tolist()]
    return value
