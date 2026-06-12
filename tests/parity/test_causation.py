from __future__ import annotations

import numpy as np
import pytest
from _r import nns
from _tolerances import EXACT

from pynns import causal_matrix, nns_causation

SIZES = [50, 200, 1000]
RELATIONSHIPS = ["linear", "independent", "quadratic", "sin", "asymmetric"]
TS_TOLERANCE = 7e-2


@pytest.mark.parity
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("relationship", RELATIONSHIPS)
def test_nns_causation_matches_r(
    rng: np.random.Generator,
    size: int,
    relationship: str,
) -> None:
    x, y = _relationship(relationship, size, rng)

    expected = nns("NNS.caus", x.tolist(), y.tolist(), False, 0, False, False)
    actual = np.fromiter(nns_causation(x, y).values(), dtype=np.float64)

    np.testing.assert_allclose(actual, _vector(expected), atol=EXACT)


@pytest.mark.parity
def test_causal_matrix_matches_r() -> None:
    rng = np.random.default_rng(123)
    x = rng.normal(size=100)
    variable = np.column_stack(
        (
            x,
            x**2 + 0.1 * rng.normal(size=100),
            np.sin(x) + 0.05 * rng.normal(size=100),
        )
    )

    expected = nns("NNS.caus", variable.tolist(), None, False, 0, False, False)
    actual = causal_matrix(variable)

    np.testing.assert_allclose(actual, _matrix(expected), atol=EXACT)


@pytest.mark.parity
@pytest.mark.parametrize("case", ["period7", "random", "short", "trend_period6"])
def test_nns_causation_ts_tau_matches_r(case: str) -> None:
    x, y = _ts_relationship(case)

    expected = nns("NNS.caus", x.tolist(), y.tolist(), False, "ts", False, False)
    actual = np.fromiter(nns_causation(x, y, tau="ts").values(), dtype=np.float64)

    np.testing.assert_allclose(actual, _vector(expected), atol=TS_TOLERANCE)


@pytest.mark.parity
def test_causal_matrix_ts_tau_matches_r() -> None:
    t = np.arange(1, 81, dtype=np.float64)
    variable = np.column_stack(
        (
            np.sin(2.0 * np.pi * t / 5.0),
            np.sin(2.0 * np.pi * t / 6.0 + 0.2),
            np.sin(2.0 * np.pi * t / 7.0 + 0.5),
        )
    )

    expected = nns("NNS.caus", variable.tolist(), None, False, "ts", False, False)
    actual = causal_matrix(variable, tau="ts")

    np.testing.assert_allclose(actual, _matrix(expected), atol=TS_TOLERANCE)


def _vector(value: object) -> np.ndarray:
    assert isinstance(value, np.ndarray)
    return value.astype(np.float64)


def _matrix(value: object) -> np.ndarray:
    assert isinstance(value, np.ndarray)
    return value.astype(np.float64)


def _relationship(
    relationship: str,
    size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    x = rng.normal(size=size)
    if relationship == "linear":
        noise = rng.normal(size=size)
        return x, 0.7 * x + np.sqrt(1.0 - 0.7**2) * noise
    if relationship == "independent":
        return x, rng.normal(size=size)
    if relationship == "quadratic":
        return x, x * x + 0.1 * rng.normal(size=size)
    if relationship == "sin":
        return x, np.sin(x) + 0.05 * rng.normal(size=size)
    return x, x * x + 0.1 * rng.normal(size=size)


def _ts_relationship(case: str) -> tuple[np.ndarray, np.ndarray]:
    if case == "short":
        x = np.array([1.0, 2.0, 1.5, 2.5], dtype=np.float64)
        return x, np.array([0.5, 0.75, 0.6, 0.9], dtype=np.float64)
    if case == "random":
        rng = np.random.default_rng(123)
        return rng.normal(size=50), rng.normal(size=50)
    if case == "trend_period6":
        t = np.arange(1, 401, dtype=np.float64)
        x = 0.02 * t + np.sin(2.0 * np.pi * t / 12.0)
        y = np.roll(x, 2) + 0.1 * np.cos(t / 5.0)
        return x, y
    t = np.arange(1, 71, dtype=np.float64)
    x = np.sin(2.0 * np.pi * t / 7.0)
    y = np.roll(x, 1) + 0.05 * np.cos(t / 3.0)
    return x, y
