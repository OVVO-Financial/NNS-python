from __future__ import annotations

import numpy as np
import pytest
from _r import nns
from _tolerances import EXACT

from pynns import nns_copula

SIZES = [50, 200, 1000]
RELATIONSHIPS = ["correlated_normal", "independent", "anti_monotonic"]


@pytest.mark.parity
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("relationship", RELATIONSHIPS)
def test_nns_copula_matches_r(
    rng: np.random.Generator,
    size: int,
    relationship: str,
) -> None:
    x, y = _relationship(relationship, size, rng)

    expected = nns("NNS.copula", np.column_stack((x, y)).tolist(), None, True, False, False)
    actual = nns_copula(x, y)

    np.testing.assert_allclose(actual, _scalar(expected), atol=EXACT)


@pytest.mark.parity
def test_nns_copula_target_matches_r() -> None:
    x = np.linspace(-2.0, 2.0, 200)
    y = np.sin(x)

    expected = nns(
        "NNS.copula",
        np.column_stack((x, y)).tolist(),
        [0.25, -0.1],
        True,
        False,
        False,
    )
    actual = nns_copula(x, y, target_x=0.25, target_y=-0.1)

    np.testing.assert_allclose(actual, _scalar(expected), atol=EXACT)


def _scalar(value: object) -> float:
    assert isinstance(value, np.ndarray)
    return float(value)


def _relationship(
    relationship: str,
    size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    x = rng.normal(size=size)
    if relationship == "correlated_normal":
        noise = rng.normal(size=size)
        return x, 0.7 * x + np.sqrt(1.0 - 0.7**2) * noise
    if relationship == "independent":
        return x, rng.normal(size=size)
    if relationship == "monotonic_nonlinear":
        return x, x**3
    return x, -x
