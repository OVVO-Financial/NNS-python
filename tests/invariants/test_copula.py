from __future__ import annotations

import numpy as np
import pytest

from nns import nns_copula


def test_nns_copula_is_bounded() -> None:
    x = np.linspace(-2.0, 2.0, 200)
    y = np.sin(x)

    result = nns_copula(x, y)

    assert result >= 0.0
    assert result <= 1.0


def test_nns_copula_is_symmetric() -> None:
    x = np.linspace(-2.0, 2.0, 200)
    y = x**3

    assert nns_copula(x, y) == pytest.approx(nns_copula(y, x), abs=1e-12)


def test_nns_copula_identical_pair_is_unit() -> None:
    x = np.linspace(-2.0, 2.0, 200)

    assert nns_copula(x, x) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("x", "y", "kwargs"),
    [
        (np.array([]), np.array([]), {}),
        (np.array([0.1, np.nan, 0.3, 0.4]), np.array([1.0, 2.0, 3.0, 4.0]), {}),
        (np.array([0.1, 0.2, 0.3, np.inf]), np.array([1.0, 2.0, 3.0, 4.0]), {}),
        (np.array([0.1, 0.2, 0.3, 0.4]), np.array([1.0, -np.inf, 3.0, 4.0]), {}),
        (np.array([0.1, 0.2, 0.3, 0.4]), np.array([1.0, 2.0, 3.0, 4.0]), {"target_x": np.nan}),
        (np.array([0.1, 0.2, 0.3, 0.4]), np.array([1.0, 2.0, 3.0, 4.0]), {"target_y": np.inf}),
    ],
)
def test_nns_copula_rejects_empty_and_non_finite_bivariate_input(
    x: np.ndarray, y: np.ndarray, kwargs: dict[str, float]
) -> None:
    # R NNS.copula stops on missing / non-finite data. The bivariate form used to
    # segfault on empty input and return 1.0 for a NaN.
    with pytest.raises(ValueError):
        nns_copula(x, y, **kwargs)


def test_nns_copula_rejects_empty_matrix() -> None:
    with pytest.raises(ValueError):
        nns_copula(np.empty((0, 2)))
