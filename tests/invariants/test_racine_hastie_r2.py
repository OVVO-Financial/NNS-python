from __future__ import annotations

import numpy as np

from nns import nns_m_reg, nns_reg
from nns._reg_engine import _r2 as engine_r2
from nns.regression import _r2 as legacy_r2


def _predictive_r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    return 1.0 - float(np.sum((actual - predicted) ** 2)) / float(
        np.sum((actual - np.mean(actual)) ** 2)
    )


def test_r2_is_squared_observed_fitted_correlation() -> None:
    actual = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    predicted = np.array([8.0, 4.0, 0.0, -4.0, -8.0])
    expected = float(np.corrcoef(actual, predicted)[0, 1] ** 2)

    np.testing.assert_allclose(expected, 1.0, atol=1e-14)
    np.testing.assert_allclose(engine_r2(actual, predicted), expected, atol=1e-14)
    np.testing.assert_allclose(legacy_r2(actual, predicted), expected, atol=1e-14)


def test_r2_never_inherits_negative_predictive_r2() -> None:
    actual = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    predicted = np.array([20.0, 10.0, 0.0, -10.0, -20.0])

    assert _predictive_r2(actual, predicted) < 0.0
    for value in (engine_r2(actual, predicted), legacy_r2(actual, predicted)):
        assert 0.0 <= value <= 1.0
        np.testing.assert_allclose(value, 1.0, atol=1e-14)


def test_r2_constant_series_degeneracy_is_explicit() -> None:
    exact = np.full(5, 3.0)
    mismatch = np.full(5, 4.0)
    varying = np.arange(1.0, 6.0)

    for helper in (engine_r2, legacy_r2):
        assert helper(exact, exact) == 1.0
        assert helper(exact, mismatch) == 0.0
        assert helper(varying, exact) == 0.0


def test_public_regression_r2_values_are_bounded() -> None:
    x = np.linspace(-2.0, 2.0, 40)
    y = np.sin(4.0 * x) + x**2

    univariate = nns_reg(x, y, order=1)
    multivariate = nns_m_reg(np.column_stack((x, x**2)), y, order=1, n_best=1)

    assert 0.0 <= univariate["R2"] <= 1.0
    assert 0.0 <= multivariate["R2"] <= 1.0
