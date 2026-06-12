from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pynns import nns_arma
from pynns.arma import _numeric_seasonal_weights

finite_arrays = arrays(
    dtype=np.float64,
    shape=st.integers(min_value=10, max_value=100),
    elements=st.floats(
        min_value=-100.0,
        max_value=100.0,
        allow_nan=False,
        allow_infinity=False,
        width=64,
    ),
)


def _valid_interval_estimates(values: np.ndarray) -> bool:
    if values.size < 2 or not np.all(np.isfinite(values)) or np.ptp(values) <= 1e-8:
        return False
    time = np.arange(1, values.size + 1, dtype=np.float64)
    fitted = np.polyval(np.polyfit(time, values, 1), time)
    residuals = values - fitted
    if np.ptp(residuals) <= 1e-8 or np.std(residuals) <= 1e-8:
        return False
    return bool(np.unique(np.round(residuals, decimals=12)).size >= 3)


@given(
    finite_arrays,
    st.integers(min_value=1, max_value=5),
    st.sampled_from([1, 4]),
    st.sampled_from(["lin", "nonlin", "both", "means"]),
)
def test_nns_arma_random_explicit_lag_shape(
    variable: np.ndarray,
    h: int,
    seasonal_factor: int,
    method: str,
) -> None:
    assume(np.ptp(variable) > 0.0)
    assume(np.unique(variable).size > 8)
    assume(np.all(np.isfinite(_numeric_seasonal_weights(variable, np.array([seasonal_factor])))))

    result = nns_arma(variable, h=h, seasonal_factor=seasonal_factor, method=method)

    assert result.shape == (h,)


@pytest.mark.stochastic
@given(
    finite_arrays,
    st.just(5),
    st.just(4),
    st.sampled_from([0.8, 0.95]),
    st.integers(min_value=0, max_value=10000),
)
def test_nns_arma_pred_int_random_explicit_lag_shape(
    variable: np.ndarray,
    h: int,
    seasonal_factor: int,
    pred_int: float,
    seed: int,
) -> None:
    variable = variable + 0.01 * np.arange(variable.size, dtype=np.float64)
    variable = variable + 0.1 * np.sin(np.arange(variable.size, dtype=np.float64) / 3.0)
    assume(np.ptp(variable) > 1e-8)
    assume(np.unique(np.round(variable, decimals=12)).size > 8)
    estimates = nns_arma(variable, h=h, seasonal_factor=seasonal_factor, method="nonlin")
    assert isinstance(estimates, np.ndarray)
    assume(_valid_interval_estimates(estimates))
    result = nns_arma(
        variable,
        h=h,
        seasonal_factor=seasonal_factor,
        method="nonlin",
        pred_int=pred_int,
        random_seed=seed,
    )

    assert isinstance(result, dict)
    assert result["Estimates"].shape == (h,)
    assert result[f"Lower {int(pred_int * 100)}% pred.int"].shape == (h,)
    assert result[f"Upper {int(pred_int * 100)}% pred.int"].shape == (h,)
    for value in result.values():
        assert np.all(np.isfinite(value))
