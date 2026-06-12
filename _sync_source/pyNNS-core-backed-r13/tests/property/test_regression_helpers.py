from __future__ import annotations

import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pynns import lpm_var, nns_mode, nns_rescale, upm_var

finite_arrays = arrays(
    dtype=np.float64,
    shape=st.integers(min_value=4, max_value=80),
    elements=st.floats(
        min_value=-1e6,
        max_value=1e6,
        allow_nan=False,
        allow_infinity=False,
        width=64,
    ),
)

positive_arrays = arrays(
    dtype=np.float64,
    shape=st.integers(min_value=4, max_value=80),
    elements=st.floats(
        min_value=1e-6,
        max_value=1e6,
        allow_nan=False,
        allow_infinity=False,
        width=64,
    ),
)


@given(finite_arrays, st.floats(-100.0, 100.0), st.floats(-100.0, 100.0))
def test_nns_rescale_minmax_bounds_hold(values: np.ndarray, a: float, b: float) -> None:
    assume(a != b)
    assume(np.ptp(values) > 0.0)

    result = nns_rescale(values, a, b)
    low = min(a, b)
    high = max(a, b)

    assert np.all(result >= low - 1e-9)
    assert np.all(result <= high + 1e-9)


@given(
    positive_arrays,
    st.floats(min_value=1e-3, max_value=1e3),
    st.floats(min_value=-0.5, max_value=0.5),
    st.floats(min_value=1e-6, max_value=10.0),
    st.sampled_from(["Terminal", "Discounted"]),
)
def test_nns_rescale_riskneutral_mean_target_holds(
    values: np.ndarray,
    spot: float,
    rate: float,
    time_to_maturity: float,
    target_type: str,
) -> None:
    result = nns_rescale(values, spot, rate, "riskneutral", time_to_maturity, target_type)

    target = spot if target_type == "Discounted" else spot * np.exp(rate * time_to_maturity)
    assert np.isfinite(result).all()
    assert abs(float(np.mean(result)) - float(target)) <= 1e-9 * max(1.0, abs(float(target)))


@given(finite_arrays, st.floats(min_value=0.0, max_value=1.0), st.sampled_from([0.0, 1.0, 2.0]))
def test_var_outputs_are_inside_observed_range(
    values: np.ndarray,
    percentile: float,
    degree: float,
) -> None:
    assume(np.ptp(values) > 0.0)

    lower = lpm_var(percentile, degree, values)
    upper = upm_var(percentile, degree, values)

    assert float(np.min(values)) <= lower <= float(np.max(values))
    assert float(np.min(values)) <= upper <= float(np.max(values))


@given(finite_arrays, st.booleans(), st.booleans())
def test_nns_mode_is_finite_and_within_observed_range(
    values: np.ndarray,
    discrete: bool,
    multi: bool,
) -> None:
    result = np.asarray(nns_mode(values, discrete=discrete, multi=multi), dtype=np.float64)

    assert np.all(np.isfinite(result))
    assert np.all(result >= np.min(values) - 1.0)
    assert np.all(result <= np.max(values) + 1.0)
