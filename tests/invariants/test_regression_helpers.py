from __future__ import annotations

import numpy as np
import pytest

from nns import lpm_var, nns_gravity, nns_mode, nns_rescale, upm_var
from nns._helpers import _fast_lm, _is_fcl


@pytest.mark.invariant
def test_nns_rescale_minmax_spans_requested_bounds() -> None:
    values = np.array([-3.0, -1.0, 0.0, 2.0, 4.0])

    result = nns_rescale(values, -2.0, 3.0)

    assert float(np.min(result)) == pytest.approx(-2.0)
    assert float(np.max(result)) == pytest.approx(3.0)


@pytest.mark.invariant
def test_nns_rescale_minmax_constant_returns_midpoint() -> None:
    result = nns_rescale(np.array([7.0, 7.0, 7.0]), -2.0, 4.0)

    np.testing.assert_allclose(result, np.array([1.0, 1.0, 1.0]))


@pytest.mark.invariant
@pytest.mark.parametrize("target_type", ["Terminal", "Discounted"])
def test_nns_rescale_riskneutral_mean_matches_target(target_type: str) -> None:
    values = np.array([11.0, 12.0, 15.0, 20.0, 25.0])
    result = nns_rescale(values, 100.0, 0.05, "riskneutral", 1.25, target_type)

    target = 100.0 if target_type == "Discounted" else 100.0 * np.exp(0.05 * 1.25)
    assert float(np.mean(result)) == pytest.approx(target)


@pytest.mark.invariant
def test_lpm_upm_var_degree_zero_match_linear_quantile_conventions() -> None:
    values = np.array([1.0, 2.0, 4.0, 8.0])

    assert lpm_var(0.25, 0.0, values) == pytest.approx(np.quantile(values, 0.25))
    assert upm_var(0.25, 0.0, values) == pytest.approx(np.quantile(values, 0.75))


@pytest.mark.invariant
@pytest.mark.parametrize("degree", [0.0, 1.0, 2.0])
def test_var_outputs_stay_in_observed_range(degree: float) -> None:
    values = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])

    lower = lpm_var(0.3, degree, values)
    upper = upm_var(0.3, degree, values)

    assert float(np.min(values)) <= lower <= float(np.max(values))
    assert float(np.min(values)) <= upper <= float(np.max(values))


@pytest.mark.invariant
def test_nns_mode_outputs_finite_value_for_finite_input() -> None:
    values = np.array([-10.0, -9.0, -8.0, 0.0, 1.0, 2.0, 2.0, 50.0])

    result = nns_mode(values)

    assert np.all(np.isfinite(np.asarray(result, dtype=np.float64)))


@pytest.mark.invariant
def test_discrete_nns_mode_breaks_ties_by_gravity_not_their_mean() -> None:
    def nearest(x: np.ndarray) -> float:
        modes = np.asarray(nns_mode(x, discrete=True, multi=True))
        return float(modes[np.argmin(np.abs(modes - nns_gravity(x)))])  # ties: the lower

    cases = ([1, 1, 3, 3], [1, 1, 2, 2], np.repeat(np.arange(20), 400), [1, 3], [1, 2])
    for case in cases:
        x = np.asarray(case, dtype=np.float64)
        result = nns_mode(x, discrete=True, multi=False)
        assert result in x
        assert result == nearest(x)
    assert nns_mode(np.array([1.0, 1.0, 3.0, 3.0]), discrete=True) == 1.0  # 2 before
    assert nns_mode(np.repeat(np.arange(20.0), 400), discrete=True) == 9.0
    np.testing.assert_array_equal(
        nns_mode(np.array([1.0, 3.0]), discrete=True, multi=True), [1.0, 3.0]
    )
    beyond_int = np.array([3e9, 3e9, -5e9, -5e9, 1.0])
    np.testing.assert_array_equal(nns_mode(beyond_int, discrete=True, multi=True), [-5e9, 3e9])
    assert nns_mode(beyond_int, discrete=True, multi=False) == -5e9


@pytest.mark.invariant
def test_fast_lm_matches_known_line() -> None:
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = 3.0 + 2.0 * x

    intercept, slope = _fast_lm(x, y)

    assert intercept == pytest.approx(3.0)
    assert slope == pytest.approx(2.0)


@pytest.mark.invariant
def test_fast_lm_constant_x_returns_mean_and_zero_slope() -> None:
    intercept, slope = _fast_lm(np.array([2.0, 2.0, 2.0]), np.array([1.0, 3.0, 5.0]))

    assert intercept == pytest.approx(3.0)
    assert slope == pytest.approx(0.0)


@pytest.mark.invariant
def test_is_fcl_maps_python_numeric_and_non_numeric_dtypes() -> None:
    assert not _is_fcl(np.array([1.0, 2.0]))
    assert not _is_fcl(np.array([1, 2]))
    assert _is_fcl(np.array([True, False]))
    assert _is_fcl(np.array(["a", "b"]))
    assert _is_fcl(np.array([object(), object()], dtype=object))
