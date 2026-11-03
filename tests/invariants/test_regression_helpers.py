from __future__ import annotations

import numpy as np
import pytest

from nns import lpm_ratio, lpm_var, nns_gravity, nns_mode, nns_rescale, upm_ratio, upm_var
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


def _var_root_data() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    return {
        "symmetric": np.array([-0.003, -0.002, -0.001, 0.001, 0.002, 0.003]),
        "skewed": np.append(rng.exponential(size=40) - 1.0, 5.0),
        "returns": rng.normal(0.0, 0.01, 500),
    }


@pytest.mark.invariant
def test_integer_degree_var_finds_the_symmetric_median_at_small_scale() -> None:
    # R LPM.VaR(.5, 4, x) gave -0.001 here: the root tolerance was absolute in x^4 units.
    x = _var_root_data()["symmetric"]
    assert lpm_var(0.5, 4, x) == pytest.approx(0.0, abs=1e-12)
    assert upm_var(0.5, 4, x) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.invariant
@pytest.mark.parametrize("degree", [1, 2, 3, 4, 5])
def test_var_scales_with_the_data_and_inverts_its_ratio(degree: int) -> None:
    p = [0.01, 0.05, 0.5, 0.95]
    for x in _var_root_data().values():
        spread = float(np.ptp(x))
        lower = lpm_var(p, degree, x)
        upper = upm_var(p, degree, x)
        for k in (1e-4, 1.0, 1e4):
            y = k * x
            assert np.max(np.abs(lpm_var(p, degree, y) - k * lower)) < 1e-10 * k * spread
            assert np.max(np.abs(upm_var(p, degree, y) - k * upper)) < 1e-10 * k * spread
            # Degree 5 uses the general solver, held to 1e-7 like other fractional degrees.
            tol = 1e-8 if degree <= 4 else 1e-7
            lr = [float(lpm_ratio(degree, lpm_var(q, degree, y), y)) for q in p]
            ur = [float(upm_ratio(degree, upm_var(q, degree, y), y)) for q in p]
            assert np.max(np.abs(np.subtract(lr, p))) < tol
            assert np.max(np.abs(np.subtract(ur, p))) < tol


@pytest.mark.invariant
@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_integer_degree_var_inverts_the_ratio_where_it_is_steep(degree: int) -> None:
    x = np.append(np.zeros(99999), 1.0)
    assert abs(float(lpm_ratio(degree, lpm_var(0.036, degree, x), x)) - 0.036) < 1e-10
    assert abs(float(upm_ratio(degree, upm_var(0.964, degree, x), x)) - 0.964) < 1e-10


@pytest.mark.invariant
def test_integer_degree_var_matches_r_main() -> None:
    # R main LPM.VaR(c(.01, .05, .5, .95), d, x) for d = 2, 3 on 1e-4 * x; R 13.3
    # returned bracket endpoints (-5e-05, -2e-05, 0.00012, ...).
    x = 1e-4 * np.array([-0.9, -0.5, -0.2, 0.1, 0.3, 0.4, 1.2, 5.0])
    p = [0.01, 0.05, 0.5, 0.95]
    np.testing.assert_allclose(
        lpm_var(p, 2, x),
        [
            -3.4961654878956205e-05,
            7.3758108967875851e-06,
            0.00013040283320663138,
            0.00031087187473751696,
        ],
        rtol=1e-13,
    )
    np.testing.assert_allclose(
        lpm_var(p, 3, x),
        [
            9.1953590717989913e-06,
            5.4438723908451524e-05,
            0.00016130986351900996,
            0.00028812474290722652,
        ],
        rtol=1e-13,
    )


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
