from __future__ import annotations

import warnings
from typing import Any
from unittest import mock

import numpy as np
import pytest

import nns.arma as arma_module
from nns import nns_arma, nns_arma_optim, nns_mc, nns_var
from nns.arma import (
    _default_arma_optim_objective,
    _generate_vectors,
    _linear_forecast_for_lag,
    _numeric_seasonal_weights,
)
from nns.meboot import _target_rho
from nns.var import lpm_var, upm_var


def test_nns_arma_output_length_matches_h() -> None:
    variable = np.sin(np.arange(1, 80, dtype=np.float64) / 4.0) + 2.0

    result = nns_arma(variable, h=12, seasonal_factor=4, method="nonlin")

    assert result.shape == (12,)


def test_nns_arma_repeated_calls_are_deterministic() -> None:
    variable = np.sin(np.arange(1, 60, dtype=np.float64) / 3.0) + 2.0

    first = nns_arma(variable, h=5, seasonal_factor=4, method="both")
    second = nns_arma(variable, h=5, seasonal_factor=4, method="both")

    np.testing.assert_array_equal(first, second)


def test_nns_arma_numeric_seasonal_dynamic_raises() -> None:
    variable = np.sin(np.arange(1, 40, dtype=np.float64))

    with pytest.raises(ValueError, match="dynamic"):
        nns_arma(variable, h=3, seasonal_factor=5, dynamic=True)


def test_nns_arma_numeric_weights_dynamic_raises() -> None:
    variable = np.sin(np.arange(1, 40, dtype=np.float64)) + 2.0

    with pytest.raises(ValueError, match="dynamic"):
        nns_arma(
            variable, h=3, seasonal_factor=False, weights=np.array([1.0, 0.0]), dynamic=True
        )


def test_nns_arma_dynamic_equal_weights_match_repeated_one_step_forecasts() -> None:
    # As in patched R, weights="equal" applies to the lags each dynamic step selects
    # (they used to be replaced by the NNS.seas weights).
    variable = 100.0 + np.arange(60.0) + 15.0 * np.sin(np.arange(60.0) * np.pi / 6.0)
    options: dict[str, Any] = {"seasonal_factor": False, "best_periods": 2, "weights": "equal"}

    dynamic = nns_arma(variable, h=3, dynamic=True, **options)
    path = variable.copy()
    for _ in range(3):
        path = np.append(path, nns_arma(path, h=1, **options))

    np.testing.assert_allclose(dynamic, path[-3:], rtol=1e-12)


def test_numeric_seasonal_weights_constant_subsample_is_finite() -> None:
    # Regression: a lag-subsample with zero variance gives CV 0, so the seasonal
    # weighting 1 / (CV / baseline) = 1 / 0 = Inf and the normalisation collapsed to
    # Inf / Inf = NaN -- which produced an all-NaN forecast. Now guarded.
    rng = np.random.default_rng(0)
    variable = rng.uniform(50.0, 150.0, size=40)
    variable[::-4] = 10.0  # every 4th point (from the end) identical -> std 0

    weights = _numeric_seasonal_weights(variable, np.array([4], dtype=np.int64))
    assert np.isfinite(weights).all()
    np.testing.assert_allclose(weights.sum(), 1.0)

    forecast = nns_arma(variable, h=8, seasonal_factor=4, method="lin")
    assert np.isfinite(forecast).all()


def test_target_rho_uses_finite_candidate_when_midpoint_is_degenerate() -> None:
    # With tied three-point residuals, the 50/50 blend can be constant even though
    # the endpoint rank arrangements are valid. A non-finite midpoint must not
    # abort correlation targeting when a finite candidate exists.
    orig_res = np.array([-1.0, 2.0, -1.0], dtype=np.float64)
    res_mat = np.array([[0.0], [1.0], [1.0]], dtype=np.float64)

    result = _target_rho(res_mat, orig_res, rho=1.0, type_="spearman")

    assert result.shape == (3, 1)
    assert np.all(np.isfinite(result))
    assert np.ptp(result[:, 0]) > 0.0


@pytest.mark.stochastic
def test_nns_arma_pred_int_returns_interval_dict() -> None:
    variable = np.sin(np.arange(1, 40, dtype=np.float64) / 3.0) + 2.0

    result = nns_arma(
        variable,
        h=3,
        seasonal_factor=4,
        method="nonlin",
        pred_int=0.95,
        random_seed=123,
    )

    assert isinstance(result, dict)
    assert list(result) == ["Estimates", "Lower 95% pred.int", "Upper 95% pred.int"]
    assert all(value.shape == (3,) for value in result.values())


def test_nns_arma_nonlin_pred_int_widens_by_linear_residual_like_r() -> None:
    # R fits the per-lag linear models for method="nonlin" whenever pred.int is set
    # and widens the band by lin.resid = mean(abs(e - mean(e))) of the last step's
    # linear forecasts e (one per lag).
    t = np.arange(1, 81, dtype=np.float64)
    variable = 50.0 + 0.5 * t + 5.0 * np.sin(2.0 * np.pi * t / 4.0) + 2.0 * np.cos(t / 2.0)
    lags = np.array([4, 8], dtype=np.int64)
    h = 6

    point = nns_arma(variable, h=h, seasonal_factor=lags, method="nonlin")
    result = nns_arma(
        variable, h=h, seasonal_factor=lags, method="nonlin", pred_int=0.9, random_seed=3
    )

    generated = _generate_vectors(np.concatenate((variable, point[:-1])), lags)
    lin = np.array(
        [
            _linear_forecast_for_lag(index, series)
            for index, series in zip(
                generated["Component.index"], generated["Component.series"], strict=True
            )
        ]
    )
    lin_resid = float(np.mean(np.abs(lin - np.mean(lin))))
    assert lin_resid > 0.0

    mc = nns_mc(point, lower_rho=-1.0, upper_rho=1.0, by=0.2, random_seed=3)
    replicates = mc["replicates"]
    assert isinstance(replicates, dict)
    rows = np.column_stack(list(replicates.values()))
    upper = np.array([upm_var(0.05, 0.0, row) for row in rows]) + lin_resid
    lower = np.array([abs(lpm_var(0.05, 0.0, row)) for row in rows]) - lin_resid

    assert isinstance(result, dict)
    np.testing.assert_array_equal(result["Estimates"], point)
    np.testing.assert_allclose(result["Upper 90% pred.int"], np.maximum(point, upper), rtol=1e-12)
    np.testing.assert_allclose(result["Lower 90% pred.int"], np.minimum(point, lower), rtol=1e-12)


@pytest.mark.stochastic
def test_nns_arma_pred_int_seed_reproducibility() -> None:
    variable = np.sin(np.arange(1, 45, dtype=np.float64) / 3.0) + 2.0

    first = nns_arma(variable, h=4, seasonal_factor=4, method="nonlin", pred_int=0.8, random_seed=1)
    second = nns_arma(
        variable, h=4, seasonal_factor=4, method="nonlin", pred_int=0.8, random_seed=1
    )
    third = nns_arma(variable, h=4, seasonal_factor=4, method="nonlin", pred_int=0.8, random_seed=2)

    assert isinstance(first, dict)
    assert isinstance(second, dict)
    assert isinstance(third, dict)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    assert not np.array_equal(first["Lower 80% pred.int"], third["Lower 80% pred.int"])


def test_nns_arma_static_linear_pred_int_returns_intervals() -> None:
    t = np.arange(1, 61, dtype=np.float64)
    variable = 50.0 + t + 5.0 * np.sin(2.0 * np.pi * t / 4.0) + np.cos(t)

    result = nns_arma(variable, h=6, seasonal_factor=4, method="lin", pred_int=0.9, random_seed=1)
    point = nns_arma(variable, h=6, seasonal_factor=4, method="lin")

    assert isinstance(result, dict)
    assert list(result) == ["Estimates", "Lower 90% pred.int", "Upper 90% pred.int"]
    np.testing.assert_array_equal(result["Estimates"], point)
    assert np.all(result["Lower 90% pred.int"] <= point)
    assert np.all(result["Upper 90% pred.int"] >= point)


def test_nns_arma_static_linear_pred_int_exact_line_matches_installed_r_error() -> None:
    # An exactly linear forecast path has constant meboot residuals; R fails the same way.
    variable = np.arange(1, 21, dtype=np.float64)

    with pytest.raises(ValueError, match="cannot be evaluated at initial parameters"):
        nns_arma(variable, h=5, seasonal_factor=4, method="lin", pred_int=0.95)


def test_nns_arma_pred_int_h_one_returns_lin_residual_band() -> None:
    # R fails with "incorrect number of dimensions"; a single forecast has no MC path
    # to bootstrap, so the band is the point forecast widened by lin.resid.
    variable = np.sin(np.arange(1, 40, dtype=np.float64) / 3.0) + 2.0

    single = nns_arma(variable, h=1, seasonal_factor=4, method="nonlin", pred_int=0.95)
    assert isinstance(single, dict)
    point = nns_arma(variable, h=1, seasonal_factor=4, method="nonlin")
    for value in single.values():
        np.testing.assert_array_equal(value, point)

    lags = np.array([3, 4], dtype=np.int64)
    multi = nns_arma(variable, h=1, seasonal_factor=lags, method="lin", pred_int=0.95)
    assert isinstance(multi, dict)
    generated = _generate_vectors(variable, lags)
    lin = np.array(
        [
            _linear_forecast_for_lag(index, series)
            for index, series in zip(
                generated["Component.index"], generated["Component.series"], strict=True
            )
        ]
    )
    lin_resid = float(np.mean(np.abs(lin - np.mean(lin))))
    estimate = multi["Estimates"]
    np.testing.assert_allclose(multi["Upper 95% pred.int"], estimate + lin_resid, rtol=1e-12)
    np.testing.assert_allclose(multi["Lower 95% pred.int"], estimate - lin_resid, rtol=1e-12)


@pytest.mark.parametrize("training_set", [0, -1, 40])
def test_nns_arma_rejects_out_of_range_training_set(training_set: int) -> None:
    # R: 0 -> "[y] must contain at least two observations.", > n -> "[y] contains
    # missing values."; Python used to slice silently or raise IndexError.
    variable = np.sin(np.arange(1, 40, dtype=np.float64) / 3.0) + 2.0

    with pytest.raises(ValueError, match="training_set must be between 1"):
        nns_arma(variable, h=2, training_set=training_set, seasonal_factor=4)


@pytest.mark.parametrize("length", [1, 2, 3, 4])
def test_nns_arma_short_series_raise_value_error(length: int) -> None:
    variable = np.arange(1, length + 1, dtype=np.float64)

    with pytest.raises(ValueError):
        nns_arma(variable, h=2)


def test_nns_arma_short_series_with_numeric_lag_still_forecasts() -> None:
    # R 13.3 + patches (no rounding of integer-x regression points):
    # NNS.ARMA(c(1,2,4), h=2, seasonal.factor=1) = 5.8333333333333339 7.5714285714285721.
    result = nns_arma(np.array([1.0, 2.0, 4.0]), h=2, seasonal_factor=1)

    np.testing.assert_allclose(result, [5.8333333333333339, 7.5714285714285721], rtol=1e-12)


@pytest.mark.parametrize("seasonal_factor", [0, 40, [4, 0]])
def test_nns_arma_rejects_out_of_range_lags(seasonal_factor: int | list[int]) -> None:
    variable = np.sin(np.arange(1, 40, dtype=np.float64) / 3.0) + 2.0

    with pytest.raises(ValueError, match="seasonal_factor lags must be between 1"):
        nns_arma(variable, h=2, seasonal_factor=seasonal_factor)


def test_nns_arma_seasonal_factor_none_behaves_like_r_false() -> None:
    variable = np.sin(np.arange(1, 60, dtype=np.float64) / 3.0) + 0.1 * np.arange(1, 60)

    np.testing.assert_array_equal(
        nns_arma(variable, h=3, seasonal_factor=None),  # type: ignore[arg-type]
        nns_arma(variable, h=3, seasonal_factor=False),
    )
    with pytest.raises(TypeError, match="seasonal_factor must be"):
        nns_arma(variable, h=3, seasonal_factor="12")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("seasonal_factor", "weights"),
    [([3, 4], [1.0]), ([3, 4], [0.2, 0.3, 0.5]), (4, [0.5, 0.5]), (False, [0.5, 0.5])],
)
def test_nns_arma_rejects_weights_of_wrong_length(
    seasonal_factor: bool | int | list[int], weights: list[float]
) -> None:
    variable = np.sin(np.arange(1, 60, dtype=np.float64) / 3.0) + 2.0

    with pytest.raises(ValueError, match="one weight per lag"):
        nns_arma(variable, h=2, seasonal_factor=seasonal_factor, weights=np.asarray(weights))


def test_nns_arma_optim_returns_forecast_dictionary() -> None:
    variable = np.sin(np.arange(1, 40, dtype=np.float64) / 3.0) + 2.0

    result = nns_arma_optim(
        variable,
        h=3,
        seasonal_factor=[3, 4, 5, 6, 7],
        lin_only=True,
        print_trace=False,
    )

    assert list(result) == [
        "periods",
        "weights",
        "obj.fn",
        "method",
        "shrink",
        "nns.regress",
        "bias.shift",
        "errors",
        "results",
        "lower.pred.int",
        "upper.pred.int",
    ]
    assert result["results"].shape == (3,)
    assert result["lower.pred.int"].shape == (3,)
    assert result["upper.pred.int"].shape == (3,)
    assert np.all(np.isfinite(result["results"]))


def test_nns_arma_optim_validation_matches_installed_r() -> None:
    variable = np.sin(np.arange(1, 40, dtype=np.float64) / 3.0) + 2.0

    with pytest.raises(ValueError, match=r"larger \[training.set\]"):
        nns_arma_optim(
            variable,
            training_set=19,
            seasonal_factor=[3, 4],
            lin_only=True,
            print_trace=False,
        )



def test_nns_arma_optim_bias_shift_is_never_negative_zero() -> None:
    variable = np.sin(np.arange(1, 40, dtype=np.float64) / 3.0) + 2.0

    with mock.patch.object(arma_module, "_finite_gravity", return_value=0.0):
        result = nns_arma_optim(
            variable, h=3, seasonal_factor=[3, 4, 5], lin_only=True, print_trace=False
        )

    assert result["bias.shift"] == 0.0
    assert not np.signbit(result["bias.shift"])


@pytest.mark.parametrize("dim_red_method", ["cor", "NNS.dep", "NNS.caus", "all"])
def test_nns_var_public_supported_paths_return_output_contract(dim_red_method: str) -> None:
    variables = np.column_stack(
        (
            np.sin(np.arange(1, 40, dtype=np.float64) / 3.0),
            np.cos(np.arange(1, 40, dtype=np.float64) / 4.0),
        )
    )

    result = nns_var(variables, h=3, tau=2, dim_red_method=dim_red_method)

    assert set(result) == {
        "interpolated_and_extrapolated",
        "relevant_variables",
        "univariate",
        "multivariate",
        "ensemble",
        "names",
    }
    assert result["interpolated_and_extrapolated"].shape == variables.shape
    assert result["univariate"].shape == (3, 2)
    assert result["multivariate"].shape == (3, 2)
    assert result["ensemble"].shape == (3, 2)
    assert result["names"] == ["x1", "x2"]


@pytest.mark.parametrize("values", [np.array([1.0, np.nan, 3.0]), np.array([1.0, np.inf, 3.0])])
def test_nns_arma_non_finite_inputs_raise(values: np.ndarray) -> None:
    with pytest.raises(ValueError):
        nns_arma(values)


def test_nns_arma_finite_where_r_is_finite_case() -> None:
    variable = np.arange(1, 21, dtype=np.float64)

    result = nns_arma(variable, h=5, seasonal_factor=4, method="lin")

    assert np.all(np.isfinite(result))


def test_arma_degenerate_objective_returns_nan_without_warning() -> None:
    values = np.ones(3, dtype=np.float64)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        result = _default_arma_optim_objective(values, values)

    assert np.isnan(result)
    assert [warning for warning in caught if issubclass(warning.category, RuntimeWarning)] == []


def test_arma_constant_numeric_seasonal_weights_are_equal_without_warning() -> None:
    # Patched R: a constant series (CV 0/0 for every lag) gets equal weights.
    values = np.full(20, 5.0, dtype=np.float64)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        result = _numeric_seasonal_weights(values, np.array([4, 6], dtype=np.int64))

    np.testing.assert_array_equal(result, [0.5, 0.5])
    assert [warning for warning in caught if issubclass(warning.category, RuntimeWarning)] == []


@pytest.mark.parametrize("native", [True, False])
def test_linear_forecasts_match_component_path_bitwise(native: bool) -> None:
    rng = np.random.default_rng(41)
    current = np.cumsum(rng.normal(size=137)) + 50.0
    lags = np.array([1, 2, 5, 12, 24, 136, 137], dtype=np.int64)
    pool = np.arange(1, 400, dtype=np.float64)
    generated = arma_module._generate_vectors(current, lags)
    expected = [
        arma_module._linear_forecast_for_lag(
            generated["Component.index"][i], generated["Component.series"][i]
        )
        for i in range(lags.size)
    ]
    if native:
        actual = arma_module._linear_forecasts(current, lags, pool)
    else:
        with mock.patch.object(arma_module, "native_fn", return_value=None):
            actual = arma_module._linear_forecasts(current, lags, pool)
    np.testing.assert_array_equal(actual, np.asarray(expected))
    with pytest.raises(ValueError, match="non-empty"):
        arma_module._linear_forecasts(current, np.array([200], dtype=np.int64), pool)
