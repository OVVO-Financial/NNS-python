from __future__ import annotations

import warnings

import numpy as np
import pytest

from pynns import nns_arma, nns_arma_optim, nns_var
from pynns.arma import _default_arma_optim_objective, _numeric_seasonal_weights


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


def test_nns_arma_static_linear_pred_int_matches_installed_r_error() -> None:
    variable = np.arange(1, 21, dtype=np.float64)

    with pytest.raises(TypeError, match="non-numeric argument"):
        nns_arma(variable, h=5, seasonal_factor=4, method="lin", pred_int=0.95)


def test_nns_arma_pred_int_h_one_matches_installed_r_error() -> None:
    variable = np.sin(np.arange(1, 40, dtype=np.float64) / 3.0) + 2.0

    with pytest.raises(ValueError, match="incorrect number of dimensions"):
        nns_arma(variable, h=1, seasonal_factor=4, method="nonlin", pred_int=0.95)


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

    with pytest.raises(TypeError, match="non-numeric argument"):
        nns_arma_optim(
            variable,
            h=3,
            seasonal_factor=[3, 4, 5, 6, 7],
            lin_only=True,
            pred_int=None,
            print_trace=False,
        )


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


def test_arma_constant_numeric_seasonal_weights_return_nan_without_warning() -> None:
    values = np.full(20, 5.0, dtype=np.float64)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        result = _numeric_seasonal_weights(values, np.array([4], dtype=np.int64))

    assert np.isnan(result).all()
    assert [warning for warning in caught if issubclass(warning.category, RuntimeWarning)] == []
