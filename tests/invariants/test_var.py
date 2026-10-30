from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from nns.var import _var_interpolate_and_extrapolate, _var_multivariate_stack_stage, nns_var


def test_var_interpolate_and_extrapolate_shape_and_names() -> None:
    variables = np.column_stack(
        (
            np.arange(1.0, 21.0, dtype=float),
            np.arange(2.0, 41.0, 2.0, dtype=float),
        )
    )
    result = cast(
        dict[str, Any],
        _var_interpolate_and_extrapolate(variables, h=3, tau=2, names=["x1", "x2"]),
    )
    interpolated = cast(np.ndarray, result["interpolated_and_extrapolated"])
    univariate = cast(np.ndarray, result["univariate"])

    assert interpolated.shape == variables.shape
    assert result["names"] == ["x1", "x2"]
    assert univariate.shape == (3, 2)


def test_var_interpolate_and_extrapolate_h0_returns_only_interpolated() -> None:
    variables = np.array(
        [
            [1.0, np.nan],
            [np.nan, 5.0],
            [3.0, 6.0],
            [4.0, 7.0],
        ],
        dtype=float,
    )
    result = cast(
        dict[str, np.ndarray],
        _var_interpolate_and_extrapolate(variables, h=0, tau=1, names=["a", "b"]),
    )
    interpolated = result["interpolated_and_extrapolated"]

    assert "univariate" not in result
    assert interpolated.shape == variables.shape
    assert not np.isnan(interpolated).any()
    np.testing.assert_array_equal(result["names"], ["a", "b"])


def test_var_interpolate_and_extrapolate_is_deterministic_for_repeat_call() -> None:
    variables = np.array(
        [[1.0, 2.0], [2.0, np.nan], [4.0, 6.0], [5.0, 8.0], [6.0, 10.0]],
        dtype=float,
    )
    first = cast(
        dict[str, np.ndarray],
        _var_interpolate_and_extrapolate(variables, h=2, tau=1, names=["x1", "x2"]),
    )
    second = cast(
        dict[str, np.ndarray],
        _var_interpolate_and_extrapolate(variables, h=2, tau=1, names=["x1", "x2"]),
    )
    first_interpolated = first["interpolated_and_extrapolated"]
    second_interpolated = second["interpolated_and_extrapolated"]
    first_univariate = first["univariate"]
    second_univariate = second["univariate"]

    np.testing.assert_allclose(first_interpolated, second_interpolated)
    np.testing.assert_allclose(first_univariate, second_univariate, equal_nan=False)


def test_var_multivariate_stack_stage_shape_and_determinism() -> None:
    variables = np.column_stack(
        (
            np.arange(1.0, 21.0, dtype=float),
            np.arange(2.0, 41.0, 2.0, dtype=float),
        )
    )
    first = cast(
        dict[str, np.ndarray],
        _var_interpolate_and_extrapolate(variables, h=3, tau=2, names=["x1", "x2"]),
    )
    first_stage = _var_multivariate_stack_stage(
        first["interpolated_and_extrapolated"],
        first["univariate"],
        h=3,
        tau=2,
        names=["x1", "x2"],
        dim_red_method="cor",
    )
    second = cast(
        dict[str, np.ndarray],
        _var_interpolate_and_extrapolate(variables, h=3, tau=2, names=["x1", "x2"]),
    )
    second_stage = _var_multivariate_stack_stage(
        second["interpolated_and_extrapolated"],
        second["univariate"],
        h=3,
        tau=2,
        names=["x1", "x2"],
        dim_red_method="cor",
    )

    first_multivariate = cast(np.ndarray, first_stage["multivariate"])
    second_multivariate = cast(np.ndarray, second_stage["multivariate"])
    first_relevant = cast(np.ndarray, first_stage["relevant_variables"])
    second_relevant = cast(np.ndarray, second_stage["relevant_variables"])

    assert first_multivariate.shape == (3, 2)
    assert first_multivariate.shape == second_multivariate.shape
    assert first_relevant.shape == second_relevant.shape
    np.testing.assert_array_equal(first_multivariate, second_multivariate)
    assert np.array_equal(first_relevant, second_relevant)


def _driven_variables(n: int = 60, seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal(n))
    y = np.concatenate(([0.0], 0.8 * x[:-1])) + 0.3 * rng.standard_normal(n)
    z = rng.standard_normal(n)
    return np.column_stack((x, y, z))


def _two_driver_variables(n: int = 80, seed: int = 2) -> np.ndarray:
    """y is driven by lagged x and lagged w, so its fit is genuinely multivariate."""
    rng = np.random.default_rng(seed)

    def ar(phi: float) -> np.ndarray:
        out = np.zeros(n)
        shocks = rng.standard_normal(n)
        for t in range(n):
            out[t] = shocks[t] + (phi * out[t - 1] if t else 0.0)
        return out

    x, w = ar(0.6), ar(0.6)
    y = np.concatenate(([0.0], x[:-1] + w[:-1])) + 0.2 * rng.standard_normal(n)
    return np.column_stack((x, y, w))


@pytest.mark.parametrize(
    ("variables", "tau"), [(_driven_variables(), 2), (_two_driver_variables(), 1)]
)
def test_nns_var_multivariate_is_nns_stack_on_exactly_the_relevant_variables(
    variables: np.ndarray, tau: int
) -> None:
    from nns.stack import nns_stack
    from nns.var import (
        _METHOD1_GUARD_MESSAGE,
        _var_default_objective,
        _var_lag_matrix,
        _var_lags,
        _var_mse_objective,
        _var_relevant,
    )

    h = 3
    result = nns_var(variables, h, tau=tau)
    names = result["names"]
    relevant = result["relevant_variables"]

    new_values = np.vstack((result["interpolated_and_extrapolated"], result["univariate"]))
    lagged, lagged_names, _ = _var_lag_matrix(new_values, _var_lags(tau, 3)["lags"], names)
    train = lagged[:-h]
    for i, name in enumerate(names):
        iv = np.delete(train, i, axis=1)
        iv_names = lagged_names[:i] + lagged_names[i + 1 :]
        ts_test = max(2 * h, int(np.ceil(0.2 * iv.shape[0])))
        columns = _var_relevant(iv, train[:, i], ts_test=ts_test, obj_fn=_var_default_objective,
                                objective="min", dim_red_method="cor")
        selected = [value for value in relevant[:, i].tolist() if value is not None]
        assert selected == [iv_names[k] for k in columns]
        assert f"{name}_tau_0" not in selected

        # Only the relevant regressors enter the stack, so the estimate cannot
        # depend on anything outside relevant_variables.
        args: dict[str, Any] = {"ivs_test": iv[-h:, columns], "ts_test": ts_test, "folds": 1,
                                "method": 1, "order": None, "stack": False, "dist": None}
        try:
            fit = nns_stack(iv[:, columns], train[:, i], obj_fn=_var_default_objective, **args)
        except ValueError as error:  # nns_var's MSE fallback, as in R
            assert str(error) == _METHOD1_GUARD_MESSAGE
            fit = nns_stack(iv[:, columns], train[:, i], obj_fn=_var_mse_objective, **args)
        np.testing.assert_allclose(result["multivariate"][:, i], fit["stack"])


def test_var_relevant_keeps_the_regressors_meeting_the_threshold() -> None:
    from nns.var import _var_default_objective, _var_lag_matrix, _var_lags, _var_relevant

    lagged, lagged_names, _ = _var_lag_matrix(
        _two_driver_variables(), _var_lags(1, 3)["lags"], ["x", "y", "w"]
    )
    iv, dv = np.delete(lagged, 1, axis=1), lagged[:, 1]
    iv_names = lagged_names[:1] + lagged_names[2:]
    relevant = _var_relevant(iv, dv, ts_test=16, obj_fn=_var_default_objective,
                             objective="min", dim_red_method="cor")
    assert relevant == sorted(set(relevant))
    assert {iv_names[k] for k in relevant} >= {"x_tau_1", "w_tau_1"}
    # A single regressor is relevant by definition.
    assert _var_relevant(iv[:, :1], dv, ts_test=16, obj_fn=_var_default_objective,
                         objective="min", dim_red_method="cor") == [0]


def test_nns_var_selects_every_driver() -> None:
    driven = nns_var(_driven_variables(), 3, tau=2)["relevant_variables"]
    assert "x1_tau_1" in [value for value in driven[:, 1].tolist() if value is not None]
    two = nns_var(_two_driver_variables(), 3, tau=1)["relevant_variables"]
    assert {value for value in two[:, 1].tolist() if value is not None} >= {"x1_tau_1", "x3_tau_1"}


def test_nns_var_own_lag_weights_follow_relevant_variables() -> None:
    result = nns_var(_two_driver_variables(), 2, tau=2, naive_weights=False)
    relevant = result["relevant_variables"]
    for i, name in enumerate(result["names"]):
        selected = [value for value in relevant[:, i].tolist() if value is not None]
        own = sum(value.startswith(f"{name}_tau_") for value in selected) / len(selected)
        np.testing.assert_allclose(
            result["ensemble"][:, i],
            own * result["univariate"][:, i] + (1.0 - own) * result["multivariate"][:, i],
        )


@pytest.mark.parametrize(
    "tau", [[1, 2], [1, 1, 1, 1], 0, 1.5, "2", [[1, 2], [1]], [[0, 1], [1], [1]]]
)
def test_nns_var_rejects_invalid_tau_before_fitting(tau: Any) -> None:
    with pytest.raises(ValueError, match="tau"):
        nns_var(_driven_variables(40), 2, tau=tau)


@pytest.mark.parametrize("h", [-1, 1.5, True, "2"])
def test_nns_var_rejects_invalid_h(h: Any) -> None:
    with pytest.raises((TypeError, ValueError), match="h must"):
        nns_var(_driven_variables(40), h)


def test_nns_var_rejects_too_short_samples() -> None:
    with pytest.raises(ValueError, match="Not enough observations"):
        nns_var(_driven_variables(40)[:10], 4, tau=2)


def test_nns_var_accepts_numpy_integer_tau() -> None:
    variables = _driven_variables(40)
    result = nns_var(variables, 1, tau=np.int64(2))
    assert result["ensemble"].shape == (1, 3)
    vector = nns_var(variables, 2, tau=np.array([1, 2, 1]))
    assert vector["ensemble"].shape == (2, 3)


def test_nns_var_forwards_obj_fn_to_both_stages() -> None:
    calls = {"count": 0}

    def mae(predicted: np.ndarray, actual: np.ndarray) -> float:
        calls["count"] += 1
        return float(np.mean(np.abs(predicted - actual)))

    result = nns_var(_driven_variables(40), 2, tau=1, obj_fn=mae)
    assert calls["count"] > 0
    assert np.all(np.isfinite(result["ensemble"]))


def test_nns_var_keeps_dataframe_names_and_unique_names() -> None:
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(_driven_variables(40), columns=["gdp", "gdp", "rate - us"])
    result = nns_var(frame, 2, tau=1)
    assert result["names"] == ["gdp", "gdp.1", "rate...us"]
