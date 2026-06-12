from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pynns._helpers import _fast_lm
from pynns.co_moments import co_lpm, co_upm
from pynns.dependence import _gravity
from pynns.mc import nns_mc
from pynns.regression import nns_reg
from pynns.seasonality import nns_seas
from pynns.var import lpm_var, upm_var


def nns_arma_optim(
    variable: NDArray[np.float64],
    h: int | None = None,
    training_set: int | None = None,
    seasonal_factor: NDArray[np.int64] | list[int] | None = None,
    *,
    lin_only: bool = False,
    negative_values: bool = False,
    obj_fn: Any = None,
    objective: str = "min",
    linear_approximation: bool = True,
    ncores: int | None = None,
    pred_int: float | None = 0.95,
    print_trace: bool = True,
    plot: bool = False,
) -> dict[str, Any]:
    """Optimize seasonal factors for :func:`nns_arma` like R's ``NNS.ARMA.optim``."""
    del ncores, print_trace, plot

    values = _as_variable(variable)
    original_values = values.copy()
    n = values.size
    objective_l = objective.lower()
    if objective_l not in {"min", "max"}:
        raise ValueError("objective must be 'min' or 'max'.")
    if obj_fn is None:
        objective_fn = _default_arma_optim_objective
    elif callable(obj_fn):
        objective_fn = obj_fn
    else:
        raise TypeError("obj_fn must be callable or None.")

    if training_set is None and h is None:
        raise ValueError(
            "Please use the length of the variable less the desired forecast period as the "
            "[training.set] value, or provide a value for [h]."
        )
    if float(np.min(values)) < 0.0:
        negative_values = True

    h_oos = int(h) if h is not None and int(h) > 0 else None
    train_n = math.floor(0.8 * n) if training_set is None else int(training_set)
    h_eval = int(n - train_n)
    actual = values[-h_eval:]
    if train_n <= 0.5 * n:
        raise ValueError("Please provide a larger [training.set] value (integer) or a smaller [h].")
    if train_n == n:
        raise ValueError(
            "Please provide a [training.set] value (integer) less than the length of the variable."
        )
    if h_eval < 1:
        raise ValueError("training_set must leave at least one validation observation.")

    seasonals = _valid_arma_optim_seasonals(seasonal_factor, train_n)
    methods = ["lin"] if lin_only else ["lin", "nonlin", "both"]

    previous_seasonals: list[list[NDArray[np.int64]]] = []
    previous_estimates: list[NDArray[np.float64]] = []
    overall_seasonals: list[NDArray[np.int64]] = []
    overall_estimates: list[float] = []
    nonlin_predicted = np.array([], dtype=np.float64)

    for method in methods:
        current_seasonals: list[NDArray[np.int64]] = []
        current_estimates: list[float] = []

        for step in range(1, seasonals.size + 1):
            if step == 1:
                if linear_approximation and method != "lin":
                    if not overall_seasonals:
                        break
                    combs = overall_seasonals[0].reshape(-1, 1)
                    current_seasonals = [overall_seasonals[0].astype(np.int64, copy=True)]
                else:
                    combs = seasonals.reshape(1, -1)
            else:
                if linear_approximation and method != "lin":
                    continue
                previous = current_seasonals[step - 2]
                remaining = seasonals[~np.isin(seasonals, previous)]
                if remaining.size == 0:
                    break
                combs = np.vstack((np.tile(previous.reshape(-1, 1), remaining.size), remaining))

            if combs.ndim != 2 or combs.shape[1] == 0:
                break

            if method == "lin":
                estimates = np.asarray(
                    [
                        _evaluate_arma_optim(
                            values,
                            actual,
                            train_n,
                            h_eval,
                            combs[:, col],
                            "lin",
                            negative_values,
                            None,
                            objective_fn,
                        )
                        for col in range(combs.shape[1])
                    ],
                    dtype=np.float64,
                )
            elif method == "nonlin" and linear_approximation:
                predicted = _arma_optim_prediction(
                    values,
                    train_n,
                    h_eval,
                    overall_seasonals[0],
                    method,
                    negative_values,
                    None,
                )
                nonlin_predicted = predicted
                estimates = np.asarray([float(objective_fn(predicted, actual))], dtype=np.float64)
            elif method == "both" and linear_approximation:
                lin_predicted = _arma_optim_prediction(
                    values,
                    train_n,
                    h_eval,
                    overall_seasonals[0],
                    "lin",
                    negative_values,
                    None,
                )
                predicted = (lin_predicted + nonlin_predicted) / 2.0
                estimates = np.asarray([float(objective_fn(predicted, actual))], dtype=np.float64)
            else:
                estimates = np.asarray(
                    [
                        _evaluate_arma_optim(
                            values,
                            actual,
                            train_n,
                            h_eval,
                            combs[:, col],
                            method,
                            negative_values,
                            None,
                            objective_fn,
                        )
                        for col in range(combs.shape[1])
                    ],
                    dtype=np.float64,
                )

            estimates = _replace_nan_objectives(estimates, objective_l)
            if objective_l == "min":
                best_index = int(np.argmin(estimates))
                best_score = float(np.min(estimates))
                worsened = bool(current_estimates and best_score > current_estimates[-1])
            else:
                best_index = int(np.argmax(estimates))
                best_score = float(np.max(estimates))
                worsened = bool(current_estimates and best_score < current_estimates[-1])
            if worsened:
                break

            if not (linear_approximation and method != "lin" and current_seasonals):
                current_seasonals.append(combs[:, best_index].astype(np.int64, copy=True))
            current_estimates.append(best_score)

            method_index = ["lin", "nonlin", "both"].index(method)
            if method_index > 0 and step - 1 < len(previous_seasonals[method_index - 1]):
                previous = previous_seasonals[method_index - 1][step - 1]
                current = current_seasonals[step - 1]
                same_periods = (
                    np.isin(current.astype(float), previous.astype(float)).sum() == current.size
                )
                if same_periods:
                    previous_score = previous_estimates[method_index - 1][step - 1]
                    if (objective_l == "min" and best_score >= previous_score) or (
                        objective_l == "max" and best_score <= previous_score
                    ):
                        break

            if method != "lin" and linear_approximation:
                break

        previous_seasonals.append(current_seasonals)
        previous_estimates.append(np.asarray(current_estimates, dtype=np.float64))
        if current_estimates:
            overall_seasonals.append(current_seasonals[-1])
            overall_estimates.append(current_estimates[-1])

    if not overall_estimates:
        raise ValueError("No ARMA optimizer candidates were evaluated.")

    overall = np.asarray(overall_estimates, dtype=np.float64)
    selected_index = int(np.argmin(overall) if objective_l == "min" else np.argmax(overall))
    nns_periods = overall_seasonals[selected_index].astype(np.int64, copy=True)
    nns_method = "lin" if lin_only else methods[selected_index]
    nns_score = float(np.min(overall) if objective_l == "min" else np.max(overall))

    predicted = _arma_optim_prediction(
        values,
        train_n,
        h_eval,
        nns_periods,
        nns_method,
        negative_values,
        None,
    )
    nns_weights: NDArray[np.float64] | None = None
    errors = predicted - actual
    bias = _finite_gravity(errors)
    predicted_shifted = predicted - bias
    bias_score = float(objective_fn(predicted_shifted, actual))

    if nns_periods.size > 1:
        weight_score = float(objective_fn(predicted, actual))
        if _improves(weight_score, nns_score, objective_l):
            nns_weights = np.full(nns_periods.size, 1.0 / float(nns_periods.size), dtype=np.float64)
            predicted = _arma_optim_prediction(
                values,
                train_n,
                h_eval,
                nns_periods,
                nns_method,
                negative_values,
                nns_weights,
            )
            errors = predicted - actual
            bias = _finite_gravity(errors)
            predicted_shifted = predicted - bias
            bias_score = float(objective_fn(predicted_shifted, actual))
            if not np.isfinite(bias_score) or not _improves(bias_score, weight_score, objective_l):
                bias = 0.0
        elif not np.isfinite(bias_score) or not _improves_or_ties_bias(
            bias_score, nns_score, objective_l
        ):
            bias = 0.0
    elif not np.isfinite(bias_score) or not _improves_or_ties_bias(
        bias_score, nns_score, objective_l
    ):
        bias = 0.0

    final_predicted = predicted

    shrink_predicted = _arma_optim_prediction(
        values,
        train_n,
        h_eval,
        nns_periods,
        nns_method,
        negative_values,
        nns_weights,
        shrink=True,
    )
    shrink_score = float(objective_fn(shrink_predicted, actual))
    nns_shrink = _improves(shrink_score, nns_score, objective_l)
    if nns_shrink:
        final_predicted = shrink_predicted

    regressed_values = _smooth_regressed_variable(values)
    smooth_predicted = _arma_optim_prediction(
        regressed_values,
        train_n,
        h_eval,
        nns_periods,
        nns_method,
        negative_values,
        nns_weights,
        shrink=True,
    )
    smooth_score = float(objective_fn(smooth_predicted, actual))
    nns_regress = _improves(smooth_score, nns_score, objective_l)
    if nns_regress:
        values = regressed_values
        final_predicted = smooth_predicted

    del final_predicted, values

    if pred_int is None:
        raise TypeError("non-numeric argument to mathematical function")
    pi_width = abs(float(upm_var((1.0 - float(pred_int)) / 2.0, 0.0, errors))) + abs(bias)

    if h_oos is None:
        result_h = h_eval
        model_results = _arma_optim_prediction(
            original_values,
            train_n,
            result_h,
            nns_periods,
            nns_method,
            negative_values,
            nns_weights,
            shrink=nns_shrink,
        )
    else:
        result_h = h_oos
        model_results = _arma_optim_prediction(
            original_values,
            None,
            result_h,
            nns_periods,
            nns_method,
            negative_values,
            nns_weights,
            shrink=nns_shrink,
        )
    model_results = model_results - bias
    lower_pi = model_results - pi_width
    upper_pi = model_results + pi_width
    if not negative_values:
        model_results = np.maximum(0.0, model_results)
        lower_pi = np.maximum(0.0, lower_pi)
        upper_pi = np.maximum(0.0, upper_pi)

    return {
        "periods": nns_periods,
        "weights": nns_weights,
        "obj.fn": nns_score,
        "method": nns_method,
        "shrink": bool(nns_shrink),
        "nns.regress": bool(nns_regress),
        "bias.shift": -float(bias),
        "errors": errors,
        "results": model_results,
        "lower.pred.int": lower_pi,
        "upper.pred.int": upper_pi,
    }


def nns_arma(
    variable: NDArray[np.float64],
    h: int = 1,
    training_set: int | None = None,
    seasonal_factor: bool | int | list[int] | NDArray[np.int64] = True,
    weights: NDArray[np.float64] | str | None = None,
    best_periods: int | None = 1,
    modulo: int | list[int] | NDArray[np.int64] | None = None,
    mod_only: bool = True,
    negative_values: bool = False,
    method: str = "nonlin",
    dynamic: bool = False,
    shrink: bool = False,
    plot: bool = False,
    seasonal_plot: bool = True,
    pred_int: float | None = None,
    random_seed: int | None = None,
) -> NDArray[np.float64] | dict[str, NDArray[np.float64]]:
    """Autoregressive NNS forecast matching R's installed NNS.ARMA behavior."""
    del plot, seasonal_plot

    horizon = int(h)
    if horizon < 1:
        raise ValueError("h must be a positive integer.")
    values = _as_variable(variable)
    if _is_numeric_seasonal(seasonal_factor) and dynamic:
        raise ValueError(
            'Hmmm...Seems you have "seasonal.factor" specified and "dynamic = TRUE".  '
            'Nothing dynamic about static seasonal factors!  Please set "dynamic = FALSE" '
            'or "seasonal.factor = FALSE"'
        )

    method_l = method.lower()
    if method_l not in {"lin", "nonlin", "both", "means"}:
        raise ValueError("method must be one of 'lin', 'nonlin', 'both', or 'means'.")
    if method_l == "means":
        shrink = False
    if float(np.min(values)) < 0.0:
        negative_values = True

    if training_set is not None:
        train_n = int(training_set)
        values = values[:train_n].astype(np.float64, copy=True)
    else:
        values = values.astype(np.float64, copy=True)

    estimates = np.zeros(horizon, dtype=np.float64)
    if not _is_numeric_seasonal(seasonal_factor) and np.ptp(values) == 0.0:
        return _with_prediction_intervals(
            estimates,
            lin_residual=0.0,
            pred_int=pred_int,
            random_seed=random_seed,
        )
    lags, lag_weights = _resolve_lags_and_weights(
        values,
        seasonal_factor=seasonal_factor,
        weights=weights,
        best_periods=best_periods,
        modulo=modulo,
        mod_only=mod_only,
    )

    if method_l == "lin" and _is_numeric_seasonal(seasonal_factor) and lags.size == 1:
        if pred_int is not None:
            raise TypeError("non-numeric argument to binary operator")
        estimates = _linear_static_numeric_forecast(
            values,
            int(lags[0]),
            horizon,
            float(lag_weights[0]),
            negative_values=negative_values,
            method=method_l,
            shrink=shrink,
        )
        return estimates

    current = values
    lin_regression_estimates = np.array([], dtype=np.float64)
    for index in range(horizon):
        if dynamic:
            lags, lag_weights = _resolve_lags_and_weights(
                current,
                seasonal_factor=seasonal_factor,
                weights=None,
                best_periods=best_periods,
                modulo=modulo,
                mod_only=mod_only,
            )

        generated = _generate_vectors(current, lags)
        component_index = generated["Component.index"]
        component_series = generated["Component.series"]

        nonlin_estimate = math.nan
        if method_l in {"nonlin", "both"}:
            regression_estimates = np.asarray(
                [
                    _nonlinear_forecast_for_lag(component_index[i], component_series[i])
                    for i in range(lags.size)
                ],
                dtype=np.float64,
            )
            regression_estimates = np.maximum(0.0, regression_estimates)
            nonlin_estimate = float(np.sum(regression_estimates * lag_weights))

        lin_estimate = math.nan
        if method_l in {"lin", "both", "means"}:
            linear_estimates = np.asarray(
                [
                    _linear_forecast_for_lag(component_index[i], component_series[i])
                    for i in range(lags.size)
                ],
                dtype=np.float64,
            )
            if method_l == "means" or shrink:
                means = np.asarray(
                    [_means_forecast_for_lag(series) for series in component_series],
                    dtype=np.float64,
                )
                if shrink:
                    linear_estimates = (linear_estimates + means) / 2.0
                else:
                    linear_estimates = means

            lin_estimate = float(np.sum(linear_estimates * lag_weights))
            if not negative_values:
                lin_estimate = float(np.maximum(0.0, lin_estimate))
            lin_regression_estimates = linear_estimates

        if method_l == "lin":
            estimate = float(np.sum(lin_estimate * lag_weights))
        elif method_l == "both":
            estimate = float(np.mean(np.array([lin_estimate, nonlin_estimate], dtype=np.float64)))
        elif method_l == "nonlin":
            estimate = float(np.sum(nonlin_estimate * lag_weights))
        else:
            estimate = 0.0

        estimates[index] = estimate
        current = np.concatenate((current, np.array([estimate], dtype=np.float64)))

    lin_resid = 0.0
    if pred_int is not None and method_l != "means" and lin_regression_estimates.size:
        lin_mean = float(np.mean(lin_regression_estimates))
        lin_resid = float(np.mean(np.abs(lin_regression_estimates - lin_mean)))
        if not np.isfinite(lin_resid):
            lin_resid = 0.0

    return _with_prediction_intervals(
        estimates,
        lin_residual=lin_resid,
        pred_int=pred_int,
        random_seed=random_seed,
    )


def _valid_arma_optim_seasonals(
    seasonal_factor: NDArray[np.int64] | list[int] | None,
    training_set: int,
) -> NDArray[np.int64]:
    if seasonal_factor is None:
        raise ValueError("seasonal_factor must be provided.")
    seasonals = np.asarray(seasonal_factor, dtype=np.int64).reshape(-1)
    denominator = min(4, max(3, _r_round_half_up(float(training_set) / 100.0)))
    limit = float(training_set) / float(denominator)
    seasonals = np.unique(seasonals[seasonals <= limit])
    if seasonals.size == 0:
        raise ValueError(
            "Please ensure [seasonal.factor] contains elements less than "
            f"{limit}, otherwise use cross-validation of seasonal factors as demonstrated "
            "in the vignette >>> Getting Started with NNS: Forecasting"
        )
    return seasonals


def _r_round_half_up(value: float) -> int:
    return int(math.floor(value) if value % 1.0 < 0.5 else math.ceil(value))


def _arma_optim_prediction(
    variable: NDArray[np.float64],
    training_set: int | None,
    h: int,
    seasonal_factor: NDArray[np.int64],
    method: str,
    negative_values: bool,
    weights: NDArray[np.float64] | None,
    *,
    shrink: bool = False,
) -> NDArray[np.float64]:
    result = nns_arma(
        variable,
        h=h,
        training_set=training_set,
        seasonal_factor=seasonal_factor,
        method=method,
        weights=weights,
        negative_values=negative_values,
        shrink=shrink,
    )
    if isinstance(result, dict):
        return np.asarray(result["Estimates"], dtype=np.float64)
    return np.asarray(result, dtype=np.float64)


def _evaluate_arma_optim(
    variable: NDArray[np.float64],
    actual: NDArray[np.float64],
    training_set: int,
    h: int,
    seasonal_factor: NDArray[np.int64],
    method: str,
    negative_values: bool,
    weights: NDArray[np.float64] | None,
    objective_fn: Any,
) -> float:
    predicted = _arma_optim_prediction(
        variable,
        training_set,
        h,
        seasonal_factor,
        method,
        negative_values,
        weights,
    )
    return float(objective_fn(predicted, actual))


def _default_arma_optim_objective(
    predicted: NDArray[np.float64],
    actual: NDArray[np.float64],
) -> float:
    predicted_values = np.asarray(predicted, dtype=np.float64)
    actual_values = np.asarray(actual, dtype=np.float64)
    denominator = float(
        co_lpm(
            1.0,
            predicted_values,
            actual_values,
            float(np.mean(predicted_values)),
            float(np.mean(actual_values)),
        )
        + co_upm(
            1.0,
            predicted_values,
            actual_values,
            float(np.mean(predicted_values)),
            float(np.mean(actual_values)),
        )
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.mean((predicted_values - actual_values) ** 2) / denominator)


def _replace_nan_objectives(values: NDArray[np.float64], objective: str) -> NDArray[np.float64]:
    output = values.astype(np.float64, copy=True)
    output[np.isnan(output)] = math.inf if objective == "min" else -math.inf
    return output


def _improves(candidate: float, baseline: float, objective: str) -> bool:
    if not np.isfinite(candidate):
        return False
    return candidate < baseline if objective == "min" else candidate > baseline


def _improves_or_ties_bias(candidate: float, baseline: float, objective: str) -> bool:
    return candidate < baseline if objective == "min" else candidate > baseline


def _finite_gravity(values: NDArray[np.float64]) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    bias = float(_gravity(finite))
    return 0.0 if not np.isfinite(bias) else bias


def _smooth_regressed_variable(values: NDArray[np.float64]) -> NDArray[np.float64]:
    result = nns_reg(
        np.arange(1, values.size + 1, dtype=np.float64),
        values,
        smooth=True,
        plot=False,
    )
    fitted = result["Fitted.xy"]
    if not isinstance(fitted, dict):
        raise TypeError("nns_reg returned an unexpected Fitted.xy structure.")
    return np.asarray(fitted["y.hat"], dtype=np.float64)


def _as_variable(variable: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(variable, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("variable must be non-empty.")
    if np.any(np.isnan(values)):
        raise ValueError("You have some missing values, please address.")
    if np.any(np.isinf(values)):
        raise ValueError("Infinite values not allowed")
    return values


def _with_prediction_intervals(
    estimates: NDArray[np.float64],
    *,
    lin_residual: float,
    pred_int: float | None,
    random_seed: int | None,
) -> NDArray[np.float64] | dict[str, NDArray[np.float64]]:
    if pred_int is None:
        return estimates
    if estimates.size < 2:
        raise ValueError("incorrect number of dimensions")

    mc_result = nns_mc(
        estimates,
        lower_rho=-1.0,
        upper_rho=1.0,
        by=0.2,
        random_seed=random_seed,
    )
    replicates = mc_result["replicates"]
    if not isinstance(replicates, dict):
        raise TypeError("nns_mc returned an unexpected replicate structure.")
    matrices = [np.asarray(matrix, dtype=np.float64) for matrix in replicates.values()]
    if not matrices:
        raise ValueError("NNS.MC returned no prediction-interval replicates.")
    intervals = np.column_stack(matrices)

    alpha = (1.0 - float(pred_int)) / 2.0
    upper_pi = np.empty(estimates.size, dtype=np.float64)
    lower_pi = np.empty(estimates.size, dtype=np.float64)
    for row_index, row in enumerate(intervals):
        upper_pi[row_index] = upm_var(alpha, 0.0, row) + lin_residual
        lower_pi[row_index] = abs(lpm_var(alpha, 0.0, row)) - lin_residual

    pct = round(float(pred_int) * 100.0, 2)
    return {
        "Estimates": estimates,
        f"Lower {_format_r_percent(pct)}% pred.int": np.minimum(estimates, lower_pi),
        f"Upper {_format_r_percent(pct)}% pred.int": np.maximum(estimates, upper_pi),
    }


def _format_r_percent(value: float) -> str:
    if value == 0.0:
        return "0"
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return text if text != "-0" else "0"


def _is_numeric_seasonal(value: object) -> bool:
    return not isinstance(value, (bool, np.bool_))


def _resolve_lags_and_weights(
    variable: NDArray[np.float64],
    *,
    seasonal_factor: bool | int | list[int] | NDArray[np.int64],
    weights: NDArray[np.float64] | str | None,
    best_periods: int | None,
    modulo: int | list[int] | NDArray[np.int64] | None,
    mod_only: bool,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    if _is_numeric_seasonal(seasonal_factor):
        lags = np.asarray(seasonal_factor, dtype=np.int64).reshape(-1)
        if lags.size == 0:
            lags = np.array([1], dtype=np.int64)
        if weights is None:
            lag_weights = _numeric_seasonal_weights(variable, lags)
        elif isinstance(weights, str):
            raise TypeError("non-numeric weights are not supported with numeric seasonal_factor.")
        else:
            lag_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        return lags, lag_weights

    seasonality = nns_seas(variable, modulo=modulo, mod_only=mod_only, plot=False)
    table = seasonality["all.periods"]
    if not isinstance(table, dict):
        lags = np.array([1], dtype=np.int64)
        lag_weights = np.array([1.0], dtype=np.float64)
    else:
        periods = np.asarray(table["Period"], dtype=np.int64).reshape(-1)
        coef = np.asarray(table["Coefficient.of.Variation"], dtype=np.float64).reshape(-1)
        varcoef = np.asarray(table["Variable.Coefficient.of.Variation"], dtype=np.float64).reshape(
            -1
        )
        if bool(seasonal_factor):
            lags, lag_weights = _arma_seas_weighting(True, periods, coef, varcoef)
        else:
            if best_periods is not None:
                count = min(int(best_periods), periods.size)
                periods = periods[:count]
                coef = coef[:count]
                varcoef = varcoef[:count]
            lags, lag_weights = _arma_seas_weighting(False, periods, coef, varcoef)

    if weights is not None:
        if isinstance(weights, str):
            lag_weights = np.full(lags.size, 1.0 / float(lags.size), dtype=np.float64)
        else:
            lag_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    return lags, lag_weights


def _numeric_seasonal_weights(
    variable: NDArray[np.float64],
    lags: NDArray[np.int64],
) -> NDArray[np.float64]:
    output = np.empty(lags.size, dtype=np.float64)
    for index, lag in enumerate(lags):
        rev_var = variable[:: -int(lag)]
        with np.errstate(invalid="ignore", divide="ignore"):
            output[index] = abs(
                np.float64(np.std(rev_var, ddof=1)) / np.float64(np.mean(rev_var))
            )
    with np.errstate(invalid="ignore", divide="ignore"):
        baseline_cv = abs(
            np.float64(np.std(variable, ddof=1)) / np.float64(np.mean(variable))
        )
        relative = output / baseline_cv
        seasonal_weighting = 1.0 / relative
    observation_weighting = 1.0 / np.sqrt(lags.astype(np.float64))
    denom = float(np.sum(observation_weighting * seasonal_weighting))
    return (seasonal_weighting * observation_weighting) / denom


def _arma_seas_weighting(
    seasonal_factor: bool,
    periods: NDArray[np.int64],
    coefficient: NDArray[np.float64],
    variable_coefficient: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    if periods.size == 0:
        return np.array([1], dtype=np.int64), np.array([1.0], dtype=np.float64)
    if seasonal_factor:
        return np.array([int(periods[0])], dtype=np.int64), np.array([1.0], dtype=np.float64)

    lags = periods.astype(np.int64, copy=True)
    observation_weighting = 1.0 / np.sqrt(lags.astype(np.float64))
    m = min(coefficient.size, variable_coefficient.size, observation_weighting.size)
    lag_weighting = variable_coefficient[:m] - coefficient[:m]
    weights_product = lag_weighting * observation_weighting[:m]
    denom = float(np.sum(weights_product))
    if denom == 0.0:
        lag_weights = np.zeros(weights_product.size, dtype=np.float64)
    else:
        lag_weights = weights_product / denom
    return lags[:m], lag_weights


def _generate_vectors(
    variable: NDArray[np.float64],
    lags: NDArray[np.int64],
) -> dict[str, list[NDArray[np.float64]]]:
    n = variable.size
    series: list[NDArray[np.float64]] = []
    indices: list[NDArray[np.float64]] = []
    for lag_raw in lags:
        lag = int(lag_raw)
        if lag <= 0:
            series.append(np.array([], dtype=np.float64))
            indices.append(np.array([], dtype=np.float64))
            continue
        start = n % lag
        component = variable[start::lag]
        series.append(component.astype(np.float64, copy=True))
        indices.append(np.arange(1, component.size + 1, dtype=np.float64))
    return {"Component.index": indices, "Component.series": series}


def _generate_lin_vectors(
    variable: NDArray[np.float64],
    lag: int,
    h: int,
) -> dict[str, list[NDArray[np.float64]]]:
    n = variable.size
    max_fcast = min(h, lag)
    component_series: list[NDArray[np.float64]] = []
    component_index: list[NDArray[np.float64]] = []
    for i in range(1, max_fcast + 1):
        start = (n + i - 1) % lag
        component = variable[start::lag]
        component_series.append(component.astype(np.float64, copy=True))
        component_index.append(np.arange(1, component.size + 1, dtype=np.float64))

    forecast_index = _recycled_num_lists(np.arange(1, h + 1, dtype=np.float64), max_fcast)
    raw = np.empty(h, dtype=np.float64)
    for i in range(1, h + 1):
        recycled_index = ((i - 1) % lag) + 1
        ci = ((recycled_index - 1) % max(1, max_fcast)) + 1
        last_val = component_index[ci - 1].size
        forecast_increment = math.ceil(i / lag)
        raw[i - 1] = float(last_val + forecast_increment)
    forecast_values = _recycled_num_lists(raw, lag)
    return {
        "Component.index": component_index,
        "Component.series": component_series,
        "forecast.values": forecast_values,
        "forecast.index": forecast_index,
    }


def _recycled_num_lists(values: NDArray[np.float64], list_length: int) -> list[NDArray[np.float64]]:
    return [values[index::list_length].copy() for index in range(list_length)]


def _linear_static_numeric_forecast(
    variable: NDArray[np.float64],
    lag: int,
    h: int,
    weight: float,
    *,
    negative_values: bool,
    method: str,
    shrink: bool,
) -> NDArray[np.float64]:
    generated = _generate_lin_vectors(variable, lag, h)
    estimates: list[NDArray[np.float64]] = []
    means: list[NDArray[np.float64]] = []
    for i in range(min(h, lag)):
        intercept, slope = _fast_lm(
            generated["Component.index"][i],
            generated["Component.series"][i],
        )
        forecast = intercept + slope * generated["forecast.values"][i]
        estimates.append(np.asarray(forecast, dtype=np.float64))
        means.append(
            np.full(
                generated["Component.series"][i].size if forecast.size == 0 else forecast.size,
                float(np.mean(generated["Component.series"][i])),
                dtype=np.float64,
            )
        )
    ordered = np.concatenate(estimates)[np.argsort(np.concatenate(generated["forecast.index"]))]
    output = ordered * weight
    if method == "means" or shrink:
        means_ordered = np.concatenate(means)[
            np.argsort(np.concatenate(generated["forecast.index"]))
        ]
        means_weighted = means_ordered * weight
        output = (output + means_weighted) / 2.0 if shrink else means_weighted
    if not negative_values:
        output = np.maximum(0.0, output)
    return output


def _linear_forecast_for_lag(
    component_index: NDArray[np.float64],
    component_series: NDArray[np.float64],
) -> float:
    intercept, slope = _fast_lm(component_index, component_series)
    return float(intercept + slope * (component_index[-1] + 1.0))


def _means_forecast_for_lag(component_series: NDArray[np.float64]) -> float:
    return float(np.mean(component_series))


def _nonlinear_forecast_for_lag(
    component_index: NDArray[np.float64],
    component_series: NDArray[np.float64],
) -> float:
    last_y = float(component_series[-1])
    reg_points_raw = nns_reg(
        component_index,
        component_series,
        return_values=False,
        plot=False,
        multivariate_call=True,
    )
    x = np.asarray(reg_points_raw["x"], dtype=np.float64)
    y = np.asarray(reg_points_raw["y"], dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    xs = x[-1] - x
    ys = y[-1] - y
    xs = xs[:-1]
    ys = ys[:-1]
    if xs.size == 0:
        return last_y
    weights = np.arange(1, xs.size + 1, dtype=np.int64) ** 2
    run = float(np.mean(np.repeat(xs, weights)))
    rise = float(np.mean(np.repeat(ys, weights)))
    return last_y + (rise / run)
