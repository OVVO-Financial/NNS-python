from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from nns.core import lpm_ratio, upm_ratio

_R_OPTIMIZE_TOL = float(np.finfo(float).eps ** 0.25)


def nns_var(
    variables: NDArray[np.float64],
    h: int,
    tau: int | list[int] | list[list[int]] = 1,
    *,
    dim_red_method: str = "cor",
    naive_weights: bool = True,
    obj_fn: Any = None,
    objective: str = "min",
    status: bool = True,
    ncores: int | None = None,
    nowcast: bool = False,
) -> dict[str, Any]:
    """Nonparametric VAR forecast for numeric matrix-like inputs.

    The public Python path returns plain arrays keyed like R's ``NNS.VAR`` output.
    For ``h == 0`` the result is normalized to a dictionary containing
    ``interpolated_and_extrapolated`` and ``names`` instead of returning a bare
    data frame as R does.
    """
    del obj_fn, status, ncores, nowcast

    method = dim_red_method.lower()
    if method not in {"cor", "nns.dep", "nns.caus", "all"}:
        raise ValueError('dim_red_method must be one of "cor", "NNS.dep", "NNS.caus", or "all".')
    if not isinstance(h, Integral):
        raise TypeError("h must be an integer.")
    h_int = int(h)
    if h_int < 0:
        raise ValueError("h must be non-negative.")

    variables_matrix = np.asarray(variables, dtype=np.float64)
    if variables_matrix.ndim != 2:
        raise ValueError("variables must be a 2-D numeric matrix.")
    if variables_matrix.shape[0] == 0 or variables_matrix.shape[1] == 0:
        raise ValueError("variables must be non-empty.")
    names = [f"x{i + 1}" for i in range(variables_matrix.shape[1])]

    first_stage = _var_interpolate_and_extrapolate(
        variables_matrix,
        h_int,
        tau=tau,
        names=names,
    )
    if h_int == 0:
        return first_stage

    interpolated = cast(NDArray[np.float64], first_stage["interpolated_and_extrapolated"])
    univariate = cast(NDArray[np.float64], first_stage["univariate"])
    multivariate_stage = _var_multivariate_stack_stage(
        interpolated,
        univariate,
        h=h_int,
        tau=tau,
        names=names,
        dim_red_method=dim_red_method,
        objective=objective,
    )
    multivariate = cast(NDArray[np.float64], multivariate_stage["multivariate"])
    relevant_variables = cast(NDArray[Any], multivariate_stage["relevant_variables"])
    uni_weights, multi_weights = _var_ensemble_weights(
        relevant_variables,
        names,
        naive_weights=naive_weights,
    )
    ensemble = univariate * uni_weights[np.newaxis, :] + multivariate * multi_weights[np.newaxis, :]

    return {
        "interpolated_and_extrapolated": interpolated,
        "relevant_variables": relevant_variables,
        "univariate": univariate,
        "multivariate": multivariate,
        "ensemble": ensemble,
        "names": names,
    }


def _var_interpolate_and_extrapolate(
    variables: NDArray[np.float64],
    h: int,
    tau: int | Sequence[int] | Sequence[Sequence[int]] = 1,
    names: Sequence[str] | None = None,
) -> dict[str, object]:
    """Interpolate missing values and generate univariate ARMA forecasts per column."""

    vars_matrix = np.asarray(variables, dtype=np.float64)
    if vars_matrix.ndim != 2:
        raise ValueError("variables must be a 2-D matrix.")
    if h < 0:
        raise ValueError("h must be non-negative.")

    n_rows, n_vars = vars_matrix.shape
    if names is None:
        names = [f"x{i + 1}" for i in range(n_vars)]
    if len(names) != n_vars:
        raise ValueError("names length must match number of variables.")

    from nns.arma import nns_arma_optim
    from nns.regression import nns_reg
    from nns.seasonality import nns_seas
    from nns.stack import nns_stack

    interpolated = np.empty_like(vars_matrix)
    univariate_columns: list[np.ndarray] = []
    indices = np.arange(1, n_rows + 1, dtype=np.float64)

    for j in range(n_vars):
        selected_variable = np.column_stack((indices, vars_matrix[:, j]))
        missing = np.flatnonzero(np.isnan(selected_variable[:, 1]))
        variable_interpolation = np.asarray(selected_variable[:, 1], copy=True)
        complete = selected_variable[~np.isnan(selected_variable[:, 1]), :]

        if complete.size == 0:
            raise ValueError("Variable contains only missing values.")
        interpolation_point = int(complete[-1, 0])
        h_int = n_rows - interpolation_point

        if missing.size == 0:
            variable_interpolation = variable_interpolation.copy()
        elif h_int > 0:
            fill = nns_stack(
                np.column_stack((complete[:, 0], complete[:, 0])),
                complete[:, 1],
                ivs_test=np.column_stack((missing + 1, missing + 1)),
                order=None,
                folds=5,
                method=1,
                ncores=1,
                status=False,
            )["stack"]
            variable_interpolation[missing] = np.asarray(fill, dtype=np.float64)
        else:
            fitted_missing = nns_reg(
                complete[:, 0],
                complete[:, 1],
                order="max",
                ncores=1,
                point_est=np.asarray(missing, dtype=np.float64) + 1,
                plot=False,
                point_only=True,
            )["Point.est"]
            if fitted_missing.size:
                variable_interpolation[missing] = np.asarray(fitted_missing, dtype=np.float64)

        if h > 0:
            tau_i = _var_tau_for_variable(tau, j)
            try:
                periods = nns_seas(
                    variable_interpolation,
                    modulo=int(np.min(tau_i)),
                    mod_only=False,
                )["periods"]
                if not isinstance(periods, np.ndarray) or periods.size == 0:
                    periods = None
            except Exception:
                periods = None

            result = nns_arma_optim(
                variable_interpolation,
                h=h,
                seasonal_factor=None if periods is None else periods,
                negative_values=float(np.min(variable_interpolation)) < 0.0,
                ncores=1,
            )
            forecast = np.asarray(result["results"], dtype=np.float64)
            univariate_columns.append(forecast)

        interpolated[:, j] = variable_interpolation

    positive_values = np.nanmin(vars_matrix, axis=0)
    for j in range(n_vars):
        if positive_values[j] > 0.0:
            interpolated[:, j] = np.maximum(0.0, interpolated[:, j])

    if h == 0:
        return {
            "interpolated_and_extrapolated": interpolated,
            "names": list(names),
        }

    univariate = np.column_stack(univariate_columns)
    return {
        "interpolated_and_extrapolated": interpolated,
        "univariate": univariate,
        "names": list(names),
    }


def _var_multivariate_stack_stage(
    interpolated: NDArray[np.float64],
    univariate: NDArray[np.float64],
    h: int,
    tau: int | Sequence[int] | Sequence[Sequence[int]] = 1,
    names: Sequence[str] | None = None,
    dim_red_method: str = "cor",
    obj_fn: Any = None,
    objective: str = "min",
) -> dict[str, object]:
    """Build multivariate R-compatible stack forecasts from interpolated VAR inputs."""

    interpolated_matrix = np.asarray(interpolated, dtype=np.float64)
    univariate_matrix = np.asarray(univariate, dtype=np.float64)
    if interpolated_matrix.ndim != 2:
        raise ValueError("interpolated must be a 2-D matrix.")
    if univariate_matrix.ndim != 2:
        raise ValueError("univariate must be a 2-D matrix.")
    if h <= 0:
        raise ValueError("h must be positive for multivariate stage.")
    n_rows, n_vars = interpolated_matrix.shape
    if n_rows == 0 or n_vars == 0:
        raise ValueError("interpolated must be non-empty.")
    if univariate_matrix.shape != (h, n_vars):
        raise ValueError("univariate shape must be (h, n_variables).")

    from nns.co_moments import co_lpm, co_upm
    from nns.stack import _spearman_scores, nns_stack

    if names is None:
        names = [f"x{i + 1}" for i in range(n_vars)]
    if len(names) != n_vars:
        raise ValueError("names length must match number of variables.")
    names_list = list(names)

    h_cols = [
        np.concatenate(
            (interpolated_matrix[:, col], univariate_matrix[:, col]),
            dtype=np.float64,
        )
        for col in range(n_vars)
    ]
    new_values = np.column_stack(h_cols)
    lagged_new_values, lagged_names = _lag_mtx(new_values, tau, names=names_list)
    if lagged_new_values.shape[0] < h:
        raise ValueError("Not enough rows after lag construction for requested h.")

    lagged_train = lagged_new_values[: lagged_new_values.shape[0] - h, :]
    univariate_forecast = univariate_matrix.copy()
    multivariate_outputs: list[np.ndarray] = []
    relevant_variables: list[list[str]] = []

    if lagged_train.shape[0] == 0:
        raise ValueError("Lagged training block is empty after removing forecast horizon.")

    if lagged_train.shape[0] < h:
        raise ValueError("Not enough lagged training rows for this forecast horizon.")

    objective_value = objective.lower()
    if objective_value not in {"min", "max"}:
        raise ValueError("objective must be 'min' or 'max'.")
    dim_red_value = dim_red_method.lower()
    dim_red_threshold_method = str(dim_red_method)
    for i in range(n_vars):
        lagged_iv = np.column_stack((lagged_train[:, :i], lagged_train[:, i + 1 :]))
        lagged_dv = lagged_train[:, i]

        if lagged_iv.size == 0:
            ivs_test = np.empty((0, 0), dtype=np.float64)
        else:
            ivs_test = lagged_iv[-h:, :]

        ts_test = max(2 * h, math.ceil(0.2 * lagged_dv.size))

        def var_obj_fn(predicted: np.ndarray, actual: np.ndarray) -> float:
            predicted_values = np.asarray(predicted, dtype=np.float64)
            actual_values = np.asarray(actual, dtype=np.float64)
            if not (predicted_values.size and actual_values.size):
                return float("inf")
            divisor = co_lpm(
                1.0,
                predicted_values,
                actual_values,
                float(np.mean(predicted_values)),
                float(np.mean(actual_values)),
            ) + co_upm(
                1.0,
                predicted_values,
                actual_values,
                float(np.mean(predicted_values)),
                float(np.mean(actual_values)),
            )
            if divisor == 0.0:
                return float("inf")
            return float(np.mean((predicted_values - actual_values) ** 2) / divisor)

        result = nns_stack(
            lagged_iv,
            lagged_dv,
            ivs_test=ivs_test,
            obj_fn=cast(Any, var_obj_fn),
            objective=cast(Any, objective_value),
            folds=1,
            method=(1, 2),
            order=None,
            stack=True,
            dim_red_method=cast(Any, dim_red_threshold_method),
            ts_test=ts_test,
        )

        nns_dv = np.asarray(result["stack"], dtype=np.float64)
        nns_dv = nns_dv[:h].copy()
        missing = np.isnan(nns_dv)
        if np.any(missing):
            replacement = univariate_forecast[:, i]
            nns_dv[missing] = replacement[missing]
        multivariate_outputs.append(nns_dv)

        threshold = float(np.asarray(result["NNS.dim.red.threshold"], dtype=np.float64))
        threshold = 0.0 if not np.isfinite(threshold) else threshold

        lagged_target_name = lagged_names[i]
        lagged_iv_names = lagged_names[:i] + lagged_names[i + 1 :]
        lagged_iv_matrix = np.column_stack((lagged_dv, lagged_iv))

        if dim_red_value == "cor":
            rel = _spearman_scores(lagged_iv_matrix, lagged_iv_matrix[:, 0])[1:]
        elif dim_red_value == "nns.dep":
            rel = _dependence_scores(lagged_iv_matrix)[1:]
        elif dim_red_value == "nns.caus":
            rel = _causation_scores(lagged_iv_matrix)[1:]
        else:
            rel = _combined_scores(lagged_iv_matrix)[1:]

        rel_vars: list[str] = [
            name
            for name, value in zip(lagged_iv_names, rel.tolist(), strict=False)
            if value > threshold and name != lagged_target_name
        ]

        if len(rel_vars) == 0:
            rel_vars = lagged_names.copy()

        relevant_variables.append(rel_vars)

    max_relevant = max((len(col) for col in relevant_variables), default=0)
    rv_matrix = np.full((max_relevant, n_vars), None, dtype=object)
    for col_idx, col in enumerate(relevant_variables):
        rv_matrix[: len(col), col_idx] = col

    return {
        "multivariate": np.column_stack(multivariate_outputs),
        "relevant_variables": rv_matrix,
        "names": names_list,
    }


def _dependence_scores(values: NDArray[np.float64]) -> NDArray[np.float64]:
    from nns.dependence import nns_dep

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("values must be a 2-D matrix.")
    if matrix.shape[1] == 0:
        return np.empty(0, dtype=np.float64)
    scores = np.empty(matrix.shape[1], dtype=np.float64)
    target = matrix[:, 0]
    for col in range(matrix.shape[1]):
        scores[col] = float(nns_dep(target, matrix[:, col])["Dependence"])
    return scores


def _causation_scores(values: NDArray[np.float64]) -> NDArray[np.float64]:
    from nns.causation import causal_matrix

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("values must be a 2-D matrix.")
    if matrix.shape[1] == 0:
        return np.empty(0, dtype=np.float64)
    return np.asarray(causal_matrix(matrix)[0, :], dtype=np.float64)


def _combined_scores(values: NDArray[np.float64]) -> NDArray[np.float64]:
    from nns.stack import _spearman_scores

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("values must be a 2-D matrix.")
    if matrix.shape[1] == 0:
        return np.empty(0, dtype=np.float64)
    cor = _spearman_scores(matrix, matrix[:, 0])
    dep = _dependence_scores(matrix)
    caus = _causation_scores(matrix)
    return (cor + dep + caus) / 3.0


def _var_ensemble_weights(
    relevant_variables: NDArray[Any],
    names: Sequence[str],
    *,
    naive_weights: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    names_list = list(names)
    n_vars = len(names_list)
    uni = np.full(n_vars, 0.5, dtype=np.float64)
    multi = np.full(n_vars, 0.5, dtype=np.float64)
    if naive_weights:
        return uni, multi

    rv = np.asarray(relevant_variables, dtype=object)
    if rv.ndim != 2 or rv.shape[1] != n_vars:
        raise ValueError("relevant_variables shape must match variable names.")

    for i, given_var in enumerate(names_list):
        observed = [
            str(value).split("_tau", 1)[0]
            for value in rv[:, i].tolist()
            if value is not None and not (isinstance(value, float) and np.isnan(value))
        ]
        if not observed:
            continue
        equal_tau = sum(value == given_var for value in observed)
        unequal_tau = len(observed) - equal_tau
        total = equal_tau + unequal_tau
        if total > 0:
            uni[i] = equal_tau / total
            multi[i] = 1.0 - uni[i]

    return uni, multi


def _var_tau_for_variable(
    tau: int | Sequence[int] | Sequence[Sequence[int]],
    index: int,
) -> NDArray[np.int64]:
    if isinstance(tau, Integral):
        return np.array([int(tau)], dtype=np.int64)

    if isinstance(tau, str):
        raise TypeError("tau must be numeric.")

    tau_values = list(cast(Sequence[Any], tau))
    if len(tau_values) == 0:
        raise ValueError("tau must include at least one lag.")

    if all(isinstance(item, Integral) for item in tau_values):
        values = np.asarray(tau_values, dtype=np.int64)
        if values.size == 0:
            raise ValueError("tau must include at least one lag.")
        if np.any(values < 0):
            raise ValueError("tau values must be non-negative integers.")
        return values

    has_vector = any(
        isinstance(item, Sequence) and not isinstance(item, (str, bytes)) for item in tau_values
    )
    if not has_vector:
        raise TypeError("tau must be an integer, a numeric sequence, or a list of lag vectors.")

    selected = tau_values[min(index, len(tau_values) - 1)]
    if isinstance(selected, Integral):
        out = np.array([int(selected)], dtype=np.int64)
    else:
        out = np.asarray(selected, dtype=np.int64).reshape(-1)
    if out.size == 0:
        raise ValueError("tau list entries must be non-empty.")
    if np.any(out < 0):
        raise ValueError("tau values must be non-negative integers.")
    return out


def _lag_mtx(
    x: np.ndarray,
    tau: int | Sequence[int] | Sequence[Sequence[int]],
    names: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build an R-compatible lag matrix for VAR-style feature construction."""

    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("x must be a 2-D array.")
    n_rows, n_vars = arr.shape

    if isinstance(tau, int):
        lag_by_var: list[list[int]] = [list(range(tau + 1)) for _ in range(n_vars)]
        tau_values = [tau]
        filter_columns = False
    else:
        raw_tau = list(tau)
        if len(raw_tau) != n_vars:
            raise ValueError("tau must have one entry per variable.")

        is_scalar_tau = all(isinstance(item, Integral) for item in raw_tau)
        is_nested_tau = any(
            isinstance(item, Sequence) and not isinstance(item, (str, bytes)) for item in raw_tau
        )

        if is_nested_tau and not is_scalar_tau:
            lag_by_var = []
            tau_values = []
            for item in raw_tau:
                if not isinstance(item, Sequence):
                    raise ValueError("tau entries must be integer lag vectors.")
                values = [int(value) for value in item]
                lag_by_var.append(values)
                tau_values.extend(values)
        elif is_scalar_tau and not is_nested_tau:
            lag_by_var = []
            tau_values = []
            for item in raw_tau:
                if not isinstance(item, Integral):
                    raise ValueError("tau entries must be integers.")
                lag_by_var.append([int(item)])
                tau_values.append(int(item))
        else:
            raise ValueError("tau entries must be integers or sequences of integers.")
        filter_columns = len(tau_values) > 1

    if len(tau_values) == 0:
        raise ValueError("tau must include at least one lag.")
    if not all(isinstance(value, int) for value in tau_values):
        raise ValueError("tau values must be integers.")
    if not all(value >= 0 for value in tau_values):
        raise ValueError("tau values must be non-negative integers.")

    max_tau = max(tau_values)

    block = max_tau + 1
    lag_matrix = np.empty((n_rows - max_tau, n_vars * block), dtype=np.float64)
    lag_names: list[str] = []
    var_names = list(names) if names is not None else [f"var{idx + 1}" for idx in range(n_vars)]

    for j in range(n_vars):
        col_offset = j * block
        for i in range(block):
            lag_matrix[:, col_offset + i] = arr[max_tau - i : n_rows - i, j]
        for i in range(block):
            lag_names.append(f"{var_names[j]}_tau_{i}")

    if filter_columns:
        requested: list[int] = []
        for j in range(n_vars):
            offset = j * block
            requested.extend(offset + int(lag) for lag in lag_by_var[j])
            if 0 not in lag_by_var[j]:
                requested.append(offset)
        selected = np.array(sorted(set(requested)), dtype=int)
    else:
        selected = np.arange(lag_matrix.shape[1], dtype=int)

    tau_zero_indices = [idx for idx, name in enumerate(lag_names) if name.endswith("_tau_0")]
    zero_set = set(tau_zero_indices)
    selected_zero = [idx for idx in selected if idx in zero_set]
    selected_non_zero = [idx for idx in selected if idx not in zero_set]
    final_indices = np.array(selected_zero + selected_non_zero, dtype=int)
    reordered_names = [lag_names[idx] for idx in final_indices]
    return lag_matrix[:, final_indices], reordered_names


def lpm_var(percentile: float, degree: float, x: NDArray[np.float64]) -> float:
    """Lower partial-moment VaR matching R's LPM.VaR."""
    values = _finite_values(x)
    pct = min(max(float(percentile), 0.0), 1.0)
    if degree == 0:
        return float(np.quantile(values, pct, method="linear"))
    if _is_supported_integer_degree(degree):
        return _lpm_var_integer(pct, int(degree), values)
    xmin = float(np.min(values))
    xmax = float(np.max(values))
    if xmin == xmax:
        return xmin

    from scipy.optimize import minimize_scalar  # type: ignore[import-untyped]

    result = minimize_scalar(
        lambda target: abs(float(lpm_ratio(degree, target, values)) - pct),
        bounds=(xmin, xmax),
        method="bounded",
        options={"xatol": _R_OPTIMIZE_TOL},
    )
    return float(result.x)


def upm_var(percentile: float, degree: float, x: NDArray[np.float64]) -> float:
    """Upper partial-moment VaR matching R's UPM.VaR."""
    values = _finite_values(x)
    pct = min(max(float(percentile), 0.0), 1.0)
    if degree == 0:
        return float(np.quantile(values, 1.0 - pct, method="linear"))
    if _is_supported_integer_degree(degree):
        # UPM.ratio(t) = p is equivalent to LPM.ratio(t) = 1 - p.
        return _lpm_var_integer(1.0 - pct, int(degree), values)
    xmin = float(np.min(values))
    xmax = float(np.max(values))
    if xmin == xmax:
        return xmin

    from scipy.optimize import minimize_scalar

    result = minimize_scalar(
        lambda target: abs(float(upm_ratio(degree, target, values)) - pct),
        bounds=(xmin, xmax),
        method="bounded",
        options={"xatol": _R_OPTIMIZE_TOL},
    )
    return float(result.x)


_UNIROOT_TOL = float(np.finfo(float).eps ** 0.5)


def _is_supported_integer_degree(degree: float) -> bool:
    return (
        math.isfinite(float(degree))
        and float(degree) == int(degree)
        and 1 <= int(degree) <= 4
    )


def _lpm_var_integer(pct: float, degree: int, values: NDArray[np.float64]) -> float:
    """Exact integer-degree LPM VaR inversion matching R NNS 13.0.

    Mirrors R's ``.NNS_LPM_VaR_integer``: within an interval between sorted
    unique observations, LPM_d(t) and UPM_d(t) are degree-d polynomials in t,
    so VaR solves ``(1 - p) * LPM_d(t) - p * UPM_d(t) = 0`` exactly on the
    located order-statistic interval.
    """
    n = values.size
    x_sorted = np.sort(values)
    # Center at the sorted median: (t - x) is translation-invariant, so results
    # are mathematically identical, but prefix powers are built at deviation
    # scale, eliminating catastrophic cancellation for level-shifted data.
    shift = float(x_sorted[(n + 1) // 2 - 1])
    x_sorted = x_sorted - shift
    x_min = float(x_sorted[0])
    x_max = float(x_sorted[-1])
    if n == 1 or x_min == x_max:
        return x_min + shift

    unique_x, counts = np.unique(x_sorted, return_counts=True)
    k_break = np.cumsum(counts)

    prefix_power = np.zeros((degree + 1, unique_x.size), dtype=np.float64)
    total_power = np.zeros(degree + 1, dtype=np.float64)
    prefix_power[0] = k_break
    total_power[0] = float(n)
    for j in range(1, degree + 1):
        x_power = x_sorted**j
        prefix_power[j] = np.cumsum(x_power)[k_break - 1]
        total_power[j] = float(np.sum(x_power))

    lpm = np.zeros(unique_x.size, dtype=np.float64)
    upm = np.zeros(unique_x.size, dtype=np.float64)
    for j in range(degree + 1):
        coef = float(math.comb(degree, j))
        lpm += coef * (-1.0) ** j * unique_x ** (degree - j) * prefix_power[j]
        suffix_power = total_power[j] - prefix_power[j]
        upm += coef * (-1.0) ** (degree - j) * unique_x ** (degree - j) * suffix_power

    with np.errstate(invalid="ignore", divide="ignore"):
        ratio_break = lpm / (lpm + upm)
    ratio_break[np.isnan(ratio_break)] = 0.0
    ratio_break = np.clip(ratio_break, 0.0, 1.0)
    # Protect interval location from tiny floating-point non-monotonicity.
    ratio_break = np.maximum.accumulate(ratio_break)
    ratio_break[0] = 0.0
    ratio_break[-1] = 1.0

    if pct <= 0.0:
        return x_min + shift
    if pct >= 1.0:
        return x_max + shift

    interval = int(np.searchsorted(ratio_break, pct, side="right"))
    interval = max(interval, 1)
    interval = min(interval, unique_x.size - 1)

    lower = float(unique_x[interval - 1])
    upper = float(unique_x[interval])
    if lower == upper:
        return lower + shift

    interval_prefix = prefix_power[:, interval - 1]

    def root_value(t: float) -> float:
        lpm_raw = 0.0
        upm_raw = 0.0
        for j in range(degree + 1):
            coef = float(math.comb(degree, j))
            lpm_raw += coef * (-1.0) ** j * t ** (degree - j) * float(interval_prefix[j])
            suffix = float(total_power[j]) - float(interval_prefix[j])
            upm_raw += coef * (-1.0) ** (degree - j) * t ** (degree - j) * suffix
        return (1.0 - pct) * lpm_raw - pct * upm_raw

    f_lower = root_value(lower)
    f_upper = root_value(upper)
    tol = _UNIROOT_TOL

    if math.isfinite(f_lower) and abs(f_lower) <= tol:
        return lower + shift
    if math.isfinite(f_upper) and abs(f_upper) <= tol:
        return upper + shift

    if math.isfinite(f_lower) and math.isfinite(f_upper) and f_lower * f_upper <= 0.0:
        from scipy.optimize import brentq

        root = float(brentq(root_value, lower, upper, xtol=tol))
        return min(max(root, lower), upper) + shift

    # Numerical safety fallback, mirroring R's optimize() on |f|.
    from scipy.optimize import minimize_scalar

    result = minimize_scalar(
        lambda t: abs(root_value(t)),
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": _R_OPTIMIZE_TOL},
    )
    return min(max(float(result.x), lower), upper) + shift


def _finite_values(x: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(x, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("x must contain at least one finite value.")
    return finite
