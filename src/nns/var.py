"""NNS.VAR nonparametric vector autoregression, plus the LPM/UPM VaR helpers.

``nns_var`` ports R's ``NNS.VAR``. ``lpm_var`` / ``upm_var`` (R's ``LPM.VaR`` /
``UPM.VaR``, partial-moment Value-at-Risk) live in this module too for
historical import paths; they are unrelated to the vector autoregression.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from numbers import Real
from typing import Any, NotRequired, TypedDict, cast

import numpy as np
from numpy.typing import NDArray

from nns._helpers import _warn_unsupported
from nns.core import lpm_ratio, upm_ratio

_R_OPTIMIZE_TOL = float(np.finfo(float).eps ** 0.25)

ObjectiveFn = Callable[[NDArray[np.float64], NDArray[np.float64]], float]

_METHOD1_GUARD_MESSAGE = (
    "No Method 1 candidate produced a finite complete-coverage OOF objective."
)


class VarResult(TypedDict):
    """``nns_var`` result. ``h=0`` returns only the interpolated matrix and names."""

    interpolated_and_extrapolated: NDArray[np.float64]
    names: list[str]
    relevant_variables: NotRequired[NDArray[Any]]
    univariate: NotRequired[NDArray[np.float64]]
    multivariate: NotRequired[NDArray[np.float64]]
    ensemble: NotRequired[NDArray[np.float64]]


class _LagSpec(TypedDict):
    lags: list[list[int]]
    modulo: list[int]
    max: int


def nns_var(
    variables: Any,
    h: int,
    tau: int | Sequence[int] | Sequence[int | Sequence[int]] = 1,
    *,
    dim_red_method: str = "cor",
    naive_weights: bool = True,
    obj_fn: ObjectiveFn | None = None,
    objective: str = "min",
    status: bool = True,
    ncores: int | None = None,
    nowcast: bool = False,
) -> VarResult:
    """Nonparametric VAR forecast, matching R's ``NNS.VAR``.

    ``variables`` is a 2-D numeric array-like (one column per series; a pandas
    DataFrame's column names are kept) and may contain NaN for missing or
    mixed-frequency observations. ``tau`` is a positive integer (lags
    ``1..tau`` for every variable), a sequence with one lag per variable, or a
    sequence of lag sequences (one per variable). ``obj_fn`` is an optional
    ``obj_fn(predicted, actual) -> float`` used for both the univariate
    ``nns_arma_optim`` and the multivariate stages; ``None`` uses R's MSE /
    co-movement default.

    ``relevant_variables`` lists, per variable, the lagged regressors whose
    dimension reduction coefficient meets the cross-validated threshold. They
    are the only regressors in the ``multivariate`` estimate:
    ``nns_stack(method=1, stack=False)`` (``nns_reg``) on them, each as its own
    dimension.

    For ``h == 0`` the result is a dictionary with ``interpolated_and_extrapolated``
    and ``names`` instead of R's bare data frame. ``ncores`` and ``nowcast``
    (row labelling only in R) are accepted for API compatibility.
    """
    _warn_unsupported(ncores=ncores is not None, nowcast=bool(nowcast))
    del status  # R console progress flag; NNS Python emits no console output.

    h_value = _var_validate_h(h)
    method = _var_validate_dim_red_method(dim_red_method)
    objective_value = _var_validate_objective(objective)
    if obj_fn is not None and not callable(obj_fn):
        raise TypeError("obj_fn must be callable or None.")
    if not isinstance(naive_weights, (bool, np.bool_)):
        raise TypeError("naive_weights must be True or False.")

    variables_matrix, names = _var_as_matrix(variables)
    lag_spec = _var_lags(tau, variables_matrix.shape[1])
    if h_value > 0:
        _var_check_sample_size(variables_matrix.shape[0], h_value, lag_spec["max"])

    first_stage = _var_interpolate_and_extrapolate(
        variables_matrix,
        h_value,
        tau=tau,
        names=names,
        obj_fn=obj_fn,
        objective=objective_value,
    )
    if h_value == 0:
        return first_stage

    interpolated = first_stage["interpolated_and_extrapolated"]
    univariate = first_stage["univariate"]
    multivariate_stage = _var_multivariate_stack_stage(
        interpolated,
        univariate,
        h=h_value,
        tau=tau,
        names=names,
        dim_red_method=method,
        obj_fn=obj_fn,
        objective=objective_value,
    )
    multivariate = cast(NDArray[np.float64], multivariate_stage["multivariate"])
    relevant_variables = cast(NDArray[Any], multivariate_stage["relevant_variables"])
    uni_weights, multi_weights = _var_ensemble_weights(
        relevant_variables,
        names,
        naive_weights=bool(naive_weights),
        tau=tau,
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


# --------------------------------------------------------------------------
# Validation and input coercion
# --------------------------------------------------------------------------

def _var_validate_h(h: Any) -> int:
    if isinstance(h, (bool, np.bool_)) or not isinstance(h, Real):
        raise TypeError("h must be a single non-negative integer.")
    value = float(h)
    if not math.isfinite(value) or value < 0 or value != math.floor(value):
        raise ValueError("h must be a single non-negative integer.")
    return int(value)


def _var_validate_dim_red_method(dim_red_method: Any) -> str:
    if not isinstance(dim_red_method, str) or dim_red_method.lower() not in {
        "cor",
        "nns.dep",
        "nns.caus",
        "all",
    }:
        raise ValueError('dim_red_method must be one of "cor", "NNS.dep", "NNS.caus", or "all".')
    return dim_red_method.lower()


def _var_validate_objective(objective: Any) -> str:
    if not isinstance(objective, str) or objective.lower() not in {"min", "max"}:
        raise ValueError("objective must be 'min' or 'max'.")
    return objective.lower()


def _var_as_matrix(variables: Any) -> tuple[NDArray[np.float64], list[str]]:
    """Coerce ``variables`` to a float matrix with unique, non-empty names."""
    raw_names: list[str] | None = None
    columns = getattr(variables, "columns", None)
    if columns is not None:
        raw_names = ["" if name is None else str(name) for name in list(columns)]
    try:
        matrix = np.asarray(variables, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("variables must be numeric.") from error
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2:
        raise ValueError("variables must be a 2-D numeric matrix.")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("variables must be non-empty.")
    if np.any(np.isinf(matrix)):
        raise ValueError("variables contains infinite values.")
    empty = np.flatnonzero(np.all(np.isnan(matrix), axis=0))
    if empty.size:
        raise ValueError(
            "Variable(s) " + ", ".join(str(j + 1) for j in empty) + " contain only missing values."
        )

    n_vars = matrix.shape[1]
    if raw_names is None or len(raw_names) != n_vars:
        raw_names = [""] * n_vars
    names = [
        (name if name else f"x{j + 1}").replace(" - ", "...") for j, name in enumerate(raw_names)
    ]
    return matrix, _make_unique(names)


def _make_unique(names: Sequence[str]) -> list[str]:
    """R ``make.unique``: later duplicates get ``.1``, ``.2``, ... suffixes."""
    seen = set(names)
    counts: dict[str, int] = {}
    out: list[str] = []
    used: set[str] = set()
    for name in names:
        if name not in used:
            out.append(name)
            used.add(name)
            continue
        k = counts.get(name, 0)
        while True:
            k += 1
            candidate = f"{name}.{k}"
            if candidate not in seen and candidate not in used:
                break
        counts[name] = k
        out.append(candidate)
        used.add(candidate)
    return out


def _var_lag_values(value: Any) -> list[int]:
    """Validate one lag specification (an integer or a sequence of them)."""
    items = [value] if _is_lag_scalar(value) else list(value) if _is_sequence(value) else None
    if not items or not all(_is_lag_scalar(item) for item in items):
        _invalid_tau()
    lags = [int(item) for item in cast(list[Any], items)]
    if any(lag < 1 for lag in lags):
        _invalid_tau()
    return sorted(set(lags))


def _is_lag_scalar(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        return False
    number = float(value)
    return math.isfinite(number) and number == math.floor(number)


def _is_sequence(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return value.ndim == 1
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _invalid_tau() -> None:
    raise ValueError(
        "tau must be a positive integer, a sequence of positive integers with one lag per "
        "variable, or a sequence of positive-integer lag sequences (one per variable, or one "
        "shared by every variable)."
    )


def _var_lags(
    tau: int | Sequence[int] | Sequence[int | Sequence[int]],
    n_vars: int,
) -> _LagSpec:
    """Expand ``tau`` to one lag list per variable, mirroring R ``NNS.VAR``.

    ``modulo`` is each variable's ``nns_seas`` modulo, R's ``min(tau_i)``: the
    smallest lag of the variable's own lag vector for a list ``tau``, and the
    smallest lag of ``tau`` itself otherwise.
    """
    if _is_lag_scalar(tau):
        lag = _var_lag_values(tau)[0]
        return {"lags": [list(range(1, lag + 1)) for _ in range(n_vars)],
                "modulo": [lag] * n_vars, "max": lag}
    if isinstance(tau, (str, bytes)) or not _is_sequence(tau):
        _invalid_tau()
    entries = list(cast(Sequence[Any], tau))
    if not entries:
        _invalid_tau()

    if all(_is_lag_scalar(entry) for entry in entries):
        # A flat vector (R c(...)): one lag per variable, or a scalar.
        values = [_var_lag_values(entry)[0] for entry in entries]
        if len(values) == 1:
            return _var_lags(values[0], n_vars)
        if len(values) != n_vars:
            _invalid_tau()
        return {"lags": [[value] for value in values], "modulo": [min(values)] * n_vars,
                "max": max(values)}

    # A list of lag vectors (R list(...)): one per variable, or one shared.
    if len(entries) == 1:
        entries = entries * n_vars
    if len(entries) != n_vars:
        _invalid_tau()
    lags = [_var_lag_values(entry) for entry in entries]
    return {
        "lags": lags,
        "modulo": [min(lag) for lag in lags],
        "max": max(max(lag) for lag in lags),
    }


def _var_check_sample_size(n_rows: int, h: int, max_tau: int) -> None:
    # The multivariate stage holds out max(2h, 20%) of the lagged training rows
    # for validation and needs at least three fitting rows ahead of that window.
    n_lagged = n_rows - max_tau
    needed = max(2 * h + 3, 4)
    if n_lagged < needed:
        raise ValueError(
            f"Not enough observations: with h = {h} and max(tau) = {max_tau} the lagged "
            f"training sample has {max(n_lagged, 0)} rows but needs at least {needed}."
        )


# --------------------------------------------------------------------------
# Stage 1: interpolation / extrapolation and univariate forecasts
# --------------------------------------------------------------------------

def _var_interpolate_and_extrapolate(
    variables: NDArray[np.float64],
    h: int,
    tau: int | Sequence[int] | Sequence[int | Sequence[int]] = 1,
    names: Sequence[str] | None = None,
    obj_fn: ObjectiveFn | None = None,
    objective: str = "min",
) -> VarResult:
    """Fill missing values and generate univariate ARMA forecasts per column.

    With trailing NaNs every missing value comes from an ``nns_stack`` Method 1
    fit on the index; otherwise interior NaNs come from an ``nns_reg`` fit on
    the index. Observed values are never smoothed.
    """
    from nns.arma import nns_arma_optim
    from nns.seasonality import nns_seas
    from nns.stack import nns_stack

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
    lag_spec = _var_lags(tau, n_vars)

    interpolated = np.empty_like(vars_matrix)
    univariate_columns: list[NDArray[np.float64]] = []

    for j in range(n_vars):
        x = vars_matrix[:, j]
        observed = np.flatnonzero(~np.isnan(x))
        if observed.size == 0:
            raise ValueError("Variable contains only missing values.")
        missing = np.flatnonzero(np.isnan(x))
        index = observed.astype(np.float64) + 1.0
        filled = x.copy()

        if missing.size and observed[-1] < n_rows - 1:
            # Trailing NaN(s): nns_stack on the index, duplicated into two
            # identical predictors so the multivariate machinery has two columns.
            at = missing.astype(np.float64) + 1.0
            fill = nns_stack(
                np.column_stack((index, index)),
                x[observed],
                ivs_test=np.column_stack((at, at)),
                order=None,
                folds=5,
                method=1,
                status=False,
                dist=None,
            )["stack"]
            filled[missing] = np.asarray(fill, dtype=np.float64).reshape(-1)
        elif missing.size:
            filled[missing] = _var_index_fit(index, x[observed], missing.astype(np.float64) + 1.0)
        interpolated[:, j] = filled

        if h > 0:
            periods: NDArray[np.int64] | None
            try:
                periods = nns_seas(filled, modulo=lag_spec["modulo"][j], mod_only=False)["periods"]
                if not isinstance(periods, np.ndarray) or periods.size == 0:
                    periods = None
            except Exception:
                periods = None
            result = nns_arma_optim(
                filled,
                h=h,
                seasonal_factor=periods,
                negative_values=float(np.min(filled)) < 0.0,
                obj_fn=obj_fn,
                objective=objective,
            )
            univariate_columns.append(np.asarray(result["results"], dtype=np.float64).reshape(-1))

    positive = np.nanmin(vars_matrix, axis=0) > 0.0
    interpolated[:, positive] = np.maximum(0.0, interpolated[:, positive])

    if h == 0:
        return {
            "interpolated_and_extrapolated": interpolated,
            "names": list(names),
        }

    return {
        "interpolated_and_extrapolated": interpolated,
        "univariate": np.column_stack(univariate_columns),
        "names": list(names),
    }


def _var_index_fit(
    index: NDArray[np.float64],
    values: NDArray[np.float64],
    at: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Fit a series on its index (``order="max"``) and evaluate it at ``at``."""
    if index.size < 2:
        return np.full(at.size, float(values[0]))
    from nns.regression import nns_reg

    fitted = nns_reg(
        index,
        values,
        order="max",
        point_est=at,
        plot=False,
        point_only=True,
        dist=None,
    )["Point.est"]
    return np.asarray(fitted, dtype=np.float64).reshape(-1)


# --------------------------------------------------------------------------
# Stage 2: multivariate estimates
# --------------------------------------------------------------------------

def _r_mean(values: NDArray[np.float64]) -> float:
    """R's two-pass ``mean()``: exact for constant vectors, so a constant
    prediction has exactly zero co-movement (as in R) rather than a ~1e-17
    residue that turns the ratio objective into a huge finite value."""
    mean = math.fsum(values.tolist()) / values.size
    if math.isfinite(mean):
        mean += math.fsum((values - mean).tolist()) / values.size
    return mean


def _var_default_objective(predicted: NDArray[np.float64], actual: NDArray[np.float64]) -> float:
    """R NNS.VAR default: MSE / (Co.LPM + Co.UPM), Inf when the denominator is 0."""
    from nns.co_moments import co_lpm, co_upm

    predicted_values = np.asarray(predicted, dtype=np.float64)
    actual_values = np.asarray(actual, dtype=np.float64)
    if not (predicted_values.size and actual_values.size):
        return float("inf")
    target_p = _r_mean(predicted_values)
    target_a = _r_mean(actual_values)
    divisor = float(
        co_lpm(1.0, predicted_values, actual_values, target_p, target_a)
        + co_upm(1.0, predicted_values, actual_values, target_p, target_a)
    )
    if divisor == 0.0:
        return float("inf")
    return float(np.mean((predicted_values - actual_values) ** 2) / divisor)


def _var_multivariate_stack_stage(
    interpolated: NDArray[np.float64],
    univariate: NDArray[np.float64],
    h: int,
    tau: int | Sequence[int] | Sequence[int | Sequence[int]] = 1,
    names: Sequence[str] | None = None,
    dim_red_method: str = "cor",
    obj_fn: ObjectiveFn | None = None,
    objective: str = "min",
) -> dict[str, object]:
    """Multivariate ``nns_stack`` forecasts over the lagged VAR design.

    Per variable, the relevant variables are the regressors the cross-validated
    dimension reduction threshold keeps, and the estimate is ``nns_stack`` on
    exactly those regressors, trained on the observed lagged rows and evaluated
    on the last ``h`` of them (R's ``tail(IV, h)``).
    """
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
    objective_value = _var_validate_objective(objective)
    dim_red_value = _var_validate_dim_red_method(dim_red_method)

    if names is None:
        names = [f"x{i + 1}" for i in range(n_vars)]
    if len(names) != n_vars:
        raise ValueError("names length must match number of variables.")
    names_list = list(names)

    lag_spec = _var_lags(tau, n_vars)
    _var_check_sample_size(n_rows, h, lag_spec["max"])
    new_values = np.vstack((interpolated_matrix, univariate_matrix))
    lagged, lagged_names, _ = _var_lag_matrix(new_values, lag_spec["lags"], names_list)
    lagged_train = lagged[:-h, :]
    stack_obj_fn = _var_default_objective if obj_fn is None else obj_fn
    # Mirror R NNS.VAR: only the built-in co-movement ratio objective (no user
    # obj_fn, objective == "min") may fall back to plain MSE when every Method 1
    # candidate is non-finite. A user objective is never overridden.
    use_default_objective = obj_fn is None and objective_value == "min"

    multivariate_outputs: list[NDArray[np.float64]] = []
    relevant_variables: list[list[str]] = []
    for i in range(n_vars):
        # Column i is variable i's contemporaneous (tau_0) column.
        lagged_iv = np.delete(lagged_train, i, axis=1)
        lagged_dv = lagged_train[:, i]
        iv_names = lagged_names[:i] + lagged_names[i + 1 :]
        ts_test = max(2 * h, math.ceil(0.2 * lagged_dv.size))

        # Relevant variables (NNS 11.6): the regressors whose dimension
        # reduction coefficient meets the cross-validated threshold
        relevant = _var_relevant(
            lagged_iv,
            lagged_dv,
            ts_test=ts_test,
            obj_fn=stack_obj_fn,
            objective=objective_value,
            dim_red_method=dim_red_value,
        )
        relevant_variables.append([iv_names[k] for k in relevant])

        # Multivariate estimate: nns_stack (Method 1, nns_reg) on exactly the
        # relevant variables, each its own dimension.
        result = _var_fit_stack(
            lagged_iv[:, relevant],
            lagged_dv,
            lagged_iv[-h:, relevant],
            ts_test=ts_test,
            obj_fn=stack_obj_fn,
            objective=objective_value,
            use_default_objective=use_default_objective,
            dim_red_method=dim_red_value,
        )

        nns_dv = np.asarray(result["stack"], dtype=np.float64).reshape(-1)[:h].copy()
        missing = np.isnan(nns_dv)
        nns_dv[missing] = univariate_matrix[missing, i]
        multivariate_outputs.append(nns_dv)

    max_relevant = max((len(col) for col in relevant_variables), default=0)
    rv_matrix = np.full((max_relevant, n_vars), None, dtype=object)
    for col_idx, col in enumerate(relevant_variables):
        rv_matrix[: len(col), col_idx] = col

    return {
        "multivariate": np.column_stack(multivariate_outputs),
        "relevant_variables": rv_matrix,
        "names": names_list,
    }


def _var_fit_stack(
    ivs_train: NDArray[np.float64],
    dv_train: NDArray[np.float64],
    ivs_test: NDArray[np.float64],
    *,
    ts_test: int,
    obj_fn: ObjectiveFn,
    objective: str,
    use_default_objective: bool,
    dim_red_method: str,
) -> dict[str, Any]:
    """NNS.VAR's ``nns_stack`` call (``method=1``, ``stack=False``). The built-in
    co-movement ratio objective falls back to plain MSE when it is undefined for
    every Method 1 candidate; a user objective is never overridden."""
    from nns.stack import Objective, nns_stack

    stack_args: dict[str, Any] = {
        "ivs_test": ivs_test,
        "ts_test": ts_test,
        "folds": 1,
        "method": 1,
        "dim_red_method": dim_red_method,
        "order": None,
        "stack": False,
        "dist": None,
    }
    try:
        return nns_stack(
            ivs_train, dv_train, obj_fn=obj_fn, objective=cast(Objective, objective), **stack_args
        )
    except ValueError as error:
        if not (use_default_objective and str(error) == _METHOD1_GUARD_MESSAGE):
            raise
        return nns_stack(
            ivs_train, dv_train, obj_fn=_var_mse_objective, objective="min", **stack_args
        )


def _var_mse_objective(predicted: NDArray[np.float64], actual: NDArray[np.float64]) -> float:
    predicted_values = np.asarray(predicted, dtype=np.float64)
    actual_values = np.asarray(actual, dtype=np.float64)
    return float(np.mean((predicted_values - actual_values) ** 2))


def _var_relevant(
    ivs_train: NDArray[np.float64],
    dv_train: NDArray[np.float64],
    *,
    ts_test: int,
    obj_fn: ObjectiveFn,
    objective: str,
    dim_red_method: str,
) -> list[int]:
    """R ``.nns_var_relevant``: relevant variables as NNS 11.6's NNS.VAR found them.

    Cross-validates the dimension reduction threshold over NNS 11.6's
    ``NNS.stack`` cutoff grid (validated on the last ``ts_test`` rows) and
    returns the column indices of every regressor whose coefficient meets it.
    """
    from nns.central_tendencies import nns_gravity
    from nns.regression import nns_reg
    from nns.stack import _spearman_signed

    x = np.asarray(ivs_train, dtype=np.float64)
    y = np.asarray(dv_train, dtype=np.float64)
    n_rows, n = x.shape
    if n == 1:
        return [0]
    reg_args: dict[str, Any] = {"plot": False, "residual_plot": False, "order": None, "ncores": 1}

    def coefficients(xs: NDArray[np.float64], ys: NDArray[np.float64]) -> NDArray[np.float64]:
        if dim_red_method == "cor":
            return _spearman_signed(xs, ys)
        fit = nns_reg(xs, ys, dim_red_method=dim_red_method, point_only=True, **reg_args)
        return np.asarray(fit["equation"]["Coefficient"], dtype=np.float64)[:n]

    def rounded(values: NDArray[np.float64]) -> NDArray[np.float64]:
        # R abs(round(x, 2)): Python's round() also rounds the exact binary value.
        return np.array([np.nan if np.isnan(v) else abs(round(float(v), 2)) for v in values])

    split = n_rows - ts_test
    cv_x, cv_x_test = x[:split], x[split:]
    cv_y, actual = y[:split], y[split:]

    cutoffs_1 = rounded(coefficients(x, y))
    cutoffs_2 = rounded(coefficients(cv_x, cv_y))
    var_cutoffs = np.minimum(cutoffs_1, (cutoffs_1 + cutoffs_2) / 2.0)
    in_range = (var_cutoffs < 1.0) & (var_cutoffs >= 0.0)
    var_cutoffs = var_cutoffs[np.isnan(var_cutoffs) | in_range]
    var_cutoffs[np.isnan(var_cutoffs)] = 0.0
    cutoffs = sorted(set(var_cutoffs.tolist()), reverse=True)[1:] or [0.0]
    if n == 2 and 0.0 not in cutoffs:
        cutoffs.append(0.0)

    scores: list[float] = []
    for cutoff in cutoffs:
        fit = nns_reg(cv_x, cv_y, point_est=cv_x_test, dim_red_method=dim_red_method,
                      threshold=cutoff, point_only=True, **reg_args)
        predicted = np.asarray(fit["Point.est"], dtype=np.float64).reshape(-1).copy()
        missing = np.isnan(predicted)
        if missing.any() and not missing.all():
            predicted[missing] = nns_gravity(predicted[~missing])
        scores.append(float(obj_fn(predicted, actual)))

    # R which.min / which.max: first optimum, NaN ignored
    values = np.asarray(scores, dtype=np.float64)
    candidates = np.flatnonzero(~np.isnan(values))
    best = 0
    if candidates.size:
        subset = values[candidates]
        best = int(candidates[np.argmin(subset) if objective == "min" else np.argmax(subset)])
    threshold = cutoffs[best]

    # Every regressor the threshold keeps (positive or negative coefficient)
    fit = nns_reg(x, y, dim_red_method=dim_red_method, threshold=threshold, point_only=True,
                  **reg_args)
    coefficient = np.asarray(fit["equation"]["Coefficient"], dtype=np.float64)[:n]
    return np.flatnonzero(coefficient != 0.0).tolist() or list(range(n))


def _var_ensemble_weights(
    relevant_variables: NDArray[Any],
    names: Sequence[str],
    *,
    naive_weights: bool,
    tau: int | Sequence[int] | Sequence[int | Sequence[int]] = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Univariate share = fraction of the relevant regressors that are the
    variable's own lags; lagged names map to variables exactly (never parsed)."""
    names_list = list(names)
    n_vars = len(names_list)
    uni = np.full(n_vars, 0.5, dtype=np.float64)
    if naive_weights:
        return uni, 1.0 - uni

    rv = np.asarray(relevant_variables, dtype=object)
    if rv.ndim != 2 or rv.shape[1] != n_vars:
        raise ValueError("relevant_variables shape must match variable names.")

    lag_spec = _var_lags(tau, n_vars)
    source = dict(zip(*_var_lag_names(names_list, lag_spec["lags"]), strict=True))
    for i, given_var in enumerate(names_list):
        observed = [
            source[str(value)]
            for value in rv[:, i].tolist()
            if value is not None
            and not (isinstance(value, float) and np.isnan(value))
            and str(value) in source
        ]
        if observed:
            uni[i] = sum(value == given_var for value in observed) / len(observed)

    return uni, 1.0 - uni


# --------------------------------------------------------------------------
# Lag matrix
# --------------------------------------------------------------------------

def _var_lag_names(
    names: Sequence[str],
    lags: Sequence[Sequence[int]],
) -> tuple[list[str], list[str]]:
    """Lagged column names and the variable behind each column: every
    variable's ``_tau_0`` column first, then each variable's lags."""
    var_names = [name.replace('"', "") for name in names]
    col_names = [f"{name}_tau_0" for name in var_names]
    col_source = list(names)
    for j, name in enumerate(var_names):
        for lag in lags[j]:
            col_names.append(f"{name}_tau_{lag}")
            col_source.append(names[j])
    return _make_unique(col_names), col_source


def _var_lag_matrix(
    x: NDArray[np.float64],
    lags: Sequence[Sequence[int]],
    names: Sequence[str],
) -> tuple[NDArray[np.float64], list[str], list[str]]:
    arr = np.asarray(x, dtype=np.float64)
    n_rows, n_vars = arr.shape
    max_tau = max(max(lag) for lag in lags)
    if n_rows <= max_tau:
        raise ValueError("Not enough rows to build the lag matrix for the requested tau.")
    rows = np.arange(max_tau, n_rows)
    columns = [arr[rows, j] for j in range(n_vars)]
    for j in range(n_vars):
        columns.extend(arr[rows - lag, j] for lag in lags[j])
    col_names, col_source = _var_lag_names(names, lags)
    return np.column_stack(columns), col_names, col_source


def _lag_mtx(
    x: NDArray[np.float64],
    tau: int | Sequence[int] | Sequence[int | Sequence[int]],
    names: Sequence[str] | None = None,
) -> tuple[NDArray[np.float64], list[str]]:
    """Build an R-compatible lag matrix for VAR-style feature construction."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("x must be a 2-D array.")
    n_vars = arr.shape[1]
    var_names = list(names) if names is not None else [f"var{idx + 1}" for idx in range(n_vars)]
    matrix, col_names, _ = _var_lag_matrix(arr, _var_lags(tau, n_vars)["lags"], var_names)
    return matrix, col_names


# --------------------------------------------------------------------------
# LPM / UPM Value-at-Risk (R LPM.VaR / UPM.VaR)
# --------------------------------------------------------------------------

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
