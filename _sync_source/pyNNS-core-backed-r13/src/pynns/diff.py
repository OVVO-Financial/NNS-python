from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

DiffResult = dict[str, float]
DyDxResult = float | dict[str, NDArray[np.float64]]

_RESULT_KEYS = [
    "Value of f(x) at point",
    "Final y-intercept (B)",
    "DERIVATIVE",
    "Inferred h",
    "iterations",
    "converged",
    "termination.code",
    "Initial h finite step: f(x-h)",
    "Initial h finite step: f(x+h)",
    "Initial h averaged finite step",
    "Inferred h finite step: f(x-h)",
    "Inferred h finite step: f(x+h)",
    "Inferred h averaged finite step",
    "Complex Step Derivative (Inferred h)",
]


def nns_diff(
    f: Callable[[float | complex | NDArray[np.float64]], float | complex | NDArray[np.float64]],
    point: float,
    h: float | None = None,
    tol: float = 1e-10,
    max_iter: int | None = None,
    digits: int = 12,
) -> DiffResult:
    """Numerically differentiate a scalar callable, matching R's NNS.diff."""
    point = _finite_scalar(point, "point")
    h_value = abs(point) * 0.1 + 0.01 if h is None else _finite_scalar(h, "h")
    if h_value <= 0.0:
        raise ValueError("h must be > 0.")
    tol = _finite_scalar(tol, "tol")
    if tol <= 0.0:
        raise ValueError("tol must be > 0.")
    max_iter_value = 100 if max_iter is None else int(max_iter)
    if max_iter_value < 1:
        raise ValueError("max_iter must be >= 1.")
    if digits < 0:
        raise ValueError("digits must be >= 0.")

    f_x = _eval_real(f, point, "f(point)")
    f_lower = _eval_real(f, point - h_value, "f(point - h)")
    f_upper = _eval_real(f, point + h_value, "f(point + h)")

    left_slope = (f_x - f_lower) / h_value
    right_slope = (f_upper - f_x) / h_value
    b1 = f_x - left_slope * point
    b2 = f_x - right_slope * point
    lower_b = min(b1, b2)
    upper_b = max(b1, b2)

    if np.isclose(lower_b, upper_b, rtol=np.sqrt(np.finfo(float).eps), atol=0.0):
        slope = float(np.mean([left_slope, right_slope]))
        return _rounded_result(
            [
                f_x,
                b1,
                slope,
                0.0,
                0.0,
                1.0,
                0.0,
                left_slope,
                right_slope,
                slope,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
            digits,
        )

    high_b = max(b1, b2)
    new_b = float(np.mean([lower_b, upper_b]))
    iteration = 1
    converged = False
    termination_code = 2
    inferred_h = np.nan

    while iteration >= 1:
        current_b = new_b

        def new_f(x: float, intercept: float = current_b) -> float:
            return -f_x + ((f_x - _eval_real(f, point - x, "f(point - x)")) / x) * point + intercept

        inferred_h = _uniroot_extend(new_f, -2.0 * h_value, 2.0 * h_value)
        if not np.isfinite(inferred_h):
            termination_code = 2
            break
        if abs(inferred_h) < tol:
            converged = True
            termination_code = 0
            break
        if iteration >= max_iter_value:
            termination_code = 1
            break

        if b1 == high_b:
            if np.sign(inferred_h) < 0:
                lower_b = new_b
            else:
                upper_b = new_b
        else:
            if np.sign(inferred_h) < 0:
                upper_b = new_b
            else:
                lower_b = new_b
        new_b = float(np.mean([lower_b, upper_b]))
        iteration += 1

    final_b = float(np.mean([upper_b, lower_b]))
    if np.isfinite(inferred_h):
        inferred_h = abs(float(inferred_h))

    if abs(point) < np.sqrt(np.finfo(float).eps):
        slope = float(np.mean(_finite_step(f, point, h_value)[:2]))
    else:
        slope = (f_x - final_b) / point

    complex_step = np.nan
    if np.isfinite(inferred_h) and inferred_h != 0.0:
        try:
            f_z = f(complex(point, inferred_h))
            if np.isscalar(f_z):
                complex_step = float(np.imag(f_z) / inferred_h)
        except (ArithmeticError, ValueError, TypeError, OverflowError):
            complex_step = np.nan

    initial = _finite_step(f, point, h_value)
    inferred = (
        _finite_step(f, point, inferred_h)
        if np.isfinite(inferred_h) and inferred_h != 0.0
        else (np.nan, np.nan, np.nan)
    )
    return _rounded_result(
        [
            f_x,
            final_b,
            slope,
            inferred_h,
            float(iteration),
            float(int(converged)),
            float(termination_code),
            initial[0],
            initial[1],
            initial[2],
            inferred[0],
            inferred[1],
            inferred[2],
            complex_step,
        ],
        digits,
    )


def dy_dx(
    x: NDArray[Any],
    y: NDArray[Any],
    eval_point: str | float | NDArray[np.float64] | None = None,
) -> DyDxResult:
    """Partial derivative wrapper for R's dy.dx paths."""
    x_values = np.asarray(x, dtype=np.float64).reshape(-1)
    y_values = np.asarray(y, dtype=np.float64).reshape(-1)
    if x_values.size != y_values.size:
        raise ValueError("x and y must have the same length.")
    if np.any(np.isnan(np.column_stack((x_values, y_values)))):
        raise ValueError("You have some missing values, please address.")
    if isinstance(eval_point, str):
        if eval_point.lower() != "overall":
            raise ValueError("eval_point must be 'overall', numeric, or None.")
        from pynns.regression import nns_reg

        result = nns_reg(
            x_values,
            y_values,
            plot=False,
        )
        fitted = result["Fitted.xy"]
        if not isinstance(fitted, dict):
            raise TypeError("nns_reg returned an unexpected fitted table.")
        return float(np.mean(np.asarray(fitted["gradient"], dtype=np.float64)))
    if eval_point is None:
        raise ValueError("some columns are not in the data.table: [eval.point]")
    return _dy_dx_numeric(x_values, y_values, np.asarray(eval_point, dtype=np.float64).reshape(-1))


def dy_d(
    x: NDArray[Any],
    y: NDArray[Any],
    wrt: int | NDArray[np.int64],
    eval_points: str | float | NDArray[np.float64] = "obs",
    *,
    mixed: bool = False,
    messages: bool = True,
) -> dict[str, NDArray[np.float64]]:
    """Partial derivative wrapper for R's ``dy.d_`` numeric matrix path."""
    del messages
    x_values = np.asarray(x, dtype=np.float64)
    if x_values.ndim != 2:
        raise ValueError("Please ensure (x) is a matrix or data.frame type object.")
    if x_values.shape[1] < 2:
        raise ValueError("Please use NNS::dy.dx(...) for univariate partial derivatives.")
    y_values = np.asarray(y, dtype=np.float64).reshape(-1)
    if y_values.size != x_values.shape[0]:
        raise ValueError("x and y must have compatible row counts.")
    if np.any(np.isnan(np.column_stack((x_values, y_values)))):
        raise ValueError("You have some missing values, please address.")
    wrt_values = np.asarray(wrt, dtype=np.int64).reshape(-1)

    if wrt_values.size > 1:
        outputs = [
            _dy_d_scalar(x_values, y_values, int(wrt_index) - 1, eval_points, mixed=bool(mixed))
            for wrt_index in wrt_values
        ]
        return _combine_dy_d_outputs(outputs)

    wrt_index = int(wrt_values[0]) - 1
    return _dy_d_scalar(
        x_values,
        y_values,
        wrt_index,
        eval_points,
        mixed=bool(mixed),
    )


def _combine_dy_d_outputs(
    outputs: list[dict[str, NDArray[np.float64]]],
) -> dict[str, NDArray[np.float64]]:
    return {
        key: np.column_stack(
            [np.asarray(output[key], dtype=np.float64).reshape(-1) for output in outputs]
        )
        for key in ("First", "Second", "Mixed")
        if all(key in output for output in outputs)
    }


def _dy_d_scalar(
    x_values: NDArray[np.float64],
    y_values: NDArray[np.float64],
    wrt_index: int,
    eval_points: str | float | NDArray[np.float64],
    mixed: bool,
) -> dict[str, NDArray[np.float64]]:
    if wrt_index < 0 or wrt_index >= x_values.shape[1]:
        raise ValueError("wrt must select an existing regressor using R's 1-based indexing.")
    if x_values.shape[1] != 2:
        mixed = False

    eval_values, vector_branch = _dy_d_eval_points(x_values, wrt_index, eval_points)
    h_s = _derivative_bandwidths(x_values.shape[0])
    results: list[dict[str, NDArray[np.float64]] | None] = [None] * h_s.size
    cumulative_step = 0.0
    from pynns.dependence import _gravity

    for h_value in h_s:
        # R overwrites duplicate rounded bandwidths at their first result slot.
        result_index = int(np.flatnonzero(h_s == h_value)[0])
        h_step = _dy_d_h_step(x_values[:, wrt_index], int(h_s[result_index]), _gravity)
        cumulative_step += h_step
        if vector_branch:
            first, second, mixed_values = _dy_d_vector_band(
                x_values,
                y_values,
                wrt_index,
                eval_values.reshape(-1),
                int(h_value),
                h_step,
                cumulative_step,
                mixed=bool(mixed),
            )
        else:
            first, second, mixed_values = _dy_d_matrix_band(
                x_values,
                y_values,
                wrt_index,
                _as_eval_matrix(eval_values, x_values.shape[1]),
                int(h_value),
                h_step,
                cumulative_step,
                mixed=bool(mixed),
            )
        result = {"First": first, "Second": second}
        if mixed_values is not None:
            result["Mixed"] = mixed_values
        results[result_index] = result

    active_results = [result for result in results if result is not None]
    output = {
        "First": _weighted_band_average([result["First"] for result in active_results]),
        "Second": _weighted_band_average([result["Second"] for result in active_results]),
    }
    if mixed and "Mixed" in active_results[0]:
        output["Mixed"] = _weighted_band_average([result["Mixed"] for result in active_results])
    return output


def _dy_dx_numeric(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    eval_points: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    from pynns.dependence import _gravity
    from pynns.regression import nns_reg

    if eval_points.size == 0:
        raise ValueError("eval_point must contain at least one value.")
    if np.any(~np.isfinite(eval_points)):
        raise ValueError("eval_point must be finite.")
    n = x.size
    root_n = int(np.floor(np.sqrt(n)))
    h_s = np.rint(np.exp(np.linspace(np.log(2.0), np.log(float(root_n)), 5))).astype(np.int64)
    spacing = float(_gravity(np.abs(np.diff(x))))
    rows: list[NDArray[np.float64]] = []
    for h_value in h_s:
        indices = np.flatnonzero(h_s == h_value).astype(np.float64) + 1.0
        h_step = spacing * indices
        length = max(eval_points.size, h_step.size)
        eval_recycled = np.resize(eval_points, length)
        h_recycled = np.resize(h_step, length)
        lower = np.maximum(float(np.min(x)), eval_recycled - h_recycled)
        upper = np.minimum(float(np.max(x)), eval_recycled + h_recycled)
        rows.append(np.column_stack((lower, eval_recycled, upper)))

    deriv_points = np.vstack(rows)
    point_est = np.concatenate((deriv_points[:, 0], deriv_points[:, 1], deriv_points[:, 2]))
    reg_output = nns_reg(
        x,
        y,
        point_est=point_est,
        point_only=True,
        smooth=True,
        plot=False,
    )
    estimates = np.asarray(reg_output["Point.est"], dtype=np.float64).reshape(3, -1).T
    eval_col = deriv_points[:, 1]
    run_1 = deriv_points[:, 2] - deriv_points[:, 1]
    run_2 = deriv_points[:, 1] - deriv_points[:, 0]

    zero_upper = run_1 == 0.0
    zero_lower = run_2 == 0.0
    if np.any(zero_upper) or np.any(zero_lower):
        fallback_step = (abs(float(np.max(x) - np.min(x))) / float(n)) * float(len(h_s))
        deriv_points[zero_upper, 2] = deriv_points[zero_upper, 1] - fallback_step
        deriv_points[zero_lower, 2] = deriv_points[zero_lower, 1] - fallback_step
        run_1 = deriv_points[:, 2] - deriv_points[:, 1]
        run_2 = deriv_points[:, 1] - deriv_points[:, 0]

    rise_1 = estimates[:, 2] - estimates[:, 1]
    rise_2 = estimates[:, 1] - estimates[:, 0]
    first = (rise_1 + rise_2) / (run_1 + run_2)
    second = (rise_1 / run_1 - rise_2 / run_2) / ((run_1 + run_2) / 2.0)

    unique_eval = np.array(sorted(set(float(v) for v in eval_col)), dtype=np.float64)
    first_out = np.empty(unique_eval.size, dtype=np.float64)
    second_out = np.empty(unique_eval.size, dtype=np.float64)
    for index, point in enumerate(unique_eval):
        mask = eval_col == point
        first_out[index] = float(np.mean(first[mask]))
        second_out[index] = float(np.mean(second[mask]))
    return {
        "eval.point": unique_eval,
        "first.derivative": first_out,
        "second.derivative": second_out,
    }


def _dy_d_eval_points(
    x: NDArray[np.float64],
    wrt_index: int,
    eval_points: str | float | NDArray[np.float64],
) -> tuple[NDArray[np.float64], bool]:
    if isinstance(eval_points, str):
        option = eval_points.lower()
        if option == "median":
            return np.median(x, axis=0).reshape(1, -1), False
        if option == "last":
            return x[-1:, :].copy(), False
        if option == "mean":
            return np.mean(x, axis=0).reshape(1, -1), False
        if option == "apd":
            return x[:, wrt_index].copy(), True
        return x.copy(), False

    values = np.asarray(eval_points, dtype=np.float64)
    if values.ndim == 0:
        return values.reshape(1), True
    if values.ndim == 1:
        return values.copy(), True
    if values.ndim == 2:
        return values.copy(), False
    raise ValueError("eval_points must be a scalar, vector, matrix, or supported string.")


def _dy_d_matrix_band(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    wrt_index: int,
    eval_points: NDArray[np.float64],
    h_value: int,
    h_step: float,
    perturbation_step: float,
    *,
    mixed: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
    from pynns.regression import nns_reg

    n = eval_points.shape[0]
    lower_points = eval_points.copy()
    upper_points = eval_points.copy()
    lower_points[:, wrt_index] -= perturbation_step
    upper_points[:, wrt_index] += perturbation_step
    deriv_points = np.vstack((lower_points, eval_points, upper_points))
    estimates = np.asarray(
        nns_reg(
            x,
            y,
            point_est=deriv_points,
            dim_red_method="equal",
            threshold=0.0,
            order=None,
            point_only=True,
            smooth=True,
            plot=False,
        )["Point.est"],
        dtype=np.float64,
    )
    lower = estimates[:n]
    fx = estimates[n : 2 * n]
    upper = estimates[2 * n :]
    first = (upper - fx + fx - lower) / (2.0 * h_step)
    second = (upper - 2.0 * fx + lower) / (h_step**2)
    mixed_values = (
        _dy_d_mixed(x, y, eval_points, h_value, wrt_index=wrt_index, matrix_points=True)
        if mixed
        else None
    )
    return first, second, mixed_values


def _dy_d_vector_band(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    wrt_index: int,
    eval_values: NDArray[np.float64],
    h_value: int,
    h_step: float,
    perturbation_step: float,
    *,
    mixed: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
    from pynns.dependence import _gravity, nns_dep
    from pynns.norm import nns_norm
    from pynns.regression import nns_reg
    from pynns.var import lpm_var

    eval_vector = eval_values.reshape(-1)
    lower_eval = eval_vector - perturbation_step
    upper_eval = eval_vector + perturbation_step
    norm_col = nns_norm(x[:, wrt_index].reshape(-1, 1)).reshape(-1)
    zz = max(
        float(nns_dep(x[:, wrt_index], y, asym=True)["Dependence"]),
        _nns_copula_matrix(np.column_stack((x[:, wrt_index], x[:, wrt_index], y))),
        _nns_copula_matrix(np.column_stack((norm_col, norm_col, y))),
    )
    seq_by = max(0.01, (1.0 - zz) / 2.0)
    probs = _r_seq_0_1(seq_by)
    base = np.column_stack(
        [
            np.asarray([lpm_var(float(prob), 1.0, x[:, col]) for prob in probs])
            for col in range(x.shape[1])
        ]
    )
    sampsize = probs.size
    deriv_points = np.vstack([base.copy() for _ in range(3 * eval_vector.size)])
    replacement = np.repeat(
        np.ravel(np.vstack((lower_eval, eval_vector, upper_eval)), order="F"),
        sampsize,
    )[: deriv_points.shape[0]]
    deriv_points[:, wrt_index] = replacement
    estimates = np.asarray(
        nns_reg(
            x,
            y,
            point_est=deriv_points,
            dim_red_method="equal",
            threshold=0.0,
            order=None,
            point_only=True,
            smooth=True,
            plot=False,
        )["Point.est"],
        dtype=np.float64,
    )
    position = np.resize(
        np.repeat(np.array(["l", "m", "u"], dtype=object), sampsize), estimates.size
    )
    ids = np.resize(np.repeat(np.arange(eval_vector.size), 3 * sampsize), estimates.size)
    lower = np.empty(eval_vector.size, dtype=np.float64)
    fx = np.empty(eval_vector.size, dtype=np.float64)
    upper = np.empty(eval_vector.size, dtype=np.float64)
    for index in range(eval_vector.size):
        lower[index] = _gravity(estimates[(ids == index) & (position == "l")])
        fx[index] = _gravity(estimates[(ids == index) & (position == "m")])
        upper[index] = _gravity(estimates[(ids == index) & (position == "u")])
    first = (upper - fx + fx - lower) / (2.0 * h_step)
    second = (upper - 2.0 * fx + lower) / (h_step**2)
    mixed_values = (
        _dy_d_mixed(x, y, eval_vector, h_value, wrt_index=wrt_index, matrix_points=False)
        if mixed
        else None
    )
    return first, second, mixed_values


def _nns_copula_matrix(values: NDArray[np.float64]) -> float:
    from pynns.dependence import _dpm_nd
    from pynns.pm_matrix import pm_matrix

    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("NNS.copula matrix input must have at least two columns.")

    n_cols = data.shape[1]
    target = np.mean(data, axis=0)
    upper = np.triu_indices(n_cols, k=1)

    discrete_pm = pm_matrix(0.0, 0.0, target, data, pop_adj=False)
    discrete_co_pm = float(np.sum(discrete_pm["cupm"][upper]) + np.sum(discrete_pm["clpm"][upper]))
    if discrete_co_pm == 1.0 or discrete_co_pm == 0.0:
        return 1.0

    continuous_pm = pm_matrix(1.0, 1.0, target, data, pop_adj=True, norm=True)
    continuous_co_pm = float(
        np.sum(continuous_pm["cupm"][upper]) + np.sum(continuous_pm["clpm"][upper])
    )

    independent_co_pm = 0.25 * (n_cols**2 - n_cols)
    discrete_dep = min(max(abs(discrete_co_pm - independent_co_pm) / independent_co_pm, 0.0), 1.0)
    continuous_dep = min(
        max(abs(continuous_co_pm - independent_co_pm) / independent_co_pm, 0.0), 1.0
    )

    discrete_d_pm = _dpm_nd(data, target, 0.0, norm=True)
    continuous_d_pm = _dpm_nd(data, target, 1.0, norm=True)
    independent_d_pm = 1.0 - (0.5**n_cols)
    n_dim_discrete_dep = abs(discrete_d_pm - independent_d_pm) / independent_d_pm
    n_dim_continuous_dep = abs(continuous_d_pm - independent_d_pm) / independent_d_pm

    return float(
        np.sqrt(
            np.mean(
                [
                    discrete_dep,
                    continuous_dep,
                    n_dim_discrete_dep,
                    n_dim_continuous_dep,
                ]
            )
        )
    )


def _dy_d_mixed(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    eval_points: NDArray[np.float64],
    h_value: int,
    *,
    wrt_index: int,
    matrix_points: bool,
) -> NDArray[np.float64]:
    from pynns.dependence import _gravity
    from pynns.regression import nns_reg

    if x.shape[1] != 2:
        raise ValueError("Mixed Derivatives are only for 2 IV")
    if matrix_points:
        points = _as_eval_matrix(eval_points, 2)
        h1 = _dy_d_h_step(x[:, 0], h_value, _gravity)
        h2 = _dy_d_h_step(x[:, 1], h_value, _gravity)
        mixed_points = np.vstack(
            (
                np.column_stack((points[:, 0] + h1, points[:, 1] + h2)),
                np.column_stack((points[:, 0] - h1, points[:, 1] + h2)),
                np.column_stack((points[:, 0] + h1, points[:, 1] - h2)),
                np.column_stack((points[:, 0] - h1, points[:, 1] - h2)),
            )
        )
        denom: float | NDArray[np.float64] = 4.0 * h1 * h2
        n = points.shape[0]
    else:
        vector = eval_points.reshape(-1)
        if vector.size != 2:
            raise ValueError("Mixed Derivatives are only for 2 IV")
        h_step = _dy_d_h_step(x[:, wrt_index], h_value, _gravity)
        mixed_points = np.asarray(
            [
                vector + h_step,
                [vector[0] - h_step, vector[1] + h_step],
                [vector[0] + h_step, vector[1] - h_step],
                vector - h_step,
            ],
            dtype=np.float64,
        )
        denom = 4.0 * h_step**2
        n = 1
    estimates = np.asarray(
        nns_reg(
            x,
            y,
            point_est=mixed_points,
            dim_red_method="equal",
            threshold=0.0,
            order=None,
            point_only=True,
            smooth=True,
            plot=False,
        )["Point.est"],
        dtype=np.float64,
    )
    z = estimates.reshape(4, n).T
    return (z[:, 0] + z[:, 3] - z[:, 1] - z[:, 2]) / denom


def _dy_d_h_step(
    values: NDArray[np.float64],
    h_value: int,
    gravity_fn: Callable[[NDArray[np.float64]], float],
) -> float:
    h_step = float(gravity_fn(np.abs(np.diff(values)))) * float(h_value)
    if h_step == 0.0:
        h_step = (abs(float(np.max(values) - np.min(values))) / float(values.size)) * float(h_value)
    return h_step


def _as_eval_matrix(values: NDArray[np.float64], n_cols: int) -> NDArray[np.float64]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        if matrix.size != n_cols:
            raise ValueError("eval_points row length must match x column count.")
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[1] != n_cols:
        raise ValueError("eval_points matrix must have one column per regressor.")
    return matrix


def _weighted_band_average(values: list[NDArray[np.float64]]) -> NDArray[np.float64]:
    matrix = np.column_stack(values)
    weights = np.arange(matrix.shape[1], 0, -1, dtype=np.int64)
    return np.asarray(
        [np.mean(np.repeat(row, weights)) for row in matrix],
        dtype=np.float64,
    )


def _derivative_bandwidths(n: int) -> NDArray[np.int64]:
    root_n = int(np.floor(np.sqrt(n)))
    return np.rint(np.exp(np.linspace(np.log(2.0), np.log(float(root_n)), 5))).astype(np.int64)


def _r_seq_0_1(by: float) -> NDArray[np.float64]:
    values: list[float] = []
    current = 0.0
    while current <= 1.0 + np.finfo(float).eps:
        values.append(min(current, 1.0))
        current += by
    return np.asarray(values, dtype=np.float64)


def _finite_step(
    f: Callable[[float | complex | NDArray[np.float64]], float | complex | NDArray[np.float64]],
    point: float,
    h: float,
) -> tuple[float, float, float]:
    f_x = _eval_real(f, point, "f(point)")
    neg_step = (f_x - _eval_real(f, point - h, "f(point - h)")) / h
    pos_step = (_eval_real(f, point + h, "f(point + h)") - f_x) / h
    return neg_step, pos_step, float(np.mean([neg_step, pos_step]))


def _uniroot_extend(fn: Callable[[float], float], lower: float, upper: float) -> float:
    eps = np.finfo(float).eps
    lo = lower if lower != 0.0 else -eps
    hi = upper if upper != 0.0 else eps
    try:
        f_lo = fn(lo)
        f_hi = fn(hi)
        for _ in range(100):
            if np.isfinite(f_lo) and np.isfinite(f_hi) and f_lo * f_hi <= 0.0:
                from scipy import optimize  # type: ignore[import-untyped]

                return float(
                    optimize.brentq(
                        fn,
                        lo,
                        hi,
                        xtol=1e-14,
                        rtol=np.finfo(float).eps * 4.0,
                        maxiter=1000,
                    )
                )
            lo *= 2.0
            hi *= 2.0
            f_lo = fn(lo)
            f_hi = fn(hi)
    except (ArithmeticError, ValueError, TypeError, OverflowError):
        return np.nan
    return np.nan


def _eval_real(
    f: Callable[[float | complex | NDArray[np.float64]], float | complex | NDArray[np.float64]],
    value: float,
    label: str,
) -> float:
    result = f(value)
    if not np.isscalar(result):
        raise ValueError(f"{label} must return a scalar.")
    scalar = cast(float | complex, result)
    if isinstance(scalar, complex):
        if scalar.imag != 0.0:
            raise ValueError(f"{label} must return a real value.")
        scalar = scalar.real
    out = float(scalar)
    if not np.isfinite(out):
        raise ValueError(f"{label} must return a finite value.")
    return out


def _finite_scalar(value: float, name: str) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite.")
    return out


def _rounded_result(values: list[float], digits: int) -> DiffResult:
    rounded = np.round(np.asarray(values, dtype=np.float64), decimals=digits)
    return {key: float(value) for key, value in zip(_RESULT_KEYS, rounded, strict=True)}
