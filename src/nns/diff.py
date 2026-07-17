from __future__ import annotations

from collections.abc import Callable, Sequence
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
        from nns.regression import nns_reg

        result = nns_reg(
            x_values,
            y_values,
            plot=False,
            dist=None,
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
    factor_levels: Sequence[Sequence[Any] | None] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Faithful Python port of R's ``dy.d_``.

    ``wrt`` uses R-style 1-based indexing and selects a column of the
    factor-expanded predictor matrix. Numeric scalar/vector ``eval_points``
    evaluate only the selected regressor; a two-dimensional array evaluates
    complete predictor tuples. ``factor_levels`` supplies R factor levels when
    NumPy inputs do not retain categorical metadata.
    """
    raw_x = np.asarray(x)
    if raw_x.ndim != 2:
        raise ValueError("Please ensure (x) is a matrix or data.frame type object.")
    if raw_x.shape[1] < 2:
        raise ValueError("Please use NNS::dy.dx(...) for univariate partial derivatives.")

    raw_y = np.asarray(y).reshape(-1)
    if raw_y.size != raw_x.shape[0]:
        raise ValueError("x and y must have compatible row counts.")
    if _dy_d_has_missing(raw_x) or _dy_d_has_missing(raw_y):
        raise ValueError("You have some missing values, please address.")

    x_values = _dy_d_expand_predictors(x, factor_levels=factor_levels)
    y_values = np.asarray(raw_y, dtype=np.float64)
    if _dy_d_has_missing(x_values) or _dy_d_has_missing(y_values):
        raise ValueError("You have some missing values, please address.")

    wrt_values = _dy_d_validate_wrt(wrt, x_values.shape[1])
    outputs = [
        _dy_d_scalar(
            x_values,
            y_values,
            int(wrt_value) - 1,
            eval_points,
            mixed=bool(mixed),
            messages=bool(messages),
            wrt_label=int(wrt_value),
        )
        for wrt_value in wrt_values
    ]
    if len(outputs) == 1:
        return outputs[0]
    return _combine_dy_d_outputs(outputs)


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


def dy_d_best(
    x: NDArray[Any],
    y: NDArray[Any],
    wrt: int | NDArray[np.int64],
    eval_points: str | float | NDArray[np.float64] = "obs",
    *,
    mixed: bool = False,
    messages: bool = False,
    factor_levels: Sequence[Sequence[Any] | None] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Reconciled ``dy.d_`` partial-derivative estimator.

    This is the reconciliation of the original NNS 0.5.7 ``dy.d_`` (Vinod & Viole
    2020, SSRN 3681436) with the current regression engine. It keeps the v0.5.7
    finite-difference scaffolding but makes two changes that make the estimates
    uniform across identically-distributed regressors and independent of the old
    data.table machinery:

    * **Step (``h_step``) shares the ``dy.dx`` logic** - a locally-adaptive step
      centred on the evaluation point's percentile,
      ``h_step = VaR(p + H, 1, x) - VaR(p - H, 1, x)`` with
      ``p = LPM.ratio(1, eval, x)`` - instead of a global quantile spacing with a
      cumulative window. This is what removes the cross-regressor scatter.
    * **Estimates come from** ``nns_stack`` on the equal-weight synthetic regressor
      ``X*`` via the increased-dimension trick ``cbind(X*, X*)``, with
      ``method=(1, 2)``, ``dim_red_method="equal"``, ``order="max"`` and
      ``folds=5``. The stack's cross-validated ``n.best`` regularises the (sharper)
      current engine back toward the paper's regime.

    Bandwidths follow v0.5.7: ``h_s = 1/log(size(x), [2, 10])`` extended by
    ``10 * h_s`` and doubled when ``nns_dep(x[:, wrt], y) < 0.5``. First and second
    derivatives use the central-difference forms
    ``First = (upper - lower) / (2 h_step)`` and
    ``Second = (upper - 2 f(x) + lower) / h_step ** 2`` (matching ``dy.dx``),
    blended across bandwidths with a plain ``nanmean``.

    ``wrt`` uses R-style 1-based indexing into the factor-expanded predictor
    matrix. A scalar/1-D ``eval_points`` evaluates only the selected regressor
    (averaged over the distribution of the other regressors); a 2-D array
    evaluates complete predictor tuples.
    """
    raw_x = np.asarray(x)
    if raw_x.ndim != 2:
        raise ValueError("Please ensure (x) is a matrix or data.frame type object.")
    if raw_x.shape[1] < 2:
        raise ValueError("Please use NNS::dy.dx(...) for univariate partial derivatives.")

    raw_y = np.asarray(y).reshape(-1)
    if raw_y.size != raw_x.shape[0]:
        raise ValueError("x and y must have compatible row counts.")
    if _dy_d_has_missing(raw_x) or _dy_d_has_missing(raw_y):
        raise ValueError("You have some missing values, please address.")

    x_values = _dy_d_expand_predictors(x, factor_levels=factor_levels)
    y_values = np.asarray(raw_y, dtype=np.float64)
    if _dy_d_has_missing(x_values) or _dy_d_has_missing(y_values):
        raise ValueError("You have some missing values, please address.")

    wrt_values = _dy_d_validate_wrt(wrt, x_values.shape[1])

    # Every regressor shares the same X* = rowMeans(x) training design, so the
    # NNS.stack fit (CV n.best, dimension-reduction coefficients, blend weight)
    # is identical for all of them and for every bandwidth. Gather the test
    # blocks from all regressors and run a SINGLE NNS.stack fit + prediction,
    # then hand each regressor's slice back to its reducer.
    plans = [
        _dy_d_best_plan(
            x_values,
            y_values,
            int(wrt_value) - 1,
            eval_points,
            mixed=bool(mixed),
            messages=bool(messages),
            wrt_label=int(wrt_value),
        )
        for wrt_value in wrt_values
    ]

    all_chunks: list[NDArray[np.float64]] = []
    spans: list[tuple[int, int, Any]] = []
    for chunks, reduce in plans:
        spans.append((len(all_chunks), len(chunks), reduce))
        all_chunks.extend(chunks)

    sizes = [chunk.shape[0] for chunk in all_chunks]
    predictions = _dy_d_best_reg_estimates(x_values, y_values, np.vstack(all_chunks))
    offsets = np.concatenate(([0], np.cumsum(sizes)))
    parts = [predictions[offsets[i] : offsets[i + 1]] for i in range(len(sizes))]

    outputs = [reduce(parts[start : start + count]) for start, count, reduce in spans]
    if len(outputs) == 1:
        return outputs[0]
    return _combine_dy_d_outputs(outputs)


def _dy_d_best_reg_estimates(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    test_points: NDArray[np.float64],
) -> NDArray[np.float64]:
    """f(x +/- h) via NNS.stack on the equal-weight synthetic regressor X*.

    Reduces both the training design and the test points to the equal-weight
    synthetic regressor X* = rowMeans(.), then estimates with the NNS
    increased-dimension trick cbind(X*, X*) under
    ``method=(1, 2), dim_red_method="equal", order="max", folds=5``. The
    cross-validated ``n.best`` is what regularises the current engine.
    """
    from nns.stack import nns_stack

    x_star = np.asarray(x, dtype=np.float64).mean(axis=1)
    test_star = np.asarray(test_points, dtype=np.float64).mean(axis=1)
    result = nns_stack(
        ivs_train=np.column_stack([x_star, x_star]),
        dv_train=y,
        ivs_test=np.column_stack([test_star, test_star]),
        method=(1, 2),
        dim_red_method="equal",
        status=False,
        order="max",
        folds=5,
        ncores=1,
        dist=None,
    )
    return np.asarray(result["stack"], dtype=np.float64).reshape(-1)


def _r_seq(start: float, stop: float, by: float) -> NDArray[np.float64]:
    """Reproduce R's seq(start, stop, by)."""
    if not np.isfinite(by) or by == 0.0:
        return np.asarray([start], dtype=np.float64)
    count = int(np.floor((stop - start) / by + 1e-9))
    return start + by * np.arange(0, count + 1, dtype=np.float64)


def _v057_bandwidths(x_size: int, wrt_dependence: float) -> NDArray[np.float64]:
    """h_s = 1/log(length(x), c(2, 10)); c(h_s, 10*h_s); doubled if dependence < .5."""
    base = np.log(np.asarray([2.0, 10.0])) / np.log(float(x_size))
    h_s = np.concatenate([base, 10.0 * base])
    if wrt_dependence < 0.5:
        h_s = 2.0 * h_s
    return h_s


def _dydx_step(column: NDArray[np.float64], eval_value: float, band: float) -> float:
    """Locally-adaptive dy.dx step: VaR(p + H, 1, x) - VaR(p - H, 1, x), p = CDF(eval)."""
    from nns.core import lpm_ratio
    from nns.var import lpm_var

    p = float(lpm_ratio(1.0, float(eval_value), column))
    upper = lpm_var(min(1.0, p + band), 1.0, column)
    lower = lpm_var(max(0.0, p - band), 1.0, column)
    return float(upper - lower)


def _dy_d_best_plan(
    x_values: NDArray[np.float64],
    y_values: NDArray[np.float64],
    wrt_index: int,
    eval_points: str | float | NDArray[np.float64],
    *,
    mixed: bool,
    messages: bool,
    wrt_label: int,
) -> tuple[
    list[NDArray[np.float64]],
    Callable[[list[NDArray[np.float64]]], dict[str, NDArray[np.float64]]],
]:
    """Build every test block this regressor needs, plus a reducer.

    The reducer turns the predicted chunks back into the derivative dict. All
    regressors share the same X* training design, so ``dy_d_best`` gathers the
    chunks from every plan and runs a single NNS.stack fit for the whole call.
    """
    from nns.dependence import nns_dep
    from nns.var import lpm_var

    _n_rows, n_predictors = x_values.shape
    if wrt_index < 0 or wrt_index >= n_predictors:
        raise ValueError("`wrt` must select exactly one column of the expanded predictor matrix.")
    if n_predictors != 2:
        mixed = False

    if messages:
        print(
            "Currently generating NNS.reg finite difference estimates...Regressor "
            f"{wrt_label}\r"
        )

    eval_values, vector_branch = _dy_d_eval_points(x_values, wrt_index, eval_points)

    column = x_values[:, wrt_index]
    dependence = float(nns_dep(column, y_values)["Dependence"])
    h_s = _v057_bandwidths(x_values.size, dependence)

    # The NNS.stack fit (CV n.best, dimension-reduction coefficients, blend
    # weight) depends only on (x, y), never on the evaluation points, so every
    # bandwidth - and the mixed-derivative corners - are predicted from a single
    # fit. Gather every test block, run ONE NNS.stack, then slice it back.
    chunks: list[NDArray[np.float64]] = []
    sizes: list[int] = []

    def _add(block: NDArray[np.float64]) -> int:
        chunks.append(block)
        sizes.append(block.shape[0])
        return len(chunks) - 1

    def _row_nanmean(bands: list[NDArray[np.float64]]) -> NDArray[np.float64]:
        matrix = np.column_stack([np.asarray(b, dtype=np.float64).reshape(-1) for b in bands])
        with np.errstate(invalid="ignore"):
            return np.asarray(np.nanmean(matrix, axis=1), dtype=np.float64)

    steps_per_band: list[NDArray[np.float64]] = []
    main_chunk_idx: list[int] = []
    mixed_eval: NDArray[np.float64] | None = None

    if vector_branch:
        eval_vec = np.asarray(eval_values, dtype=np.float64).reshape(-1)
        grid = np.column_stack(
            [
                np.asarray(
                    [lpm_var(float(p), 0.0, x_values[:, col]) for p in _r_seq(0.0, 1.0, 0.05)],
                    dtype=np.float64,
                )
                for col in range(n_predictors)
            ]
        )
        sample_size = grid.shape[0]
        k = eval_vec.size
        position = np.tile(
            np.repeat(np.asarray(["l", "m", "u"], dtype=object), sample_size), k
        )
        ids = np.repeat(np.arange(k), 3 * sample_size)
        for band in h_s:
            steps = np.array([_dydx_step(column, ev, float(band)) for ev in eval_vec])
            blocks: list[NDArray[np.float64]] = []
            for g in range(k):
                lower = grid.copy()
                middle = grid.copy()
                upper = grid.copy()
                lower[:, wrt_index] = eval_vec[g] - steps[g]
                middle[:, wrt_index] = eval_vec[g]
                upper[:, wrt_index] = eval_vec[g] + steps[g]
                blocks.extend((lower, middle, upper))
            steps_per_band.append(steps)
            main_chunk_idx.append(_add(np.vstack(blocks)))
        if k == 2:
            mixed_eval = eval_vec.reshape(1, 2)
    else:
        eval_mat = _as_eval_matrix(eval_values, n_predictors)
        n_eval = eval_mat.shape[0]
        for band in h_s:
            steps = np.array(
                [_dydx_step(column, eval_mat[i, wrt_index], float(band)) for i in range(n_eval)]
            )
            lower = eval_mat.copy()
            upper = eval_mat.copy()
            lower[:, wrt_index] = eval_mat[:, wrt_index] - steps
            upper[:, wrt_index] = eval_mat[:, wrt_index] + steps
            steps_per_band.append(steps)
            main_chunk_idx.append(_add(np.vstack((lower, eval_mat, upper))))
        mixed_eval = eval_mat

    # Mixed-derivative corners (also predicted from the same single fit).
    mixed_meta: list[tuple[int | None, NDArray[np.float64]]] = []
    if mixed:
        if mixed_eval is None or mixed_eval.shape[1] != 2:
            raise ValueError("Mixed Derivatives are only for 2 IV")
        for band in h_s:
            corner_blocks: list[NDArray[np.float64]] = []
            scales: list[float] = []
            for point in mixed_eval:
                s1 = _dydx_step(x_values[:, 0], point[0], float(band))
                s2 = _dydx_step(x_values[:, 1], point[1], float(band))
                if np.isfinite(s1) and np.isfinite(s2) and s1 != 0.0 and s2 != 0.0:
                    corner_blocks.append(
                        np.array(
                            [
                                [point[0] + s1, point[1] + s2],
                                [point[0] - s1, point[1] + s2],
                                [point[0] + s1, point[1] - s2],
                                [point[0] - s1, point[1] - s2],
                            ]
                        )
                    )
                    scales.append(4.0 * s1 * s2)
                else:
                    scales.append(np.nan)
            scale_arr = np.asarray(scales, dtype=np.float64)
            if corner_blocks:
                mixed_meta.append((_add(np.vstack(corner_blocks)), scale_arr))
            else:
                mixed_meta.append((None, scale_arr))

    # ---- reduction: given this plan's predicted chunks, build the result -----
    def reduce(parts: list[NDArray[np.float64]]) -> dict[str, NDArray[np.float64]]:
        firsts: list[NDArray[np.float64]] = []
        seconds: list[NDArray[np.float64]] = []
        for bi, steps in enumerate(steps_per_band):
            block = parts[main_chunk_idx[bi]]
            if vector_branch:
                k = steps.size
                f = np.empty(k)
                s = np.empty(k)
                for g in range(k):
                    lo = np.mean(block[(position == "l") & (ids == g)])
                    mid = np.mean(block[(position == "m") & (ids == g)])
                    up = np.mean(block[(position == "u") & (ids == g)])
                    h = steps[g]
                    if np.isfinite(h) and h != 0.0:
                        f[g] = (up - lo) / (2.0 * h)
                        s[g] = (up - 2.0 * mid + lo) / (h**2)
                    else:
                        f[g] = np.nan
                        s[g] = np.nan
            else:
                n_eval = steps.size
                lo = block[:n_eval]
                mid = block[n_eval : 2 * n_eval]
                up = block[2 * n_eval :]
                finite = np.isfinite(steps) & (steps != 0.0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    f = (up - lo) / (2.0 * steps)
                    s = (up - 2.0 * mid + lo) / (steps**2)
                f = np.where(finite, f, np.nan)
                s = np.where(finite, s, np.nan)
            firsts.append(f)
            seconds.append(s)

        result = {"First": _row_nanmean(firsts), "Second": _row_nanmean(seconds)}

        if mixed:
            mixeds: list[NDArray[np.float64]] = []
            for chunk_idx, scale_arr in mixed_meta:
                vals = np.full(scale_arr.size, np.nan)
                if chunk_idx is not None:
                    z = parts[chunk_idx]
                    pos = 0
                    for m in range(scale_arr.size):
                        if np.isfinite(scale_arr[m]):
                            c = z[pos : pos + 4]
                            vals[m] = (c[0] + c[3] - c[1] - c[2]) / scale_arr[m]
                            pos += 4
                mixeds.append(vals)
            result["Mixed"] = _row_nanmean(mixeds)
        return result

    return chunks, reduce




def _dy_d_scalar(
    x_values: NDArray[np.float64],
    y_values: NDArray[np.float64],
    wrt_index: int,
    eval_points: str | float | NDArray[np.float64],
    *,
    mixed: bool,
    messages: bool,
    wrt_label: int,
) -> dict[str, NDArray[np.float64]]:
    from nns.central_tendencies import nns_rescale
    from nns.copula import nns_copula
    from nns.dependence import _gravity, nns_dep
    from nns.var import lpm_var

    n, n_predictors = x_values.shape
    if wrt_index < 0 or wrt_index >= n_predictors:
        raise ValueError("`wrt` must select exactly one column of the expanded predictor matrix.")
    if n_predictors != 2:
        mixed = False

    if messages:
        print(
            "Currently generating NNS.reg finite difference estimates...Regressor "
            f"{wrt_label}\r"
        )

    eval_values, vector_branch = _dy_d_eval_points(x_values, wrt_index, eval_points)

    norm_matrix = np.column_stack(
        [nns_rescale(x_values[:, col], 0.0, 1.0) for col in range(n_predictors)]
    )
    zz_candidates = np.asarray(
        [
            float(nns_dep(x_values[:, wrt_index], y_values, asym=True)["Dependence"]),
            float(
                nns_copula(
                    np.column_stack(
                        (x_values[:, wrt_index], x_values[:, wrt_index], y_values)
                    )
                )
            ),
            float(
                nns_copula(
                    np.column_stack(
                        (
                            norm_matrix[:, wrt_index],
                            norm_matrix[:, wrt_index],
                            y_values,
                        )
                    )
                )
            ),
        ],
        dtype=np.float64,
    )
    non_missing_zz = zz_candidates[~np.isnan(zz_candidates)]
    zz = float(np.max(non_missing_zz)) if non_missing_zz.size else float("-inf")

    root_n = int(np.floor(np.sqrt(n)))
    if root_n < 2:
        raise ValueError("Insufficient observations to construct finite-difference bandwidths.")
    h_s = _derivative_bandwidths(n)

    base_h = float(_gravity(np.abs(np.diff(x_values[:, wrt_index]))))
    if not np.isfinite(base_h) or base_h == 0.0:
        value_range = abs(
            float(np.max(x_values[:, wrt_index]) - np.min(x_values[:, wrt_index]))
        )
        if not np.isfinite(value_range) or value_range == 0.0:
            raise ValueError("Regressor `wrt` is constant; derivative is undefined.")
        base_h = value_range / float(n)

    deriv_grid: NDArray[np.float64] | None = None
    sample_size = 0
    if vector_branch:
        seq_by = max(0.01, (1.0 - zz) / 2.0)
        probs = _r_seq_0_1(seq_by)
        deriv_grid = np.column_stack(
            [
                np.asarray(
                    [lpm_var(float(prob), 1.0, x_values[:, col]) for prob in probs],
                    dtype=np.float64,
                )
                for col in range(n_predictors)
            ]
        )
        if deriv_grid.ndim != 2 or deriv_grid.shape[1] != n_predictors:
            deriv_grid = np.asarray(deriv_grid, dtype=np.float64).reshape(
                -1, n_predictors, order="F"
            )
        sample_size = deriv_grid.shape[0]
    else:
        eval_values = _as_eval_matrix(eval_values, n_predictors)

    results: list[dict[str, NDArray[np.float64]]] = []
    for index, h_value in enumerate(h_s, start=1):
        h_step = base_h * float(h_value)
        if not np.isfinite(h_step) or h_step <= 0.0:
            raise ValueError("A non-positive finite-difference step was generated.")

        if vector_branch:
            assert deriv_grid is not None
            first, second = _dy_d_vector_band(
                x_values,
                y_values,
                wrt_index,
                np.asarray(eval_values, dtype=np.float64).reshape(-1),
                deriv_grid,
                sample_size,
                h_step,
                index=index,
                total=h_s.size,
                messages=messages,
            )
            mixed_eval_points: NDArray[np.float64] | None = None
        else:
            matrix_eval = _as_eval_matrix(eval_values, n_predictors)
            first, second = _dy_d_matrix_band(
                x_values,
                y_values,
                wrt_index,
                matrix_eval,
                h_step,
                index=index,
                total=h_s.size,
                messages=messages,
            )
            mixed_eval_points = matrix_eval

        result: dict[str, NDArray[np.float64]] = {
            "First": first,
            "Second": second,
        }
        if mixed:
            if vector_branch:
                vector_eval = np.asarray(eval_values, dtype=np.float64).reshape(-1)
                if vector_eval.size != 2:
                    raise ValueError(
                        "Mixed derivatives require a complete two-predictor evaluation tuple."
                    )
                mixed_eval_points = vector_eval.reshape(1, 2)
            assert mixed_eval_points is not None
            result["Mixed"] = _dy_d_mixed(
                x_values,
                y_values,
                mixed_eval_points,
                int(h_value),
            )
        results.append(result)

    output = {
        "First": _weighted_band_average([result["First"] for result in results]),
        "Second": _weighted_band_average([result["Second"] for result in results]),
    }
    if mixed:
        output["Mixed"] = _weighted_band_average([result["Mixed"] for result in results])

    if messages:
        print("\r")
    return output


def _dy_d_expand_predictors(
    x: NDArray[Any],
    *,
    factor_levels: Sequence[Sequence[Any] | None] | None,
) -> NDArray[np.float64]:
    from nns.categorical import factor_2_dummy_fr

    raw = np.asarray(x)
    n_columns = raw.shape[1]
    supplied_levels: list[Sequence[Any] | None]
    if factor_levels is None:
        supplied_levels = [None] * n_columns
    else:
        supplied_levels = list(factor_levels)
        if len(supplied_levels) != n_columns:
            raise ValueError("factor_levels must contain one entry per original predictor.")

    expanded: list[NDArray[np.float64]] = []
    for col in range(n_columns):
        column_source: Any
        detected_levels: Sequence[Any] | None = supplied_levels[col]
        if hasattr(x, "iloc"):
            series = x.iloc[:, col]
            column_source = np.asarray(series)
            if detected_levels is None and str(getattr(series, "dtype", "")) == "category":
                detected_levels = list(series.cat.categories)
        else:
            column_source = raw[:, col]

        dummy_columns = factor_2_dummy_fr(column_source, levels=detected_levels)
        expanded.extend(
            np.asarray(column, dtype=np.float64).reshape(-1)
            for column in dummy_columns.values()
        )

    if not expanded:
        raise ValueError("The expanded predictor matrix contains no columns.")
    return np.column_stack(expanded).astype(np.float64, copy=False)


def _dy_d_validate_wrt(wrt: int | NDArray[np.int64], n_predictors: int) -> NDArray[np.int64]:
    raw = np.asarray(wrt)
    if raw.size == 0 or raw.dtype.kind == "b" or raw.dtype.kind not in {"i", "u", "f"}:
        raise ValueError("`wrt` must select exactly one column of the expanded predictor matrix.")
    numeric = raw.astype(np.float64).reshape(-1)
    if (
        np.any(~np.isfinite(numeric))
        or np.any(numeric != np.floor(numeric))
        or np.any(numeric < 1.0)
        or np.any(numeric > float(n_predictors))
    ):
        raise ValueError("`wrt` must select exactly one column of the expanded predictor matrix.")
    return numeric.astype(np.int64)


def _dy_d_has_missing(values: NDArray[Any]) -> bool:
    for value in np.asarray(values, dtype=object).reshape(-1):
        if value is None:
            return True
        try:
            if bool(np.isnan(value)):
                return True
        except (TypeError, ValueError):
            continue
    return False


def _dy_dx_numeric(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    eval_points: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    from nns.dependence import _gravity
    from nns.regression import nns_reg

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
        dist=None,
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
        if option == "obs":
            return x.copy(), False
        raise ValueError("Unknown `eval_points` option.")

    values = np.asarray(eval_points, dtype=np.float64)
    if values.ndim == 0:
        return values.reshape(1), True
    if values.ndim == 1:
        return values.copy(), True
    if values.ndim == 2:
        if values.shape[1] == 1:
            return values.reshape(-1), True
        return values.copy(), False
    raise ValueError("eval_points must be a scalar, vector, matrix, or supported string.")


def _dy_d_matrix_band(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    wrt_index: int,
    eval_points: NDArray[np.float64],
    h_step: float,
    *,
    index: int,
    total: int,
    messages: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    lower_points = eval_points.copy()
    upper_points = eval_points.copy()
    lower_points[:, wrt_index] = eval_points[:, wrt_index] - h_step
    upper_points[:, wrt_index] = eval_points[:, wrt_index] + h_step
    deriv_points = np.vstack((lower_points, eval_points, upper_points))

    if messages:
        print(
            "Currently generating NNS.reg finite difference estimates...bandwidth "
            f"{index} of {total}\r",
            end="",
            flush=True,
        )

    estimates = _dy_d_stack_estimates(x, y, deriv_points)
    n_eval = eval_points.shape[0]
    if estimates.size != 3 * n_eval:
        raise ValueError("NNS.reg returned an unexpected number of point estimates.")
    lower = estimates[:n_eval]
    fx = estimates[n_eval : 2 * n_eval]
    upper = estimates[2 * n_eval :]
    first = (upper - lower) / (2.0 * h_step)
    second = (upper - 2.0 * fx + lower) / (h_step**2)
    return first, second


def _dy_d_vector_band(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    wrt_index: int,
    eval_values: NDArray[np.float64],
    deriv_grid: NDArray[np.float64],
    sample_size: int,
    h_step: float,
    *,
    index: int,
    total: int,
    messages: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    from nns.dependence import _gravity

    blocks: list[NDArray[np.float64]] = []
    for eval_value in eval_values:
        lower_grid = deriv_grid.copy()
        middle_grid = deriv_grid.copy()
        upper_grid = deriv_grid.copy()
        lower_grid[:, wrt_index] = eval_value - h_step
        middle_grid[:, wrt_index] = eval_value
        upper_grid[:, wrt_index] = eval_value + h_step
        blocks.extend((lower_grid, middle_grid, upper_grid))
    deriv_points = np.vstack(blocks)

    ids = np.repeat(np.arange(eval_values.size), 3 * sample_size)
    position = np.tile(
        np.repeat(np.asarray(["l", "m", "u"], dtype=object), sample_size),
        eval_values.size,
    )

    if messages:
        print(
            f"Currently evaluating the {deriv_points.shape[0]} required points "
            f"{index} of {total}\r",
            end="",
            flush=True,
        )

    estimates = _dy_d_stack_estimates(x, y, deriv_points)
    if estimates.size != deriv_points.shape[0]:
        raise ValueError("NNS.reg returned an unexpected number of point estimates.")

    lower = np.empty(eval_values.size, dtype=np.float64)
    fx = np.empty(eval_values.size, dtype=np.float64)
    upper = np.empty(eval_values.size, dtype=np.float64)
    for eval_id in range(eval_values.size):
        lower[eval_id] = float(_gravity(estimates[(position == "l") & (ids == eval_id)]))
        fx[eval_id] = float(_gravity(estimates[(position == "m") & (ids == eval_id)]))
        upper[eval_id] = float(_gravity(estimates[(position == "u") & (ids == eval_id)]))

    first = (upper - lower) / (2.0 * h_step)
    second = (upper - 2.0 * fx + lower) / (h_step**2)
    return first, second


def _dy_d_stack_estimates(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    test_points: NDArray[np.float64],
) -> NDArray[np.float64]:
    from nns.stack import nns_stack

    result = nns_stack(
        ivs_train=x,
        dv_train=y,
        ivs_test=test_points,
        method=(1, 2),
        dim_red_method="equal",
        status=False,
        order=None,
        folds=1,
        ncores=1,
        dist=None,
    )
    return np.asarray(result["stack"], dtype=np.float64).reshape(-1)


def _dy_d_mixed(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    eval_points: NDArray[np.float64],
    h_value: int,
) -> NDArray[np.float64]:
    from nns.dependence import _gravity

    points = _as_eval_matrix(eval_points, 2)
    h_step_1 = float(_gravity(np.abs(np.diff(x[:, 0])))) * float(h_value)
    h_step_2 = float(_gravity(np.abs(np.diff(x[:, 1])))) * float(h_value)

    if not np.isfinite(h_step_1) or h_step_1 == 0.0:
        h_step_1 = (
            abs(float(np.max(x[:, 0]) - np.min(x[:, 0]))) / float(x.shape[0])
        ) * float(h_value)
    if not np.isfinite(h_step_2) or h_step_2 == 0.0:
        h_step_2 = (
            abs(float(np.max(x[:, 1]) - np.min(x[:, 1]))) / float(x.shape[0])
        ) * float(h_value)
    if h_step_1 <= 0.0 or h_step_2 <= 0.0:
        raise ValueError("Unable to construct valid mixed-derivative bandwidths.")

    blocks = []
    for point in points:
        blocks.append(
            np.asarray(
                [
                    [point[0] + h_step_1, point[1] + h_step_2],
                    [point[0] - h_step_1, point[1] + h_step_2],
                    [point[0] + h_step_1, point[1] - h_step_2],
                    [point[0] - h_step_1, point[1] - h_step_2],
                ],
                dtype=np.float64,
            )
        )
    mixed_points = np.vstack(blocks)
    estimates = _dy_d_stack_estimates(x, y, mixed_points)
    if estimates.size != 4 * points.shape[0]:
        raise ValueError("NNS.reg returned an unexpected number of mixed-derivative estimates.")
    z = estimates.reshape(points.shape[0], 4)
    return (z[:, 0] - z[:, 1] - z[:, 2] + z[:, 3]) / (
        4.0 * h_step_1 * h_step_2
    )


def _as_eval_matrix(values: NDArray[np.float64], n_cols: int) -> NDArray[np.float64]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        if matrix.size != n_cols:
            raise ValueError("eval_points row length must match x column count.")
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[1] != n_cols:
        raise ValueError(
            "Matrix/data-frame `eval_points` must have one column per expanded predictor."
        )
    return matrix


def _weighted_band_average(values: list[NDArray[np.float64]]) -> NDArray[np.float64]:
    matrix = np.column_stack(
        [np.asarray(value, dtype=np.float64).reshape(-1) for value in values]
    )
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    weights = np.arange(matrix.shape[1], 0, -1, dtype=np.float64)
    return np.asarray(matrix @ (weights / np.sum(weights)), dtype=np.float64).reshape(-1)


def _derivative_bandwidths(n: int) -> NDArray[np.int64]:
    root_n = int(np.floor(np.sqrt(n)))
    if root_n < 2:
        raise ValueError("Insufficient observations to construct finite-difference bandwidths.")
    return np.rint(
        np.exp(np.linspace(np.log(2.0), np.log(float(root_n)), 5))
    ).astype(np.int64)


def _r_seq_0_1(by: float) -> NDArray[np.float64]:
    if not np.isfinite(by) or by <= 0.0:
        raise ValueError("A positive finite probability increment is required.")
    values: list[float] = []
    current = 0.0
    tolerance = np.finfo(float).eps * 8.0
    while current <= 1.0 + tolerance:
        values.append(min(current, 1.0))
        current += by
    if not values or values[-1] < 1.0:
        values.append(1.0)
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
