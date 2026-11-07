from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

DiffResult = dict[str, float]
DyDxResult = float | dict[str, NDArray[np.float64]]
_DyDReducer = Callable[[list[NDArray[np.float64]]], dict[str, NDArray[np.float64]]]

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
    *,
    plot: bool = False,
) -> DiffResult:
    """Numerically differentiate a scalar callable, matching R's NNS.diff.

    The search stops when the inferred ``h`` is below ``tol`` or when the bracket on
    the y-intercept fixes the derivative to within about ``tol * max(1, |f'(point)|)``.
    ``plot=True`` also draws ``f`` around ``point`` with the secant lines
    (``nns.plotting.plot_nns_diff``), as R's ``plot = TRUE`` does.
    """
    result = _nns_diff(f, point, h=h, tol=tol, max_iter=max_iter, digits=digits)
    if plot:
        from nns.plotting.differentiation import plot_nns_diff

        plot_nns_diff(f, float(point), h=h)
    return result



def _nns_diff(
    f: Callable[[float | complex | NDArray[np.float64]], float | complex | NDArray[np.float64]],
    point: float,
    h: float | None = None,
    tol: float = 1e-10,
    max_iter: int | None = None,
    digits: int = 12,
) -> DiffResult:
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

    if _r_all_equal(lower_b, upper_b, tolerance=math.sqrt(_DBL_EPS)):
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
            if x == 0.0:
                # R's 0/0 is NaN, which uniroot (and _zeroin2) replaces and steps past.
                return math.nan
            return -f_x + ((f_x - _eval_real(f, point - x, "f(point - x)")) / x) * point + intercept

        # Solve for h to the convergence tolerance (uniroot's default
        # tol of about 1.2e-4 made |inferred_h| < tol a test of its stopping point).
        inferred_h = _uniroot_extend(new_f, -2.0 * h_value, 2.0 * h_value, tol=tol / 10.0)
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
        # Converged once the bisection bracket on B is tight: inferred_h cannot reach
        # |h| < tol in double precision once it is accurate. The derivative is
        # (f(x) - B) / point, so the bracket is scaled to the derivative, not to |B|.
        if abs(upper_b - lower_b) <= tol * max(abs(point), abs(f_x - new_b)):
            converged = True
            termination_code = 0
            break
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
    if eval_point is None:
        # R's dy.dx(x, y) with the default eval.point = NULL also fails, but
        # deep inside with "undefined columns selected".
        raise ValueError("eval_point must be provided: a numeric point/vector or 'overall'")
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
    """Partial derivatives of ``y`` with respect to column ``wrt`` (R's ``dy.d_``).

    Finite differences of the multivariate ``nns_reg`` fit on the actual
    regressors, ``f(x +/- h e_wrt)``, with a locally-adaptive ``dy.dx`` step
    ``h = VaR(p + H, 1, x_wrt) - VaR(p - H, 1, x_wrt)``, ``p = LPM.ratio(1, eval,
    x_wrt)``. Bandwidths follow NNS 0.5.7: ``H`` in ``1/log(size(x), [2, 10])``
    extended by ``10 * H`` and doubled when ``nns_dep(x[:, wrt], y) < 0.5``. First
    and second derivatives use the central-difference forms
    ``(upper - lower) / (2 h)`` and ``(upper - 2 f(x) + lower) / h ** 2``, averaged
    over the bandwidths. This follows the patched R: NNS 13.3
    estimated ``f`` from ``NNS.stack`` on the equal-weight index ``rowMeans(x)``,
    so every regressor received about the index slope / p whatever its true effect.

    ``wrt`` uses R-style 1-based indexing and selects a column of the
    factor-expanded predictor matrix. A scalar/1-D ``eval_points`` evaluates
    only the selected regressor (averaged over the distribution of the other
    regressors); a two-dimensional array evaluates complete predictor tuples.
    The string modes ``"mean"``, ``"median"``, ``"last"``, ``"obs"`` and
    ``"apd"`` mirror R. ``factor_levels`` supplies R factor levels when NumPy
    inputs do not retain categorical metadata.
    """
    return _dy_d_reconciled(
        x,
        y,
        wrt,
        eval_points,
        mixed=mixed,
        messages=messages,
        factor_levels=factor_levels,
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
    """Deprecated alias of :func:`dy_d` (they always returned the same numbers)."""
    warnings.warn(
        "dy_d_best is deprecated and will be removed; use dy_d (same estimator).",
        DeprecationWarning,
        stacklevel=2,
    )
    return dy_d(
        x,
        y,
        wrt,
        eval_points,
        mixed=mixed,
        messages=messages,
        factor_levels=factor_levels,
    )


def _dy_d_reconciled(
    x: NDArray[Any],
    y: NDArray[Any],
    wrt: int | NDArray[np.int64],
    eval_points: str | float | NDArray[np.float64] = "obs",
    *,
    mixed: bool = False,
    messages: bool = False,
    factor_levels: Sequence[Sequence[Any] | None] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Shared ``dy.d_`` engine backing :func:`dy_d`."""
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

    # Every regressor and bandwidth uses the same multivariate nns_reg fit on the
    # actual regressors. Gather the test blocks from all regressors and run a
    # SINGLE fit + prediction, then hand each regressor's slice back to its reducer.
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
    spans: list[tuple[int, int, _DyDReducer]] = []
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
    """f(x +/- h e_wrt) from the multivariate nns_reg fit on the actual regressors.

    Patched R: ``NNS.reg(x, y, point.est = pts, order = NULL)``. The previous
    ``NNS.stack`` fit on the equal-weight index ``rowMeans(x)`` made every estimate
    depend on ``mean(x)`` only (y = 2*x1 gave dy/dx2 of about 1).
    """
    from nns.regression import nns_reg

    result = nns_reg(
        np.asarray(x, dtype=np.float64),
        y,
        point_est=np.asarray(test_points, dtype=np.float64),
        order=None,
        plot=False,
    )
    return np.asarray(result["Point.est"], dtype=np.float64).reshape(-1)


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


def _r14_fd_weights(
    nodes: NDArray[np.float64], at: float, degree: int
) -> NDArray[np.float64]:
    """R .nns_fd_weights: scaled Vandermonde finite-difference coefficients."""
    n = int(nodes.size)
    if n <= degree or np.unique(nodes).size != n:
        return np.full(n, np.nan)
    scale = float(np.max(np.abs(nodes - at)))
    if not np.isfinite(scale) or scale == 0.0:
        return np.full(n, np.nan)
    z = (nodes - at) / scale
    powers = z[None, :] ** np.arange(n)[:, None]
    rhs = np.zeros(n, dtype=np.float64)
    rhs[degree] = float(math.factorial(degree))
    try:
        return np.linalg.solve(powers, rhs) / (scale**degree)
    except np.linalg.LinAlgError:
        return np.full(n, np.nan)


def _r14_dy_stencil(
    column: NDArray[np.float64], at: float, probability_width: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """R .nns_dy_stencil with local degree-1 partial-moment quantiles."""
    from nns.core import lpm_ratio
    from nns.var import lpm_var

    support = np.unique(column)
    lower, upper = float(support[0]), float(support[-1])
    if not np.isfinite(at) or at < lower or at > upper:
        raise ValueError(
            "Derivative evaluation points must lie inside the observed coordinate support."
        )
    if support.size < 2:
        z = np.asarray([at], dtype=np.float64)
        return z, np.asarray([np.nan]), np.asarray([np.nan])
    if support.size == 2:
        nodes = np.asarray(support, dtype=np.float64)
    else:
        share = float(lpm_ratio(1.0, at, column))
        lo = float(lpm_var(max(0.0, share - probability_width), 1.0, column))
        hi = float(lpm_var(min(1.0, share + probability_width), 1.0, column))
        h = min((hi - lo) / 2.0, (upper - lower) / 4.0)
        if not np.isfinite(h) or h <= 0.0:
            h = float(np.min(np.diff(support)))
        if at == lower:
            nodes = np.asarray([at, at + h, at + 2.0 * h])
        elif at == upper:
            nodes = np.asarray([at - 2.0 * h, at - h, at])
        else:
            nodes = np.asarray([max(lower, at - h), at, min(upper, at + h)])
    return nodes, _r14_fd_weights(nodes, at, 1), _r14_fd_weights(nodes, at, 2)


def _r14_spread_rows(n: int, k: int) -> NDArray[np.intp]:
    """R .nns_spread_rows: golden-ratio/Weyl phase rows, 0-based."""
    if k >= n:
        return np.arange(n, dtype=np.intp)
    phase = np.mod(np.arange(1, n + 1, dtype=np.float64) * 0.6180339887498949, 1.0)
    return np.sort(np.argsort(phase, kind="stable")[:k]).astype(np.intp)


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
    """R 14.0 dy.d_: two locally capped probability windows and polynomial stencils.

    R/ dy_d_wrt.R builds every perturbed context from the original observed
    regressors, fits NNS.reg once, then subtracts the first evaluation before
    applying each stencil. No old tenfold/four-band CV approximation is used.
    """
    from nns._dep_r14 import dep_local_r14

    n, p = x_values.shape
    if wrt_index < 0 or wrt_index >= p:
        raise ValueError("wrt must identify one expanded predictor column.")

    eval_values, marginal = _dy_d_eval_points(x_values, wrt_index, eval_points)
    if marginal:
        points = np.asarray(eval_values, dtype=np.float64).reshape(-1)
        context_count = min(n, max(21, 10000 // max(1, int(points.size))))
        other = x_values[_r14_spread_rows(n, context_count)]
        contexts = []
        for point in points:
            z = other.copy()
            z[:, wrt_index] = point
            contexts.append(z)
    else:
        locations = np.asarray(eval_values, dtype=np.float64)
        if locations.ndim != 2 or locations.shape[1] != p:
            raise ValueError(
                "eval.points must have one column per expanded predictor."
            )
        if np.any(~np.isfinite(locations)) or np.any(
            locations < np.min(x_values, axis=0)
        ) or np.any(locations > np.max(x_values, axis=0)):
            raise ValueError(
                "Derivative evaluation points must lie inside the observed coordinate support."
            )
        contexts = [locations[i : i + 1] for i in range(locations.shape[0])]
        points = locations[:, wrt_index]

    if points.size == 0:
        raise ValueError("eval.points must contain at least one point.")
    # R: 1/log(max(nrow(x),3), c(2,10)), half and full windows capped at .25.
    base = np.log(np.asarray([2.0, 10.0])) / np.log(float(max(n, 3)))
    widths = np.minimum(0.25, np.r_[base / 2.0, base])
    if dep_local_r14(x_values[:, wrt_index], y_values, asym=False) < 0.5:
        widths = np.minimum(0.25, 2.0 * widths)

    chunks: list[NDArray[np.float64]] = []
    specifications: list[
        tuple[int, int, int, int, NDArray[np.float64], NDArray[np.float64]]
    ] = []
    column = x_values[:, wrt_index]
    for band, width in enumerate(widths):
        for i, context in enumerate(contexts):
            nodes, w_first, w_second = _r14_dy_stencil(
                column, float(points[i]), float(width)
            )
            blocks = []
            for node in nodes:
                amended = context.copy()
                amended[:, wrt_index] = node
                blocks.append(amended)
            specifications.append(
                (len(chunks), band, i, context.shape[0], w_first, w_second)
            )
            chunks.append(np.vstack(blocks))

    mixed_specs: list[tuple[int, int, int, NDArray[np.float64]]] = []
    if mixed:
        if p != 2 or marginal:
            raise ValueError(
                "Mixed derivatives require two predictors and explicit matrix evaluation points."
            )
        for band, width in enumerate(widths):
            for i, context in enumerate(contexts):
                at = context[0]
                s1, weight1, _ = _r14_dy_stencil(x_values[:, 0], float(at[0]), float(width))
                s2, weight2, _ = _r14_dy_stencil(x_values[:, 1], float(at[1]), float(width))
                grid = np.asarray([[v1, v2] for v2 in s2 for v1 in s1], dtype=np.float64)
                weights = np.outer(weight1, weight2).ravel(order="F")
                mixed_specs.append((len(chunks), band, i, weights))
                chunks.append(grid)

    if messages:
        print(f"Generating local NNS.reg finite-difference estimates...Regressor {wrt_label}")

    def reduce(parts: list[NDArray[np.float64]]) -> dict[str, NDArray[np.float64]]:
        first = np.full((points.size, widths.size), np.nan)
        second = np.full((points.size, widths.size), np.nan)
        for index, band, point_index, context_rows, w_first, w_second in specifications:
            predicted = np.asarray(parts[index], dtype=np.float64)
            means = np.mean(predicted.reshape(-1, context_rows), axis=1)
            centered = means - means[0]
            first[point_index, band] = float(np.sum(w_first * centered))
            second[point_index, band] = float(np.sum(w_second * centered))

        def average(a: NDArray[np.float64]) -> NDArray[np.float64]:
            with np.errstate(invalid="ignore"):
                numerator = np.nansum(a, axis=1)
                count = np.sum(~np.isnan(a), axis=1)
                return np.divide(
                    numerator, count,
                    out=np.full(a.shape[0], np.nan),
                    where=count > 0,
                )

        result = {"First": average(first), "Second": average(second)}
        if mixed:
            matrix = np.full((points.size, widths.size), np.nan)
            for index, band, point_index, weights in mixed_specs:
                predicted = np.asarray(parts[index], dtype=np.float64)
                matrix[point_index, band] = float(
                    np.sum(weights * (predicted - predicted[0]))
                )
            result["Mixed"] = average(matrix)
        return result

    return chunks, reduce


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
    h_s = np.rint(np.exp(np.linspace(np.log(2.0), np.log(float(root_n)), 5)))
    # Typical spacing between neighbouring x values: diff() of x
    # in row order measured the gap between consecutive rows, so unsorted input
    # gave steps spanning the whole range and depended on row order.
    spacing = float(_gravity(np.diff(np.unique(x))))
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    rows: list[NDArray[np.float64]] = []
    # Every bandwidth position counts, repeated ones too, as in R.
    for h_value in h_s:
        h_step = spacing * float(h_value)
        lower = np.maximum(x_min, eval_points - h_step)
        upper = np.minimum(x_max, eval_points + h_step)
        rows.append(np.column_stack((lower, eval_points, upper)))

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

    rise_1 = estimates[:, 2] - estimates[:, 1]
    rise_2 = estimates[:, 1] - estimates[:, 0]
    # At min(x) or max(x) one run is 0: R uses the one-sided slope for the
    # first derivative and the second derivative is NaN/Inf.
    with np.errstate(divide="ignore", invalid="ignore"):
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


def _finite_step(
    f: Callable[[float | complex | NDArray[np.float64]], float | complex | NDArray[np.float64]],
    point: float,
    h: float,
) -> tuple[float, float, float]:
    f_x = _eval_real(f, point, "f(point)")
    neg_step = (f_x - _eval_real(f, point - h, "f(point - h)")) / h
    pos_step = (_eval_real(f, point + h, "f(point + h)") - f_x) / h
    return neg_step, pos_step, float(np.mean([neg_step, pos_step]))


_DBL_EPS = float(np.finfo(float).eps)
_DBL_MAX = float(np.finfo(float).max)


def _uniroot_extend(
    fn: Callable[[float], float], lower: float, upper: float, *, tol: float = _DBL_EPS**0.25
) -> float:
    """Root of ``fn`` like R's ``uniroot(fn, c(lower, upper), extendInt = "yes", tol)``.

    ``tol`` defaults to R's ``.Machine$double.eps^0.25``; ``maxiter = 1000``:
    the interval is widened additively on both sides until the end values
    bracket a root, then refined with Brent's method as in R's ``zeroin2``.
    Returns NaN where R's ``uniroot`` stops with an error.
    """
    maxiter = 1000
    try:
        f_lower = float(fn(lower))
        f_upper = float(fn(upper))
        if math.isnan(f_lower) or math.isnan(f_upper):
            return np.nan
        if _truncate(f_lower) * _truncate(f_upper) > 0.0:
            delta = [0.01 * max(1e-4, abs(lower)), 0.01 * max(1e-4, abs(upper))]
            it = 0
            while f_lower * f_upper > 0.0 and (math.isfinite(lower) or math.isfinite(upper)):
                it += 1
                if it > maxiter:
                    return np.nan
                if math.isfinite(lower):
                    old, old_f = lower, f_lower
                    lower -= delta[0]
                    f_lower = float(fn(lower))
                    if math.isnan(f_lower):
                        lower, f_lower = old, old_f
                        delta[0] /= 4.0
                if math.isfinite(upper):
                    old, old_f = upper, f_upper
                    upper += delta[1]
                    f_upper = float(fn(upper))
                    if math.isnan(f_upper):
                        upper, f_upper = old, old_f
                        delta[1] /= 4.0
                delta = [2.0 * delta[0], 2.0 * delta[1]]
        if not np.sign(f_lower) * np.sign(f_upper) <= 0.0:
            return np.nan
        return _zeroin2(fn, lower, upper, _truncate(f_lower), _truncate(f_upper), tol, maxiter)
    except (ArithmeticError, ValueError, TypeError, OverflowError):
        return np.nan


def _r_all_equal(target: float, current: float, *, tolerance: float) -> bool:
    """R's ``isTRUE(all.equal(target, current, tolerance))`` for two scalars.

    The mean absolute difference is taken relative to ``|target|`` when that
    exceeds ``tolerance`` and absolute otherwise.
    """
    diff = abs(target - current)
    scale = abs(target)
    if math.isfinite(scale) and scale > tolerance:
        diff /= scale
    return not math.isnan(diff) and diff <= tolerance


def _truncate(value: float) -> float:
    return min(max(value, -_DBL_MAX), _DBL_MAX)


def _zeroin2(
    fn: Callable[[float], float],
    a: float,
    b: float,
    fa: float,
    fb: float,
    tol: float,
    maxiter: int,
) -> float:
    """Brent's root finder, a line-by-line port of R's ``R_zeroin2`` (zeroin.c)."""

    def f(x: float) -> float:
        value = float(fn(x))
        if math.isfinite(value):
            return value
        return -_DBL_MAX if value == -math.inf else _DBL_MAX

    c, fc = a, fa
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    for _ in range(maxiter + 1):
        prev_step = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol_act = 2.0 * _DBL_EPS * abs(b) + tol / 2.0
        new_step = (c - b) / 2.0
        if abs(new_step) <= tol_act or fb == 0.0:
            return b
        if abs(prev_step) >= tol_act and abs(fa) > abs(fb):
            cb = c - b
            if a == c:
                t1 = fb / fa
                p = cb * t1
                q = 1.0 - t1
            else:
                q = fa / fc
                t1 = fb / fc
                t2 = fb / fa
                p = t2 * (cb * q * (q - t1) - (b - a) * (t1 - 1.0))
                q = (q - 1.0) * (t1 - 1.0) * (t2 - 1.0)
            if p > 0.0:
                q = -q
            else:
                p = -p
            if p < (0.75 * cb * q - abs(tol_act * q) / 2.0) and p < abs(prev_step * q / 2.0):
                new_step = p / q
        if abs(new_step) < tol_act:
            new_step = tol_act if new_step > 0.0 else -tol_act
        a, fa = b, fb
        b += new_step
        fb = f(b)
        if (fb > 0.0 and fc > 0.0) or (fb < 0.0 and fc < 0.0):
            c, fc = a, fa
    return b


def _eval_real(
    f: Callable[[float | complex | NDArray[np.float64]], float | complex | NDArray[np.float64]],
    value: float,
    label: str,
) -> float:
    result = f(value)
    if not np.isscalar(result):
        raise ValueError(f"{label} must return a scalar.")
    scalar = result
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
