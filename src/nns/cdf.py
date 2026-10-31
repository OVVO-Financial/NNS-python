from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray

from nns._native import native_fn
from nns.core import lpm_ratio
from nns.dependence import co_lpm_nd
from nns.regression import RegResult, nns_reg

if TYPE_CHECKING:
    from nns.multivariate_regression import MRegResult


def nns_cdf(
    variable: NDArray[np.float64],
    degree: float = 0,
    target: float | NDArray[np.float64] | None = None,
    type: str = "CDF",
    plot: bool = False,
    names: Sequence[str] | None = None,
) -> dict[str, object]:
    """Partial-moment CDF wrapper matching R's non-plotting NNS.CDF paths."""
    type_value = type.lower()
    if type_value not in {"cdf", "survival", "hazard", "cumulative hazard"}:
        raise ValueError("invalid type")

    if names is None:
        columns = getattr(variable, "columns", None)
        if columns is not None:
            names = [str(column) for column in columns]
    values = np.asarray(variable, dtype=np.float64)
    if values.ndim == 0:
        values = values.reshape(1)
    if values.ndim == 1 or (values.ndim == 2 and values.shape[1] == 1):
        result = _univariate_cdf(values.reshape(-1), float(degree), target, type_value)
        if plot:
            _render_cdf(result, target)
        return result
    if values.ndim == 2:
        # Multivariate CDF has no faithful single-Axes plot; plot is a no-op here.
        return _multivariate_cdf(values, float(degree), target, type_value, names)
    raise ValueError("variable must be a vector or 2D matrix.")


def _render_cdf(result: dict[str, object], target: float | NDArray[np.float64] | None) -> None:
    """Render the univariate NNS.CDF figure as a side effect of ``plot=True``."""
    from nns.plotting.partial_moments import plot_nns_cdf

    plot_target: float | None = None
    if target is not None:
        coords = np.asarray(target, dtype=np.float64).reshape(-1)
        if coords.size == 1:
            plot_target = float(coords[0])
    plot_nns_cdf(result, target=plot_target)


def _univariate_cdf(
    values: NDArray[np.float64],
    degree: float,
    target: float | NDArray[np.float64] | None,
    type_value: str,
) -> dict[str, object]:
    if values.size == 0:
        raise ValueError("variable must be non-empty.")
    target_value = _univariate_target(target, values)
    x = np.sort(values[~np.isnan(values)])
    pval = _r_lpm_ratio(degree, x, values)
    column_name = {
        "cdf": "CDF",
        "survival": "S(x)",
        "hazard": "h(x)",
        "cumulative hazard": "H(x)",
    }[type_value]

    y = pval.copy()
    fit: RegResult | MRegResult | None = None
    if type_value == "survival":
        y = 1.0 - y
    elif type_value == "hazard":
        proxy = _hazard_proxy(x, pval)
        point_est = None if target_value is None else float(target_value)
        fit = nns_reg(
            x,
            np.maximum(proxy, 1e-10),
            order=None,
            n_best=1,
            point_est=point_est,
            plot=False,
        )
        fitted = cast(dict[str, NDArray[np.float64]], fit["Fitted.xy"])
        y = np.minimum(
            np.maximum(fitted["y.hat"] / np.maximum(1.0 - pval, 1e-10), 0.0),
            1e6,
        )
    elif type_value == "cumulative hazard":
        y = np.maximum(-np.log(np.maximum(1.0 - pval, 1e-10)), 0.0)

    if target_value is None:
        pv = np.array([], dtype=np.float64)
    else:
        pv = _r_lpm_ratio(degree, np.array([target_value], dtype=np.float64), values)
        if type_value == "survival":
            pv = 1.0 - pv
        elif type_value == "hazard":
            if fit is None:
                raise RuntimeError("hazard fit was not computed.")
            point = np.asarray(fit["Point.est"], dtype=np.float64).reshape(-1)
            nearest = int(np.argmin(np.abs(x - target_value)))
            pv = point / np.maximum(1.0 - pval[nearest], 1e-10)
        elif type_value == "cumulative hazard":
            point_fit = nns_reg(
                x,
                y,
                order=None,
                n_best=1,
                point_est=float(target_value),
                plot=False,
            )
            pv = np.asarray(point_fit["Point.est"], dtype=np.float64).reshape(-1)

    return {"Function": {"x": x, column_name: y}, "target.value": np.asarray(pv, dtype=np.float64)}


def _multivariate_cdf(
    values: NDArray[np.float64],
    degree: float,
    target: float | NDArray[np.float64] | None,
    type_value: str,
    names: Sequence[str] | None,
) -> dict[str, object]:
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("variable must have at least one row and one column.")
    if not np.all(np.isfinite(values)):
        raise ValueError("variable must contain only finite values.")
    target_values = _multivariate_target(target, values)
    column_names = _matrix_names(values.shape[1], names)

    cdf = _co_lpm_nd_rows(values, degree)
    if type_value == "survival":
        marginal_probs = _marginal_lpm_ratios(values, degree)
        cdf = np.maximum(0.0, np.minimum(1.0, 1.0 - np.sum(marginal_probs, axis=1) + cdf))
    elif type_value == "hazard":
        fit = nns_reg(values, np.maximum(cdf, 1e-10), order="max", plot=False)
        fitted = cast(dict[str, NDArray[np.float64]], fit["Fitted.xy"])
        marginal_probs = _marginal_lpm_ratios(values, degree)
        denominator = np.maximum(1.0 - np.sum(marginal_probs, axis=1) + cdf, 1e-10)
        cdf = np.maximum(fitted["y.hat"] / denominator, 0.0)
    elif type_value == "cumulative hazard":
        marginal_probs = _marginal_lpm_ratios(values, degree)
        survival = np.maximum(1.0 - np.sum(marginal_probs, axis=1) + cdf, 1e-10)
        cdf = np.maximum(-np.log(survival), 0.0)

    pv = np.array([], dtype=np.float64)
    if target_values is not None:
        target_cdf = float(co_lpm_nd(values, target_values, degree=degree))
        if type_value == "cdf":
            pv = np.array([target_cdf], dtype=np.float64)
        elif type_value == "survival":
            marg_target = np.array(
                [
                    _r_lpm_ratio(
                        degree,
                        np.array([target_values[col]], dtype=np.float64),
                        values[:, col],
                    )[0]
                    for col in range(values.shape[1])
                ],
                dtype=np.float64,
            )
            target_survival = max(0.0, min(1.0, 1.0 - float(np.sum(marg_target)) + target_cdf))
            pv = np.array([target_survival], dtype=np.float64)
        elif type_value == "hazard":
            point_fit = nns_reg(values, cdf, order="max", plot=False, point_est=target_values)
            point = np.asarray(point_fit["Point.est"], dtype=np.float64).reshape(-1)
            pv = point / np.maximum(1.0 - target_cdf, 1e-10)
        elif type_value == "cumulative hazard":
            pv = np.array([max(-math.log(max(1.0 - target_cdf, 1e-10)), 0.0)], dtype=np.float64)

    function = {column_names[col]: values[:, col].copy() for col in range(values.shape[1])}
    function["CDF"] = cdf
    return {"Function": function, "target.value": pv}


def _univariate_target(
    target: float | NDArray[np.float64] | None,
    values: NDArray[np.float64],
) -> float | None:
    if target is None:
        return None
    target_array = np.asarray(target, dtype=np.float64)
    if target_array.ndim != 0 and target_array.size != 1:
        raise ValueError("target must be scalar for univariate NNS.CDF.")
    target_value = float(target_array.reshape(-1)[0])
    if np.isnan(values).any():
        raise ValueError("missing value where TRUE/FALSE needed")
    if target_value < float(np.min(values)) or target_value > float(np.max(values)):
        raise ValueError("target out of bounds")
    return target_value


def _multivariate_target(
    target: float | NDArray[np.float64] | None,
    values: NDArray[np.float64],
) -> NDArray[np.float64] | None:
    if target is None:
        return None
    target_values = np.asarray(target, dtype=np.float64).reshape(-1)
    if target_values.size < 2:
        raise ValueError("target must contain at least two coordinates for multivariate NNS.CDF.")
    if (
        target_values[0] < float(np.min(values[:, 0]))
        or target_values[0] > float(np.max(values[:, 0]))
        or target_values[1] < float(np.min(values[:, 1]))
        or target_values[1] > float(np.max(values[:, 1]))
    ):
        raise ValueError("target out of bounds")
    if target_values.size != values.shape[1]:
        raise ValueError("target length must match number of columns in variable.")
    if not np.all(np.isfinite(target_values)):
        raise ValueError("target must be finite.")
    return target_values


def _matrix_names(column_count: int, names: Sequence[str] | None) -> list[str]:
    if names is None:
        # R's NNS.CDF labels an unnamed matrix's columns by their index ("1", "2").
        return [str(index + 1) for index in range(column_count)]
    column_names = [str(name) for name in names]
    if len(column_names) != column_count:
        raise ValueError("names length must match the number of columns in variable.")
    return column_names


def _r_lpm_ratio(
    degree: float,
    targets: NDArray[np.float64],
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """R's ``LPM.ratio(degree, targets, values)``, including its NA handling.

    R's ratio kernels skip NA observations in the partial-moment sums but keep them in
    the denominator ``n``.  For ``degree > 0`` the ``n`` cancels in ``LPM / (LPM + UPM)``,
    so the ratio equals the ratio over the non-missing values; for ``degree == 0`` the
    ratio is the non-missing count scaled by ``n_finite / n``.
    """
    target_values = np.asarray(targets, dtype=np.float64).reshape(-1)
    variable_values = np.asarray(values, dtype=np.float64).reshape(-1)
    if variable_values.size == 0:
        raise ValueError("variable must be non-empty.")
    if np.isinf(variable_values).any() or not np.all(np.isfinite(target_values)):
        # R's comparisons give Inf - Inf = NaN, which drops out of both partial sums;
        # reproduce that rare quirk with the direct (target x observation) evaluation.
        return _dense_lpm_ratio(degree, target_values, variable_values)
    finite = variable_values[~np.isnan(variable_values)]
    if target_values.size == 0:
        return np.array([], dtype=np.float64)
    if finite.size == 0:
        fill = 0.0 if degree == 0.0 else np.nan
        return np.full(target_values.size, fill, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.asarray(lpm_ratio(degree, target_values, finite), dtype=np.float64)
    if degree == 0.0 and finite.size != variable_values.size:
        ratio = ratio * (finite.size / variable_values.size)
    return ratio


def _dense_lpm_ratio(
    degree: float,
    targets: NDArray[np.float64],
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    with np.errstate(invalid="ignore", divide="ignore"):
        diff = targets[:, np.newaxis] - values[np.newaxis, :]
        lower = np.where(diff >= 0.0, np.abs(diff), 0.0)
        upper = np.where(-diff > 0.0, np.abs(diff), 0.0)
        if degree == 0.0:
            return np.asarray(np.mean(diff >= 0.0, axis=1), dtype=np.float64)
        lpm_values = np.mean(np.where(diff >= 0.0, lower**degree, 0.0), axis=1)
        upm_values = np.mean(np.where(-diff > 0.0, upper**degree, 0.0), axis=1)
        return np.asarray(lpm_values / (lpm_values + upm_values), dtype=np.float64)


def _co_lpm_nd_rows(values: NDArray[np.float64], degree: float) -> NDArray[np.float64]:
    """R's ``Co.LPM_nD(values, values, degree, norm = TRUE)`` evaluated at every row."""
    n, d = values.shape
    native = native_fn("clpm_nd_batch")
    if native is not None:
        flat = np.ascontiguousarray(values.ravel(order="F"))
        return np.asarray(native(flat, n, d, flat, n, float(degree), True), dtype=np.float64)
    out = np.empty(n, dtype=np.float64)
    # Pure-NumPy fallback, chunked so memory stays O(chunk * n * d).
    chunk = max(1, 2_000_000 // max(1, n * d))
    for start in range(0, n, chunk):
        rows = values[start : start + chunk]
        diff = values[np.newaxis, :, :] - rows[:, np.newaxis, :]
        lower_mask = np.all(diff <= 0.0, axis=2)
        if degree == 0.0:
            out[start : start + chunk] = np.mean(lower_mask, axis=1)
            continue
        upper_mask = np.all(diff >= 0.0, axis=2)
        abs_pow = np.prod(np.abs(diff) ** degree, axis=2)
        clpm = np.mean(np.where(lower_mask, abs_pow, 0.0), axis=1)
        cupm = np.mean(np.where(upper_mask, abs_pow, 0.0), axis=1)
        dpm = np.mean(np.where(~(lower_mask | upper_mask), abs_pow, 0.0), axis=1)
        total = clpm + cupm + dpm
        out[start : start + chunk] = np.divide(
            clpm, total, out=np.zeros_like(clpm), where=total > 0.0
        )
    return out


def _hazard_proxy(x: NDArray[np.float64], pval: NDArray[np.float64]) -> NDArray[np.float64]:
    n = x.size
    if n == 0:
        return np.array([], dtype=np.float64)
    half_window = min(10, n - 1) // 2
    index = np.arange(n)
    lo = np.maximum(0, index - half_window)
    hi = np.minimum(n - 1, index + half_window)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray((pval[hi] - pval[lo]) / (x[hi] - x[lo]), dtype=np.float64)


def _marginal_lpm_ratios(values: NDArray[np.float64], degree: float) -> NDArray[np.float64]:
    out = np.empty_like(values, dtype=np.float64)
    for col in range(values.shape[1]):
        out[:, col] = _r_lpm_ratio(degree, values[:, col], values[:, col])
    return out
