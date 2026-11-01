from __future__ import annotations

import itertools
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
    # As R's NNS.CDF: drop missing observations (complete rows for a matrix) so the
    # evaluation grid and the denominator agree; infinite values are rejected below.
    if values.ndim == 1:
        values = values[~np.isnan(values)]
    elif values.ndim == 2:
        values = values[~np.isnan(values).any(axis=1)]
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
        raise ValueError("variable must contain at least one non-missing value.")
    if not np.all(np.isfinite(values)):
        raise ValueError("variable must contain only finite values.")
    target_value = _univariate_target(target, values)
    x = np.sort(values)
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
        raise ValueError("variable must have at least one complete row and one column.")
    if not np.all(np.isfinite(values)):
        raise ValueError("variable must contain only finite values.")
    target_values = _multivariate_target(target, values)
    column_names = _matrix_names(values.shape[1], names)

    cdf = _co_lpm_nd_rows(values, degree)
    survival = np.array([], dtype=np.float64)
    if type_value != "cdf":
        survival = _joint_survival(values, values, cdf, degree)
    fit: RegResult | MRegResult | None = None
    if type_value == "survival":
        cdf = np.maximum(0.0, np.minimum(1.0, survival))
    elif type_value == "hazard":
        fit = nns_reg(
            values, np.maximum(cdf, 1e-10), order="max", plot=False, point_est=target_values
        )
        fitted = cast(dict[str, NDArray[np.float64]], fit["Fitted.xy"])
        cdf = np.maximum(fitted["y.hat"] / np.maximum(survival, 1e-10), 0.0)
    elif type_value == "cumulative hazard":
        cdf = np.maximum(-np.log(np.maximum(survival, 1e-10)), 0.0)

    # The target value is the same function as the rows, evaluated at target.
    pv = np.array([], dtype=np.float64)
    if target_values is not None:
        target_cdf = float(co_lpm_nd(values, target_values, degree=degree))
        target_survival = 0.0
        if type_value != "cdf":
            target_survival = float(
                _joint_survival(
                    values, target_values.reshape(1, -1), np.array([target_cdf]), degree
                )[0]
            )
        if type_value == "cdf":
            pv = np.array([target_cdf], dtype=np.float64)
        elif type_value == "survival":
            pv = np.array([max(0.0, min(1.0, target_survival))], dtype=np.float64)
        elif type_value == "hazard":
            if fit is None:
                raise RuntimeError("hazard fit was not computed.")
            point = float(np.asarray(fit["Point.est"], dtype=np.float64).reshape(-1)[0])
            pv = np.array([max(point / max(target_survival, 1e-10), 0.0)], dtype=np.float64)
        elif type_value == "cumulative hazard":
            pv = np.array(
                [max(-math.log(max(target_survival, 1e-10)), 0.0)], dtype=np.float64
            )

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
    if not np.isfinite(target_value):
        raise ValueError("target must be finite.")
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
    if target_values.size != values.shape[1]:
        raise ValueError("target must have one value per column of variable")
    if not np.all(np.isfinite(target_values)):
        raise ValueError("target must be finite.")
    if np.any((target_values < np.min(values, axis=0)) | (target_values > np.max(values, axis=0))):
        raise ValueError("target out of bounds")
    return target_values


def _matrix_names(column_count: int, names: Sequence[str] | None) -> list[str]:
    if names is None:
        # R's NNS.CDF labels an unnamed matrix's columns by their index ("1", "2").
        return [str(index + 1) for index in range(column_count)]
    column_names = [str(name) for name in names]
    if len(column_names) != column_count:
        raise ValueError("names length must match the number of columns in variable.")
    # R keeps duplicate data.frame names; a dict cannot, so suffix them like make.unique.
    return _make_unique(["CDF", *column_names])[1:]


def _make_unique(names: list[str]) -> list[str]:
    """R's ``make.unique``: later duplicates get the first free ``.1``, ``.2``, ... suffix."""
    taken = set(names)
    seen: set[str] = set()
    counters: dict[str, int] = {}
    out: list[str] = []
    for name in names:
        if name in seen:
            suffix = counters.get(name, 0)
            while True:
                suffix += 1
                candidate = f"{name}.{suffix}"
                if candidate not in taken:
                    break
            counters[name] = suffix
            taken.add(candidate)
            name = candidate
        seen.add(name)
        out.append(name)
    return out


def _r_lpm_ratio(
    degree: float,
    targets: NDArray[np.float64],
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """R's ``LPM.ratio(degree, targets, values)`` on finite ``values``."""
    target_values = np.asarray(targets, dtype=np.float64).reshape(-1)
    if target_values.size == 0:
        return np.array([], dtype=np.float64)
    return np.asarray(lpm_ratio(degree, target_values, values), dtype=np.float64)


def _co_lpm_nd_rows(values: NDArray[np.float64], degree: float) -> NDArray[np.float64]:
    """R's ``Co.LPM_nD(values, values, degree, norm = TRUE)`` evaluated at every row."""
    return _co_lpm_nd_batch(values, values, degree)


def _co_lpm_nd_batch(
    values: NDArray[np.float64], points: NDArray[np.float64], degree: float
) -> NDArray[np.float64]:
    """R's ``Co.LPM_nD.batch(values, points, degree, norm = TRUE)``: one value per point row."""
    n, d = values.shape
    m = points.shape[0]
    native = native_fn("clpm_nd_batch")
    if native is not None:
        flat = np.ascontiguousarray(values.ravel(order="F"))
        flat_points = np.ascontiguousarray(points.ravel(order="F"))
        return np.asarray(
            native(flat, n, d, flat_points, m, float(degree), True), dtype=np.float64
        )
    out = np.empty(m, dtype=np.float64)
    # Pure-NumPy fallback, chunked so memory stays O(chunk * n * d).
    chunk = max(1, 2_000_000 // max(1, n * d))
    for start in range(0, m, chunk):
        rows = points[start : start + chunk]
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


def _joint_survival(
    values: NDArray[np.float64],
    points: NDArray[np.float64],
    joint_cdf: NDArray[np.float64],
    degree: float,
) -> NDArray[np.float64]:
    """Joint survival ``P(X_1 > t_1, ..., X_d > t_d)`` at each row of ``points``.

    Inclusion-exclusion over all subsets A of the coordinates,
    ``S = sum_A (-1)^|A| F_A(t_A)`` with ``F_A`` the joint CDF of the coordinates in A
    (``F_{}`` = 1, the full set is ``joint_cdf``); for d = 2 this is
    ``1 - F_1 - F_2 + F_12``.  Mirrors R NNS.CDF's ``joint.survival``.
    """
    d = values.shape[1]
    marginals = np.column_stack(
        [_r_lpm_ratio(degree, points[:, col], values[:, col]) for col in range(d)]
    )
    survival = 1.0 - np.sum(marginals, axis=1)
    for k in range(2, d):
        sign = -1.0 if k % 2 else 1.0
        for subset in itertools.combinations(range(d), k):
            cols = list(subset)
            survival = survival + sign * _co_lpm_nd_batch(
                np.ascontiguousarray(values[:, cols]), np.ascontiguousarray(points[:, cols]), degree
            )
    return np.asarray(survival + (-1.0) ** d * joint_cdf, dtype=np.float64)


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

