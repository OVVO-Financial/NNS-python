from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline  # type: ignore[import-untyped]


class RSmoothSpline:
    def __init__(
        self,
        *,
        knots: NDArray[np.float64],
        coef: NDArray[np.float64],
        x_min: float,
        x_range: float,
    ) -> None:
        self._spline = BSpline(knots, coef, 3, extrapolate=True)
        self.x_min = x_min
        self.x_range = x_range

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        values = np.asarray(x, dtype=np.float64)
        scaled = (values - self.x_min) / self.x_range
        return np.asarray(self._spline(scaled), dtype=np.float64)


def r_smooth_spline_fixed_spar(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    spar: float,
) -> RSmoothSpline:
    """Fit the fixed-spar subset of R's stats::smooth.spline used by NNS.reg."""
    x_values = np.asarray(x, dtype=np.float64).reshape(-1)
    y_values = np.asarray(y, dtype=np.float64).reshape(-1)
    if x_values.size != y_values.size:
        raise ValueError("x and y must have the same length.")
    if not np.all(np.isfinite(x_values)) or not np.all(np.isfinite(y_values)):
        raise ValueError("x and y must contain only finite values.")

    unique_x, y_bar, weights = _r_unique_xy(x_values, y_values)
    if unique_x.size <= 3:
        raise ValueError("need at least four unique x values")
    x_range = float(unique_x[-1] - unique_x[0])
    if x_range <= 0.0:
        raise ValueError("x must span a positive range.")
    x_scaled = (unique_x - unique_x[0]) / x_range
    knots = _r_knot_sequence(x_scaled)
    n_coef = knots.size - 4

    basis = _basis_matrix(knots, x_scaled, n_coef)
    sigma = _sigma_matrix(knots, n_coef)
    weighted_basis = basis * weights[:, np.newaxis]
    xwx = basis.T @ weighted_basis
    xwy = basis.T @ (weights * y_bar)

    interior = slice(2, n_coef - 3)
    sigma_trace = float(np.sum(np.diag(sigma)[interior]))
    xwx_trace = float(np.sum(np.diag(xwx)[interior]))
    if sigma_trace == 0.0:
        raise ValueError("smoothing spline penalty matrix is degenerate.")
    ratio = xwx_trace / sigma_trace
    lam = ratio * (16.0 ** (6.0 * float(spar) - 2.0))
    coef = np.linalg.solve(xwx + lam * sigma, xwy)
    return RSmoothSpline(
        knots=knots,
        coef=coef.astype(np.float64),
        x_min=float(unique_x[0]),
        x_range=x_range,
    )


def _r_unique_xy(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    tol = 1e-6 * _iqr(x)
    if not math.isfinite(tol) or tol <= 0.0:
        raise ValueError("'tol' must be strictly positive and finite")
    rounded = np.round((x - float(np.mean(x))) / tol)
    order = np.argsort(x, kind="mergesort")
    x_ordered = x[order]
    y_ordered = y[order]
    rounded_ordered = rounded[order]
    groups = np.concatenate(([0], np.flatnonzero(rounded_ordered[:-1] < rounded_ordered[1:]) + 1))
    unique_x = x_ordered[groups]
    counts = np.diff(np.concatenate((groups, [x.size]))).astype(np.float64)
    y_bar = np.empty(groups.size, dtype=np.float64)
    for index, start in enumerate(groups):
        stop = groups[index + 1] if index + 1 < groups.size else x.size
        y_bar[index] = float(np.mean(y_ordered[start:stop]))
    return unique_x.astype(np.float64), y_bar, counts


def _iqr(values: NDArray[np.float64]) -> float:
    quantiles = np.quantile(values, [0.25, 0.75], method="linear")
    return float(quantiles[1] - quantiles[0])


def _r_knot_sequence(x_scaled: NDArray[np.float64]) -> NDArray[np.float64]:
    n = x_scaled.size
    nknots = _r_nknots_smspl(n)
    if nknots == n:
        inner = x_scaled
    else:
        indices = np.trunc(np.linspace(1.0, float(n), nknots)).astype(np.int64) - 1
        inner = x_scaled[indices]
    return np.concatenate(
        (
            np.repeat(x_scaled[0], 3),
            inner,
            np.repeat(x_scaled[-1], 3),
        )
    ).astype(np.float64)


def _r_nknots_smspl(n: int) -> int:
    if n < 50:
        return n
    a1 = math.log2(50)
    a2 = math.log2(100)
    a3 = math.log2(140)
    a4 = math.log2(200)
    if n < 200:
        return _trunc_int(2.0 ** (a1 + (a2 - a1) * (n - 50) / 150))
    if n < 800:
        return _trunc_int(2.0 ** (a2 + (a3 - a2) * (n - 200) / 600))
    if n < 3200:
        return _trunc_int(2.0 ** (a3 + (a4 - a3) * (n - 800) / 2400))
    return _trunc_int(200 + (n - 3200) ** 0.2)


def _trunc_int(value: float) -> int:
    return math.trunc(value)


def _basis_matrix(
    knots: NDArray[np.float64],
    x_scaled: NDArray[np.float64],
    n_coef: int,
) -> NDArray[np.float64]:
    eye = np.eye(n_coef, dtype=np.float64)
    columns = [BSpline(knots, eye[index], 3, extrapolate=True)(x_scaled) for index in range(n_coef)]
    return np.column_stack(columns).astype(np.float64)


def _sigma_matrix(knots: NDArray[np.float64], n_coef: int) -> NDArray[np.float64]:
    eye = np.eye(n_coef, dtype=np.float64)
    second = [
        BSpline(knots, eye[index], 3, extrapolate=True).derivative(2) for index in range(n_coef)
    ]
    sigma = np.zeros((n_coef, n_coef), dtype=np.float64)
    nodes, weights = np.polynomial.legendre.leggauss(3)
    unique_knots = np.unique(knots)
    for left, right in pairwise(unique_knots):
        if right <= left:
            continue
        mid = 0.5 * (left + right)
        half = 0.5 * (right - left)
        eval_points = mid + half * nodes
        values = np.asarray([fn(eval_points) for fn in second], dtype=np.float64)
        sigma += half * (values * weights[np.newaxis, :]) @ values.T
    return sigma
