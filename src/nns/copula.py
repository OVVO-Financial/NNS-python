from __future__ import annotations

import math
import warnings
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from nns._helpers import _as_pair
from nns._native import native_fn
from nns.core import lpm, lpm_ratio
from nns.dependence import _dpm_nd
from nns.pm_matrix import pm_matrix


def nns_copula(
    x: NDArray[np.float64],
    y: NDArray[np.float64] | None = None,
    target_x: float | None = None,
    target_y: float | None = None,
    *,
    continuous: bool = True,
    target: NDArray[np.float64] | Sequence[float] | None = None,
    plot: bool = False,
) -> float:
    """Return R's ``NNS.copula`` higher-dimension dependence value in ``[0, 1]``.

    Two input conventions are supported, matching R's ``NNS.copula(X, ...)``:

    * Bivariate: pass two equal-length 1-D vectors ``x`` and ``y``. The optional
      ``target_x`` / ``target_y`` override the per-column targets (which default
      to the column means).
    * Multivariate: pass a single 2-D matrix ``x`` (with ``y=None``) whose rows
      are observations and whose columns are variables. Any number of columns
      ``>= 2`` is accepted (e.g. three-column inputs). Per-column targets default
      to the column means and can be overridden with ``target``.

    ``continuous=True`` (default) blends the discrete (degree-0) and continuous
    (degree-1) partial-moment dependence measures, exactly as R's
    ``NNS.copula(..., continuous=TRUE)``. ``continuous=False`` reuses the
    discrete partial moments for both terms, matching ``continuous=FALSE``.

    Inputs and targets must be non-empty and finite (R stops on missing or
    non-finite data); a ``ValueError`` is raised otherwise.

    ``plot=True`` also scatters the data with R's orthant colouring
    (``nns.plotting.plot_nns_copula``): a 2-D scatter for two variables and a
    3-D one for three. More than three variables cannot be drawn; a warning
    says so.
    """
    values, targets = _prepare(x, y, target_x, target_y, target)
    result = _copula(values, targets, continuous)
    if plot:
        if values.shape[1] > 3:
            warnings.warn(
                "plot=True draws two or three variables only; nothing is drawn.",
                UserWarning,
                stacklevel=2,
            )
        else:
            from nns.plotting.copula import plot_nns_copula

            plot_nns_copula(values)
    return result


def _copula(
    values: NDArray[np.float64],
    target: NDArray[np.float64],
    continuous: bool,
) -> float:
    if values.ndim != 2 or values.shape[0] == 0:
        # The native kernel assumes at least one observation.
        raise ValueError("copula input must be non-empty.")
    # A constant margin is independent of the rest and has no degree-1 lower share.
    vary = np.any(values != values[0], axis=0)
    if not np.all(vary):
        if np.count_nonzero(vary) < 2:
            return 0.0
        values = np.ascontiguousarray(values[:, vary])
        target = target[vary]
    native_copula = native_fn("copula_nd")
    if native_copula is not None:
        return float(
            native_copula(
                np.ascontiguousarray(np.ravel(values, order="F")),
                values.shape[0],
                values.shape[1],
                np.ascontiguousarray(target),
                bool(continuous),
            )
        )

    n = values.shape[1]
    upper = np.triu_indices(n, k=1)

    discrete_pm_cov = pm_matrix(0.0, 0.0, target, values, pop_adj=False)
    discrete_co_pm = float(
        discrete_pm_cov["cupm"][upper].sum() + discrete_pm_cov["clpm"][upper].sum()
    )
    # Perfect dependence: every pair fully concordant (1) or fully discordant (0).
    pair_co_pm = discrete_pm_cov["cupm"][upper] + discrete_pm_cov["clpm"][upper]
    if np.all((np.abs(pair_co_pm - 1.0) < 1e-9) | (np.abs(pair_co_pm) < 1e-9)):
        return 1.0

    discrete_d_pm = _dpm_nd(values, target, 0.0, norm=True)

    if continuous:
        continuous_pm_cov = pm_matrix(1.0, 1.0, target, values, pop_adj=True, norm=True)
        continuous_co_pm = float(
            continuous_pm_cov["cupm"][upper].sum() + continuous_pm_cov["clpm"][upper].sum()
        )
        continuous_d_pm = _dpm_nd(values, target, 1.0, norm=True)
    else:
        continuous_co_pm = discrete_co_pm
        continuous_d_pm = discrete_d_pm

    # Independence anchors from each margin's own lower-tail share at its target.
    p0 = np.array([float(lpm(0.0, target[j], values[:, j])) for j in range(n)])
    p1 = (
        np.array([float(lpm_ratio(1.0, target[j], values[:, j])) for j in range(n)])
        if continuous
        else p0
    )
    discrete_co_anchor, discrete_d_anchor = _independence_anchors(p0, upper)
    continuous_co_anchor, continuous_d_anchor = _independence_anchors(p1, upper)
    discrete_dep = min(max(_relative_deviation(discrete_co_pm, discrete_co_anchor), 0.0), 1.0)
    continuous_dep = min(
        max(_relative_deviation(continuous_co_pm, continuous_co_anchor), 0.0), 1.0
    )
    n_dim_discrete_dep = min(_relative_deviation(discrete_d_pm, discrete_d_anchor), 1.0)
    n_dim_continuous_dep = min(_relative_deviation(continuous_d_pm, continuous_d_anchor), 1.0)

    return math.sqrt(
        (discrete_dep + continuous_dep + n_dim_discrete_dep + n_dim_continuous_dep) / 4.0
    )


def _independence_anchors(
    p: NDArray[np.float64], upper: tuple[NDArray[np.intp], NDArray[np.intp]]
) -> tuple[float, float]:
    """Pairwise co-moment sum and nD divergent share under independence (R NNS.copula)."""
    co = float(np.sum((np.outer(p, p) + np.outer(1.0 - p, 1.0 - p))[upper]))
    return co, 1.0 - float(np.prod(p)) - float(np.prod(1.0 - p))


def _relative_deviation(observed: float, anchor: float) -> float:
    # A boundary target can zero an anchor; an observed 0 then matches it exactly.
    if observed == anchor:
        return 0.0
    with np.errstate(divide="ignore"):
        return float(np.divide(abs(observed - anchor), anchor))


def _prepare(
    x: NDArray[np.float64],
    y: NDArray[np.float64] | None,
    target_x: float | None,
    target_y: float | None,
    target: NDArray[np.float64] | Sequence[float] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if y is None:
        values = _as_matrix(x)
        if target_x is not None or target_y is not None:
            raise ValueError("target_x/target_y only apply to the bivariate (x, y) form.")
        targets = _matrix_target(values, target)
        return values, targets

    if target is not None:
        raise ValueError("Use target_x/target_y (not target) with the bivariate (x, y) form.")
    x_values, y_values = _as_pair(x, y)
    values = np.column_stack((x_values, y_values))
    targets = _r_col_means(values)
    if target_x is not None:
        targets[0] = float(target_x)
    if target_y is not None:
        targets[1] = float(target_y)
    if not np.all(np.isfinite(targets)):
        raise ValueError("target must contain only finite values.")
    return values, targets


def _as_matrix(x: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Multivariate copula input must be a 2D matrix (rows=observations).")
    if values.shape[0] == 0:
        raise ValueError("copula input must be non-empty.")
    if values.shape[1] < 2:
        raise ValueError("copula requires at least two variables (columns).")
    if not np.all(np.isfinite(values)):
        raise ValueError("copula input must contain only finite values.")
    return values


def _matrix_target(
    values: NDArray[np.float64],
    target: NDArray[np.float64] | Sequence[float] | None,
) -> NDArray[np.float64]:
    if target is None:
        return _r_col_means(values)
    targets = np.asarray(target, dtype=np.float64).reshape(-1)
    if targets.size != values.shape[1]:
        raise ValueError("target length must match the number of variables (columns).")
    if not np.all(np.isfinite(targets)):
        raise ValueError("target must contain only finite values.")
    return targets


def _r_col_means(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Column means computed exactly as R's ``colMeans`` on a double matrix.

    R accumulates each column sequentially in long double and divides in long
    double before rounding to double. ``np.mean`` uses pairwise double summation
    and can differ in the last ulp, which is enough to flip the copula's
    exact-equality shortcut when an observation sits on the mean.
    On platforms where ``np.longdouble`` is plain double this degrades to a
    sequential double sum.
    """
    n = values.shape[0]
    out = np.empty(values.shape[1], dtype=np.float64)
    for column in range(values.shape[1]):
        total = np.cumsum(values[:, column], dtype=np.longdouble)[-1]
        out[column] = float(total / n)
    return out
