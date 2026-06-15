from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast

import numpy as np
from numpy.typing import NDArray

from nns._native import nnscore
from nns.co_moments import _as_pair
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
    """
    values, targets = _prepare(x, y, target_x, target_y, target)
    return _copula(values, targets, continuous)


def _copula(
    values: NDArray[np.float64],
    target: NDArray[np.float64],
    continuous: bool,
) -> float:
    native = nnscore()
    if native is not None and hasattr(native, "copula_nd"):
        return float(
            native.copula_nd(
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
    if discrete_co_pm == 1.0 or discrete_co_pm == 0.0:
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

    indep_co_pm = 0.25 * (n**2 - n)
    discrete_dep = min(max(abs(discrete_co_pm - indep_co_pm) / indep_co_pm, 0.0), 1.0)
    continuous_dep = min(max(abs(continuous_co_pm - indep_co_pm) / indep_co_pm, 0.0), 1.0)

    indep_d_pm = 1.0 - 0.5**n
    n_dim_discrete_dep = abs(discrete_d_pm - indep_d_pm) / indep_d_pm
    n_dim_continuous_dep = abs(continuous_d_pm - indep_d_pm) / indep_d_pm

    return math.sqrt(
        (discrete_dep + continuous_dep + n_dim_discrete_dep + n_dim_continuous_dep) / 4.0
    )


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
    targets = cast(NDArray[np.float64], np.mean(values, axis=0))
    if target_x is not None:
        targets[0] = float(target_x)
    if target_y is not None:
        targets[1] = float(target_y)
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
        return cast(NDArray[np.float64], np.mean(values, axis=0))
    targets = np.asarray(target, dtype=np.float64).reshape(-1)
    if targets.size != values.shape[1]:
        raise ValueError("target length must match the number of variables (columns).")
    if not np.all(np.isfinite(targets)):
        raise ValueError("target must contain only finite values.")
    return targets
