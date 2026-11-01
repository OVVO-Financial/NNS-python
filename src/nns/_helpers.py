from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray

from nns._native import external_stacklevel, native_fn


def _warn_unsupported(**was_set: bool) -> None:
    """Warn about R-compatibility parameters that have no effect in NNS Python.

    Each keyword names a parameter; pass True when the caller deviated from the
    default and would therefore expect the parameter to do something.
    """
    ignored = [name for name, flag in was_set.items() if flag]
    if ignored:
        warnings.warn(
            f"{', '.join(ignored)}: accepted for R NNS API compatibility "
            "but not implemented in NNS Python; ignored.",
            UserWarning,
            stacklevel=external_stacklevel(),
        )


def _as_matrix(x: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    """Return x as a non-empty, all-finite 2D float64 matrix."""
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"{name} must be 2D.")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")
    return values


def _as_pair(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return x and y as equal-length, non-empty, all-finite 1D float64 vectors."""
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    if x_values.ndim != 1 or y_values.ndim != 1:
        raise ValueError("x and y must be 1D.")
    if x_values.size == 0:
        raise ValueError("x and y must be non-empty.")
    if x_values.size != y_values.size:
        raise ValueError("x and y must have the same length.")
    if not np.all(np.isfinite(x_values)) or not np.all(np.isfinite(y_values)):
        raise ValueError("x and y must contain only finite values.")
    return x_values, y_values


def _fast_lm(x: NDArray[np.float64], y: NDArray[np.float64]) -> tuple[float, float]:
    """Return intercept and slope matching R's fast_lm helper."""
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    if x_values.ndim != 1 or y_values.ndim != 1:
        raise ValueError("x and y must be 1D.")
    if x_values.size != y_values.size:
        raise ValueError("x and y must have the same length.")
    if x_values.size == 0:
        raise ValueError("x and y must be non-empty.")

    native_fast_lm = native_fn("fast_lm")
    if native_fast_lm is not None:
        result = native_fast_lm(np.ascontiguousarray(x_values), np.ascontiguousarray(y_values))
        coef = result["coef"]
        return float(coef[0]), float(coef[1])

    # Same arithmetic as NNS-core's fast_lm (R's Rcpp kernel): long-double two-pass
    # means and sequential double sums. np.mean/np.sum sum pairwise and can differ in
    # the last bit, which reorders tied residuals (e.g. nns_meboot's ordxx).
    from nns._rmean import r_mean

    mean_x = r_mean(x_values)
    mean_y = r_mean(y_values)
    dx = x_values - mean_x
    dy = y_values - mean_y
    var_x = float(np.cumsum(dx * dx)[-1])
    if var_x == 0.0:
        return mean_y, 0.0

    slope = float(np.cumsum(dx * dy)[-1]) / var_x
    intercept = mean_y - slope * mean_x
    return intercept, slope


def _is_fcl(x: object) -> bool:
    """Return whether x maps to R factor/character/logical input."""
    values = np.asarray(x)
    if values.dtype.kind in {"O", "S", "U", "b"}:
        return True
    return False
