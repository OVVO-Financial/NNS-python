from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray

from nns._native import native_fn


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
            stacklevel=3,
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


def _as_vector_or_matrix(x: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    """Return x as a non-empty, all-finite 2D float64 matrix, promoting 1D to a column."""
    values = np.asarray(x, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty numeric vector or matrix.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")
    return values


def _as_point_matrix(x: NDArray[np.float64], n_cols: int) -> NDArray[np.float64]:
    """Return evaluation points as an all-finite matrix with the training column count."""
    values = np.asarray(x, dtype=np.float64)
    if values.ndim == 1:
        if n_cols == 1:
            values = values.reshape(-1, 1)
        else:
            values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != n_cols:
        raise ValueError("ivs_test must have the same column count as ivs_train.")
    if not np.all(np.isfinite(values)):
        raise ValueError("ivs_test must contain only finite values.")
    return values


def _as_flat_vector(x: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    """Return x flattened to a non-empty, all-finite 1D float64 vector."""
    values = np.asarray(x, dtype=np.float64).reshape(-1)
    if values.size == 0:
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

    mean_x = float(np.mean(x_values))
    mean_y = float(np.mean(y_values))
    dx = x_values - mean_x
    var_x = float(np.sum(dx * dx))
    if var_x == 0.0:
        return mean_y, 0.0

    slope = float(np.sum(dx * (y_values - mean_y)) / var_x)
    intercept = mean_y - slope * mean_x
    return intercept, slope


def _is_fcl(x: object) -> bool:
    """Return whether x maps to R factor/character/logical input."""
    values = np.asarray(x)
    if values.dtype.kind in {"O", "S", "U", "b"}:
        return True
    return False
