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
