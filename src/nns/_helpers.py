from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from nns._native import nnscore


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

    native = nnscore()
    if native is not None and hasattr(native, "fast_lm"):
        result = native.fast_lm(np.ascontiguousarray(x_values), np.ascontiguousarray(y_values))
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
