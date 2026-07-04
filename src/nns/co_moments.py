from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from nns._native import native_fn
from nns.core import _as_degree, _as_targets


def co_lpm(
    degree_lpm: float,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    target_x: float | NDArray[np.float64],
    target_y: float | NDArray[np.float64],
    degree_y: float | None = None,
) -> float | NDArray[np.float64]:
    degree_y = degree_lpm if degree_y is None else degree_y
    return _co_moment(_lower, _lower, degree_lpm, degree_y, x, y, target_x, target_y)


def co_upm(
    degree_upm: float,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    target_x: float | NDArray[np.float64],
    target_y: float | NDArray[np.float64],
    degree_y: float | None = None,
) -> float | NDArray[np.float64]:
    degree_y = degree_upm if degree_y is None else degree_y
    return _co_moment(_upper, _upper, degree_upm, degree_y, x, y, target_x, target_y)


def d_lpm(
    degree_lpm: float,
    degree_upm: float,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    target_x: float | NDArray[np.float64],
    target_y: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    return _co_moment(_upper, _lower, degree_upm, degree_lpm, x, y, target_x, target_y)


def d_upm(
    degree_lpm: float,
    degree_upm: float,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    target_x: float | NDArray[np.float64],
    target_y: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    return _co_moment(_lower, _upper, degree_lpm, degree_upm, x, y, target_x, target_y)


def _co_moment(
    x_side: object,
    y_side: object,
    degree_x: float,
    degree_y: float,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    target_x: float | NDArray[np.float64],
    target_y: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    x_values, y_values = _as_pair(x, y)
    x_targets = _as_targets(target_x)
    y_targets = _as_targets(target_y)
    degree_x = _as_degree(degree_x)
    degree_y = _as_degree(degree_y)

    native, swap_degrees = _native_co_moment(x_side, y_side)
    if native is not None and x_targets.size > 0 and y_targets.size > 0:
        first_degree, second_degree = (
            (degree_y, degree_x) if swap_degrees else (degree_x, degree_y)
        )
        native_result = native(
            first_degree,
            second_degree,
            np.ascontiguousarray(x_values),
            np.ascontiguousarray(y_values),
            np.ascontiguousarray(x_targets),
            np.ascontiguousarray(y_targets),
        )
        moments = np.asarray(native_result, dtype=np.float64).reshape(-1)
        if np.asarray(target_x).ndim == 0 and np.asarray(target_y).ndim == 0:
            return float(moments[0])
        return moments

    target_count = max(x_targets.size, y_targets.size)
    moments = np.empty(target_count, dtype=np.float64)
    for index in range(target_count):
        x_target = x_targets[index % x_targets.size]
        y_target = y_targets[index % y_targets.size]
        x_deviation = _deviation(x_side, degree_x, x_target, x_values)
        y_deviation = _deviation(y_side, degree_y, y_target, y_values)
        moments[index] = np.mean(x_deviation * y_deviation)

    if np.asarray(target_x).ndim == 0 and np.asarray(target_y).ndim == 0:
        return float(moments[0])
    return moments


def _native_co_moment(x_side: object, y_side: object) -> tuple[Any | None, bool]:
    """Return the native kernel for a side pair and whether it swaps degree order."""
    if x_side is _lower and y_side is _lower:
        return native_fn("co_lpm_v"), False
    if x_side is _upper and y_side is _upper:
        return native_fn("co_upm_v"), False
    if x_side is _upper and y_side is _lower:
        return native_fn("d_lpm_v"), True
    return native_fn("d_upm_v"), False


def _as_pair(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    if x_values.ndim != 1 or y_values.ndim != 1:
        raise ValueError("x and y must be 1D.")
    if x_values.size != y_values.size:
        raise ValueError("x and y must have the same length.")
    return x_values, y_values


def _deviation(
    side: object,
    degree: float,
    target: float,
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    if side is _lower:
        if degree == 0:
            return (values <= target).astype(np.float64)
        return np.maximum(0.0, target - values) ** degree

    if degree == 0:
        return (values > target).astype(np.float64)
    return np.maximum(0.0, values - target) ** degree


_lower = object()
_upper = object()
