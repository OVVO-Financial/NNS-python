from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from nns._native import native_fn
from nns.core import _as_degree, _as_targets, _is_scalar, _require_finite


def co_lpm(
    degree_lpm: float,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    target_x: float | NDArray[np.float64] | None,
    target_y: float | NDArray[np.float64] | None,
    degree_y: float | None = None,
) -> float | NDArray[np.float64]:
    """Co-lower partial moment (R ``Co.LPM``).

    ``mean(max(target_x - x, 0) ** degree_lpm * max(target_y - y, 0) ** degree_y)``:
    both variables at or below their targets (degree 0 counts ``x <= target_x`` and
    ``y <= target_y``). ``degree_y`` defaults to ``degree_lpm``. Targets may be
    scalars (returns a float) or 1D arrays (returns an array); ``None`` uses the mean.
    ``x`` and ``y`` must be non-empty and the same length, and ``target_x`` and
    ``target_y`` non-empty and the same length or one of them length 1.
    """
    degree_y = degree_lpm if degree_y is None else degree_y
    return _co_moment(_lower, _lower, degree_lpm, degree_y, x, y, target_x, target_y)


def co_upm(
    degree_upm: float,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    target_x: float | NDArray[np.float64] | None,
    target_y: float | NDArray[np.float64] | None,
    degree_y: float | None = None,
) -> float | NDArray[np.float64]:
    """Co-upper partial moment (R ``Co.UPM``).

    ``mean(max(x - target_x, 0) ** degree_upm * max(y - target_y, 0) ** degree_y)``:
    both variables above their targets (degree 0 counts ``x > target_x`` and
    ``y > target_y``). ``degree_y`` defaults to ``degree_upm``. Targets may be
    scalars or 1D arrays; ``None`` uses the mean.
    """
    degree_y = degree_upm if degree_y is None else degree_y
    return _co_moment(_upper, _upper, degree_upm, degree_y, x, y, target_x, target_y)


def d_lpm(
    degree_lpm: float,
    degree_upm: float,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    target_x: float | NDArray[np.float64] | None,
    target_y: float | NDArray[np.float64] | None,
) -> float | NDArray[np.float64]:
    """Divergent lower partial moment (R ``D.LPM``).

    ``mean(max(x - target_x, 0) ** degree_upm * max(target_y - y, 0) ** degree_lpm)``:
    ``x`` above its target while ``y`` is at or below its target. Together with
    ``co_lpm``, ``co_upm`` and ``d_upm`` the four quadrants partition the data, and
    ``co_lpm + co_upm - d_lpm - d_upm`` at degree 1 and the means is the population
    covariance. Targets may be scalars or 1D arrays; ``None`` uses the mean.
    """
    return _co_moment(_upper, _lower, degree_upm, degree_lpm, x, y, target_x, target_y)


def d_upm(
    degree_lpm: float,
    degree_upm: float,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    target_x: float | NDArray[np.float64] | None,
    target_y: float | NDArray[np.float64] | None,
) -> float | NDArray[np.float64]:
    """Divergent upper partial moment (R ``D.UPM``).

    ``mean(max(target_x - x, 0) ** degree_lpm * max(y - target_y, 0) ** degree_upm)``:
    ``x`` at or below its target while ``y`` is above its target. Targets may be
    scalars or 1D arrays; ``None`` uses the mean.
    """
    return _co_moment(_lower, _upper, degree_lpm, degree_upm, x, y, target_x, target_y)


def _co_moment(
    x_side: object,
    y_side: object,
    degree_x: float,
    degree_y: float,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    target_x: float | NDArray[np.float64] | None,
    target_y: float | NDArray[np.float64] | None,
) -> float | NDArray[np.float64]:
    x_values, y_values = _as_pair(x, y)
    # ``None`` targets default to the variable means, as in R's Co.LPM/D.LPM wrappers.
    x_targets = _as_targets(target_x, x_values)
    y_targets = _as_targets(target_y, y_values)
    if x_targets.size == 0 or y_targets.size == 0:
        raise ValueError("target_x and target_y must be non-empty.")
    if x_targets.size != y_targets.size and 1 not in (x_targets.size, y_targets.size):
        raise ValueError(
            "target_x and target_y must have the same length, or one of them length 1."
        )
    # As R's Co.LPM/Co.UPM/D.LPM/D.UPM: NaN would be counted in n but in no quadrant.
    _require_finite(x_targets, "target_x")
    _require_finite(y_targets, "target_y")
    degree_x = _as_degree(degree_x)
    degree_y = _as_degree(degree_y)

    native, swap_degrees = _native_co_moment(x_side, y_side)
    if native is not None:
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
        if _is_scalar(target_x) and _is_scalar(target_y):
            return float(moments[0])
        return moments

    target_count = max(x_targets.size, y_targets.size)
    moments = np.empty(target_count, dtype=np.float64)
    for index in range(target_count):
        x_target = x_targets[index % x_targets.size]
        y_target = y_targets[index % y_targets.size]
        x_deviation = _deviation(x_side, degree_x, x_target, x_values)
        y_deviation = _deviation(y_side, degree_y, y_target, y_values)
        # R's C++ accumulates x1 * y1 sequentially in double and divides by n.
        moments[index] = float(np.cumsum(x_deviation * y_deviation)[-1]) / x_values.size

    if _is_scalar(target_x) and _is_scalar(target_y):
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
    if x_values.size == 0:
        raise ValueError("x and y must be non-empty.")
    _require_finite(x_values, "x")
    _require_finite(y_values, "y")
    return x_values, y_values


def _deviation(
    side: object,
    degree: float,
    target: float,
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """R's C++ per-observation term: an indicator at degree 0, otherwise the
    clamped deviation raised by repeated multiplication (integer degrees) or
    libm ``pow``."""
    from nns.dependence import _r_pm_terms

    if side is _lower:
        if degree == 0:
            return (values <= target).astype(np.float64)
        return _r_pm_terms(np.maximum(0.0, target - values), degree)

    if degree == 0:
        return (values > target).astype(np.float64)
    return _r_pm_terms(np.maximum(0.0, values - target), degree)


_lower = object()
_upper = object()
