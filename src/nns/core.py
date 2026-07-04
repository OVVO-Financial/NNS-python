from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from nns._native import native_fn


def lpm(
    degree: float,
    target: float | NDArray[np.float64],
    x: NDArray[np.float64],
) -> float | NDArray[np.float64]:
    return _moment(degree, target, x, lower=True)


def upm(
    degree: float,
    target: float | NDArray[np.float64],
    x: NDArray[np.float64],
) -> float | NDArray[np.float64]:
    return _moment(degree, target, x, lower=False)


def lpm_ratio(
    degree: float,
    target: float | NDArray[np.float64],
    x: NDArray[np.float64],
) -> float | NDArray[np.float64]:
    return _ratio(degree, target, x, lower=True)


def upm_ratio(
    degree: float,
    target: float | NDArray[np.float64],
    x: NDArray[np.float64],
) -> float | NDArray[np.float64]:
    return _ratio(degree, target, x, lower=False)


def _moment(
    degree: float,
    target: float | NDArray[np.float64],
    x: NDArray[np.float64],
    *,
    lower: bool,
) -> float | NDArray[np.float64]:
    values = _as_1d_values(x)
    targets = _as_targets(target)
    degree = _as_degree(degree)

    native = native_fn("lpm" if lower else "upm")
    if native is not None and targets.size > 0 and _native_safe(values, targets):
        native_target = float(targets[0]) if np.asarray(target).ndim == 0 else targets
        native_result = native(degree, native_target, np.ascontiguousarray(values))
        return _result_for_target(np.asarray(native_result, dtype=np.float64).reshape(-1), target)

    if degree == 0:
        # R convention: equality with the target counts toward the lower moment.
        grid = targets[:, np.newaxis]
        moments = np.mean(values <= grid if lower else values > grid, axis=1)
        return _result_for_target(moments, target)

    deviations = targets[:, np.newaxis] - values if lower else values - targets[:, np.newaxis]
    moments = np.mean(np.maximum(0.0, deviations) ** degree, axis=1)
    return _result_for_target(moments, target)


def _ratio(
    degree: float,
    target: float | NDArray[np.float64],
    x: NDArray[np.float64],
    *,
    lower: bool,
) -> float | NDArray[np.float64]:
    values = _as_1d_values(x)
    targets = _as_targets(target)
    degree = _as_degree(degree)

    native = native_fn("lpm_ratio_v" if lower else "upm_ratio_v")
    if native is not None and targets.size > 0 and _native_safe(values, targets):
        native_result = native(
            degree,
            np.ascontiguousarray(targets),
            np.ascontiguousarray(values),
        )
        return _result_for_target(np.asarray(native_result, dtype=np.float64).reshape(-1), target)

    if degree == 0:
        return _moment(degree, target, x, lower=lower)

    lower_moment = np.asarray(lpm(degree, target, x))
    upper_moment = np.asarray(upm(degree, target, x))
    numerator = lower_moment if lower else upper_moment
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = numerator / (lower_moment + upper_moment)
    return _result_for_target(np.asarray(ratio).reshape(-1), target)


def _native_safe(
    values: NDArray[np.float64],
    targets: NDArray[np.float64],
) -> bool:
    return bool(np.all(np.isfinite(values)) and np.all(np.isfinite(targets)))


def _as_1d_values(x: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("x must be 1D.")
    if values.size == 0:
        raise ValueError("x must be non-empty.")
    return values


def _as_targets(target: float | NDArray[np.float64]) -> NDArray[np.float64]:
    targets = np.asarray(target, dtype=np.float64)
    if targets.ndim == 0:
        return targets.reshape(1)
    if targets.ndim != 1:
        raise ValueError("target must be scalar or 1D.")
    return targets


def _as_degree(degree: float) -> float:
    degree = float(degree)
    if degree < 0:
        raise ValueError("degree must be non-negative.")
    return degree


def _result_for_target(
    moments: NDArray[np.float64],
    target: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    if np.asarray(target).ndim == 0:
        return float(moments[0])
    return moments
