from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from nns._native import native_fn

# Targets per block in the NumPy fallback, so memory stays O(block * n) rather than n x n.
_TARGET_BLOCK = 1024


def lpm(
    degree: float,
    target: float | NDArray[np.float64] | None,
    x: NDArray[np.float64],
) -> float | NDArray[np.float64]:
    """Lower partial moment ``mean(max(target - x, 0) ** degree)`` (R ``LPM``).

    ``degree=0`` gives the share of observations ``<= target`` (ties count as lower).
    ``target`` may be a scalar (returns a float) or a 1D array (returns an array);
    ``None`` uses ``mean(x)``.  ``x`` must be finite.
    """
    return _moment(degree, target, x, lower=True)


def upm(
    degree: float,
    target: float | NDArray[np.float64] | None,
    x: NDArray[np.float64],
) -> float | NDArray[np.float64]:
    """Upper partial moment ``mean(max(x - target, 0) ** degree)`` (R ``UPM``).

    ``degree=0`` gives the share of observations ``> target``.  ``target`` may be a
    scalar or a 1D array; ``None`` uses ``mean(x)``.  ``x`` must be finite.
    """
    return _moment(degree, target, x, lower=False)


def lpm_ratio(
    degree: float,
    target: float | NDArray[np.float64] | None,
    x: NDArray[np.float64],
) -> float | NDArray[np.float64]:
    """``LPM / (LPM + UPM)`` at ``target`` (R ``LPM.ratio``).

    ``degree=0`` is the empirical CDF; ``degree>0`` gives a continuous CDF.  ``None``
    uses ``mean(x)``.  ``x`` must be finite.
    """
    return _ratio(degree, target, x, lower=True)


def upm_ratio(
    degree: float,
    target: float | NDArray[np.float64] | None,
    x: NDArray[np.float64],
) -> float | NDArray[np.float64]:
    """``UPM / (LPM + UPM)`` at ``target`` (R ``UPM.ratio``); ``1 - lpm_ratio``."""
    return _ratio(degree, target, x, lower=False)


def _moment(
    degree: float,
    target: float | NDArray[np.float64] | None,
    x: NDArray[np.float64],
    *,
    lower: bool,
) -> float | NDArray[np.float64]:
    values = _as_1d_values(x)
    targets = _as_targets(target, values)
    degree = _as_degree(degree)
    if not np.all(np.isfinite(targets)):
        raise ValueError("`target` must be finite.")

    native = native_fn("lpm" if lower else "upm")
    if native is not None and targets.size > 0 and _native_safe(values, targets):
        native_target = float(targets[0]) if _is_scalar(target) else targets
        native_result = native(degree, native_target, np.ascontiguousarray(values))
        return _result_for_target(np.asarray(native_result, dtype=np.float64).reshape(-1), target)

    moments = np.empty(targets.size, dtype=np.float64)
    for start in range(0, targets.size, _TARGET_BLOCK):
        grid = targets[start : start + _TARGET_BLOCK, np.newaxis]
        if degree == 0:
            # R convention: equality with the target counts toward the lower moment.
            block = np.mean(values <= grid if lower else values > grid, axis=1)
        else:
            deviations = grid - values if lower else values - grid
            block = np.mean(np.maximum(0.0, deviations) ** degree, axis=1)
        moments[start : start + _TARGET_BLOCK] = block
    return _result_for_target(moments, target)


def _ratio(
    degree: float,
    target: float | NDArray[np.float64] | None,
    x: NDArray[np.float64],
    *,
    lower: bool,
) -> float | NDArray[np.float64]:
    values = _as_1d_values(x)
    targets = _as_targets(target, values)
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
    if not np.all(np.isfinite(values)):
        # Matches R's LPM/UPM ("`variable` must be finite"); NaN would otherwise be
        # counted in n but in neither tail, so degree-0 moments stop summing to 1.
        raise ValueError("x must contain only finite values.")
    return values


def _is_scalar(target: object) -> bool:
    return target is None or np.asarray(target).ndim == 0


def _as_targets(
    target: float | NDArray[np.float64] | None,
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    if target is None:
        return np.array([float(np.mean(values))], dtype=np.float64)
    targets = np.asarray(target, dtype=np.float64)
    if targets.ndim == 0:
        targets = targets.reshape(1)
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
    target: float | NDArray[np.float64] | None,
) -> float | NDArray[np.float64]:
    if _is_scalar(target):
        return float(moments[0])
    return moments
