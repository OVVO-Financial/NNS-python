from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from nns._native import native_fn
from nns._rmean import r_mean

# Targets per block in the NumPy fallback, so memory stays O(block * n) rather than n x n.
_TARGET_BLOCK = 1024


def lpm(
    degree: float,
    target: float | NDArray[np.float64] | None,
    x: NDArray[np.float64],
    excess_ret: bool = False,
) -> float | NDArray[np.float64]:
    """Lower partial moment ``mean(max(target - x, 0) ** degree)`` (R ``LPM``).

    ``degree=0`` gives the share of observations ``<= target`` (ties count as lower).
    ``target`` may be a scalar (returns a float) or a 1D array (returns an array);
    ``None`` uses ``mean(x)``.  ``x`` must be finite.

    With ``excess_ret=True`` (R ``excess_ret = TRUE``) the target is element-wise:
    ``target`` has length 1 or ``len(x)`` and the result is the single value
    ``mean(max(target_i - x_i, 0) ** degree)`` (degree 0 counts ``x_i <= target_i``).
    """
    if excess_ret:
        return _excess_moment(degree, target, x, lower=True)
    return _moment(degree, target, x, lower=True)


def upm(
    degree: float,
    target: float | NDArray[np.float64] | None,
    x: NDArray[np.float64],
    excess_ret: bool = False,
) -> float | NDArray[np.float64]:
    """Upper partial moment ``mean(max(x - target, 0) ** degree)`` (R ``UPM``).

    ``degree=0`` gives the share of observations ``> target``.  ``target`` may be a
    scalar or a 1D array; ``None`` uses ``mean(x)``.  ``x`` must be finite.

    With ``excess_ret=True`` the target is element-wise (length 1 or ``len(x)``) and
    the result is ``mean(max(x_i - target_i, 0) ** degree)``.
    """
    if excess_ret:
        return _excess_moment(degree, target, x, lower=False)
    return _moment(degree, target, x, lower=False)


def _excess_moment(
    degree: float,
    target: float | NDArray[np.float64] | None,
    x: NDArray[np.float64],
    *,
    lower: bool,
) -> float:
    """R ``LPM``/``UPM`` with ``excess_ret = TRUE``: element-wise targets, one mean.

    At degree 0 the lower moment counts ties (``x_i <= target_i``) as the non-excess
    path does, so ``LPM0 + UPM0 = 1``.
    """
    values = _as_1d_values(x)
    targets = _as_targets(target, values)
    degree = _as_degree(degree)
    _require_finite(targets, "target")
    if targets.size not in (1, values.size):
        raise ValueError(
            "When excess_ret=True, target must be length 1 or same length as x."
        )
    diff = targets - values if lower else values - targets
    # R's C++ excess path: one term per element (degree 0 counts ties as
    # lower), then Rcpp's mean(), which is R's two-pass long-double mean.
    if degree == 0:
        hits = diff >= 0.0 if lower else diff > 0.0
        return r_mean(hits.astype(np.float64))
    keep = diff > 0.0
    return r_mean(np.where(keep, _c_components(diff, degree, True), 0.0))


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

    if targets.size > _DIRECT_PATH_MAX_TARGETS:
        # R's LPM_CPv/UPM_CPv switch to the sorted-anchor prefix backend beyond
        # 32 targets for (near-)integer degrees up to 32.
        degree_int = _prefix_degree(degree)
        if degree_int is not None:
            prefix = _prefix_moments(values, targets, degree_int, lower)
            return _result_for_target(prefix, target)

    moments = np.empty(targets.size, dtype=np.float64)
    for start in range(0, targets.size, _TARGET_BLOCK):
        grid = targets[start : start + _TARGET_BLOCK, np.newaxis]
        deviations = grid - values if lower else values - grid
        # R's LPM_C / UPM_C: one component per observation, summed in order
        # in double, then divided by n.
        components = _c_components(deviations, degree, lower)
        moments[start : start + _TARGET_BLOCK] = np.cumsum(components, axis=1)[:, -1] / values.size
    return _result_for_target(moments, target)


def _c_components(
    deviations: NDArray[np.float64], degree: float, lower: bool
) -> NDArray[np.float64]:
    """Per-observation partial-moment terms exactly as R's C++ forms them.

    Degree 0 counts ``deviation >= 0`` (lower; ties count as lower) or ``> 0``
    (upper). An integer degree is repeated multiplication starting from 1.0
    (``repeatMultiplication``); any other degree is the C library ``pow``,
    which NumPy's vectorised ``power`` does not reproduce bit for bit.
    """
    keep = deviations >= 0.0 if lower else deviations > 0.0
    if degree == 0:
        return keep.astype(np.float64)
    positive = np.where(keep, deviations, 0.0)
    if degree == int(degree):
        out = np.ones_like(positive)
        for _ in range(int(degree)):
            out = out * positive
    else:
        out = np.asarray(_libm_pow(positive, degree), dtype=np.float64)
    return np.where(keep, out, 0.0)


_libm_pow = np.frompyfunc(math.pow, 2, 1)

# R's LPM_CPv/UPM_CPv scan each target directly up to this many targets.
_DIRECT_PATH_MAX_TARGETS = 32
_PREFIX_MAX_DEGREE = 32


def _prefix_degree(degree: float) -> int | None:
    """R's ``prefix_supported_degree``: a degree within 1e-12 of an integer <= 32."""
    if not math.isfinite(degree) or degree < 0.0:
        return None
    rounded = math.floor(degree + 0.5)  # std::round on a non-negative value
    if abs(degree - rounded) > 1e-12 or rounded > _PREFIX_MAX_DEGREE:
        return None
    return int(rounded)


def _prefix_binomials(degree: int) -> list[list[float]]:
    width = degree + 1
    binom = [[0.0] * width for _ in range(width)]
    for j in range(width):
        binom[j][0] = 1.0
        for k in range(1, j + 1):
            binom[j][k] = binom[j - 1][k - 1] + (binom[j - 1][k] if k < j else 0.0)
    return binom


def _prefix_table(
    ordered: list[float], degree: int, binom: list[list[float]], lower: bool
) -> NDArray[np.float64]:
    """R's ``PrefixPartialMomentBackend`` lower or upper table, ``(n + 1) x (degree + 1)``.

    Row ``k`` holds the power sums anchored at the adjacent order statistic,
    built with R's recurrence ``next_j = [j == 0] + sum_l C(j, l) delta^(j - l)
    prev_l`` in R's operation order (zero terms skipped, as R does).
    """
    n = len(ordered)
    width = degree + 1
    table = [[0.0] * width for _ in range(n + 1)]
    steps = range(1, n) if lower else range(n - 1, 0, -1)
    start = 1 if lower else n - 1
    table[start][0] = 1.0
    for k in steps:
        delta = ordered[k] - ordered[k - 1]
        powers = [1.0] * width
        for p in range(1, width):
            powers[p] = powers[p - 1] * delta
        prev = table[k]
        nxt = table[k + 1] if lower else table[k - 1]
        for j in range(width):
            acc = 1.0 if j == 0 else 0.0
            row = binom[j]
            for m in range(j + 1):
                if prev[m] != 0.0:
                    acc += row[m] * powers[j - m] * prev[m]
            nxt[j] = acc
    return np.asarray(table, dtype=np.float64)


def _prefix_moments(
    values: NDArray[np.float64], targets: NDArray[np.float64], degree: int, lower: bool
) -> NDArray[np.float64]:
    """R's prefix-backend ``lpm(t)`` / ``upm(t)`` for every target."""
    ordered = np.sort(values)
    n = ordered.size
    count = np.searchsorted(ordered, targets, side="right")
    if degree == 0:
        hits = count if lower else n - count
        return np.asarray(hits / float(n), dtype=np.float64)
    binom = _prefix_binomials(degree)
    table = _prefix_table(ordered.tolist(), degree, binom, lower)
    if lower:
        empty = count == 0
        gap = targets - ordered[np.maximum(count - 1, 0)]
    else:
        empty = count == n
        gap = ordered[np.minimum(count, n - 1)] - targets
    sums = table[count]
    out = np.zeros(targets.size, dtype=np.float64)
    gap_power = np.ones(targets.size, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        for j in range(degree, -1, -1):
            column = sums[:, j]
            out = np.where(column != 0.0, out + binom[degree][j] * gap_power * column, out)
            gap_power = gap_power * gap
    return np.asarray(np.where(empty, 0.0, out / float(n)), dtype=np.float64)


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
    # R's LPM.ratio/UPM.ratio reject a non-finite target as LPM() does.
    _require_finite(targets, "target")

    native = native_fn("lpm_ratio_v" if lower else "upm_ratio_v")
    if native is not None and targets.size > 0 and _native_safe(values, targets):
        native_result = native(
            degree,
            np.ascontiguousarray(targets),
            np.ascontiguousarray(values),
        )
        ratio = np.asarray(native_result, dtype=np.float64).reshape(-1)
        ratio = _fill_degenerate_ratio(ratio, degree, targets, values, lower)
        return _result_for_target(ratio, target)

    if degree == 0:
        return _moment(degree, target, x, lower=lower)

    lower_moment = np.asarray(lpm(degree, target, x))
    upper_moment = np.asarray(upm(degree, target, x))
    numerator = lower_moment if lower else upper_moment
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.asarray(numerator / (lower_moment + upper_moment)).reshape(-1)
    ratio = _fill_degenerate_ratio(ratio, degree, targets, values, lower)
    return _result_for_target(ratio, target)


def _fill_degenerate_ratio(
    ratio: NDArray[np.float64],
    degree: float,
    targets: NDArray[np.float64],
    values: NDArray[np.float64],
    lower: bool,
) -> NDArray[np.float64]:
    """Replace the degree > 0 ``0/0`` ratio by the degree-0 ratio (R ``fill_degenerate_ratio``).

    When every observation equals the target, ``LPM + UPM`` is 0; R returns the
    frequency ratio there (the degree -> 0 limit, ECDF convention ``x <= target``),
    i.e. ``LPM.ratio`` 1 and ``UPM.ratio`` 0.  Other NaNs (``Inf/Inf`` from overflow)
    stay NaN.
    """
    if not degree > 0:
        return ratio
    missing = np.isnan(ratio)
    if not missing.any():
        return ratio
    constant = values[0]
    if not np.all(values == constant):
        return ratio
    out = ratio.copy()
    out[missing & (targets == constant)] = 1.0 if lower else 0.0
    return out


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


def _require_finite(values: NDArray[np.float64], name: str) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")


def _is_scalar(target: object) -> bool:
    return target is None or np.asarray(target).ndim == 0


def _as_targets(
    target: float | NDArray[np.float64] | None,
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    if target is None:
        return np.array([r_mean(values)], dtype=np.float64)
    targets = np.asarray(target, dtype=np.float64)
    if targets.ndim == 0:
        targets = targets.reshape(1)
    if targets.ndim != 1:
        raise ValueError("target must be scalar or 1D.")
    return targets


def _as_degree(degree: float) -> float:
    degree = float(degree)
    if not degree >= 0:
        raise ValueError("degree must be non-negative.")
    return degree


def _result_for_target(
    moments: NDArray[np.float64],
    target: float | NDArray[np.float64] | None,
) -> float | NDArray[np.float64]:
    if _is_scalar(target):
        return float(moments[0])
    return moments
