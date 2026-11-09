from __future__ import annotations

import math
import warnings

import numpy as np
from numpy.typing import NDArray

from nns._native import native_fn as native_fn
from nns.dependence import (
    _gravity_python,
    _quartiles_like_r_code,
    _small_median,
    _sparse_bin_counts,
)


def nns_rescale(
    x: NDArray[np.float64],
    a: float,
    b: float,
    method: str = "minmax",
    time_to_maturity: float | None = None,
    type: str = "Terminal",
) -> NDArray[np.float64]:
    """Rescale a vector using R's NNS.rescale conventions."""
    values = np.asarray(x, dtype=np.float64)
    method_l = method.lower()
    type_l = type.lower()

    if method_l == "minmax":
        finite = values[np.isfinite(values)]
        if finite.size == 0 or float(np.min(finite)) == float(np.max(finite)):
            warnings.warn("All x identical: returning midpoint values", UserWarning, stacklevel=2)
            return np.full(values.shape, (a + b) / 2.0, dtype=np.float64)
        xmin = float(np.min(finite))
        xmax = float(np.max(finite))
        return a + (b - a) * ((values - xmin) / (xmax - xmin))

    if method_l == "riskneutral":
        if time_to_maturity is None:
            raise ValueError("time_to_maturity must be provided for riskneutral method.")
        if not a > 0.0:
            raise ValueError("S_0 (a) must be positive for riskneutral method.")
        finite = values[np.isfinite(values)]
        # R's C++: one long-double pass over the finite values, divided by the count.
        mean_x = (
            float(np.cumsum(finite.astype(np.longdouble))[-1] / finite.size)
            if finite.size
            else float("nan")
        )
        if not np.isfinite(mean_x) or mean_x <= 0.0:
            raise ValueError("Mean(x) must be positive/finite for riskneutral scaling.")
        target = a if type_l == "discounted" else a * math.exp(b * time_to_maturity)
        theta = math.log(target / mean_x)
        return values * math.exp(theta)

    raise ValueError("Invalid method: use 'minmax' or 'riskneutral'.")


def nns_mode(
    x: NDArray[np.float64],
    discrete: bool = False,
    multi: bool = False,
) -> float | NDArray[np.float64]:
    """Mode of a distribution matching R's ``NNS.mode``.

    ``discrete=True`` returns the most frequent value after rounding to integers,
    and of tied values the one nearest ``nns_gravity(x)``, then the lower;
    otherwise the mode of a fixed-width histogram. ``multi=True`` returns every
    detected mode as an array and ``multi=False`` a single float.

    Note the default differs from R: ``NNS.mode`` defaults to ``multi = TRUE``,
    while ``nns_mode`` defaults to ``multi=False`` (one value). Pass
    ``multi=True`` for R's default output. Explicit ``multi`` matches R.
    """
    values = np.asarray(x, dtype=np.float64)
    finite = values[np.isfinite(values)]
    n = finite.size
    if n == 0:
        return np.array([np.nan], dtype=np.float64) if multi else float("nan")

    native = native_fn("mode_r14")
    if native is not None:
        result = np.asarray(native(np.ascontiguousarray(finite), discrete, multi), dtype=np.float64)
    else:
        result = _discrete_mode(finite, multi) if discrete else _continuous_mode(finite, multi)
    return result if multi else float(result[0])


def nns_gravity(x: NDArray[np.float64], discrete: bool = False) -> float:
    """Alternative central tendency matching R's NNS.gravity."""
    values = np.asarray(x, dtype=np.float64)
    finite = np.sort(values[np.isfinite(values)])
    n = finite.size
    if n == 0:
        return float("nan")
    if n <= 3:
        median = float(np.median(finite))
        return _nearest_int_half_up(median) if discrete else median
    if bool(np.all(finite == finite[0])):
        return float(finite[0])

    # A versioned name avoids accidentally selecting an old R13 extension.
    native = native_fn("gravity_exact_r14")
    out = (
        float(native(np.ascontiguousarray(finite)))
        if native is not None
        else _gravity_python(finite)
    )
    return _nearest_int_half_up(out) if discrete else out


def _discrete_mode(values: NDArray[np.float64], multi: bool) -> NDArray[np.float64]:
    """NumPy port of NNS-core ``nns::mode(discrete=True)`` (R ``NNS.mode``)."""
    # Keyed on the rounded value; "+ 0.0" folds -0.0 into 0.0.
    integerized = _nearest_int_half_up_array(values) + 0.0
    modes, counts = np.unique(integerized, return_counts=True)
    sorted_modes: NDArray[np.float64] = modes[counts == int(np.max(counts))].astype(np.float64)
    if multi or sorted_modes.size == 1:
        return sorted_modes
    # Tied modes: the one nearest the gravity, then the lower; a mean of tied classes is no class.
    gravity = nns_gravity(values)
    return sorted_modes[[int(np.argmin(np.abs(sorted_modes - gravity)))]]


def _continuous_mode(values: NDArray[np.float64], multi: bool) -> NDArray[np.float64]:
    """R 14.0 NNS_mode_cpp: discrete-valued peaks and observed-bin centers."""
    n = values.size
    if n <= 3:
        return np.array([_small_median(values)], dtype=np.float64)
    if bool(np.all(values == values[0])):
        return np.array([values[0]], dtype=np.float64)

    ordered = np.sort(values)
    span = abs(float(ordered[-1]) - float(ordered[0]))
    if span == 0.0:
        return np.array([ordered[0]], dtype=np.float64)

    # Mode's histogram width intentionally still uses R's original quartiles;
    # gravity, which chooses among tied modes, uses type-7 quartiles separately.
    q1, _, q3 = _quartiles_like_r_code(ordered)
    width = (q3 - q1) * n**-0.5
    if not (width > 0.0 and math.isfinite(width)):
        width = span / 128.0

    distinct, frequencies = np.unique(ordered, return_counts=True)
    if distinct.size >= 2 and bool(np.all(np.diff(distinct) > width)):
        # R #114: when bins merely separate discrete observations, detect
        # contiguous plateaux in the exact frequency sequence, including ends.
        mode_indices: list[int] = []
        i = 0
        while i < distinct.size:
            j = i + 1
            while j < distinct.size and frequencies[j] == frequencies[i]:
                j += 1
            above_left = i == 0 or frequencies[i - 1] < frequencies[i]
            above_right = j == distinct.size or frequencies[j] < frequencies[i]
            if above_left and above_right:
                mode_indices.extend(range(i, j))
            i = j
        idx = np.asarray(mode_indices, dtype=np.intp)
        modes = distinct[idx]
        if multi:
            return np.asarray(modes, dtype=np.float64)
        top = int(np.max(frequencies[idx]))
        top_modes = modes[frequencies[idx] == top]
        gravity = nns_gravity(ordered)
        return np.asarray([top_modes[int(np.argmin(np.abs(top_modes - gravity)))]])

    names, counts = _core_bin_list(ordered, width, float(ordered[0]))
    lz = counts.size
    max_count = int(np.max(counts))
    # Truncated empty runs have count zero; the prefix exactly indexes the
    # contiguous observations in each histogram window.
    starts = np.r_[0, np.cumsum(counts, dtype=np.int64)]

    def observed_center(lo: int, hi: int) -> float:
        a = int(starts[lo])
        e = int(starts[hi + 1])
        if e <= a:
            return float(names[lo])
        if ordered[a] == ordered[e - 1]:
            return float(ordered[a])
        vals = ordered[a:e].astype(np.longdouble)
        return float(np.cumsum(vals)[-1] / np.longdouble(e - a))

    cs = _smooth_counts_tri7(counts)
    candidates: list[int] = []
    if lz >= 7:
        max_height = float(np.max(cs))
        for i in range(lz):
            height = float(cs[i])
            if height <= 0.0:
                continue
            if (i < 3 or i > lz - 4) and height < max_height:
                continue
            left_max = max((cs[i - d] for d in (1, 2, 3) if i - d >= 0), default=-math.inf)
            right_max = max((cs[i + d] for d in (1, 2, 3) if i + d < lz), default=-math.inf)
            if not (height > left_max and height > right_max):
                continue
            left = cs[i - 1] if i > 0 else cs[i + 1]
            right = cs[i + 1] if i + 1 < lz else cs[i - 1]
            if left - 2.0 * height + right < 0.0:
                candidates.append(i)

    if candidates:
        # Stable highest-first ordering, suppress peaks within three bins.
        ranked = sorted(candidates, key=lambda i: -cs[i])
        kept: list[int] = []
        for i in ranked:
            if all(abs(i - j) > 3 for j in kept):
                kept.append(i)
        centers = np.array(
            [observed_center(max(0, i - 3), min(lz - 1, i + 3)) for i in kept],
            dtype=np.float64,
        )
        if multi:
            return np.sort(centers)
        highest = max(cs[i] for i in kept)
        top = centers[np.array([cs[i] == highest for i in kept])]
        gravity = nns_gravity(ordered)
        return np.asarray([sorted(top, key=lambda v: (abs(v - gravity), v))[0]], dtype=np.float64)

    winners = np.flatnonzero(counts == max_count)
    if winners.size > 1:
        centers = np.array(
            [observed_center(int(i), int(i)) for i in winners],
            dtype=np.float64,
        )
        if multi:
            return np.sort(centers)
        gravity = nns_gravity(ordered)
        return np.asarray(
            [sorted(centers, key=lambda v: (abs(v - gravity), v))[0]], dtype=np.float64
        )

    center = int(winners[0])
    return np.array(
        [observed_center(max(0, center - 1), min(lz - 1, center + 1))],
        dtype=np.float64,
    )


def _core_bin_list(
    sorted_values: NDArray[np.float64], width: float, origin: float
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """NNS-core ``simple_bin_counts``: bin names and counts with long empty runs shortened."""
    bin_count, width, occupied, counts = _sparse_bin_counts(sorted_values, width, origin)
    starts = np.r_[0, occupied + 1]
    ends = np.r_[occupied - 1, bin_count - 1]
    lengths = np.maximum(ends - starts + 1, 0)
    keep = np.minimum(lengths, 2 * _HALF_EMPTY_RUN)
    gap = np.repeat(np.arange(starts.size), keep)
    offset = np.arange(gap.size) - np.repeat(np.cumsum(keep) - keep, keep)
    long_run = lengths[gap] > 2 * _HALF_EMPTY_RUN
    tail = long_run & (offset >= _HALF_EMPTY_RUN)
    empty = np.where(tail, ends[gap] - (2 * _HALF_EMPTY_RUN - 1) + offset, starts[gap] + offset)
    bins = np.concatenate((occupied, empty))
    bin_counts = np.concatenate((counts, np.zeros(empty.size, dtype=np.int64)))
    order = np.argsort(bins, kind="stable")
    names = origin + bins[order].astype(np.float64) * width
    return names, bin_counts[order]


# NNS-core keeps the first and last 8 bins of every run of more than 16 empty
# bins; mode and gravity only read bins within 6 of an occupied one.
_HALF_EMPTY_RUN = 8


def _weighted_bin_mean(names: NDArray[np.float64], counts: NDArray[np.int64]) -> float:
    zero = np.longdouble(0.0)
    numerator = zero + np.cumsum(names.astype(np.longdouble) * counts.astype(np.longdouble))[-1]
    denominator = int(np.sum(counts))
    if denominator <= 0:
        return float(names[names.size // 2])
    return float(numerator / np.longdouble(denominator))


def _long_double_mean(values: NDArray[np.float64]) -> float:
    total = np.longdouble(0.0) + np.cumsum(values.astype(np.longdouble))[-1]
    return float(total / np.longdouble(values.size))


def _smooth_counts_tri7(counts: NDArray[np.int64]) -> NDArray[np.float64]:
    """7-tap [1, 2, 3, 4, 3, 2, 1] / 16 smoothing with mirrored, clamped edges."""
    n = counts.size
    index = np.arange(n)[:, None] + np.arange(-3, 4)[None, :]
    index = np.where(index < 0, -index, index)
    index = np.where(index >= n, 2 * n - 2 - index, index)
    index = np.clip(index, 0, n - 1)
    weights = np.array([1, 2, 3, 4, 3, 2, 1], dtype=np.int64)
    acc = np.sum(counts[index].astype(np.int64) * weights[None, :], axis=1)
    return np.asarray(acc, dtype=np.float64) / 16.0


def _peak_indices(smoothed: NDArray[np.float64]) -> NDArray[np.int64]:
    n = smoothed.size
    if n < 7:
        return np.zeros(0, dtype=np.int64)
    i = np.arange(3, n - 3)
    center = smoothed[i]
    left = np.maximum(np.maximum(smoothed[i - 1], smoothed[i - 2]), smoothed[i - 3])
    right = np.maximum(np.maximum(smoothed[i + 1], smoothed[i + 2]), smoothed[i + 3])
    curvature = smoothed[i - 1] - 2.0 * center + smoothed[i + 1]
    peak = (center > 0.0) & (center > left) & (center > right) & (curvature < 0.0)
    return np.asarray(i[peak], dtype=np.int64)


def _nearest_int_half_up(value: float) -> float:
    # Float floor/ceil, as in C++: ceil(-0.4) is -0.0 (math.ceil returns int 0).
    floor = float(np.floor(value))
    return floor if value - floor < 0.5 else float(np.ceil(value))


def _nearest_int_half_up_array(values: NDArray[np.float64]) -> NDArray[np.float64]:
    floors = np.floor(values)
    with np.errstate(invalid="ignore"):
        return np.where(values - floors < 0.5, floors, np.ceil(values)).astype(np.float64)
