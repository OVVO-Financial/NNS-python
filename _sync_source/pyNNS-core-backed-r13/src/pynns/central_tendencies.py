from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from pynns._native import nnscore
from pynns.dependence import _quartiles_like_r_code, _simple_bin_counts


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
        if finite.size == 0:
            return np.full(values.shape, (a + b) / 2.0, dtype=np.float64)
        xmin = float(np.min(finite))
        xmax = float(np.max(finite))
        if xmax == xmin:
            return np.full(values.shape, (a + b) / 2.0, dtype=np.float64)
        return a + (b - a) * ((values - xmin) / (xmax - xmin))

    if method_l == "riskneutral":
        if time_to_maturity is None:
            raise ValueError("time_to_maturity must be provided for riskneutral method.")
        if not a > 0.0:
            raise ValueError("S_0 (a) must be positive for riskneutral method.")
        finite = values[np.isfinite(values)]
        mean_x = float(np.mean(finite)) if finite.size else float("nan")
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
    """Mode of a distribution matching R's NNS.mode."""
    values = np.asarray(x, dtype=np.float64)
    finite = values[np.isfinite(values)]
    n = finite.size
    if n == 0:
        return np.array([np.nan], dtype=np.float64) if multi else float("nan")

    native = nnscore()
    if native is not None and hasattr(native, "mode"):
        native_result = np.asarray(
            native.mode(np.ascontiguousarray(finite), discrete, multi), dtype=np.float64
        )
        if multi:
            return native_result
        return float(native_result[0])

    if discrete:
        return _discrete_mode(finite, multi)
    return _continuous_mode(finite, multi)


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

    native = nnscore()
    if native is not None and hasattr(native, "gravity"):
        return float(native.gravity(np.ascontiguousarray(finite), discrete))

    value_range = float(abs(finite[-1] - finite[0]))
    if value_range == 0.0:
        return float(finite[0])

    q1, q2, q3 = _quartiles_like_r_code(finite)
    width = (q3 - q1) * n**-0.5
    if width <= 0.0 or not np.isfinite(width):
        width = value_range / 128.0

    bin_names, counts = _simple_bin_counts(finite, width, float(finite[0]))
    max_count = int(np.max(counts))
    max_positions = np.flatnonzero(counts == max_count)
    if max_positions.size == 1:
        center = int(max_positions[0])
        lo = max(0, center - 1)
        hi = min(counts.size - 1, center + 1)
    else:
        lo = 0
        hi = counts.size - 1

    selected_names = bin_names[lo : hi + 1]
    selected_counts = counts[lo : hi + 1]
    denominator = float(np.sum(selected_counts))
    mode_gravity = (
        float(np.sum(selected_names * selected_counts) / denominator)
        if denominator > 0.0
        else float(bin_names[(lo + hi) // 2])
    )
    out = 0.25 * (q2 + mode_gravity + float(np.mean(finite)) + 0.5 * (q1 + q3))
    if not math.isfinite(out):
        out = q2
    return _nearest_int_half_up(out) if discrete else float(out)


def _discrete_mode(values: NDArray[np.float64], multi: bool) -> float | NDArray[np.float64]:
    n = values.size
    if n <= 3:
        median = float(np.median(np.sort(values)))
        mode = _nearest_int_half_up(median)
        return mode

    integerized = _nearest_int_half_up_array(values)
    modes, counts = np.unique(integerized, return_counts=True)
    tied_modes = modes[counts == int(np.max(counts))]
    tied_modes = np.sort(tied_modes.astype(np.float64))
    if multi:
        return tied_modes
    return float(np.mean(tied_modes))


def _continuous_mode(values: NDArray[np.float64], multi: bool) -> float | NDArray[np.float64]:
    n = values.size
    if n <= 3:
        median = float(np.median(np.sort(values)))
        return median
    if bool(np.all(values == values[0])):
        return float(values[0])

    sorted_values = np.sort(values)
    value_range = float(abs(sorted_values[-1] - sorted_values[0]))
    if value_range == 0.0:
        return float(sorted_values[0])

    q1, _, q3 = _quartiles_like_r_code(sorted_values)
    width = (q3 - q1) * n**-0.5
    if width <= 0.0 or not np.isfinite(width):
        width = value_range / 128.0
    if width <= 0.0 or not np.isfinite(width):
        width = value_range / 128.0

    bin_names, counts = _simple_bin_counts(sorted_values, width, float(sorted_values[0]))
    if counts.size == 0:
        return np.array([np.nan], dtype=np.float64) if multi else float("nan")

    max_count = int(np.max(counts))
    smoothed = _smooth_counts_tri7(counts)
    peak_indices = _peak_indices(smoothed)
    if peak_indices.size:
        kept = _non_maximum_suppress(peak_indices, smoothed)
        if kept.size:
            centers = np.empty(kept.size, dtype=np.float64)
            for idx, center in enumerate(kept):
                lo = max(0, int(center) - 3)
                hi = min(counts.size - 1, int(center) + 3)
                selected_names = bin_names[lo : hi + 1]
                selected_counts = counts[lo : hi + 1]
                denominator = float(np.sum(selected_counts))
                centers[idx] = (
                    float(np.sum(selected_names * selected_counts) / denominator)
                    if denominator > 0.0
                    else float(bin_names[center])
                )
            if multi:
                return np.sort(centers)
            best = int(np.argmax(smoothed[kept]))
            return float(centers[best])

    tied = np.flatnonzero(counts == max_count)
    if tied.size > 1:
        modes = bin_names[tied].astype(np.float64)
        if multi:
            return np.sort(modes)
        return float(np.mean(modes))

    center = int(tied[0])
    lo = max(0, center - 1)
    hi = min(counts.size - 1, center + 1)
    selected_names = bin_names[lo : hi + 1]
    selected_counts = counts[lo : hi + 1]
    denominator = float(np.sum(selected_counts))
    value = (
        float(np.sum(selected_names * selected_counts) / denominator)
        if denominator > 0.0
        else float(bin_names[center])
    )
    return np.array([value], dtype=np.float64) if multi else value


def _smooth_counts_tri7(counts: NDArray[np.int64]) -> NDArray[np.float64]:
    weights = np.array([1, 2, 3, 4, 3, 2, 1], dtype=np.float64)
    n = counts.size
    if n == 1:
        return counts.astype(np.float64)
    smooth = np.zeros(n, dtype=np.float64)

    def at(index: int) -> int:
        while index < 0 or index >= n:
            if index < 0:
                index = -index
            if index >= n:
                index = 2 * n - 2 - index
        return int(counts[index])

    for i in range(n):
        smooth[i] = sum(weights[j] * at(i + j - 3) for j in range(7)) / 16.0
    return smooth


def _peak_indices(smoothed: NDArray[np.float64]) -> NDArray[np.int64]:
    peaks: list[int] = []
    for i in range(3, smoothed.size - 3):
        center = smoothed[i]
        if center <= 0.0:
            continue
        left = max(smoothed[i - 1], smoothed[i - 2], smoothed[i - 3])
        right = max(smoothed[i + 1], smoothed[i + 2], smoothed[i + 3])
        if not (center > left and center > right):
            continue
        curvature = smoothed[i - 1] - 2.0 * center + smoothed[i + 1]
        if curvature < 0.0:
            peaks.append(i)
    return np.asarray(peaks, dtype=np.int64)


def _non_maximum_suppress(
    peaks: NDArray[np.int64],
    smoothed: NDArray[np.float64],
) -> NDArray[np.int64]:
    ordered = peaks[np.argsort(-smoothed[peaks])]
    kept: list[int] = []
    for peak in ordered:
        if all(abs(int(peak) - prior) > 3 for prior in kept):
            kept.append(int(peak))
    return np.asarray(kept, dtype=np.int64)


def _nearest_int_half_up(value: float) -> float:
    floor = math.floor(value)
    return float(floor if value - floor < 0.5 else math.ceil(value))


def _nearest_int_half_up_array(values: NDArray[np.float64]) -> NDArray[np.float64]:
    floors = np.floor(values)
    return np.where(values - floors < 0.5, floors, np.ceil(values)).astype(np.float64)
