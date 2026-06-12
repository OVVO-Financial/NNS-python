from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from numpy.typing import NDArray

from pynns.co_moments import co_lpm, co_upm, d_lpm, d_upm


def nns_dep(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    asym: bool = False,
) -> dict[str, float]:
    """Return NNS nonlinear correlation and dependence for a pair of variables."""
    x_values, y_values = _as_pair(x, y)
    if _is_constant(x_values) or _is_constant(y_values):
        return {"Correlation": 0.0, "Dependence": 0.0}

    obs_req = max(8, x_values.size // 8)
    quad_xy = _xonly_partition(x_values, obs_req)
    quad_yx = _xonly_partition(y_values, obs_req)
    correlation, dependence = _dep_pair(x_values, y_values, quad_xy, quad_yx, asym)
    return {"Correlation": correlation, "Dependence": dependence}


def nns_cor(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """Return the signed NNS nonlinear correlation component."""
    return nns_dep(x, y)["Correlation"]


def _dep_pair(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    quad_xy: list[str],
    quad_yx: list[str],
    asym: bool,
) -> tuple[float, float]:
    global_cop = _finite_or_zero(_copula_signed(x, y))

    corr_xy, dep_xy = _directional_dep(x, y, quad_xy, global_cop)
    corr_yx, dep_yx = _directional_dep(y, x, quad_yx, global_cop)

    if _is_discrete_case(x, y):
        disc_cop = _copula_degree0_unsigned(x, y)
        if not math.isfinite(disc_cop):
            disc_cop = max(dep_xy, dep_yx)
        if asym:
            dep_xy = _gravity(np.array([dep_xy, disc_cop], dtype=np.float64))
        else:
            dep_sym = _gravity(np.array([max(dep_xy, dep_yx), disc_cop], dtype=np.float64))
            dep_xy = dep_sym
            dep_yx = dep_sym

    if asym:
        return corr_xy, dep_xy

    return max(corr_xy, corr_yx), max(dep_xy, dep_yx)


def _directional_dep(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    quadrants: list[str],
    fallback: float,
) -> tuple[float, float]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, quadrant in enumerate(quadrants):
        groups[quadrant].append(index)

    corr = 0.0
    dep = 0.0
    n = x.size
    for indices in groups.values():
        idx = np.asarray(indices, dtype=np.intp)
        cop = _copula_signed(x[idx], y[idx])
        if not math.isfinite(cop):
            cop = fallback
        weight = idx.size / n
        corr += cop * weight
        dep += abs(cop) * weight
    return corr, dep


def _xonly_partition(x: NDArray[np.float64], obs_req: int) -> list[str]:
    n = x.size
    max_order = max(math.ceil(math.log2(max(1, n))), 1)
    floor_order = math.floor(math.log2(max(1, n)))
    quadrants = ["q"] * n

    for depth in range(max_order):
        if depth >= floor_order:
            break

        groups: dict[str, list[int]] = defaultdict(list)
        for index, quadrant in enumerate(quadrants):
            groups[quadrant].append(index)

        to_split = [quadrant for quadrant, indices in groups.items() if len(indices) > obs_req]
        if not to_split:
            break

        centers = {
            quadrant: _gravity(x[np.asarray(groups[quadrant], dtype=np.intp)])
            for quadrant in to_split
        }
        for quadrant in to_split:
            center = centers[quadrant]
            for index in groups[quadrant]:
                quadrants[index] += "2" if x[index] > center else "1"
    return quadrants


def _copula_signed(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    n = x.size
    if n < 2:
        return 0.0

    target_x = float(np.mean(x))
    target_y = float(np.mean(y))

    d0_cupm = float(co_upm(0.0, x, y, target_x, target_y))
    d0_clpm = float(co_lpm(0.0, x, y, target_x, target_y))
    d0_co = d0_cupm + d0_clpm
    if d0_co == 1.0 or d0_co == 0.0:
        return 1.0

    c1_cupm = float(co_upm(1.0, x, y, target_x, target_y))
    c1_clpm = float(co_lpm(1.0, x, y, target_x, target_y))
    c1_dlpm = float(d_lpm(1.0, 1.0, x, y, target_x, target_y))
    c1_dupm = float(d_upm(1.0, 1.0, x, y, target_x, target_y))
    if n > 1:
        adjust = n / (n - 1)
        c1_cupm *= adjust
        c1_clpm *= adjust
        c1_dlpm *= adjust
        c1_dupm *= adjust
    total = c1_cupm + c1_dupm + c1_dlpm + c1_clpm
    if total > 0.0:
        c1_cupm /= total
        c1_clpm /= total

    data = np.column_stack((x, y))
    target = np.array([target_x, target_y], dtype=np.float64)
    dpm_d0 = _dpm_nd(data, target, 0.0, norm=True)
    dpm_d1 = _dpm_nd(data, target, 1.0, norm=True)

    discrete_dep = min(max(abs(d0_co - 0.5) / 0.5, 0.0), 1.0)
    continuous_dep = min(max(abs(c1_cupm + c1_clpm - 0.5) / 0.5, 0.0), 1.0)
    nd_disc_dep = abs(dpm_d0 - 0.75) / 0.75
    nd_cont_dep = abs(dpm_d1 - 0.75) / 0.75

    copula = math.sqrt((discrete_dep + continuous_dep + nd_disc_dep + nd_cont_dep) / 4.0)
    return copula * _ols_sign(x, y)


def _copula_degree0_unsigned(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    target_x = float(np.mean(x))
    target_y = float(np.mean(y))
    d0_co = float(co_upm(0.0, x, y, target_x, target_y)) + float(
        co_lpm(0.0, x, y, target_x, target_y)
    )
    data = np.column_stack((x, y))
    target = np.array([target_x, target_y], dtype=np.float64)
    dpm_d0 = _dpm_nd(data, target, 0.0, norm=True)
    disc_dep = min(max(abs(d0_co - 0.5) / 0.5, 0.0), 1.0)
    nd_disc = abs(dpm_d0 - 0.75) / 0.75
    return math.sqrt((disc_dep + nd_disc) / 2.0)


def _dpm_nd(
    data: NDArray[np.float64],
    target: NDArray[np.float64],
    degree: float,
    norm: bool,
) -> float:
    diff = data - target[np.newaxis, :]
    all_below = np.all(diff < 0.0, axis=1)
    all_above = np.all(diff > 0.0, axis=1)
    discordant = ~(all_below | all_above)

    if degree == 0.0:
        return float(np.mean(discordant))
    else:
        values = np.prod(np.abs(diff) ** degree, axis=1)
        dpm = float(np.mean(np.where(discordant, values, 0.0)))

    if not norm:
        return dpm

    clpm = _clpm_nd(data, target, degree)
    cupm = _cupm_nd(data, target, degree)
    total = clpm + cupm + dpm
    return dpm / total if total > 0.0 else 0.0


def _clpm_nd(data: NDArray[np.float64], target: NDArray[np.float64], degree: float) -> float:
    diff = target[np.newaxis, :] - data
    if degree == 0.0:
        return float(np.mean(np.all(diff >= 0.0, axis=1)))
    valid = np.all(diff >= 0.0, axis=1)
    return float(np.mean(np.where(valid, np.prod(diff**degree, axis=1), 0.0)))


def _cupm_nd(data: NDArray[np.float64], target: NDArray[np.float64], degree: float) -> float:
    diff = data - target[np.newaxis, :]
    if degree == 0.0:
        return float(np.mean(np.all(diff >= 0.0, axis=1)))
    valid = np.all(diff >= 0.0, axis=1)
    return float(np.mean(np.where(valid, np.prod(diff**degree, axis=1), 0.0)))


def co_lpm_nd(
    data: NDArray[np.float64],
    target: NDArray[np.float64],
    degree: float = 0.0,
    norm: bool = True,
) -> float:
    values, target_values = _as_nd_moment_inputs(data, target)
    degree = float(degree)
    clpm = _clpm_nd(values, target_values, degree)
    if not norm or degree == 0.0:
        return clpm
    cupm = _cupm_nd(values, target_values, degree)
    dpm = _dpm_nd(values, target_values, degree, norm=False)
    total = clpm + cupm + dpm
    return clpm / total if total > 0.0 else 0.0


def co_upm_nd(
    data: NDArray[np.float64],
    target: NDArray[np.float64],
    degree: float = 0.0,
    norm: bool = True,
) -> float:
    values, target_values = _as_nd_moment_inputs(data, target)
    degree = float(degree)
    cupm = _cupm_nd(values, target_values, degree)
    if not norm or degree == 0.0:
        return cupm
    clpm = _clpm_nd(values, target_values, degree)
    dpm = _dpm_nd(values, target_values, degree, norm=False)
    total = clpm + cupm + dpm
    return cupm / total if total > 0.0 else 0.0


def dpm_nd(
    data: NDArray[np.float64],
    target: NDArray[np.float64],
    degree: float = 0.0,
    norm: bool = True,
) -> float:
    values, target_values = _as_nd_moment_inputs(data, target)
    return _dpm_nd(values, target_values, float(degree), norm=bool(norm))


def _as_nd_moment_inputs(
    data: NDArray[np.float64],
    target: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    values = np.asarray(data, dtype=np.float64)
    target_values = np.asarray(target, dtype=np.float64).reshape(-1)
    if values.ndim != 2:
        raise ValueError("data must be a 2D matrix.")
    if values.shape[0] == 0:
        raise ValueError("data must have at least one row.")
    if target_values.size != values.shape[1]:
        raise ValueError("target length must match number of columns in data.")
    if not np.all(np.isfinite(values)):
        raise ValueError("data must be finite.")
    if not np.all(np.isfinite(target_values)):
        raise ValueError("target must be finite.")
    return values, target_values


def _gravity(x: NDArray[np.float64]) -> float:
    values = np.sort(x[np.isfinite(x)])
    n = values.size
    if n == 0:
        return float("nan")
    if n <= 3:
        return float(np.median(values))
    if np.all(values == values[0]):
        return float(values[0])

    value_range = float(np.ptp(values))
    if abs(value_range) == 0.0:
        return float(values[0])

    q1, q2, q3 = _quartiles_like_r_code(values)
    width = (q3 - q1) * n**-0.5
    if width <= 0.0 or not np.isfinite(width):
        width = value_range / 128.0

    bin_names, counts = _simple_bin_counts(values, width, float(values[0]))
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
    return float(0.25 * (q2 + mode_gravity + float(np.mean(values)) + 0.5 * (q1 + q3)))


def _quartiles_like_r_code(values: NDArray[np.float64]) -> tuple[float, float, float]:
    n = values.size
    p25 = n * 0.25
    p50 = n * 0.50
    p75 = n * 0.75
    if n % 2 == 0:
        return (
            float(values[max(1, math.floor(p25)) - 1]),
            float(values[max(1, math.floor(p50)) - 1]),
            float(values[max(1, math.floor(p75)) - 1]),
        )

    q1 = _interpolate_position(values, p25)
    f50 = min(max(1, math.floor(p50)), n)
    c50 = min(max(1, math.ceil(p50)), n)
    q2 = 0.5 * (values[f50 - 1] + values[c50 - 1])
    q3 = _interpolate_position(values, p75)
    return float(q1), float(q2), float(q3)


def _interpolate_position(values: NDArray[np.float64], position: float) -> float:
    n = values.size
    floor_pos = min(max(1, math.floor(position)), n)
    ceil_pos = min(max(1, math.ceil(position)), n)
    weight = position - math.floor(position)
    return float(values[floor_pos - 1] + weight * (values[ceil_pos - 1] - values[floor_pos - 1]))


def _simple_bin_counts(
    values: NDArray[np.float64],
    width: float,
    origin: float,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    int_max = np.iinfo(np.int32).max
    if width <= 0.0 or not math.isfinite(width):
        bin_count = 1
    else:
        bin_ratio = (float(values[-1]) - origin) / width + 1e-12
        if not math.isfinite(bin_ratio) or bin_ratio > int_max:
            bin_count = 1
        else:
            bin_count = math.floor(bin_ratio) + 1
    bin_count = min(max(1, bin_count), 4 * values.size)
    bin_names = origin + np.arange(bin_count, dtype=np.float64) * width
    if bin_count == 1:
        return bin_names, np.array([values.size], dtype=np.int64)
    indices = np.floor((values - origin) / width).astype(np.int64)
    indices = np.clip(indices, 0, bin_count - 1)
    counts = np.bincount(indices, minlength=bin_count)
    return bin_names, counts


def _ols_sign(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    if x.size < 2:
        return 0.0
    dx = x - np.mean(x)
    denominator = float(np.sum(dx * dx))
    if denominator == 0.0:
        return 0.0
    slope = float(np.sum(dx * (y - np.mean(y))) / denominator)
    if slope > 0.0:
        return 1.0
    if slope < 0.0:
        return -1.0
    return 0.0


def _is_discrete_case(x: NDArray[np.float64], y: NDArray[np.float64]) -> bool:
    threshold = math.sqrt(x.size)
    return np.unique(x).size < threshold and np.unique(y).size < threshold


def _finite_or_zero(value: float) -> float:
    return value if math.isfinite(value) else 0.0


def _is_constant(values: NDArray[np.float64]) -> bool:
    return bool(np.all(values == values[0]))


def _as_pair(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
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
