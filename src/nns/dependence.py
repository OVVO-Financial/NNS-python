from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, overload

import numpy as np
from numpy.typing import NDArray

from nns._helpers import _as_pair
from nns._native import native_fn


@overload
def nns_dep(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    asym: bool = False,
) -> dict[str, float]: ...


@overload
def nns_dep(
    x: Any,
    y: None = None,
    asym: bool = False,
) -> dict[str, Any]: ...


def nns_dep(
    x: Any,
    y: NDArray[np.float64] | None = None,
    asym: bool = False,
) -> dict[str, float] | dict[str, Any]:
    """Return NNS nonlinear correlation and dependence (R ``NNS.dep``).

    Bivariate form ``nns_dep(x, y)``: returns ``{"Correlation": float,
    "Dependence": float}``. ``Correlation`` is signed in ``[-1, 1]`` and
    ``Dependence`` is in ``[0, 1]``. With ``asym=False`` (default) the measure is
    symmetric, ``nns_dep(x, y) == nns_dep(y, x)``. With ``asym=True`` the
    partition is built on ``x`` only and the result measures the dependence of
    ``y`` on ``x`` (y as a function of x); e.g. ``nns_dep(x, x**2, asym=True)``
    is close to 1 while ``nns_dep(x**2, x, asym=True)`` is not.

    Matrix form ``nns_dep(X)`` (``y=None``, ``X`` a 2-D array or DataFrame with
    one column per variable): returns ``{"Correlation": M, "Dependence": M}``
    with ones on the diagonal, like R's ``NNS.dep(X)``. With ``asym=True`` entry
    ``[i, j]`` is ``nns_dep(X[:, i], X[:, j], asym=True)``, i.e. row variable ->
    column variable. A DataFrame input returns labelled DataFrames.

    Inputs must be finite. R's ``p.value`` option is not implemented.
    """
    if y is None:
        return _dep_matrix(x, asym)
    x_values, y_values = _as_pair(x, y)

    native_dep_pair = native_fn("nns_dep_pair")
    if native_dep_pair is not None and _dep_native_is_exact(x_values, y_values):
        res = native_dep_pair(
            np.ascontiguousarray(x_values),
            np.ascontiguousarray(y_values),
            bool(asym),
        )
        return {"Correlation": float(res["Correlation"]), "Dependence": float(res["Dependence"])}

    if _is_constant(x_values) or _is_constant(y_values):
        return {"Correlation": 0.0, "Dependence": 0.0}

    obs_req = max(8, x_values.size // 8)
    quad_xy = _xonly_partition(x_values, obs_req)
    quad_yx = _xonly_partition(y_values, obs_req)
    correlation, dependence = _dep_pair(x_values, y_values, quad_xy, quad_yx, asym)
    return {"Correlation": correlation, "Dependence": dependence}


def _dep_matrix(x: Any, asym: bool) -> dict[str, Any]:
    """R NNS.dep.matrix: pairwise NNS.dep over the columns of ``x``."""
    labels = _column_labels(x)
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("supply both 'x' and 'y' or a matrix-like 'x'.")
    if values.shape[1] < 2:
        raise ValueError("matrix input to nns_dep needs at least two columns.")
    n_vars = values.shape[1]
    rhos = np.zeros((n_vars, n_vars), dtype=np.float64)
    deps = np.zeros((n_vars, n_vars), dtype=np.float64)
    for i in range(n_vars - 1):
        for j in range(i + 1, n_vars):
            upper = nns_dep(values[:, i], values[:, j], asym=asym)
            if asym:
                # Lower triangle: [j, i] = NNS.dep(x_j, x_i); R zeroes NaN there.
                lower = nns_dep(values[:, j], values[:, i], asym=True)
                rhos[j, i] = _finite_or_zero(lower["Correlation"])
                deps[j, i] = _finite_or_zero(lower["Dependence"])
                rhos[i, j] = upper["Correlation"]
                deps[i, j] = upper["Dependence"]
            else:
                rhos[i, j] = rhos[j, i] = upper["Correlation"]
                deps[i, j] = deps[j, i] = upper["Dependence"]
    np.fill_diagonal(rhos, 1.0)
    np.fill_diagonal(deps, 1.0)
    return {
        "Correlation": _labelled_matrix(rhos, labels),
        "Dependence": _labelled_matrix(deps, labels),
    }


def _column_labels(x: Any) -> list[Any] | None:
    """Column labels of a DataFrame-like input (R keeps colnames as dimnames)."""
    columns = getattr(x, "columns", None)
    if columns is None:
        return None
    try:
        return list(columns)
    except TypeError:
        return None


def _labelled_matrix(matrix: NDArray[np.float64], labels: list[Any] | None) -> Any:
    """Return ``matrix`` as a DataFrame indexed by ``labels`` when the input had them."""
    if labels is None or len(labels) != matrix.shape[0]:
        return matrix
    import pandas as pd  # type: ignore[import-untyped]  # labels imply a pandas-like input

    return pd.DataFrame(matrix, index=labels, columns=labels)


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
    # R (and the native kernel) add the cell terms in std::unordered_map order
    # of the hashed labels; this adds them in first-seen order, so the totals
    # can differ from R's in the last ulp (the cell copulas themselves match).
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


def _sequential_sum(values: NDArray[np.float64]) -> float:
    """Left-to-right double sum, like a C++ ``for`` loop (``np.sum`` is pairwise)."""
    if values.size == 0:
        return 0.0
    return float(np.cumsum(values)[-1])


def _copula_moments(
    x: NDArray[np.float64], y: NDArray[np.float64]
) -> tuple[float, float, NDArray[np.float64], NDArray[np.float64]]:
    """Degree-0 co-moment and DPM shares about the cell means, as R ``copula_signed``.

    R's ``NNS_dep.cpp`` takes the cell means as a sequential double sum / n and
    scales the counts by ``1 / n``. A pairwise (``np.mean``) cell mean can differ
    in the last ulp on evenly spaced or tied data, and a point sitting exactly on
    the mean then switches between the concordant and discordant counts.
    """
    n = x.size
    dx = x - _sequential_sum(x) / n
    dy = y - _sequential_sum(y) / n
    inv_n = 1.0 / n
    d0_concordant = np.count_nonzero((dx > 0.0) & (dy > 0.0)) + np.count_nonzero(
        (dx <= 0.0) & (dy <= 0.0)
    )
    aligned = ((dx < 0.0) & (dy < 0.0)) | ((dx > 0.0) & (dy > 0.0))
    d0_co = float(d0_concordant) * inv_n
    dpm_d0 = float(np.count_nonzero(~aligned)) * inv_n
    return d0_co, dpm_d0, dx, dy


def _copula_signed(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """Signed bivariate copula of one cell (R ``NNS_dep.cpp`` ``copula_signed``).

    Every sum runs in R's order (one sequential pass), so tie-sensitive comparisons
    and the result match R bit for bit.
    """
    n = x.size
    if n < 2:
        return 0.0

    d0_co, dpm_d0, dx, dy = _copula_moments(x, y)
    if d0_co == 1.0 or d0_co == 0.0:
        return 1.0

    product = dx * dy
    upper = (dx >= 0.0) & (dy >= 0.0)
    lower = ~upper & (dx <= 0.0) & (dy <= 0.0)
    discordant = ~(upper | lower)
    c1_cupm = _sequential_sum(np.where(upper, product, 0.0))
    c1_clpm = _sequential_sum(np.where(lower, product, 0.0))
    c1_dpm = _sequential_sum(np.where(discordant, np.abs(dx) * np.abs(dy), 0.0))
    c1_total = c1_cupm + c1_clpm + c1_dpm
    co_d1 = (c1_cupm + c1_clpm) / c1_total if c1_total > 0.0 else 0.0
    dpm_d1 = c1_dpm / c1_total if c1_total > 0.0 else 0.0

    discrete_dep = min(1.0, max(0.0, abs(d0_co - 0.5) / 0.5))
    continuous_dep = min(1.0, max(0.0, abs(co_d1 - 0.5) / 0.5))
    # DPM_nD counts a point as concordant when it is all-below OR all-above the
    # target (both fully-aligned orthants), so under independence the discordant
    # null is 1 - 2*0.5**n = 0.5 for the bivariate copula, not 1 - 0.5**n = 0.75.
    # The old 0.75 anchor left a fixed 1/3 residual per discordant term, i.e. a
    # ~0.41 dependence floor that never vanished for independent data.
    nd_disc_dep = abs(dpm_d0 - 0.5) / 0.5
    nd_cont_dep = abs(dpm_d1 - 0.5) / 0.5

    copula = math.sqrt((discrete_dep + continuous_dep + nd_disc_dep + nd_cont_dep) / 4.0)
    cov = _sequential_sum(product)
    varx = _sequential_sum(dx * dx)
    slope_sign = 0.0 if varx == 0.0 else (1.0 if cov > 0.0 else (-1.0 if cov < 0.0 else 0.0))
    return copula * slope_sign


def _copula_degree0_unsigned(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """R ``NNS_dep.cpp`` ``copula_degree0_unsigned`` (the discrete-case blend term)."""
    if x.size < 2:
        return 0.0
    d0_co, dpm_d0, _, _ = _copula_moments(x, y)
    disc_dep = min(1.0, max(0.0, abs(d0_co - 0.5) / 0.5))
    # Bivariate discordant independence null is 1 - 2*0.5**2 = 0.5 (see
    # _copula_signed); the old 0.75 anchor never vanished for independent data.
    nd_disc = abs(dpm_d0 - 0.5) / 0.5
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
    # Clamp before the power: rows with a negative deviation are masked out anyway,
    # and a fractional power of a negative number only produced RuntimeWarnings.
    powered = np.prod(np.maximum(diff, 0.0) ** degree, axis=1)
    return float(np.mean(np.where(valid, powered, 0.0)))


def _cupm_nd(data: NDArray[np.float64], target: NDArray[np.float64], degree: float) -> float:
    diff = data - target[np.newaxis, :]
    if degree == 0.0:
        return float(np.mean(np.all(diff >= 0.0, axis=1)))
    valid = np.all(diff >= 0.0, axis=1)
    powered = np.prod(np.maximum(diff, 0.0) ** degree, axis=1)
    return float(np.mean(np.where(valid, powered, 0.0)))


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
    """NNS gravity central tendency (R ``NNS.gravity``) used by the partitioners.

    The native ``gravity_exact`` kernel is bit-identical to the Python code below
    except that it clips its dense histogram at 1e6 bins (R does not), which moves
    the mode term onto outliers. It is used only when the bin count is
    provably below that cap.
    """
    values = np.asarray(x, dtype=np.float64).reshape(-1)
    native_gravity_exact = native_fn("gravity_exact")
    if native_gravity_exact is not None and _native_gravity_is_exact(values):
        return float(native_gravity_exact(np.ascontiguousarray(values)))
    return _gravity_python(values)


def _gravity_python(x: NDArray[np.float64]) -> float:
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

    origin = float(values[0])
    bin_count, occupied, counts = _sparse_bin_counts(values, width, origin)
    max_count = int(np.max(counts))
    max_positions = np.flatnonzero(counts == max_count)
    if max_positions.size == 1:
        center = int(occupied[max_positions[0]])
        lo = max(0, center - 1)
        hi = min(bin_count - 1, center + 1)
        selected_bins = np.arange(lo, hi + 1, dtype=np.int64)
        selected_names = origin + selected_bins.astype(np.float64) * width
        selected_counts = _counts_at(selected_bins, occupied, counts)
        denominator = float(np.sum(selected_counts))
        mode_gravity = float(np.sum(selected_names * selected_counts) / denominator)
    elif bin_count <= _MAX_DENSE_BINS:
        # All bins are averaged; keep the dense layout so the pairwise sum (and
        # hence the result) is bit-identical to the native kernel.
        dense_counts = np.zeros(bin_count, dtype=np.int64)
        dense_counts[occupied] = counts
        bin_names = origin + np.arange(bin_count, dtype=np.float64) * width
        mode_gravity = float(np.sum(bin_names * dense_counts) / float(np.sum(dense_counts)))
    else:
        # Beyond the dense range use R's own long-double sequential sum; empty
        # bins add exact zeros there, so only the occupied ones are needed.
        names = origin + occupied.astype(np.longdouble) * np.longdouble(width)
        weighted = np.cumsum(names * counts.astype(np.longdouble))[-1]
        mode_gravity = float(weighted / np.longdouble(n))
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


_MAX_DENSE_BINS = 1_000_000


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
    # R (NNS 13.x) does not cap the bin count; the guard only bounds memory
    # for pathologically narrow widths and never binds on realistic data.
    bin_count = min(max(1, bin_count), _MAX_DENSE_BINS)
    bin_names = origin + np.arange(bin_count, dtype=np.float64) * width
    if bin_count == 1:
        return bin_names, np.array([values.size], dtype=np.int64)
    indices = np.floor((values - origin) / width).astype(np.int64)
    indices = np.clip(indices, 0, bin_count - 1)
    counts = np.bincount(indices, minlength=bin_count)
    return bin_names, counts


def _bin_count(value_range: float, width: float) -> int:
    """Dense histogram size used by R's simple_bin_counts (including its int overflow)."""
    int_max = np.iinfo(np.int32).max
    if width <= 0.0 or not math.isfinite(width):
        return 1
    bin_ratio = value_range / width + 1e-12
    if not math.isfinite(bin_ratio) or bin_ratio > int_max:
        # R's (int) cast overflows and the histogram collapses to one bin.
        return 1
    return max(1, math.floor(bin_ratio) + 1)


def _sparse_bin_counts(
    values: NDArray[np.float64],
    width: float,
    origin: float,
) -> tuple[int, NDArray[np.int64], NDArray[np.int64]]:
    """Occupied bins of R's fixed-width gravity histogram, without allocating it.

    ``values`` must be sorted. Returns the dense bin count R would allocate and
    the indices and counts of the occupied bins (ascending), so any number of
    empty bins costs nothing.
    """
    bin_count = _bin_count(float(values[-1]) - origin, width)
    if bin_count == 1:
        return 1, np.zeros(1, dtype=np.int64), np.array([values.size], dtype=np.int64)
    indices = np.floor((values - origin) / width).astype(np.int64)
    indices = np.clip(indices, 0, bin_count - 1)
    # Sorted values give non-decreasing bin indices, so the bins are runs.
    starts = np.flatnonzero(np.r_[True, indices[1:] != indices[:-1]])
    counts = np.diff(np.r_[starts, indices.size]).astype(np.int64)
    return bin_count, indices[starts], counts


def _counts_at(
    bins: NDArray[np.int64],
    occupied: NDArray[np.int64],
    counts: NDArray[np.int64],
) -> NDArray[np.int64]:
    position = np.clip(np.searchsorted(occupied, bins), 0, occupied.size - 1)
    found = occupied[position] == bins
    return np.where(found, counts[position], 0).astype(np.int64)


# The native gravity_exact kernel clips its dense histogram at this many bins.
_NATIVE_GRAVITY_BIN_CAP = 1_000_000
# Ratios this close to the cap (or to R's int overflow) are treated as unsafe;
# misclassifying a safe call only costs the slower exact path.
_CAP_MARGIN = 64.0


def _ratio_exceeds_native_cap(bin_ratio: NDArray[np.float64]) -> bool:
    int_max = float(np.iinfo(np.int32).max)
    with np.errstate(invalid="ignore"):
        unsafe = (bin_ratio >= _NATIVE_GRAVITY_BIN_CAP - 1 - _CAP_MARGIN) & (
            bin_ratio <= int_max + _CAP_MARGIN
        )
    return bool(np.any(unsafe))


def _group_bin_ratios(
    sorted_values: NDArray[np.float64],
    starts: NDArray[np.intp],
    sizes: NDArray[np.intp],
) -> NDArray[np.float64]:
    """``range / width + 1e-12`` of the gravity histogram of each sorted, finite group.

    Mirrors ``_quartiles_like_r_code`` and the width rule of ``_gravity``: for even
    m the quartiles are the order statistics at m // 4 and 3m // 4 (1-based); for
    odd m they interpolate with weights (m % 4) / 4 and (3m % 4) / 4. Groups
    without a histogram (m <= 3 or zero range) get NaN.
    """
    last = starts + sizes - 1
    value_range = sorted_values[last] - sorted_values[starts]
    odd = (sizes % 2).astype(bool)
    f1 = np.minimum(starts + np.maximum(sizes // 4, 1) - 1, last)
    f3 = np.minimum(starts + np.maximum((3 * sizes) // 4, 1) - 1, last)
    c1 = np.minimum(f1 + odd, last)
    c3 = np.minimum(f3 + odd, last)
    w1 = np.where(odd, (sizes % 4) / 4.0, 0.0)
    w3 = np.where(odd, ((3 * sizes) % 4) / 4.0, 0.0)
    lo1 = sorted_values[f1]
    lo3 = sorted_values[f3]
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        q1 = lo1 + w1 * (sorted_values[c1] - lo1)
        q3 = lo3 + w3 * (sorted_values[c3] - lo3)
        width = (q3 - q1) * sizes.astype(np.float64) ** -0.5
        width = np.where((width > 0.0) & np.isfinite(width), width, value_range / 128.0)
        ratio = value_range / width + 1e-12
    out: NDArray[np.float64] = np.where((sizes > 3) & (value_range > 0.0), ratio, np.nan)
    return out


def _run_starts(change: NDArray[np.bool_]) -> NDArray[np.intp]:
    """Start offsets of the runs of a sequence given ``change[i] = (s[i + 1] != s[i])``."""
    return np.concatenate((np.zeros(1, dtype=np.intp), np.flatnonzero(change) + 1))


def _native_gravity_is_exact(values: NDArray[np.float64]) -> bool:
    """Whether the native gravity_exact kernel stays below its histogram cap."""
    finite = values if np.isfinite(values).all() else values[np.isfinite(values)]
    m = finite.size
    if m <= 3:
        return True
    if m % 2 == 0:
        i1 = max(1, m // 4) - 1
        i3 = max(1, (3 * m) // 4) - 1
        kth = [0, i1, i3, m - 1]
    else:
        p25 = m * 0.25
        p75 = m * 0.75
        f25, c25 = min(max(1, math.floor(p25)), m) - 1, min(max(1, math.ceil(p25)), m) - 1
        f75, c75 = min(max(1, math.floor(p75)), m) - 1, min(max(1, math.ceil(p75)), m) - 1
        kth = [0, f25, c25, f75, c75, m - 1]
    # After the partition every needed order statistic sits at its sorted position.
    picked = np.partition(finite, sorted(set(kth)))
    part = dict(zip(kth, picked[kth].tolist(), strict=True))
    value_range = part[m - 1] - part[0]
    if value_range == 0.0:
        return True
    if m % 2 == 0:
        q1, q3 = part[i1], part[i3]
    else:
        q1 = part[f25] + (p25 - math.floor(p25)) * (part[c25] - part[f25])
        q3 = part[f75] + (p75 - math.floor(p75)) * (part[c75] - part[f75])
    width = (q3 - q1) * m**-0.5
    if not (width > 0.0 and math.isfinite(width)):
        width = value_range / 128.0
    if not width > 0.0:
        return True  # range / 128 underflowed: one bin, as in _bin_count
    ratio = value_range / width + 1e-12
    int_max = float(np.iinfo(np.int32).max)
    return not (_NATIVE_GRAVITY_BIN_CAP - 1 - _CAP_MARGIN <= ratio <= int_max + _CAP_MARGIN)


def _grouped_gravity_is_exact(values: NDArray[np.float64], group: NDArray[np.intp]) -> bool:
    """Whether native gravity on every group ``values[group == g]`` stays below the cap."""
    finite = np.isfinite(values)
    if not np.all(finite):
        values = values[finite]
        group = group[finite]
    if values.size == 0:
        return True
    order = np.lexsort((values, group))
    sorted_group = group[order]
    starts = _run_starts(sorted_group[1:] != sorted_group[:-1])
    sizes = np.diff(np.append(starts, sorted_group.size))
    return not _ratio_exceeds_native_cap(_group_bin_ratios(values[order], starts, sizes))


def _native_partition_is_exact(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    quadrant: NDArray[np.str_],
    xonly: bool,
    regression_points: bool = True,
) -> bool:
    """Check every gravity centre a native "off" partition computed against the cap.

    The groups split at depth ``d`` are the points sharing the first ``d + 1``
    characters of their final quadrant label (each split appends one digit). The
    regression points are the prior.quadrant groups: a label without its last
    digit, i.e. a group split at the previous depth (or every point, "pq", when
    nothing was split). The native result is exact up to the first gravity call
    that hits the cap, so that call is always among the groups checked here.
    """
    n = quadrant.size
    labels = np.ascontiguousarray(quadrant)
    width = labels.dtype.itemsize // 4
    chars = labels.view(np.uint32).reshape(n, width)
    n_splits = np.count_nonzero(chars, axis=1) - 1
    max_depth = int(np.max(n_splits)) if n else 0
    if max_depth == 0:
        # No split: the only gravity calls are the single "pq" regression point.
        if not regression_points:
            return True
        return _native_gravity_is_exact(x) and _native_gravity_is_exact(y)

    if xonly and np.all(np.isfinite(x)):
        # X-only groups are x-intervals, i.e. contiguous runs once x is sorted
        # (tied x values always share a group).
        perm = np.argsort(x, kind="stable")
        chars_sorted = chars[perm]
        splits_sorted = n_splits[perm]
        same = chars_sorted[1:] == chars_sorted[:-1]
        lcp = np.where(np.all(same, axis=1), width, np.argmin(same, axis=1))
        starts_list = []
        sizes_list = []
        for depth in range(max_depth):
            starts = _run_starts(lcp < depth + 1)
            sizes = np.diff(np.append(starts, n))
            keep = splits_sorted[starts] > depth
            starts_list.append(starts[keep])
            sizes_list.append(sizes[keep])
        ratios = _group_bin_ratios(
            x[perm], np.concatenate(starts_list), np.concatenate(sizes_list)
        )
        if _ratio_exceeds_native_cap(ratios):
            return False
        if not regression_points:
            return True
        # prior.quadrant groups: same split count k and the same first k characters;
        # their x centres are the depth-(k - 1) groups above, only y is new.
        prior_change = (splits_sorted[1:] != splits_sorted[:-1]) | (lcp < splits_sorted[1:])
        prior_group = np.cumsum(np.concatenate(([0], prior_change)))
        return _grouped_gravity_is_exact(y[perm], prior_group)

    # General case (two-axis partition or non-finite x): sort each split group per
    # depth via (group id, rank) keys. Every prior.quadrant group is one of them.
    variables = [x] if xonly else [x, y]
    ranks = []
    for values in variables:
        rank = np.empty(n, dtype=np.int64)
        rank[np.argsort(values, kind="stable")] = np.arange(n, dtype=np.int64)
        ranks.append(rank)
    dense = np.zeros(n, dtype=np.int64)
    for depth in range(max_depth):
        split = n_splits > depth
        for values, rank in zip(variables, ranks, strict=True):
            idx = np.flatnonzero(split & np.isfinite(values))
            if idx.size == 0:
                continue
            order = idx[np.argsort(dense[idx] * n + rank[idx])]
            sorted_group = dense[order]
            starts = _run_starts(sorted_group[1:] != sorted_group[:-1])
            sizes = np.diff(np.append(starts, order.size))
            if _ratio_exceeds_native_cap(_group_bin_ratios(values[order], starts, sizes)):
                return False
        digit = np.where(split, chars[:, depth + 1].astype(np.int64) - 48, 0)
        _, inverse = np.unique(dense * 5 + digit, return_inverse=True)
        dense = inverse.reshape(-1).astype(np.int64)
    if regression_points and xonly:
        # X-only partitions with non-finite x: y centres of the prior groups.
        prior_chars = chars.copy()
        prior_chars[np.arange(n), n_splits] = 0
        _, prior_group = np.unique(prior_chars, axis=0, return_inverse=True)
        return _grouped_gravity_is_exact(y, prior_group.reshape(-1))
    return True


def _dep_native_is_exact(x: NDArray[np.float64], y: NDArray[np.float64]) -> bool:
    """Whether nns_dep_pair's X-only partitions stay below the native gravity cap."""
    n = x.size
    obs_req = max(8, n // 8)
    if _gravity_subsets_are_safe(x, obs_req + 1) and _gravity_subsets_are_safe(y, obs_req + 1):
        return True
    native_part = native_fn("nns_part_off")
    if native_part is None:
        return False
    order = max(math.ceil(math.log2(max(1, n))), 1)
    for values in (x, y):
        res = native_part(
            np.ascontiguousarray(values), np.ascontiguousarray(values), True, order, obs_req, False
        )
        quadrant = np.asarray(res["dt"]["quadrant"], dtype=str)
        if not _native_partition_is_exact(values, values, quadrant, True, False):
            return False
    return True


def _gravity_subsets_are_safe(values: NDArray[np.float64], min_size: int) -> bool:
    """Cheap sufficient condition that native gravity on any subset stays below the cap.

    Covers every subset of ``values`` with at least ``min_size`` finite members. For
    a subset of m >= 4 sorted values, q1 <= s[c1] and q3 >= s[f3] for subset
    positions k(m) = 3m // 4 - m // 4 - (m % 2) apart; those two values are at least
    k(m) positions apart in the full sorted sample too, so q3 - q1 >= g_k =
    min_j (s[j + k] - s[j]). With range <= R the bin ratio is at most
    sqrt(m) * R / g_k (q3 == q1 uses the range / 128 width instead), bounded on
    each block m in [M, 2M) with the smallest k of the block.
    """
    finite = values[np.isfinite(values)]
    if finite.size < values.size:
        min_size = 4  # a group's finite count can be below its size
    n = finite.size
    lo = max(4, min_size)
    if lo > n:
        return True
    sorted_values = np.sort(finite)
    value_range = float(sorted_values[-1] - sorted_values[0])
    if value_range == 0.0:
        return True
    limit = _NATIVE_GRAVITY_BIN_CAP - 1 - _CAP_MARGIN
    block = lo
    while block <= n:
        hi = min(2 * block - 1, n)
        exact = [(3 * m) // 4 - m // 4 - (m % 2) for m in range(block, min(hi, block + 3) + 1)]
        # (m - 3) // 2 <= k(m) is non-decreasing, so it bounds the rest of the block.
        k = min([*exact, (block + 1) // 2]) if hi > block + 3 else min(exact)
        if k < 1:
            return False
        if k < n:
            gap = float(np.min(sorted_values[k:] - sorted_values[:-k]))
            if not gap > 0.0 or math.sqrt(hi) * value_range / gap + 1.0 >= limit:
                return False
        block *= 2
    return True


def _is_discrete_case(x: NDArray[np.float64], y: NDArray[np.float64]) -> bool:
    threshold = math.sqrt(x.size)
    return np.unique(x).size < threshold and np.unique(y).size < threshold


def _finite_or_zero(value: float) -> float:
    return value if math.isfinite(value) else 0.0


def _is_constant(values: NDArray[np.float64]) -> bool:
    return bool(np.all(values == values[0]))
