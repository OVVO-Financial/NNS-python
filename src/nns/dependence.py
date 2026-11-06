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
    if native_dep_pair is not None:
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
    # C++ starts the sum at +0.0, so an all-(-0.0) sum is +0.0.
    return 0.0 + float(np.cumsum(values)[-1])


def _copula_moments(
    x: NDArray[np.float64], y: NDArray[np.float64]
) -> tuple[int, float, float, NDArray[np.float64], NDArray[np.float64]]:
    """Degree-0 concordant count, co-moment and DPM shares about the cell means.

    As R ``copula_signed``: a point is concordant when it is co-upper (both
    deviations > 0) or co-lower (both <= 0) and discordant otherwise.

    R's ``NNS_dep.cpp`` takes the cell means as a sequential double sum / n and
    scales the counts by ``1 / n``. A pairwise (``np.mean``) cell mean can differ
    in the last ulp on evenly spaced or tied data, and a point sitting exactly on
    the mean then switches between the concordant and discordant counts.
    """
    n = x.size
    dx = x - _sequential_sum(x) / n
    dy = y - _sequential_sum(y) / n
    inv_n = 1.0 / n
    concordant = int(
        np.count_nonzero((dx > 0.0) & (dy > 0.0)) + np.count_nonzero((dx <= 0.0) & (dy <= 0.0))
    )
    # Discordant = neither co-lower nor co-upper, the same split as the concordant
    # count, so the degree-0 shares partition the cell (a tie used to count as
    # co-lower and discordant at once before the R tie fix).
    d0_co = float(concordant) * inv_n
    dpm_d0 = float(n - concordant) * inv_n
    return concordant, d0_co, dpm_d0, dx, dy


def _copula_signed(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """Signed bivariate copula of one cell (R ``NNS_dep.cpp`` ``copula_signed``).

    Every sum runs in R's order (one sequential pass), so tie-sensitive comparisons
    and the result match R bit for bit.
    """
    n = x.size
    if n < 2:
        return 0.0

    concordant, d0_co, dpm_d0, dx, dy = _copula_moments(x, y)
    # Every point concordant: perfect positive dependence; every point discordant:
    # perfect negative dependence (as in the patched R; this used to return +1 too). The
    # count is compared with n, not the share with 1: count * (1/n) is not exactly 1
    # for many n (49 * (1/49) != 1).
    if concordant == n:
        return 1.0
    if concordant == 0:
        return -1.0

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
    _, d0_co, dpm_d0, _, _ = _copula_moments(x, y)
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
    # Discordant = neither all-lower (x <= target, the CLPM count) nor all-upper
    # (x > target, the CUPM count), so the degree-0 counts partition the sample
    # (ties used to count as discordant and lower at once). At degree > 0
    # a tied coordinate contributes a zero factor either way.
    all_below = np.all(diff <= 0.0, axis=1)
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
        # Strictly above in every coordinate, as Co.UPM (x > target): ties belong
        # to the lower side, so CLPM + CUPM + DPM = 1 at degree 0.
        return float(np.mean(np.all(diff > 0.0, axis=1)))
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
    """R 14.0 gravity for partition splits while NNS-core is being synchronized.

    The vendored native gravity_exact implements the earlier 13.3 statistic:
    using it would change the partition tree and every derived regression
    estimate. Do not silently substitute it for this R 14.0 oracle.
    """
    values = np.asarray(x, dtype=np.float64).reshape(-1)
    return _gravity_python(values)


def _gravity_python(x: NDArray[np.float64]) -> float:
    """Match R 14.0 NNS_gravity_cpp, without allocating a dense histogram.

    R 14.0 uses type-7 quartiles and the *observations* in the modal bin
    neighborhood (not the left edges of the bins). Tied maximum-frequency
    bins select the entire sample. Long-double accumulation follows R's
    compiled implementation. Sparse bin indices preserve very large gaps.
    """
    values = np.asarray(x, dtype=np.float64)
    finite = values[np.isfinite(values)]
    n = finite.size
    if n == 0:
        return float("nan")
    if n <= 3:
        return _small_median(finite)
    if np.all(finite == finite[0]):
        return float(finite[0])
    ordered = np.sort(finite)
    span = abs(float(ordered[-1]) - float(ordered[0]))
    if span == 0.0:
        return float(ordered[0])

    # R's quantile(type=7): h = (n - 1) * p, with zero-based indices.
    def type7(p: float) -> float:
        h = (n - 1) * p
        lo = math.floor(h)
        hi = min(lo + 1, n - 1)
        return float(ordered[lo] + (h - lo) * (ordered[hi] - ordered[lo]))

    q1, q2, q3 = type7(0.25), type7(0.50), type7(0.75)
    width = (q3 - q1) * n**-0.5
    if not (width > 0.0 and math.isfinite(width)):
        width = span / 128.0

    _, used_width, occupied, counts = _sparse_bin_counts(
        ordered, width, float(ordered[0])
    )
    max_count = int(np.max(counts))
    winners = np.flatnonzero(counts == max_count)

    zero = np.longdouble(0.0)
    mean = float(
        (zero + np.cumsum(ordered.astype(np.longdouble))[-1]) / np.longdouble(n)
    )
    if winners.size != 1:
        modal_mean = mean
    else:
        # Include all observations from the winning bin and its two neighbors.
        # This is a contiguous interval because the inputs are sorted.
        center = int(occupied[int(winners[0])])
        with np.errstate(over="ignore", divide="ignore"):
            positions = np.floor((ordered - float(ordered[0])) / used_width)
        selected = (positions >= center - 1) & (positions <= center + 1)
        modal_observations = ordered[selected]
        modal_mean = float(
            (zero + np.cumsum(modal_observations.astype(np.longdouble))[-1])
            / np.longdouble(modal_observations.size)
        )

    result = 0.25 * (q2 + modal_mean + mean + 0.5 * (q1 + q3))
    return float(result) if math.isfinite(result) else float(q2)


def _small_median(values: NDArray[np.float64]) -> float:
    """Median as NNS-core computes it for n <= 3 (np.median maps -0.0 to +0.0)."""
    ordered = np.sort(values)
    half = ordered.size // 2
    if ordered.size % 2:
        return float(ordered[half])
    return float(0.5 * (ordered[half - 1] + ordered[half]))


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


# NNS-core simple_bin_counts widens bins whose index would exceed 2**52 (R's
# int cast overflowed beyond INT_MAX instead and collapsed the histogram).
_MAX_BIN_INDEX = 4503599627370496.0


def _sparse_bin_counts(
    values: NDArray[np.float64],
    width: float,
    origin: float,
) -> tuple[int, float, NDArray[np.int64], NDArray[np.int64]]:
    """Occupied bins of R's fixed-width gravity histogram, without allocating it.

    ``values`` must be sorted. Bins are ``[origin + k * width, origin + (k + 1) *
    width)`` for ``k < floor((max - origin) / width + 1e-12) + 1``, as NNS-core's
    ``simple_bin_counts``. Returns that bin count, the (possibly widened) width,
    and the indices and counts of the occupied bins (ascending), so any number of
    empty bins costs nothing.
    """
    span = float(values[-1]) - origin
    with np.errstate(over="ignore", divide="ignore"):
        last = float(np.floor(np.float64(span) / width + 1e-12))
        if not last < _MAX_BIN_INDEX:
            width = span / _MAX_BIN_INDEX
            last = min(float(np.floor(np.float64(span) / width + 1e-12)), _MAX_BIN_INDEX)
    bin_count = int(last) + 1 if last >= 0.0 else 1
    with np.errstate(over="ignore", divide="ignore"):
        position = np.floor((values - origin) / width)
    indices = np.clip(position, 0.0, float(bin_count - 1)).astype(np.int64)
    # Sorted values give non-decreasing bin indices, so the bins are runs.
    starts = np.flatnonzero(np.r_[True, indices[1:] != indices[:-1]])
    counts = np.diff(np.r_[starts, indices.size]).astype(np.int64)
    return bin_count, width, indices[starts], counts


def _counts_at(
    bins: NDArray[np.int64],
    occupied: NDArray[np.int64],
    counts: NDArray[np.int64],
) -> NDArray[np.int64]:
    position = np.clip(np.searchsorted(occupied, bins), 0, occupied.size - 1)
    found = occupied[position] == bins
    return np.where(found, counts[position], 0).astype(np.int64)


def _is_discrete_case(x: NDArray[np.float64], y: NDArray[np.float64]) -> bool:
    threshold = math.sqrt(x.size)
    return np.unique(x).size < threshold and np.unique(y).size < threshold


def _finite_or_zero(value: float) -> float:
    return value if math.isfinite(value) else 0.0


def _is_constant(values: NDArray[np.float64]) -> bool:
    return bool(np.all(values == values[0]))
