"""Stochastic dominance routines matching NNS' discrete SD conventions.

Every pairwise verdict (``fsd``/``ssd``/``tsd``, their ``*_uni`` forms, the
efficient set and SD clustering) comes from one test, a port of R's native
``sd_dom_pair``: the curves are compared on the merged sample points and, for
TSD and continuous FSD, also at their closed-form extrema between those points
(LPM2 is piecewise quadratic and the continuous ratio linear-fractional, so the
difference can peak between knots). Differences up to ``64 * eps * S**d``
(``S`` the largest absolute value of the two samples; ``d`` = 1 for SSD and the
mean gate, 2 for TSD, a ratio analogue for continuous FSD) are ties, so rounding
noise does not decide dominance; discrete FSD compares exact ECDF fractions.
Equal curves are non-dominance (R's curve equality guard). Efficient-set output
follows the R C++ routine's LPM-at-global-maximum ordering and original-index
tie break.

Pairwise tests accept samples of unequal length: each sample's curve is
evaluated on the merged threshold grid with its own length, as R's
``NNS.FSD``/``NNS.SSD``/``NNS.TSD`` and (since R fix 019fc1a) the ``.uni``
walkers do.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import linkage  # type: ignore[import-untyped]
from scipy.spatial.distance import squareform  # type: ignore[import-untyped]

from nns.core import _as_1d_values

_SD_ORDER_STAT_TARGET_BLOCK_COLUMNS = 64
# R's native SD test ignores excesses up to 64 * DBL_EPSILON (relative to the data scale).
_SD_EPS64 = 64.0 * float(np.finfo(np.float64).eps)


@dataclass(frozen=True)
class _SDPrefixPrecomputed:
    values: NDArray[np.float64]
    sorted_values: NDArray[np.float64]
    prefix1: NDArray[np.float64]
    prefix2: NDArray[np.float64] | None
    identical: NDArray[np.bool_]


@dataclass(frozen=True)
class _SDOrderStatPrecomputed:
    values: NDArray[np.float64]
    sorted_values: NDArray[np.float64]
    identical: NDArray[np.bool_]


def fsd(x: NDArray[np.float64], y: NDArray[np.float64], type: str = "discrete") -> int:
    """First-order stochastic dominance; ``x`` and ``y`` may differ in length.

    ``type="discrete"`` compares empirical CDFs; ``type="continuous"`` compares the
    degree-1 ratio ``LPM1 / (LPM1 + UPM1)``, as R's ``NNS.FSD(type=)`` does. Like R,
    an unrecognised ``type`` warns and runs the continuous test; where both
    partial moments vanish the ratio is the degree-0 ratio (the CDF value), as
    in ``lpm_ratio`` and the patched R.
    """
    x_values = _as_sd_values(x, "x")
    y_values = _as_sd_values(y, "y")
    discrete = _checked_fsd_type(type) == "discrete"
    return _sd_result(x_values, y_values, 1, discrete=discrete)


def fsd_uni(x: NDArray[np.float64], y: NDArray[np.float64], type: str = "discrete") -> int:
    """Unidirectional first-order stochastic dominance: 1 if x dominates y, else 0.

    Like R's ``NNS.FSD.uni``, an unrecognised ``type`` warns and runs the discrete test.
    """
    x_values = _as_sd_values(x, "x")
    y_values = _as_sd_values(y, "y")
    discrete = _checked_fsd_type(type) != "continuous"
    return int(_dominates_uni(x_values, y_values, 1, discrete=discrete))


def ssd(x: NDArray[np.float64], y: NDArray[np.float64]) -> int:
    """Second-order stochastic dominance; ``x`` and ``y`` may differ in length."""
    x_values = _as_sd_values(x, "x")
    y_values = _as_sd_values(y, "y")
    return _sd_result(x_values, y_values, 2)


def ssd_uni(x: NDArray[np.float64], y: NDArray[np.float64]) -> int:
    """Unidirectional second-order stochastic dominance: 1 if x dominates y, else 0."""
    x_values = _as_sd_values(x, "x")
    y_values = _as_sd_values(y, "y")
    return int(_dominates_uni(x_values, y_values, 2, discrete=True))


def tsd(x: NDArray[np.float64], y: NDArray[np.float64]) -> int:
    """Third-order stochastic dominance; ``x`` and ``y`` may differ in length."""
    x_values = _as_sd_values(x, "x")
    y_values = _as_sd_values(y, "y")
    return _sd_result(x_values, y_values, 3)


def tsd_uni(x: NDArray[np.float64], y: NDArray[np.float64]) -> int:
    """Unidirectional third-order stochastic dominance: 1 if x dominates y, else 0."""
    x_values = _as_sd_values(x, "x")
    y_values = _as_sd_values(y, "y")
    return int(_dominates_uni(x_values, y_values, 3, discrete=True))


def nns_sd_cluster(
    data: NDArray[np.float64],
    degree: int = 1,
    type: str = "discrete",
    min_cluster: int = 1,
    dendrogram: bool = False,
    names: Sequence[str] | None = None,
) -> dict[str, object]:
    """Cluster variables by iteratively peeling stochastic-dominance efficient sets.

    Cluster members are named by ``names``, else by the columns of a DataFrame-like
    ``data`` (R's ``colnames``), else ``X_1``, ``X_2``, ...
    """
    column_labels = getattr(data, "columns", None)
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("data must be a 2D array.")
    if values.shape[0] == 0:
        raise ValueError("data must have at least one row.")
    degree_int = _sd_degree(degree)
    if not np.all(np.isfinite(values)):
        raise ValueError("data must contain only finite values.")
    type_value = _sd_type_value(degree_int, type)
    discrete = degree_int != 1 or type_value != "continuous"
    min_cluster = int(min_cluster)
    if min_cluster < 0:
        raise ValueError("min_cluster must be non-negative.")

    column_count = values.shape[1]
    if names is None and column_labels is not None and len(column_labels) == column_count:
        all_names = [str(name) for name in column_labels]
    elif names is None:
        all_names = [f"X_{index + 1}" for index in range(column_count)]
    else:
        all_names = [str(name) for name in names]
        if len(all_names) != column_count:
            raise ValueError("names length must match the number of data columns.")

    order_stat_dominance_matrix = None
    prefix_precomputed = None
    if degree_int == 1 and discrete:
        order_stat_precomputed = _order_stat_sd_precompute(values)
        order_stat_dominance_matrix = _dominance_matrix_from_order_stats(order_stat_precomputed)
    else:
        prefix_precomputed = _prefix_sd_precompute(values, degree_int, discrete=discrete)
    active = list(range(column_count))
    clusters: dict[str, list[str]] = {}
    iteration = 1

    while len(active) > min_cluster:
        if order_stat_dominance_matrix is not None:
            sd_set_indices = _sd_efficient_active_indices_from_matrix(
                values,
                active,
                degree_int,
                order_stat_dominance_matrix,
            )
        else:
            assert prefix_precomputed is not None
            sd_set_indices = _sd_efficient_active_indices_from_prefix_kept(
                prefix_precomputed,
                active,
                degree_int,
                discrete=discrete,
            )
        sd_set = [all_names[index] for index in sd_set_indices]
        if not sd_set:
            break

        clusters[f"Cluster_{iteration}"] = sd_set
        remove_indices = set(sd_set_indices)
        active = [index for index in active if index not in remove_indices]
        iteration += 1

        if len(active) <= min_cluster:
            # R assigns colnames(<0-column matrix>) == NULL here, which adds nothing.
            if active:
                clusters[f"Cluster_{iteration}"] = [all_names[index] for index in active]
            break

    if len(active) > min_cluster and f"Cluster_{iteration}" not in clusters:
        clusters[f"Cluster_{iteration}"] = [all_names[index] for index in active]

    if clusters:
        final_cluster_name = f"Cluster_{len(clusters)}"
        if len(clusters[final_cluster_name]) < min_cluster and len(clusters) > 1:
            previous_cluster_name = f"Cluster_{len(clusters) - 1}"
            clusters[previous_cluster_name].extend(clusters[final_cluster_name])
            del clusters[final_cluster_name]

    if dendrogram:
        all_vars = [name for cluster in clusters.values() for name in cluster]
        if len(all_vars) < 2:
            return {"Clusters": clusters, "Order": None}
        return {
            "Clusters": clusters,
            "Dendrogram": _sd_cluster_hclust(clusters, all_names),
        }

    return {"Clusters": clusters}


def _sd_cluster_hclust(
    clusters: dict[str, list[str]],
    original_names: Sequence[str],
) -> dict[str, object]:
    all_vars = [name for cluster in clusters.values() for name in cluster]
    cluster_labels = np.asarray(
        [
            cluster_index
            for cluster_index, cluster in enumerate(clusters.values(), start=1)
            for _ in cluster
        ],
        dtype=np.float64,
    )
    extraction_order = np.arange(1, len(all_vars) + 1, dtype=np.float64)
    epsilon = 0.0 if len(clusters) == 1 else 1e-3
    n = len(original_names)
    distances = n * np.abs(cluster_labels[:, np.newaxis] - cluster_labels[np.newaxis, :])
    distances = distances + epsilon * np.abs(
        extraction_order[:, np.newaxis] - extraction_order[np.newaxis, :]
    )
    condensed = squareform(distances, checks=False)
    linked = linkage(condensed, method="complete")
    merge = _r_hclust_merge(linked, len(all_vars))
    original_positions = {name: index + 1 for index, name in enumerate(original_names)}
    order = np.asarray([original_positions[name] for name in all_vars], dtype=np.int64)
    return {
        "merge": merge,
        "height": linked[:, 2].astype(np.float64),
        "order": order,
        "labels": np.asarray(all_vars, dtype=str),
        "method": "complete",
        "call": 'hclust(d = dist_matrix, method = "complete")',
        "dist.method": None,
    }


def _r_hclust_merge(linked: NDArray[np.float64], n_obs: int) -> NDArray[np.int64]:
    out = np.empty((linked.shape[0], 2), dtype=np.int64)
    cluster_to_r_id: dict[int, int] = {}
    for row_index, row in enumerate(linked):
        for col_index, cluster_id_value in enumerate(row[:2]):
            cluster_id = int(cluster_id_value)
            if cluster_id < n_obs:
                out[row_index, col_index] = -(cluster_id + 1)
            else:
                out[row_index, col_index] = cluster_to_r_id[cluster_id]
        cluster_to_r_id[n_obs + row_index] = row_index + 1
    return out


def sd_efficient_set(
    returns: NDArray[np.float64],
    degree: int,
    type: str = "discrete",
) -> list[int]:
    """Return indices of non-dominated columns at the requested SD degree.

    ``degree`` is coerced like R's ``as.integer``: 1.5 runs first-order (FSD) and
    ``"2"`` runs second-order; anything that does not truncate to 1, 2 or 3 raises.
    """
    values = np.asarray(returns, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("returns must be a 2D array.")
    if values.shape[0] == 0:
        raise ValueError("returns must have at least one row.")
    degree = _sd_degree(degree)
    if not np.all(np.isfinite(values)):
        raise ValueError("returns must contain only finite values.")

    type_value = _sd_type_value(degree, type)
    discrete = degree != 1 or type_value != "continuous"

    if values.shape[1] == 0:
        return []

    # Every pair is evaluated on its own merged threshold grid (as R does), so
    # the verdict for a pair never depends on which other columns are present.
    active = list(range(values.shape[1]))
    if degree == 1 and discrete:
        order_stat_precomputed = _order_stat_sd_precompute(values)
        dominance_matrix = _dominance_matrix_from_order_stats(order_stat_precomputed)
        return _sd_efficient_active_indices_from_matrix(
            values,
            active,
            degree,
            dominance_matrix,
        )
    prefix_precomputed = _prefix_sd_precompute(values, degree, discrete=discrete)
    return _sd_efficient_active_indices_from_prefix_kept(
        prefix_precomputed,
        active,
        degree,
        discrete=discrete,
    )


def _sd_degree(degree: object) -> int:
    """R's ``as.integer(degree)`` (truncation toward zero), restricted to 1, 2 or 3."""
    try:
        value = int(float(degree))  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("degree must be 1, 2, or 3.") from error
    if value not in (1, 2, 3):
        raise ValueError("degree must be 1, 2, or 3.")
    return value


def _sd_type_value(degree: int, type: str) -> str:
    if degree != 1:
        return "discrete"
    # R's C++ efficient set warns for degree 1 and then runs the discrete test.
    type_value = _checked_fsd_type(type, stacklevel=4)
    return "continuous" if type_value == "continuous" else "discrete"


def _checked_fsd_type(type: str, *, stacklevel: int = 3) -> str:
    type_value = type.lower()
    if type_value not in {"discrete", "continuous"}:
        warnings.warn(
            "type needs to be either 'discrete' or 'continuous'",
            UserWarning,
            stacklevel=stacklevel,
        )
    return type_value


def _prefix_sd_precompute(
    values: NDArray[np.float64],
    degree: int,
    *,
    discrete: bool,
) -> _SDPrefixPrecomputed:
    sorted_values = np.asfortranarray(np.sort(values, axis=0))
    prefix1 = _prefix_sum(sorted_values)
    prefix2 = _prefix_sum(np.asfortranarray(sorted_values * sorted_values)) if degree == 3 else None
    return _SDPrefixPrecomputed(
        values=values,
        sorted_values=sorted_values,
        prefix1=prefix1,
        prefix2=prefix2,
        identical=_identical_sorted_columns(sorted_values),
    )


def _order_stat_sd_precompute(values: NDArray[np.float64]) -> _SDOrderStatPrecomputed:
    sorted_values = np.asfortranarray(np.sort(values, axis=0))
    return _SDOrderStatPrecomputed(
        values=values,
        sorted_values=sorted_values,
        identical=_identical_sorted_columns(sorted_values),
    )


def _identical_sorted_columns(sorted_values: NDArray[np.float64]) -> NDArray[np.bool_]:
    """k x k mask of columns whose sorted samples are equal (R's identical_samples).

    Columns are bucketed by the bytes of their sorted values, so memory is O(k n)
    for the keys plus the k x k result, instead of a k x k x n comparison tensor.
    Adding 0.0 maps -0.0 to 0.0 so the byte keys agree with ``==``.
    """
    labels: dict[bytes, int] = {}
    column_labels = np.empty(sorted_values.shape[1], dtype=np.intp)
    for index in range(sorted_values.shape[1]):
        key = np.ascontiguousarray(sorted_values[:, index] + 0.0).tobytes()
        column_labels[index] = labels.setdefault(key, len(labels))
    return np.asarray(column_labels[:, np.newaxis] == column_labels[np.newaxis, :])


def _sd_efficient_active_indices_from_matrix(
    values: NDArray[np.float64],
    active: Sequence[int],
    degree: int,
    dominance_matrix: NDArray[np.bool_],
) -> list[int]:
    if not active:
        return []

    active_array = np.asarray(active, dtype=np.intp)
    tmax = float(np.max(values[:, active_array]))
    order_lpm = _lpm_at_target(values[:, active_array], tmax, degree)
    order = [
        active[int(position)]
        for position in sorted(
            range(len(active)),
            key=lambda position: (order_lpm[position], active[position]),
        )
    ]

    keep: list[int] = []
    for index in order:
        dominated = any(dominance_matrix[kept, index] for kept in keep)
        if not dominated:
            keep.append(index)
    return keep


def _sd_efficient_active_indices_from_prefix_kept(
    precomputed: _SDPrefixPrecomputed,
    active: Sequence[int],
    degree: int,
    *,
    discrete: bool,
) -> list[int]:
    if not active:
        return []

    active_array = np.asarray(active, dtype=np.intp)
    tmax = float(np.max(precomputed.values[:, active_array]))
    order_lpm = _lpm_at_target(precomputed.values[:, active_array], tmax, degree)
    order = [
        active[int(position)]
        for position in sorted(
            range(len(active)),
            key=lambda position: (order_lpm[position], active[position]),
        )
    ]

    keep: list[int] = []
    for index in order:
        dominated = _any_prefix_source_dominates(
            precomputed,
            keep,
            index,
            degree,
            discrete=discrete,
        )
        if not dominated:
            keep.append(index)
    return keep


def _dominance_matrix_from_order_stats(
    precomputed: _SDOrderStatPrecomputed,
) -> NDArray[np.bool_]:
    sorted_values = precomputed.sorted_values
    columns = sorted_values.shape[1]
    dominates = np.zeros((columns, columns), dtype=np.bool_)

    for target_start in range(0, columns, _SD_ORDER_STAT_TARGET_BLOCK_COLUMNS):
        target_stop = min(target_start + _SD_ORDER_STAT_TARGET_BLOCK_COLUMNS, columns)
        target_values = sorted_values[:, target_start:target_stop]
        ge_all = np.all(sorted_values[:, :, np.newaxis] >= target_values[:, np.newaxis, :], axis=0)
        gt_any = np.any(sorted_values[:, :, np.newaxis] > target_values[:, np.newaxis, :], axis=0)
        dominates[:, target_start:target_stop] = ge_all & gt_any

    dominates &= np.logical_not(precomputed.identical)
    np.fill_diagonal(dominates, False)
    return dominates


def _any_prefix_source_dominates(
    precomputed: _SDPrefixPrecomputed,
    source_indices: Sequence[int],
    target_index: int,
    degree: int,
    *,
    discrete: bool,
) -> bool:
    return any(
        _dominates_from_prefix_pair(
            precomputed,
            source_index,
            target_index,
            degree,
            discrete=discrete,
        )
        for source_index in source_indices
    )


def _dominates_from_prefix_pair(
    precomputed: _SDPrefixPrecomputed,
    source_index: int,
    target_index: int,
    degree: int,
    *,
    discrete: bool,
) -> bool:
    if precomputed.identical[source_index, target_index]:
        return False
    prefix2 = precomputed.prefix2
    return _dominates_sorted(
        precomputed.sorted_values[:, source_index],
        precomputed.prefix1[:, source_index],
        None if prefix2 is None else prefix2[:, source_index],
        precomputed.sorted_values[:, target_index],
        precomputed.prefix1[:, target_index],
        None if prefix2 is None else prefix2[:, target_index],
        degree,
        discrete=discrete,
    )


def _sd_result(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    degree: int,
    *,
    discrete: bool = True,
) -> int:
    if _dominates_uni(x, y, degree, discrete=discrete):
        return 1
    if _dominates_uni(y, x, degree, discrete=discrete):
        return -1
    return 0


def _dominates_uni(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    degree: int,
    *,
    discrete: bool,
) -> bool:
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    return _dominates_sorted(
        x_sorted,
        _prefix_sum_1d(x_sorted),
        _prefix_sum_1d(x_sorted * x_sorted) if degree == 3 else None,
        y_sorted,
        _prefix_sum_1d(y_sorted),
        _prefix_sum_1d(y_sorted * y_sorted) if degree == 3 else None,
        degree,
        discrete=discrete,
    )


def _dominates_sorted(
    x_sorted: NDArray[np.float64],
    x_prefix1: NDArray[np.float64],
    x_prefix2: NDArray[np.float64] | None,
    y_sorted: NDArray[np.float64],
    y_prefix1: NDArray[np.float64],
    y_prefix2: NDArray[np.float64] | None,
    degree: int,
    *,
    discrete: bool,
) -> bool:
    """True iff x dominates y: a port of R's native pairwise test ``sd_dom_pair``.

    Samples are sorted, with prefix sums ``P[k] = sum(sorted[:k])`` (and of squares
    for TSD). Each curve is evaluated on the merged unique thresholds with the
    sample's own count ``k`` of observations ``<= t``. For TSD and continuous FSD
    the closed-form extremum of the curve difference inside each interval between
    thresholds is checked too (R fix d4fc624).

    Curve values from prefix sums that are equal in exact arithmetic can differ by
    rounding noise of order ``eps * S**degree`` (``S`` the largest absolute value of
    the two samples), so differences up to ``64 * eps * S**d`` are ties: ``d`` = 1 for
    SSD and the mean gate, 2 for TSD, and a ratio analogue for continuous FSD (R fix
    2ca5aca). Discrete FSD compares exact ECDF fractions with no tolerance.
    """
    x_count = x_sorted.size
    y_count = y_sorted.size
    scale = float(max(abs(x_sorted[0]), abs(x_sorted[-1]), abs(y_sorted[0]), abs(y_sorted[-1])))
    if not x_sorted[0] >= y_sorted[0]:
        return False
    if degree > 1 and y_prefix1[-1] / y_count - x_prefix1[-1] / x_count > _SD_EPS64 * scale:
        return False
    if x_count == y_count and np.array_equal(x_sorted, y_sorted):
        return False

    thresholds = np.union1d(x_sorted, y_sorted)
    x_counts = np.searchsorted(x_sorted, thresholds, side="right")
    y_counts = np.searchsorted(y_sorted, thresholds, side="right")

    interior_excess = False
    tol: float | NDArray[np.float64]
    if degree == 1 and discrete:
        # L0/(L0+U0) == ECDF.
        x_curve = x_counts / x_count
        y_curve = y_counts / y_count
        tol = 0.0
    elif degree == 1:
        x_curve, x_total = _ratio_deg1(x_prefix1, x_counts, thresholds, x_count)
        y_curve, y_total = _ratio_deg1(y_prefix1, y_counts, thresholds, y_count)
        total_min = np.minimum(x_total, y_total)
        with np.errstate(divide="ignore"):
            tol = _SD_EPS64 * np.where(total_min > 0.0, np.maximum(1.0, scale / total_min), 1.0)
        if not np.any(x_curve - y_curve > tol):
            interior_excess = _ratio_deg1_interior_excess(
                x_prefix1, x_counts, x_count, y_prefix1, y_counts, y_count, thresholds, scale
            )
    elif degree == 2:
        # LPM1 is piecewise linear with knots at the data: checking the knots is exact.
        x_curve = (x_counts * thresholds - x_prefix1[x_counts]) / x_count
        y_curve = (y_counts * thresholds - y_prefix1[y_counts]) / y_count
        tol = _SD_EPS64 * scale
    else:
        if x_prefix2 is None or y_prefix2 is None:
            raise ValueError("degree 3 prefix evaluation requires second-moment prefixes.")
        x_curve = _lpm_deg2(x_prefix1, x_prefix2, x_counts, thresholds, x_count)
        y_curve = _lpm_deg2(y_prefix1, y_prefix2, y_counts, thresholds, y_count)
        tol = _SD_EPS64 * scale * scale
        if not np.any(x_curve - y_curve > tol):
            interior_excess = _lpm_deg2_interior_excess(
                x_prefix1,
                x_prefix2,
                x_counts,
                x_count,
                y_prefix1,
                y_prefix2,
                y_counts,
                y_count,
                thresholds,
                scale,
            )

    excess = x_curve - y_curve
    if interior_excess or np.any(excess > tol):
        return False
    # Curves equal (within the tolerance) at every threshold -> no dominance (R's
    # curve equality guard).
    return bool(np.any(np.abs(excess) > tol))


def _ratio_deg1(
    prefix1: NDArray[np.float64],
    counts: NDArray[np.intp],
    thresholds: NDArray[np.float64],
    observations: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Continuous FSD ratio LPM1 / (LPM1 + UPM1) at the thresholds, and LPM1 + UPM1."""
    sums1 = prefix1[counts]
    lower = (counts * thresholds - sums1) / observations
    upper = ((prefix1[-1] - sums1) - (observations - counts) * thresholds) / observations
    total = lower + upper
    # 0/0 (every observation equals the threshold): the degree-0 ratio, i.e. the
    # right-continuous CDF value, as LPM.ratio does (R fix 05f6d3d; it was 0).
    ratio: NDArray[np.float64] = np.divide(
        lower,
        total,
        out=np.asarray(counts / observations, dtype=np.float64),
        where=total > 0.0,
    )
    return ratio, total


def _ratio_deg1_interior_excess(
    x_prefix1: NDArray[np.float64],
    x_counts: NDArray[np.intp],
    x_count: int,
    y_prefix1: NDArray[np.float64],
    y_counts: NDArray[np.intp],
    y_count: int,
    thresholds: NDArray[np.float64],
    scale: float,
) -> bool:
    """Whether ratio_x - ratio_y peaks above 0 strictly inside an interval of thresholds.

    On ``(t, t_next)`` with fixed counts each ratio is ``(a t + b) / (g t + d)``; the
    stationary points of the difference solve ``Ax / Dx(t)^2 = Ay / Dy(t)^2`` with
    ``A = a d - b g`` and ``D > 0``, i.e. ``Dy(t) = sqrt(Ay / Ax) * Dx(t)``.
    """
    lower_knots = thresholds[:-1]
    upper_knots = thresholds[1:]
    kx = x_counts[:-1]
    ky = y_counts[:-1]
    ax, bx, gx, dx = _ratio_deg1_coefs(x_prefix1, kx, x_count)
    ay, by, gy, dy = _ratio_deg1_coefs(y_prefix1, ky, y_count)
    det_x = ax * dx - bx * gx
    det_y = ay * dy - by * gy
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        valid = (det_x != 0.0) & (det_y != 0.0) & ((det_x > 0.0) == (det_y > 0.0))
        r = np.sqrt(np.where(valid, det_y / det_x, 1.0))
        den = gy - r * gx
        valid &= den != 0.0
        ts = (r * dx - dy) / np.where(valid, den, 1.0)
        valid &= (ts > lower_knots) & (ts < upper_knots)
        denom_x = gx * ts + dx
        denom_y = gy * ts + dy
        valid &= (denom_x > 0.0) & (denom_y > 0.0)
        if not np.any(valid):
            return False
        ratio_x = (ax * ts + bx) / denom_x
        ratio_y = (ay * ts + by) / denom_y
        tol = _SD_EPS64 * np.maximum(1.0, scale / np.minimum(denom_x, denom_y))
        return bool(np.any(valid & (ratio_x - ratio_y > tol)))


def _ratio_deg1_coefs(
    prefix1: NDArray[np.float64],
    counts: NDArray[np.intp],
    observations: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """LPM1 = a t + b and LPM1 + UPM1 = g t + d on an interval with fixed count k."""
    sums1 = prefix1[counts]
    alpha = counts / observations
    beta = -sums1 / observations
    gamma = (2.0 * counts - observations) / observations
    delta = (prefix1[-1] - 2.0 * sums1) / observations
    return alpha, beta, gamma, delta


def _lpm_deg2(
    prefix1: NDArray[np.float64],
    prefix2: NDArray[np.float64],
    counts: NDArray[np.intp],
    thresholds: NDArray[np.float64],
    observations: int,
) -> NDArray[np.float64]:
    """LPM2 = mean(max(t - x, 0)^2) = (k t^2 - 2 t P1[k] + P2[k]) / m."""
    return np.asarray(
        (counts * thresholds * thresholds - 2.0 * thresholds * prefix1[counts] + prefix2[counts])
        / observations,
        dtype=np.float64,
    )


def _lpm_deg2_interior_excess(
    x_prefix1: NDArray[np.float64],
    x_prefix2: NDArray[np.float64],
    x_counts: NDArray[np.intp],
    x_count: int,
    y_prefix1: NDArray[np.float64],
    y_prefix2: NDArray[np.float64],
    y_counts: NDArray[np.intp],
    y_count: int,
    thresholds: NDArray[np.float64],
    scale: float,
) -> bool:
    """Whether LPM2_x - LPM2_y peaks above 0 strictly inside an interval of thresholds.

    On ``(t, t_next)`` the difference is ``a s^2 - 2 b s + c`` with
    ``a = kx/mx - ky/my`` and ``b = P1x/mx - P1y/my``; when ``a < 0`` it has an
    interior maximum at ``s* = b / a``, which the knots miss.
    """
    kx = x_counts[:-1]
    ky = y_counts[:-1]
    a = kx / x_count - ky / y_count
    concave = a < 0.0
    if not np.any(concave):
        return False
    kx = kx[concave]
    ky = ky[concave]
    b = x_prefix1[kx] / x_count - y_prefix1[ky] / y_count
    ts = b / a[concave]
    inside = (ts > thresholds[:-1][concave]) & (ts < thresholds[1:][concave])
    if not np.any(inside):
        return False
    kx = kx[inside]
    ky = ky[inside]
    ts = ts[inside]
    excess = _lpm_deg2(x_prefix1, x_prefix2, kx, ts, x_count) - _lpm_deg2(
        y_prefix1, y_prefix2, ky, ts, y_count
    )
    return bool(np.any(excess > _SD_EPS64 * scale * scale))


def _prefix_sum_1d(values: NDArray[np.float64]) -> NDArray[np.float64]:
    prefix = np.empty(values.size + 1, dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(values, out=prefix[1:])
    return prefix


def _prefix_sum(values: NDArray[np.float64]) -> NDArray[np.float64]:
    prefix = np.empty((values.shape[0] + 1, values.shape[1]), dtype=np.float64, order="F")
    prefix[0, :] = 0.0
    np.cumsum(values, axis=0, out=prefix[1:, :])
    return prefix


def _lpm_at_target(
    values: NDArray[np.float64],
    target: float,
    degree: int,
) -> NDArray[np.float64]:
    deviations = np.maximum(0.0, target - values)
    if degree > 1:
        deviations = deviations**degree
    return np.asarray(np.mean(deviations, axis=0), dtype=np.float64)


def _as_sd_values(x: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    values = np.asarray(x, dtype=np.float64)
    if values.ndim == 2:
        # R accepts single-column matrices and data frames.
        if values.shape[1] != 1:
            raise ValueError(f"{name} must be a single-column object or numeric vector.")
        values = values[:, 0]
    values = _as_1d_values(values)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")
    return values
