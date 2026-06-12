"""Stochastic dominance routines matching NNS' discrete SD conventions.

Dominance uses strict floating-point comparisons with no tolerance, plus R's
curve equality guard: equal LPM/CDF curves are non-dominance even when samples
differ below meaningful double precision. Efficient-set output follows the R
C++ routine's LPM-at-global-maximum ordering and original-index tie break.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import linkage  # type: ignore[import-untyped]
from scipy.spatial.distance import squareform  # type: ignore[import-untyped]

from nns.core import _as_1d_values, lpm

_SD_CLUSTER_DOMINANCE_MATRIX_MIN_COLUMNS = 75
_SD_PREFIX_PAIR_MATRIX_MIN_COLUMNS = 75
_SD_PREFIX_PAIR_TARGET_BLOCK_COLUMNS = 64
_SD_ORDER_STAT_TARGET_BLOCK_COLUMNS = 64


@dataclass(frozen=True)
class _SDPrecomputed:
    values: NDArray[np.float64]
    sorted_values: NDArray[np.float64]
    curves: NDArray[np.float64]
    curve_sums: NDArray[np.float64]
    minimums: NDArray[np.float64]
    means: NDArray[np.float64]
    identical: NDArray[np.bool_]


@dataclass(frozen=True)
class _SDPrefixPrecomputed:
    values: NDArray[np.float64]
    sorted_values: NDArray[np.float64]
    prefix1: NDArray[np.float64]
    prefix2: NDArray[np.float64] | None
    own_curves: NDArray[np.float64]
    minimums: NDArray[np.float64]
    means: NDArray[np.float64]
    identical: NDArray[np.bool_]


@dataclass(frozen=True)
class _SDOrderStatPrecomputed:
    values: NDArray[np.float64]
    sorted_values: NDArray[np.float64]
    identical: NDArray[np.bool_]


def fsd(x: NDArray[np.float64], y: NDArray[np.float64]) -> int:
    """First-order stochastic dominance."""
    x_values = _as_sd_values(x, "x")
    y_values = _as_sd_values(y, "y")
    return _sd_result(x_values, y_values, 1)


def fsd_uni(x: NDArray[np.float64], y: NDArray[np.float64], type: str = "discrete") -> int:
    """Unidirectional first-order stochastic dominance: 1 if x dominates y, else 0."""
    x_values = _as_sd_values(x, "x")
    y_values = _as_sd_values(y, "y")
    discrete = type.lower() != "continuous"
    return int(_dominates_uni(x_values, y_values, 1, discrete=discrete))


def ssd(x: NDArray[np.float64], y: NDArray[np.float64]) -> int:
    """Second-order stochastic dominance."""
    x_values = _as_sd_values(x, "x")
    y_values = _as_sd_values(y, "y")
    return _sd_result(x_values, y_values, 2)


def ssd_uni(x: NDArray[np.float64], y: NDArray[np.float64]) -> int:
    """Unidirectional second-order stochastic dominance: 1 if x dominates y, else 0."""
    x_values = _as_sd_values(x, "x")
    y_values = _as_sd_values(y, "y")
    return int(_dominates_uni(x_values, y_values, 2, discrete=True))


def tsd(x: NDArray[np.float64], y: NDArray[np.float64]) -> int:
    """Third-order stochastic dominance."""
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
    """Cluster variables by iteratively peeling stochastic-dominance efficient sets."""
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("data must be a 2D array.")
    if values.shape[0] == 0:
        raise ValueError("data must have at least one row.")
    if not 1 <= int(degree) <= 3:
        raise ValueError("degree must be 1, 2, or 3.")
    if not np.all(np.isfinite(values)):
        raise ValueError("data must contain only finite values.")
    type_value = _sd_type_value(int(degree), type)
    discrete = int(degree) != 1 or type_value != "continuous"
    min_cluster = int(min_cluster)
    if min_cluster < 0:
        raise ValueError("min_cluster must be non-negative.")

    column_count = values.shape[1]
    if names is None:
        all_names = [f"X_{index + 1}" for index in range(column_count)]
    else:
        all_names = [str(name) for name in names]
        if len(all_names) != column_count:
            raise ValueError("names length must match the number of data columns.")

    degree_int = int(degree)
    order_stat_dominance_matrix = None
    prefix_precomputed = None
    precomputed = None
    if column_count >= _SD_CLUSTER_DOMINANCE_MATRIX_MIN_COLUMNS:
        if degree_int == 1 and discrete:
            order_stat_precomputed = _order_stat_sd_precompute(values)
            order_stat_dominance_matrix = _dominance_matrix_from_order_stats(
                order_stat_precomputed
            )
        else:
            prefix_precomputed = _prefix_sd_precompute(values, degree_int, discrete=discrete)
    else:
        precomputed = _precompute_sd_table(values, degree_int, discrete=discrete)
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
        elif prefix_precomputed is not None:
            sd_set_indices = _sd_efficient_active_indices_from_prefix_kept(
                prefix_precomputed,
                active,
                degree_int,
                discrete=discrete,
            )
        else:
            assert precomputed is not None
            sd_set_indices = _sd_efficient_active_indices(precomputed, active, degree_int)
        sd_set = [all_names[index] for index in sd_set_indices]
        if not sd_set:
            break

        clusters[f"Cluster_{iteration}"] = sd_set
        remove_indices = set(sd_set_indices)
        active = [index for index in active if index not in remove_indices]
        iteration += 1

        if len(active) <= min_cluster:
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
    """Return indices of non-dominated columns at the requested SD degree."""
    values = np.asarray(returns, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("returns must be a 2D array.")
    if values.shape[0] == 0:
        raise ValueError("returns must have at least one row.")
    if not 1 <= degree <= 3:
        raise ValueError("degree must be 1, 2, or 3.")
    if not np.all(np.isfinite(values)):
        raise ValueError("returns must contain only finite values.")

    type_value = _sd_type_value(degree, type)
    discrete = degree != 1 or type_value != "continuous"

    if values.shape[1] == 0:
        return []

    active = list(range(values.shape[1]))
    if values.shape[1] >= _SD_PREFIX_PAIR_MATRIX_MIN_COLUMNS:
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
    precomputed = _precompute_sd_table(values, degree, discrete=discrete)
    return _sd_efficient_active_indices(precomputed, active, degree)


def _sd_efficient_set_names(
    values: NDArray[np.float64],
    degree: int,
    type: str,
    names: Sequence[str],
) -> list[str]:
    return [names[index] for index in sd_efficient_set(values, degree, type=type)]


def _sd_type_value(degree: int, type: str) -> str:
    type_value = type.lower()
    if degree == 1 and type_value in {"discrete", "continuous"}:
        return type_value
    return "discrete"


def _precompute_sd_table(
    values: NDArray[np.float64],
    degree: int,
    *,
    discrete: bool,
) -> _SDPrecomputed:
    sorted_values = np.sort(values, axis=0)
    curves = _sd_curve_table(sorted_values, degree, discrete=discrete)
    return _SDPrecomputed(
        values=values,
        sorted_values=sorted_values,
        curves=curves,
        curve_sums=np.sum(curves, axis=0),
        minimums=sorted_values[0, :],
        means=np.mean(values, axis=0),
        identical=np.all(
            sorted_values.T[:, np.newaxis, :] == sorted_values.T[np.newaxis, :, :],
            axis=2,
        ),
    )


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
        own_curves=_own_threshold_curves(
            sorted_values,
            prefix1,
            prefix2,
            degree,
            discrete=discrete,
        ),
        minimums=sorted_values[0, :],
        means=np.mean(values, axis=0),
        identical=np.all(
            sorted_values.T[:, np.newaxis, :] == sorted_values.T[np.newaxis, :, :],
            axis=2,
        ),
    )


def _order_stat_sd_precompute(values: NDArray[np.float64]) -> _SDOrderStatPrecomputed:
    sorted_values = np.asfortranarray(np.sort(values, axis=0))
    return _SDOrderStatPrecomputed(
        values=values,
        sorted_values=sorted_values,
        identical=np.all(
            sorted_values.T[:, np.newaxis, :] == sorted_values.T[np.newaxis, :, :],
            axis=2,
        ),
    )


def _own_threshold_curves(
    sorted_values: NDArray[np.float64],
    prefix1: NDArray[np.float64],
    prefix2: NDArray[np.float64] | None,
    degree: int,
    *,
    discrete: bool,
) -> NDArray[np.float64]:
    columns = sorted_values.shape[1]
    curves = np.empty(sorted_values.shape, dtype=np.float64, order="F")
    for index in range(columns):
        curves[:, index] = _pair_curve_values_at_thresholds(
            sorted_values[:, index],
            prefix1[:, index],
            None if prefix2 is None else prefix2[:, index],
            sorted_values[:, index],
            sorted_values.shape[0],
            degree,
            discrete=discrete,
        )
    return curves


def _sd_efficient_active_indices(
    precomputed: _SDPrecomputed,
    active: Sequence[int],
    degree: int,
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
        dominated = any(
            _dominates_from_precomputed(kept, index, degree, precomputed) for kept in keep
        )
        if not dominated:
            keep.append(index)
    return keep


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


def _dominance_matrix_from_precomputed(
    precomputed: _SDPrecomputed,
    degree: int,
) -> NDArray[np.bool_]:
    curves = precomputed.curves
    columns = curves.shape[1]
    any_gt = np.zeros((columns, columns), dtype=np.bool_)
    for start, stop in _curve_comparison_chunks(curves.shape[0], columns):
        block = curves[start:stop, :]
        any_gt |= np.any(block[:, :, np.newaxis] > block[:, np.newaxis, :], axis=0)

    dominates = np.logical_not(any_gt) & any_gt.T
    dominates &= np.logical_not(precomputed.identical)
    dominates &= precomputed.minimums[:, np.newaxis] >= precomputed.minimums[np.newaxis, :]
    if degree > 1:
        dominates &= precomputed.means[:, np.newaxis] >= precomputed.means[np.newaxis, :]
    np.fill_diagonal(dominates, False)
    return dominates


def _dominance_matrix_from_prefix_pairs(
    precomputed: _SDPrefixPrecomputed,
    degree: int,
    *,
    discrete: bool,
) -> NDArray[np.bool_]:
    sorted_values = precomputed.sorted_values
    observations, columns = sorted_values.shape
    any_gt = np.zeros((columns, columns), dtype=np.bool_)
    pair_candidates = _prefix_pair_candidate_matrix(precomputed, degree)

    for target_start in range(0, columns, _SD_PREFIX_PAIR_TARGET_BLOCK_COLUMNS):
        target_stop = min(target_start + _SD_PREFIX_PAIR_TARGET_BLOCK_COLUMNS, columns)
        block_indices = np.arange(target_start, target_stop, dtype=np.intp)

        for source_index in range(columns):
            local_candidates = pair_candidates[source_index, target_start:target_stop]
            if not np.any(local_candidates):
                continue
            target_indices = block_indices[local_candidates]
            target_thresholds = sorted_values[:, target_indices]
            target_own_curves = precomputed.own_curves[:, target_indices]
            source_curves = _pair_curve_values_at_thresholds(
                sorted_values[:, source_index],
                precomputed.prefix1[:, source_index],
                None if precomputed.prefix2 is None else precomputed.prefix2[:, source_index],
                target_thresholds,
                observations,
                degree,
                discrete=discrete,
            )
            source_gt_target = np.any(source_curves > target_own_curves, axis=0)
            target_gt_source = np.any(target_own_curves > source_curves, axis=0)
            any_gt[source_index, target_indices] |= source_gt_target
            any_gt[target_indices, source_index] |= target_gt_source

    dominates = np.logical_not(any_gt) & any_gt.T
    dominates &= _prefix_directional_candidate_matrix(precomputed, degree)
    np.fill_diagonal(dominates, False)
    return dominates


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
    if precomputed.minimums[source_index] < precomputed.minimums[target_index]:
        return False
    if degree > 1 and precomputed.means[source_index] < precomputed.means[target_index]:
        return False

    observations = precomputed.sorted_values.shape[0]
    source_sorted = precomputed.sorted_values[:, source_index]
    source_prefix1 = precomputed.prefix1[:, source_index]
    source_prefix2 = None if precomputed.prefix2 is None else precomputed.prefix2[:, source_index]
    source_own_curve = precomputed.own_curves[:, source_index]
    target_sorted = precomputed.sorted_values[:, target_index]
    target_prefix1 = precomputed.prefix1[:, target_index]
    target_prefix2 = None if precomputed.prefix2 is None else precomputed.prefix2[:, target_index]
    target_own_curve = precomputed.own_curves[:, target_index]

    source_curve_at_target = _pair_curve_values_at_thresholds(
        source_sorted,
        source_prefix1,
        source_prefix2,
        target_sorted,
        observations,
        degree,
        discrete=discrete,
    )
    if np.any(source_curve_at_target > target_own_curve):
        return False

    target_gt_source = bool(np.any(target_own_curve > source_curve_at_target))
    target_curve_at_source = _pair_curve_values_at_thresholds(
        target_sorted,
        target_prefix1,
        target_prefix2,
        source_sorted,
        observations,
        degree,
        discrete=discrete,
    )
    if np.any(source_own_curve > target_curve_at_source):
        return False

    return target_gt_source or bool(np.any(target_curve_at_source > source_own_curve))


def _prefix_pair_candidate_matrix(
    precomputed: _SDPrefixPrecomputed,
    degree: int,
) -> NDArray[np.bool_]:
    directional_candidates = _prefix_directional_candidate_matrix(precomputed, degree)
    return directional_candidates | directional_candidates.T


def _prefix_directional_candidate_matrix(
    precomputed: _SDPrefixPrecomputed,
    degree: int,
) -> NDArray[np.bool_]:
    candidates = precomputed.minimums[:, np.newaxis] >= precomputed.minimums[np.newaxis, :]
    if degree > 1:
        candidates &= precomputed.means[:, np.newaxis] >= precomputed.means[np.newaxis, :]
    candidates &= np.logical_not(precomputed.identical)
    np.fill_diagonal(candidates, False)
    return candidates


def _prefix_directional_candidates_to_target(
    precomputed: _SDPrefixPrecomputed,
    source_indices: NDArray[np.intp],
    target_index: int,
    degree: int,
) -> NDArray[np.bool_]:
    candidates = precomputed.minimums[source_indices] >= precomputed.minimums[target_index]
    if degree > 1:
        candidates &= precomputed.means[source_indices] >= precomputed.means[target_index]
    candidates &= np.logical_not(precomputed.identical[source_indices, target_index])
    return np.asarray(candidates, dtype=np.bool_)


def _pair_curve_values_at_thresholds(
    sorted_column: NDArray[np.float64],
    prefix1_column: NDArray[np.float64],
    prefix2_column: NDArray[np.float64] | None,
    thresholds: NDArray[np.float64],
    observations: int,
    degree: int,
    *,
    discrete: bool,
) -> NDArray[np.float64]:
    counts = np.searchsorted(sorted_column, thresholds, side="right")
    if degree == 1 and discrete:
        return np.asarray(counts / observations, dtype=np.float64)

    sums1 = prefix1_column[counts]
    if degree == 1:
        lower = (counts * thresholds - sums1) / observations
        totals = prefix1_column[-1]
        upper = (totals - sums1 - (observations - counts) * thresholds) / observations
        ratio: NDArray[np.float64] = np.divide(
            lower,
            lower + upper,
            out=np.zeros_like(lower, dtype=np.float64),
            where=(lower + upper) != 0,
        )
        return ratio

    if degree == 2:
        return (counts * thresholds - sums1) / observations

    if prefix2_column is None:
        raise ValueError("degree 3 prefix evaluation requires second-moment prefixes.")
    sums2 = prefix2_column[counts]
    return (counts * thresholds * thresholds - 2.0 * thresholds * sums1 + sums2) / observations


def _dominates_from_precomputed(
    x_index: int,
    y_index: int,
    degree: int,
    precomputed: _SDPrecomputed,
) -> bool:
    if precomputed.identical[x_index, y_index]:
        return False
    if precomputed.minimums[x_index] < precomputed.minimums[y_index]:
        return False
    if degree > 1 and precomputed.means[x_index] < precomputed.means[y_index]:
        return False

    x_curve = precomputed.curves[:, x_index]
    y_curve = precomputed.curves[:, y_index]
    if precomputed.curve_sums[x_index] == precomputed.curve_sums[y_index] and np.array_equal(
        x_curve,
        y_curve,
    ):
        return False
    return bool(not np.any(x_curve > y_curve))


def _sd_result(x: NDArray[np.float64], y: NDArray[np.float64], degree: int) -> int:
    if _dominates(x, y, degree):
        return 1
    if _dominates(y, x, degree):
        return -1
    return 0


def _dominates(x: NDArray[np.float64], y: NDArray[np.float64], degree: int) -> bool:
    return _dominates_uni(x, y, degree, discrete=True)


def _dominates_uni(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    degree: int,
    *,
    discrete: bool,
) -> bool:
    if x.size != y.size:
        raise ValueError("x and y must have the same length.")
    if np.array_equal(np.sort(x), np.sort(y)):
        return False
    if np.min(x) < np.min(y):
        return False
    if degree > 1 and np.mean(x) < np.mean(y):
        return False

    grid = np.sort(np.concatenate((x, y)))
    x_lpm = _dominance_curve(x, grid, degree, discrete=discrete)
    y_lpm = _dominance_curve(y, grid, degree, discrete=discrete)
    if np.array_equal(x_lpm, y_lpm):
        return False
    return bool(not np.any(x_lpm > y_lpm))


def _dominance_curve(
    values: NDArray[np.float64],
    grid: NDArray[np.float64],
    degree: int,
    *,
    discrete: bool = True,
) -> NDArray[np.float64]:
    if degree == 1:
        if discrete:
            return np.asarray(lpm(0, grid, values), dtype=np.float64)
        lower = np.asarray(lpm(1, grid, values), dtype=np.float64)
        upper = np.mean(np.maximum(0.0, values - grid[:, np.newaxis]), axis=1)
        ratio: NDArray[np.float64] = np.divide(
            lower,
            lower + upper,
            out=np.zeros_like(lower),
            where=(lower + upper) != 0,
        )
        return ratio
    return np.asarray(lpm(degree - 1, grid, values), dtype=np.float64)


def _dominates_from_curves(
    x_index: int,
    y_index: int,
    degree: int,
    curves: NDArray[np.float64],
    sorted_values: NDArray[np.float64],
    minimums: NDArray[np.float64],
    means: NDArray[np.float64],
) -> bool:
    if np.array_equal(sorted_values[:, x_index], sorted_values[:, y_index]):
        return False
    if minimums[x_index] < minimums[y_index]:
        return False
    if degree > 1 and means[x_index] < means[y_index]:
        return False

    x_curve = curves[:, x_index]
    y_curve = curves[:, y_index]
    if np.array_equal(x_curve, y_curve):
        return False
    return bool(not np.any(x_curve > y_curve))


def _sd_curve_table(
    sorted_values: NDArray[np.float64],
    degree: int,
    *,
    discrete: bool = True,
) -> NDArray[np.float64]:
    grid = np.unique(sorted_values.reshape(-1))
    observations, columns = sorted_values.shape
    curves = np.empty((grid.size, columns), dtype=np.float64)

    if degree == 1:
        if discrete:
            _fill_cdf_curves(curves, grid, sorted_values)
        else:
            _fill_continuous_fsd_curves(curves, grid, sorted_values)
        return curves

    prefix1 = _prefix_sum(sorted_values)
    if degree == 2:
        _fill_lpm_degree1_curves(curves, grid, sorted_values, prefix1, observations)
        return curves

    prefix2 = _prefix_sum(sorted_values * sorted_values)
    _fill_lpm_degree2_curves(curves, grid, sorted_values, prefix1, prefix2, observations)
    return curves


def _fill_cdf_curves(
    curves: NDArray[np.float64],
    grid: NDArray[np.float64],
    sorted_values: NDArray[np.float64],
) -> None:
    observations = sorted_values.shape[0]
    for start, stop in _grid_chunks(grid.size, sorted_values.shape[1]):
        thresholds = grid[start:stop]
        for index in range(sorted_values.shape[1]):
            counts = np.searchsorted(sorted_values[:, index], thresholds, side="right")
            curves[start:stop, index] = counts / observations


def _fill_lpm_degree1_curves(
    curves: NDArray[np.float64],
    grid: NDArray[np.float64],
    sorted_values: NDArray[np.float64],
    prefix1: NDArray[np.float64],
    observations: int,
) -> None:
    for start, stop in _grid_chunks(grid.size, sorted_values.shape[1]):
        thresholds = grid[start:stop]
        for index in range(sorted_values.shape[1]):
            counts = np.searchsorted(sorted_values[:, index], thresholds, side="right")
            sums1 = prefix1[counts, index]
            curves[start:stop, index] = (counts * thresholds - sums1) / observations


def _fill_continuous_fsd_curves(
    curves: NDArray[np.float64],
    grid: NDArray[np.float64],
    sorted_values: NDArray[np.float64],
) -> None:
    observations = sorted_values.shape[0]
    prefix1 = _prefix_sum(sorted_values)
    totals = prefix1[-1, :]
    for start, stop in _grid_chunks(grid.size, sorted_values.shape[1]):
        thresholds = grid[start:stop]
        for index in range(sorted_values.shape[1]):
            counts = np.searchsorted(sorted_values[:, index], thresholds, side="right")
            sums1 = prefix1[counts, index]
            lower = (counts * thresholds - sums1) / observations
            upper = (totals[index] - sums1 - (observations - counts) * thresholds) / observations
            curves[start:stop, index] = np.divide(
                lower,
                lower + upper,
                out=np.zeros_like(lower),
                where=(lower + upper) != 0,
            )


def _fill_lpm_degree2_curves(
    curves: NDArray[np.float64],
    grid: NDArray[np.float64],
    sorted_values: NDArray[np.float64],
    prefix1: NDArray[np.float64],
    prefix2: NDArray[np.float64],
    observations: int,
) -> None:
    for start, stop in _grid_chunks(grid.size, sorted_values.shape[1]):
        thresholds = grid[start:stop]
        for index in range(sorted_values.shape[1]):
            counts = np.searchsorted(sorted_values[:, index], thresholds, side="right")
            sums1 = prefix1[counts, index]
            sums2 = prefix2[counts, index]
            curves[start:stop, index] = (
                counts * thresholds * thresholds - 2.0 * thresholds * sums1 + sums2
            ) / observations


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


def _grid_chunks(grid_size: int, columns: int) -> Iterator[tuple[int, int]]:
    max_intermediate_bytes = 100 * 1024 * 1024
    row_bytes = columns * np.dtype(np.float64).itemsize
    chunk_size = max(1, max_intermediate_bytes // max(row_bytes, 1))
    for start in range(0, grid_size, chunk_size):
        yield start, min(start + chunk_size, grid_size)


def _curve_comparison_chunks(grid_size: int, columns: int) -> Iterator[tuple[int, int]]:
    max_intermediate_bytes = 100 * 1024 * 1024
    row_bytes = columns * columns * np.dtype(np.bool_).itemsize
    chunk_size = max(1, max_intermediate_bytes // max(row_bytes, 1))
    for start in range(0, grid_size, chunk_size):
        yield start, min(start + chunk_size, grid_size)


def _as_sd_values(x: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    values = _as_1d_values(x)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")
    return values
