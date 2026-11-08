"""Stochastic dominance routines matching NNS' discrete SD conventions.

Every pairwise verdict (``fsd``/``ssd``/``tsd``, their ``*_uni`` forms, the
efficient set and SD clustering) comes from one test, a port of R's native
``sd_dom_pair``: the curves are compared on the merged sample points and, for
TSD and continuous FSD, also at their closed-form extrema between those points
(LPM2 is piecewise quadratic and the continuous ratio linear-fractional, so the
difference can peak between knots). Each pair is compared centred on its joint
midrange, and differences up to ``64 * eps * S**d`` (``S`` the largest absolute
centred value; ``d`` = 1 for SSD and the mean gate, 2 for TSD, a ratio analogue for
continuous FSD), plus a term for the inputs' own rounding at their level, are ties,
so rounding noise does not decide dominance and a common shift leaves the verdict
unchanged; discrete FSD compares exact ECDF fractions.
Equal curves are non-dominance (R's curve equality guard). The efficient set
is every column that no column outside its dominance cycle dominates, so it does
not depend on column order; its output follows the R C++ routine's
LPM-at-global-maximum ordering and original-index tie break.

Pairwise tests accept samples of unequal length: each sample's curve is
evaluated on the merged threshold grid with its own length, as R's
``NNS.FSD``/``NNS.SSD``/``NNS.TSD`` and (in the patched R) the ``.uni``
walkers do.
"""

from __future__ import annotations

import functools
import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import ParamSpec, TypeVar

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import linkage  # type: ignore[import-untyped]
from scipy.sparse.csgraph import connected_components  # type: ignore[import-untyped]
from scipy.spatial.distance import squareform  # type: ignore[import-untyped]

_SD_ORDER_STAT_TARGET_BLOCK_COLUMNS = 64
# R's native SD test ignores excesses up to 64 * DBL_EPSILON (relative to the data scale).
_SD_EPS = float(np.finfo(np.float64).eps)
_SD_EPS64 = 64.0 * _SD_EPS
# R's own messages: the NA check in every R wrapper and the C++ empty-sample check.
_MISSING_VALUES = "You have some missing values, please address."
_EMPTY_SAMPLE = "Each sample needs at least one observation."

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _cpp_float_semantics(function: Callable[_P, _R]) -> Callable[_P, _R]:
    """Evaluate like R's C++: infinite inputs give inf/NaN intermediates silently."""

    @functools.wraps(function)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            return function(*args, **kwargs)

    return wrapper


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


@_cpp_float_semantics
def fsd(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    type: str = "discrete",
    *,
    plot: bool = False,
) -> int:
    """First-order stochastic dominance; ``x`` and ``y`` may differ in length.

    ``type="discrete"`` compares empirical CDFs; ``type="continuous"`` compares the
    degree-1 ratio ``LPM1 / (LPM1 + UPM1)``, as R's ``NNS.FSD(type=)`` does. Like R,
    an unrecognised ``type`` warns and runs the continuous test; where both
    partial moments vanish the ratio is the degree-0 ratio (the CDF value), as
    in ``lpm_ratio`` and the patched R. ``plot=True`` also draws R's FSD figure
    (``nns.plotting.plot_fsd``); R's default is ``plot = TRUE``.

    As in R, missing values raise, and so do infinite ones (R evaluates the
    plotted ``LPM.ratio`` curves, which reject them, before the verdict).
    """
    discrete = _checked_fsd_type(type) == "discrete"
    x_values, y_values = _sd_pair(x, y, finite=True)
    result = _sd_result(x_values, y_values, 1, discrete=discrete)
    if plot:
        from nns.plotting.dominance import plot_fsd

        plot_fsd(x_values, y_values, type="discrete" if discrete else "continuous")
    return result


@_cpp_float_semantics
def fsd_uni(x: NDArray[np.float64], y: NDArray[np.float64], type: str = "discrete") -> int:
    """Unidirectional first-order stochastic dominance: 1 if x dominates y, else 0.

    Like R's ``NNS.FSD.uni``, an unrecognised ``type`` warns and runs the discrete
    test, and infinite values are compared as R's C++ test compares them.
    """
    x_values, y_values = _sd_pair(x, y, finite=False)
    discrete = _checked_fsd_type(type) != "continuous"
    return int(_dominates_uni(x_values, y_values, 1, discrete=discrete))


@_cpp_float_semantics
def ssd(x: NDArray[np.float64], y: NDArray[np.float64], *, plot: bool = False) -> int:
    """Second-order stochastic dominance; ``x`` and ``y`` may differ in length.

    ``plot=True`` also draws R's SSD figure (``nns.plotting.plot_ssd``); R's
    default is ``plot = TRUE``.
    """
    x_values, y_values = _sd_pair(x, y, finite=True)
    result = _sd_result(x_values, y_values, 2)
    if plot:
        from nns.plotting.dominance import plot_ssd

        plot_ssd(x_values, y_values)
    return result


@_cpp_float_semantics
def ssd_uni(x: NDArray[np.float64], y: NDArray[np.float64]) -> int:
    """Unidirectional second-order stochastic dominance: 1 if x dominates y, else 0."""
    x_values, y_values = _sd_pair(x, y, finite=False)
    return int(_dominates_uni(x_values, y_values, 2, discrete=True))


@_cpp_float_semantics
def tsd(x: NDArray[np.float64], y: NDArray[np.float64], *, plot: bool = False) -> int:
    """Third-order stochastic dominance; ``x`` and ``y`` may differ in length.

    ``plot=True`` also draws R's TSD figure (``nns.plotting.plot_tsd``); R's
    default is ``plot = TRUE``.
    """
    x_values, y_values = _sd_pair(x, y, finite=True)
    result = _sd_result(x_values, y_values, 3)
    if plot:
        from nns.plotting.dominance import plot_tsd

        plot_tsd(x_values, y_values)
    return result


@_cpp_float_semantics
def tsd_uni(x: NDArray[np.float64], y: NDArray[np.float64]) -> int:
    """Unidirectional third-order stochastic dominance: 1 if x dominates y, else 0."""
    x_values, y_values = _sd_pair(x, y, finite=False)
    return int(_dominates_uni(x_values, y_values, 3, discrete=True))


def nns_sd_cluster(
    data: NDArray[np.float64],
    degree: int = 1,
    type: str = "discrete",
    min_cluster: float = 1,
    dendrogram: bool = False,
    names: Sequence[str] | None = None,
) -> dict[str, object]:
    """Cluster variables by iteratively peeling stochastic-dominance efficient sets.

    Cluster members are named by ``names``, else by the columns of a DataFrame-like
    ``data`` (R's ``colnames``), else ``X_1``, ``X_2``, ...
    ``dendrogram=True`` returns R's ``hclust`` fields under ``"Dendrogram"`` and, as
    R does, draws the dendrogram; with fewer than two clustered variables it warns
    and returns the clusters only.

    As in R, the data are validated by the first efficient-set search, so a
    matrix with no more than ``min_cluster`` columns is returned as one cluster
    unchecked, and ``min_cluster`` is compared as the real number it is.
    """
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return _nns_sd_cluster(data, degree, type, min_cluster, dendrogram, names)


def _nns_sd_cluster(
    data: NDArray[np.float64],
    degree: int,
    type: str,
    min_cluster: float,
    dendrogram: bool,
    names: Sequence[str] | None,
) -> dict[str, object]:
    column_labels = getattr(data, "columns", None)
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("data must be a 2D array.")
    min_cluster = float(min_cluster)

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
    degree_int = 1
    discrete = True
    verdicts: dict[tuple[int, int], bool] = {}
    active = list(range(column_count))
    clusters: dict[str, list[str]] = {}
    iteration = 1

    while len(active) > min_cluster:
        if not active:
            break  # R's efficient set of no columns is empty
        if order_stat_dominance_matrix is None and prefix_precomputed is None:
            # R validates in the first NNS.SD.efficient.set call, on every column.
            degree_int, discrete = _validated_sd_matrix(values, degree, type)
            if degree_int == 1 and discrete:
                order_stat_dominance_matrix = _dominance_matrix_from_order_stats(
                    _order_stat_sd_precompute(values)
                )
            else:
                prefix_precomputed = _prefix_sd_precompute(values, degree_int, discrete=discrete)
        if order_stat_dominance_matrix is not None:
            sd_set_indices = _sd_efficient_active_indices_from_matrix(
                values,
                active,
                degree_int,
                order_stat_dominance_matrix,
            )
        else:
            assert prefix_precomputed is not None
            sd_set_indices = _sd_efficient_active_indices_from_prefix(
                prefix_precomputed,
                active,
                degree_int,
                discrete=discrete,
                verdicts=verdicts,
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

    if active and f"Cluster_{iteration}" not in clusters:
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
            warnings.warn(
                "Not enough variables for hierarchical clustering. Returning clusters only.",
                UserWarning,
                stacklevel=2,
            )
            return {"Clusters": clusters, "Order": None}
        return {
            "Clusters": clusters,
            "Dendrogram": _sd_cluster_hclust(clusters, all_names, draw_degree=degree),
        }

    return {"Clusters": clusters}


def _sd_cluster_hclust(
    clusters: dict[str, list[str]],
    original_names: Sequence[str],
    *,
    draw_degree: object = None,
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
    if draw_degree is not None:
        _draw_sd_dendrogram(linked, all_vars, draw_degree)
    return {
        "merge": merge,
        "height": linked[:, 2].astype(np.float64),
        "order": _r_hclust_order(merge),
        "labels": np.asarray(all_vars, dtype=str),
        "method": "complete",
        "call": 'hclust(d = dist_matrix, method = "complete")',
        "dist.method": None,
    }


def _draw_sd_dendrogram(linked: NDArray[np.float64], labels: list[str], degree: object) -> None:
    """Draw the cluster dendrogram, as R's ``NNS.SD.cluster(dendrogram = TRUE)`` does."""
    from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram

    from nns.plotting._mpl import resolve_ax

    ax = resolve_ax(None)
    scipy_dendrogram(
        linked, labels=labels, ax=ax, link_color_func=lambda _: "black", leaf_rotation=90
    )
    # R pastes the degree as given; a whole double prints without decimals.
    if isinstance(degree, float) and degree.is_integer():
        degree = int(degree)
    ax.set_title(f"Hierarchical Clustering of Stochastic Dominance Sets\nSD Degree: {degree}")
    ax.set_xlabel("Variables")
    ax.set_ylabel("SD Distance")


def _r_hclust_order(merge: NDArray[np.int64]) -> NDArray[np.int64]:
    """R hclust's leaf order (hcass2): expand the last merge, left before right."""
    order: list[int] = []
    stack = [int(merge.shape[0])]
    while stack:
        node = stack.pop()
        if node < 0:
            order.append(-node)
        else:
            stack.extend((int(merge[node - 1, 1]), int(merge[node - 1, 0])))
    return np.asarray(order, dtype=np.int64)


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


@_cpp_float_semantics
def sd_efficient_set(
    returns: NDArray[np.float64],
    degree: int,
    type: str = "discrete",
) -> list[int]:
    """Return 0-based indices of the non-dominated columns, in R's output order.

    R's ``NNS.SD.efficient.set`` returns the corresponding column names. A vector
    is one column (R's ``as.matrix``). ``degree`` is coerced like R's
    ``as.integer``: 1.5 runs first-order (FSD) and ``"2"`` runs second-order;
    anything that does not truncate to 1, 2 or 3 raises. Validation follows R's
    C++ order (no columns, empty columns, degree, missing values), and infinite
    values are compared as R's C++ test compares them.
    """
    values = _sd_matrix(returns, "returns")
    if values.shape[1] == 0:
        return []
    degree, discrete = _validated_sd_matrix(values, degree, type)

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
    return _sd_efficient_active_indices_from_prefix(
        prefix_precomputed,
        active,
        degree,
        discrete=discrete,
    )


def _sd_matrix(data: object, name: str) -> NDArray[np.float64]:
    values = np.asarray(data, dtype=np.float64)
    if values.ndim < 2:
        values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise ValueError(f"{name} must be a 2D array.")
    return values


def _validated_sd_matrix(
    values: NDArray[np.float64], degree: object, type: str
) -> tuple[int, bool]:
    """R's C++ checks for a non-empty set of columns, in its order."""
    if values.shape[0] == 0:
        raise ValueError(_EMPTY_SAMPLE)
    degree_int = _sd_degree(degree)
    if np.isnan(values).any():
        raise ValueError(_MISSING_VALUES)
    type_value = _sd_type_value(degree_int, type)
    return degree_int, degree_int != 1 or type_value != "continuous"


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
    efficient = _undominated_components(dominance_matrix[np.ix_(order, order)])
    return [index for index, keep in zip(order, efficient, strict=True) if keep]


def _undominated_components(
    dominates: NDArray[np.bool_], entered_from_outside: NDArray[np.bool_] | None = None
) -> NDArray[np.bool_]:
    """Members of dominance-cycle components (SCCs) that no column outside them dominates.

    Equal to "not dominated by any column" unless tolerance-level ties make
    dominance cyclic, as in R's NNS_SD_efficient_set_parallel_cpp.
    """
    _, labels = connected_components(dominates, directed=True, connection="strong")
    component = np.asarray(labels, dtype=np.intp)
    _, target = np.nonzero(dominates & (component[:, np.newaxis] != component))
    entered = np.zeros(int(component.max()) + 1, dtype=np.bool_)
    entered[component[target]] = True
    if entered_from_outside is not None:
        entered[component[entered_from_outside]] = True
    return np.logical_not(entered[component])


def _sd_efficient_active_indices_from_prefix(
    precomputed: _SDPrefixPrecomputed,
    active: Sequence[int],
    degree: int,
    *,
    discrete: bool,
    verdicts: dict[tuple[int, int], bool] | None = None,
) -> list[int]:
    """The matrix path's efficient set, testing only the pairs it needs.

    A column nothing dominates is efficient, and one that such a column dominates
    is not. Only columns dominated by dominated columns alone can be left in an
    undominated cycle, so only their pairs are all tested. ``verdicts`` caches pair
    tests across calls, since a verdict does not depend on the other columns.
    """
    if not active:
        return []
    cache: dict[tuple[int, int], bool] = {} if verdicts is None else verdicts

    def dominates(source: int, target: int) -> bool:
        key = (source, target)
        if key not in cache:
            cache[key] = _dominates_from_prefix_pair(
                precomputed, source, target, degree, discrete=discrete
            )
        return cache[key]

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
    # LPM order puts likely dominators first, so the search for one usually stops early.
    free = [
        not any(dominates(source, target) for source in order if source != target)
        for target in order
    ]
    roots = [index for index, is_free in zip(order, free, strict=True) if is_free]
    rest = [index for index, is_free in zip(order, free, strict=True) if not is_free]
    entered = [any(dominates(source, target) for source in roots) for target in rest]
    if all(entered):
        return roots
    # Paths between the rest never pass through a root, which nothing dominates.
    sub = np.array([[i != j and dominates(i, j) for j in rest] for i in rest], dtype=np.bool_)
    rest_efficient = _undominated_components(sub, np.asarray(entered, dtype=np.bool_))
    efficient = set(roots) | {
        index for index, keep in zip(rest, rest_efficient, strict=True) if keep
    }
    return [index for index in order if index in efficient]


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
    thresholds is checked too, as the patched R does.

    Except for discrete FSD the pair is first centred on its joint midrange ``c``.
    Curve values from prefix sums that are equal in exact arithmetic can differ by
    rounding noise of order ``eps * S**degree`` (``S`` the largest absolute centred
    value), so differences up to ``64 * eps * S**d`` plus the inputs' rounding at
    ``|c|`` are ties: ``d`` = 1 for SSD and the mean gate, 2 for TSD, and a ratio
    analogue for continuous FSD. Discrete FSD compares exact ECDF fractions with no
    tolerance.
    """
    x_count = x_sorted.size
    y_count = y_sorted.size
    if not x_sorted[0] >= y_sorted[0]:
        return False
    if x_count == y_count and np.array_equal(x_sorted, y_sorted):
        return False
    level = 0.0
    if degree > 1 or not discrete:
        # Dominance is translation invariant but the tolerances scale with |value|^d,
        # so compare the pair centred on its joint midrange, as R does.
        lo, hi = float(y_sorted[0]), float(max(x_sorted[-1], y_sorted[-1]))
        gap = y_prefix1[-1] / y_count - x_prefix1[-1] / x_count
        if degree > 1 and gap > 2.0 * _SD_EPS64 * max(abs(lo), abs(hi)):
            return False
        centre = 0.5 * lo + 0.5 * hi
        # R compares a pair with an infinite value unshifted: it cannot be centred.
        if math.isfinite(centre):
            x_sorted, y_sorted = x_sorted - centre, y_sorted - centre
            x_prefix1, y_prefix1 = _prefix_sum_1d(x_sorted), _prefix_sum_1d(y_sorted)
            if degree == 3:
                x_prefix2 = _prefix_sum_1d(x_sorted * x_sorted)
                y_prefix2 = _prefix_sum_1d(y_sorted * y_sorted)
            level = abs(centre)
    scale = float(max(abs(x_sorted[0]), abs(x_sorted[-1]), abs(y_sorted[0]), abs(y_sorted[-1])))
    # Summation rounding plus the inputs' own rounding at their level |c|.
    din = 4.0 * _SD_EPS * level
    e1 = _SD_EPS64 * scale + din
    if degree > 1 and y_prefix1[-1] / y_count - x_prefix1[-1] / x_count > e1:
        return False

    thresholds, x_counts, y_counts = _merged_thresholds(x_sorted, y_sorted)

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
        total_min = _cpp_min(x_total, y_total)
        with np.errstate(divide="ignore", invalid="ignore"):
            tol = np.where(total_min > 0.0, _cpp_max(_SD_EPS64, e1 / total_min), _SD_EPS64)
        if not np.any(x_curve - y_curve > tol):
            interior_excess = _ratio_deg1_interior_excess(
                x_prefix1, x_counts, x_count, y_prefix1, y_counts, y_count, thresholds, e1
            )
        if not interior_excess and not np.any(x_curve - y_curve > tol):
            midpoint = _ratio_deg1_midpoint_verdict(
                x_prefix1, x_counts, x_count, y_prefix1, y_counts, y_count, thresholds, e1,
                np.abs(x_curve - y_curve) > tol,
            )
            if midpoint is not None:
                return midpoint
    elif degree == 2:
        # LPM1 is piecewise linear with knots at the data: checking the knots is exact.
        x_curve = (x_counts * thresholds - x_prefix1[x_counts]) / x_count
        y_curve = (y_counts * thresholds - y_prefix1[y_counts]) / y_count
        tol = e1
    else:
        if x_prefix2 is None or y_prefix2 is None:
            raise ValueError("degree 3 prefix evaluation requires second-moment prefixes.")
        x_curve = _lpm_deg2(x_prefix1, x_prefix2, x_counts, thresholds, x_count)
        y_curve = _lpm_deg2(y_prefix1, y_prefix2, y_counts, thresholds, y_count)
        tol = _tsd_tolerance(x_prefix1, x_counts, x_count, y_prefix1, y_counts, y_count,
                             thresholds, scale, din)
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
                din,
            )

    excess = x_curve - y_curve
    if interior_excess or np.any(excess > tol):
        return False
    # Curves equal (within the tolerance) at every threshold -> no dominance (R's
    # curve equality guard).
    return bool(np.any(np.abs(excess) > tol))


def _cpp_min(
    a: float | NDArray[np.float64], b: float | NDArray[np.float64]
) -> NDArray[np.float64]:
    """``std::min(a, b)``, i.e. ``b < a ? b : a``: a NaN in ``b`` yields ``a``."""
    return np.asarray(np.where(b < a, b, a), dtype=np.float64)


def _cpp_max(
    a: float | NDArray[np.float64], b: float | NDArray[np.float64]
) -> NDArray[np.float64]:
    """``std::max(a, b)``, i.e. ``a < b ? b : a``: a NaN in ``b`` yields ``a``."""
    return np.asarray(np.where(a < b, b, a), dtype=np.float64)


def _merged_thresholds(
    x_sorted: NDArray[np.float64], y_sorted: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.intp], NDArray[np.intp]]:
    """The merged unique thresholds and each sample's count of observations <= them.

    Equal to ``np.union1d`` plus ``np.searchsorted(..., side="right")`` per sample,
    but one stable merge of the two sorted runs replaces the two binary searches
    over the 2n thresholds (about 2.5x faster at n = 1e4).
    """
    merged = np.concatenate((x_sorted, y_sorted))
    order = np.argsort(merged, kind="stable")
    values = merged[order]
    from_x = np.cumsum(order < x_sorted.size)
    breaks = np.flatnonzero(values[1:] != values[:-1])
    ends = np.append(breaks, values.size - 1)
    x_counts = from_x[ends]
    y_counts = ends + 1 - x_counts
    return values[ends], x_counts, y_counts


def _lpm1_and_total(
    prefix1: NDArray[np.float64],
    counts: NDArray[np.intp],
    thresholds: NDArray[np.float64],
    observations: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """LPM1 and LPM1 + UPM1 at the thresholds, from prefix sums (R's ``lpm_upm_deg1``)."""
    sums1 = prefix1[counts]
    lower = (counts * thresholds - sums1) / observations
    upper = ((prefix1[-1] - sums1) - (observations - counts) * thresholds) / observations
    return lower, lower + upper


def _ratio_deg1(
    prefix1: NDArray[np.float64],
    counts: NDArray[np.intp],
    thresholds: NDArray[np.float64],
    observations: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Continuous FSD ratio LPM1 / (LPM1 + UPM1) at the thresholds, and LPM1 + UPM1."""
    lower, total = _lpm1_and_total(prefix1, counts, thresholds, observations)
    # 0/0 (every observation equals the threshold): the degree-0 ratio, i.e. the
    # right-continuous CDF value, as LPM.ratio does (it was 0).
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
    e1: float,
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
        valid &= (ts > lower_knots) & (ts < upper_knots) & np.isfinite(upper_knots)
        denom_x = gx * ts + dx
        denom_y = gy * ts + dy
        valid &= (denom_x > 0.0) & (denom_y > 0.0)
        if not np.any(valid):
            return False
        ratio_x = (ax * ts + bx) / denom_x
        ratio_y = (ay * ts + by) / denom_y
        tol = _cpp_max(_SD_EPS64, e1 / _cpp_min(denom_x, denom_y))
        return bool(np.any(valid & (ratio_x - ratio_y > tol)))


def _ratio_deg1_midpoint_verdict(
    x_prefix1: NDArray[np.float64],
    x_counts: NDArray[np.intp],
    x_count: int,
    y_prefix1: NDArray[np.float64],
    y_counts: NDArray[np.intp],
    y_count: int,
    thresholds: NDArray[np.float64],
    e1: float,
    knot_differs: NDArray[np.bool_],
) -> bool | None:
    """The verdict when the ratios first differ at an interval midpoint, else None.

    Two linear-fractional ratios that agree at both knots and at the midpoint agree
    on the whole interval. As in R, a midpoint is checked only while the curves have
    not yet differed (at a knot or an earlier midpoint).
    """
    if thresholds.size < 2:
        return None
    s = 0.5 * (thresholds[:-1] + thresholds[1:])
    kx = x_counts[:-1]
    ky = y_counts[:-1]
    x_lower, x_total = _lpm1_and_total(x_prefix1, kx, s, x_count)
    y_lower, y_total = _lpm1_and_total(y_prefix1, ky, s, y_count)
    # R divides directly here: strictly between knots a finite sample's total
    # deviation is positive.
    d = x_lower / x_total - y_lower / y_total
    tol = _cpp_max(_SD_EPS64, e1 / _cpp_min(x_total, y_total))
    differs = (np.abs(d) > tol) & np.isfinite(thresholds[1:])
    first_knot = np.flatnonzero(knot_differs)
    if first_knot.size:
        differs[first_knot[0] :] = False
    first = np.flatnonzero(differs)
    if not first.size:
        return None
    return not bool(d[first[0]] > tol[first[0]])


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
    din: float,
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
    upper_knots = thresholds[1:][concave]
    inside = (ts > thresholds[:-1][concave]) & (ts < upper_knots) & np.isfinite(upper_knots)
    if not np.any(inside):
        return False
    kx = kx[inside]
    ky = ky[inside]
    ts = ts[inside]
    excess = _lpm_deg2(x_prefix1, x_prefix2, kx, ts, x_count) - _lpm_deg2(
        y_prefix1, y_prefix2, ky, ts, y_count
    )
    tol = _tsd_tolerance(x_prefix1, kx, x_count, y_prefix1, ky, y_count, ts, scale, din)
    return bool(np.any(excess > tol))


def _tsd_tolerance(
    x_prefix1: NDArray[np.float64],
    x_counts: NDArray[np.intp],
    x_count: int,
    y_prefix1: NDArray[np.float64],
    y_counts: NDArray[np.intp],
    y_count: int,
    thresholds: NDArray[np.float64],
    scale: float,
    din: float,
) -> NDArray[np.float64]:
    """R's LPM2 tie bound: input rounding ``din`` moves each term by about ``2 (t - x) din``."""
    x_lpm1 = (x_counts * thresholds - x_prefix1[x_counts]) / x_count
    y_lpm1 = (y_counts * thresholds - y_prefix1[y_counts]) / y_count
    return np.asarray(
        _SD_EPS64 * scale * scale + 2.0 * din * (np.abs(x_lpm1) + np.abs(y_lpm1) + din),
        dtype=np.float64,
    )


def _prefix_sum_1d(values: NDArray[np.float64]) -> NDArray[np.float64]:
    prefix = np.empty(values.size + 1, dtype=np.float64)
    prefix[0] = 0.0
    prefix[1:] = _compensated_cumsum(values)
    return prefix


def _prefix_sum(values: NDArray[np.float64]) -> NDArray[np.float64]:
    prefix = np.empty((values.shape[0] + 1, values.shape[1]), dtype=np.float64, order="F")
    prefix[0, :] = 0.0
    prefix[1:, :] = _compensated_cumsum(values)
    return prefix


def _compensated_cumsum(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Neumaier-compensated running sums along axis 0, as R's SD ``finish_col``.

    A plain running sum drifts by up to ``m * eps * |x|``, which for large samples
    exceeds the tie tolerance of ``_dominates_sorted``. ``np.cumsum`` adds in order,
    so each step's rounding error can be recovered from the running sums.
    """
    sums = np.cumsum(values, axis=0)
    previous = np.zeros_like(sums)
    previous[1:] = sums[:-1]
    errors = np.where(
        np.abs(previous) >= np.abs(values),
        (previous - sums) + values,
        (values - sums) + previous,
    )
    return np.asarray(sums + np.cumsum(errors, axis=0), dtype=np.float64)


def _lpm_at_target(
    values: NDArray[np.float64],
    target: float,
    degree: int,
) -> NDArray[np.float64]:
    """Each column's LPM(degree, target) with the arithmetic of R's efficient-set key.

    R's C++ raises each positive deviation to the power by repeated multiplication
    and sums the rows in their original order, so columns whose keys tie in exact
    arithmetic (permutations, duplicates) keep their tie and the index tie break.
    """
    deviations = target - values
    powered = deviations
    for _ in range(degree - 1):
        powered = powered * deviations
    terms = np.where(deviations > 0.0, powered, 0.0)
    if terms.shape[0] == 0:
        return np.zeros(terms.shape[1], dtype=np.float64)
    return np.asarray(np.cumsum(terms, axis=0)[-1] / terms.shape[0], dtype=np.float64)


def _as_sd_values(x: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    """R's ``to_numeric_vector``: single-column objects and any vector shape flatten."""
    values = np.asarray(x, dtype=np.float64)
    if values.ndim == 2:
        # R accepts single-column matrices and data frames.
        if values.shape[1] != 1:
            raise ValueError(f"{name} must be a single-column object or numeric vector.")
        values = values[:, 0]
    return values.reshape(-1)


def _sd_pair(
    x: NDArray[np.float64], y: NDArray[np.float64], *, finite: bool
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Both samples, checked in R's order: shape, missing values, finiteness, size.

    ``finite`` is for ``NNS.FSD``/``NNS.SSD``/``NNS.TSD``, which evaluate
    ``LPM(.ratio)(degree, sort(c(x, y)), x)`` first: that checks the variable ``x``
    and then the target grid. The ``.uni`` tests go straight to R's C++ test,
    which accepts infinite values.
    """
    x_values = _as_sd_values(x, "x")
    y_values = _as_sd_values(y, "y")
    if np.isnan(x_values).any() or np.isnan(y_values).any():
        raise ValueError(_MISSING_VALUES)
    if finite:
        if not np.all(np.isfinite(x_values)):
            raise ValueError("`variable` must be finite.")
        if not np.all(np.isfinite(y_values)):
            raise ValueError("`target` must be finite.")
    if x_values.size == 0 or y_values.size == 0:
        raise ValueError(_EMPTY_SAMPLE)
    return x_values, y_values
