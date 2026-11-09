from __future__ import annotations

import math
from collections import defaultdict
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from nns._helpers import _as_matrix
from nns._rnmath import dexp, dlnorm_log, dnorm, dt, libm_exp
from nns._stdsort import heap_select, partial_sort, std_sort

KValue = int | Literal["all"]


def nns_distance(
    rpm: NDArray[np.float64],
    dist_estimate: NDArray[np.float64],
    k: KValue = "all",
    class_: object | None = None,
) -> float:
    """Return R's NNS.distance prediction for one target row.

    ``rpm`` is a numeric matrix whose last column is R's ``y.hat`` column.
    The value is R's ``NNS_distance_cpp`` to the last bit: range-scaled
    distances, libstdc++'s ordering of tied distances, R's nmath ``dt``,
    ``dexp``, ``dlnorm`` and ``dnorm`` weights, and sequential double sums.
    """
    features, y_hat = _split_rpm(rpm)
    dest = _as_vector(dist_estimate)
    if dest.size != features.shape[1]:
        raise ValueError("dist_estimate length must match rpm feature column count.")
    k_value = _resolve_k(k, features.shape[0])

    col_min = np.minimum(np.min(features, axis=0), dest)
    col_max = np.maximum(np.max(features, axis=0), dest)
    distances = _distances(features, dest, _inverse_ranges(col_min, col_max))
    order = _cpp_sorted_prefix(distances, k_value)
    selected_distances = distances[order]
    selected_y = y_hat[order]
    if k_value == 1:
        return float(selected_y[0])

    weights = _scalar_weights(selected_distances)
    if class_ is not None:
        return _weighted_mode(selected_y, weights)
    return _sequential_sum(selected_y * weights)


def nns_distance_path_single_bulk(
    rpm: NDArray[np.float64],
    x_test: NDArray[np.float64],
    k: KValue,
    class_: object | None = None,
) -> NDArray[np.float64]:
    """Return R's NNS.distance.path.single.bulk predictions for many target rows.

    This is R's ``NNS_distance_path_single_parallel_cpp``, which differs from the
    single-point kernel: it orders the k nearest rows by distance and then row,
    weights them with unnormalised Student-t and Gaussian kernels and a
    ``sd(1:k)`` lognormal rank weight, and scales the weighted sum by ``1 / total``.
    """
    features, y_hat = _split_rpm(rpm)
    tests = _as_matrix(x_test, "x_test")
    if tests.shape[1] != features.shape[1]:
        raise ValueError("x_test column count must match rpm feature column count.")

    k_value = _resolve_k(k, features.shape[0])
    rank_weights = _bulk_rank_weights(k_value)
    rpm_min = np.min(features, axis=0)
    rpm_max = np.max(features, axis=0)
    out = np.empty(tests.shape[0], dtype=np.float64)
    for row_index, dest in enumerate(tests):
        inverse = _inverse_ranges(np.minimum(rpm_min, dest), np.maximum(rpm_max, dest))
        distances = _distances(features, dest, inverse)
        selected = _cpp_selected_by_row(distances, k_value)
        top_distances = distances[selected]
        top_y = y_hat[selected]
        if k_value == 1:
            out[row_index] = float(top_y[0])
            continue
        out[row_index] = _bulk_prediction(top_distances, top_y, rank_weights, class_ is not None)
    return out


def nns_distance_bulk(
    rpm: NDArray[np.float64],
    x_test: NDArray[np.float64],
    k: KValue,
    class_: object | None = None,
) -> NDArray[np.float64]:
    """Return R's NNS.distance.bulk predictions for many target rows."""
    # R14 R/NNS_Distance_bulk.R routes the public bulk API through the
    # SAME eight-channel distance ensemble as the single-path bulk kernel, not
    # the old 1/d-only NNS_distance_bulk_cpp regression helper.
    return nns_distance_path_single_bulk(rpm, x_test, k, class_)


def _inverse_ranges(
    col_min: NDArray[np.float64], col_max: NDArray[np.float64]
) -> NDArray[np.float64]:
    """R's ``invR``: ``1 / range`` for a finite positive range, else 0 (column skipped)."""
    ranges = col_max - col_min
    usable = np.isfinite(ranges) & (ranges > 0.0)
    return np.where(usable, 1.0 / np.where(usable, ranges, 1.0), 0.0)


def _distances(
    features: NDArray[np.float64],
    dest: NDArray[np.float64],
    inverse: NDArray[np.float64],
) -> NDArray[np.float64]:
    """R's ``S``: per row, ``diff = (a - b) * invR`` and ``acc += diff^2 + |diff|``
    over the columns in order; an exact 0 becomes ``1e-10``."""
    acc = np.zeros(features.shape[0], dtype=np.float64)
    for column in range(features.shape[1]):
        if not inverse[column] > 0.0:
            continue
        diff = (features[:, column] - dest[column]) * inverse[column]
        acc = acc + (diff * diff + np.abs(diff))
    return np.where(acc == 0.0, 1e-10, acc)


def _has_ties(values: NDArray[np.float64]) -> bool:
    return bool(np.unique(values).size != values.size)


def _cpp_sorted_prefix(distances: NDArray[np.float64], count: int) -> NDArray[np.intp]:
    """The first ``count`` rows in R's order: ``std::partial_sort`` when
    ``count < n``, else ``std::sort``, on ``S[a] < S[b]``."""
    n = distances.size
    if not _has_ties(distances):
        return np.argsort(distances, kind="stable")[:count]
    order = list(range(n))
    key = distances.tolist()
    if count < n:
        partial_sort(order, key, count)
    else:
        std_sort(order, key)
    return np.asarray(order[:count], dtype=np.intp)


def _cpp_selected_by_row(distances: NDArray[np.float64], count: int) -> NDArray[np.intp]:
    """The bulk kernel's k nearest rows: ``std::partial_sort`` picks them, then a
    stable sort orders them by distance and row index."""
    n = distances.size
    ranked = np.argsort(distances, kind="stable")
    if count >= n or distances[ranked[count - 1]] < distances[ranked[count]]:
        return ranked[:count]
    # A tie straddles the k-th place: which tied rows are kept is libstdc++'s
    # heap selection.
    order = list(range(n))
    heap_select(order, distances.tolist(), count)
    chosen = np.asarray(order[:count], dtype=np.intp)
    return chosen[np.lexsort((chosen, distances[chosen]))]


def _sequential_sum(values: NDArray[np.float64]) -> float:
    """``std::accumulate(begin, end, 0.0)``: a running double sum."""
    if values.size == 0:
        return 0.0
    return float(np.cumsum(values)[-1])


def _normalised(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """R's ``if (sum > 0) v /= sum; else 0`` with a sequential double sum."""
    total = _sequential_sum(values)
    if total > 0.0:
        return np.asarray(values / total, dtype=np.float64)
    return np.zeros_like(values, dtype=np.float64)


def _sd_vec(values: NDArray[np.float64]) -> float:
    """R's ``sd_vec``: double mean, sequential sum of squares, ``sqrt(acc / (n - 1))``."""
    n = values.size
    if n < 2:
        return math.nan
    mean = _sequential_sum(values) / float(n)
    deviations = values - mean
    return math.sqrt(_sequential_sum(deviations * deviations) / float(n - 1))


def _var_vec(values: NDArray[np.float64]) -> float:
    """R's ``var_vec``: the square of ``sd_vec``."""
    sd = _sd_vec(values)
    return sd * sd if math.isfinite(sd) else math.nan


def _finite_or_zero(values: list[float]) -> NDArray[np.float64]:
    out = np.asarray(values, dtype=np.float64)
    return np.where(np.isfinite(out), out, 0.0)


def _scalar_weights(distances: NDArray[np.float64]) -> NDArray[np.float64]:
    """The eight-component weights of R's ``NNS_distance_cpp``."""
    count = distances.size
    ranks = np.arange(1, count + 1, dtype=np.float64)
    uniform = np.full(count, 1.0 / count, dtype=np.float64)
    t_weights = _normalised(_finite_or_zero([dt(float(d), float(count)) for d in distances]))
    empirical = _normalised(
        np.where(distances > 0.0, 1.0 / np.where(distances > 0.0, distances, 1.0), 0.0)
    )
    exponential = _normalised(_finite_or_zero([dexp(float(r), float(count)) for r in ranks]))
    lognormal = np.zeros(count, dtype=np.float64)
    sd_ranks = _sd_vec(ranks) if count >= 2 else math.nan
    if math.isfinite(sd_ranks):
        lognormal = np.abs(
            np.asarray([dlnorm_log(float(r), 0.0, sd_ranks) for r in ranks], dtype=np.float64)
        )[::-1]
    lognormal = _normalised(lognormal)
    power_law = _normalised(np.asarray([math.pow(float(r), -2.0) for r in ranks]))
    normal = np.zeros(count, dtype=np.float64)
    sd_distances = _sd_vec(distances)
    if math.isfinite(sd_distances) and sd_distances > 0.0:
        normal = _normalised(
            _finite_or_zero([dnorm(float(d), 0.0, sd_distances) for d in distances])
        )
    rbf = np.zeros(count, dtype=np.float64)
    var_distances = _var_vec(distances)
    if math.isfinite(var_distances) and var_distances > 0.0:
        rbf = _normalised(libm_exp(-distances / (2.0 * var_distances)))

    weights = uniform + t_weights + empirical + exponential + lognormal + power_law + normal + rbf
    total = _sequential_sum(weights)
    if total > 0.0:
        return np.asarray(weights / total, dtype=np.float64)
    return uniform


def _bulk_rank_weights(count: int) -> NDArray[np.float64]:
    """The bulk kernel's rank-only weights ``uniW + expW + lnormW + plW``, summed
    in that order (each normalised as R does)."""
    ranks = np.arange(1, count + 1, dtype=np.float64)
    uniform = np.full(count, 1.0 / count, dtype=np.float64)
    exponential = _normalised(np.asarray([dexp(float(r), float(count)) for r in ranks]))
    power_law = _normalised(np.asarray([math.pow(float(r), -2.0) for r in ranks]))
    lognormal = np.zeros(count, dtype=np.float64)
    if count >= 2:
        # R matches sd(1:k): sqrt(k (k + 1) / 12).
        sdlog = math.sqrt(float(count) * (count + 1.0) / 12.0)
        lognormal = _normalised(
            np.abs(np.asarray([dlnorm_log(float(r), 0.0, sdlog) for r in ranks]))[::-1]
        )
    return np.asarray(uniform + exponential + lognormal + power_law, dtype=np.float64)


def _bulk_prediction(
    distances: NDArray[np.float64],
    y: NDArray[np.float64],
    rank_weights: NDArray[np.float64],
    is_class: bool,
) -> float:
    """One test row of R's ``NNS_distance_path_single_parallel_cpp`` (k >= 2)."""
    count = distances.size
    exponent = -float(count + 1) / 2.0
    t_weights = _normalised(
        np.asarray([math.pow(1.0 + (d * d) / count, exponent) for d in distances.tolist()])
    )
    empirical = _normalised(
        np.where(distances > 0.0, 1.0 / np.where(distances > 0.0, distances, 1.0), 0.0)
    )
    normal = np.zeros(count, dtype=np.float64)
    sd_distances = _sd_vec(distances)
    if math.isfinite(sd_distances) and sd_distances > 0.0:
        z = distances / sd_distances
        normal = _normalised(libm_exp(-0.5 * z * z))
    rbf = np.zeros(count, dtype=np.float64)
    var_distances = _var_vec(distances)
    if math.isfinite(var_distances) and var_distances > 0.0:
        rbf = _normalised(libm_exp(-distances / (2.0 * var_distances)))

    raw = rank_weights + t_weights + empirical + normal + rbf
    total = _sequential_sum(raw)
    inverse_total = 1.0 / total if total > 0.0 else 1.0 / count
    if not is_class:
        if total > 0.0:
            return _sequential_sum(y * raw) * inverse_total
        return _sequential_sum(y) / count
    weights = raw * inverse_total if total > 0.0 else np.full(count, 1.0 / count)
    return _weighted_mode(y, weights)


def _weighted_mode(y: NDArray[np.float64], weights: NDArray[np.float64]) -> float:
    """R's ``mode_class_weighted``: votes ``ceil(100 w)``; among equal totals the
    smallest class value wins (R scans the classes in ascending order)."""
    if y.size == 0:
        return float("nan")
    if y.size == 1:
        return float(y[0])
    counts: defaultdict[float, int] = defaultdict(int)
    for value, weight in zip(y.tolist(), weights.tolist(), strict=True):
        votes = math.ceil(100.0 * weight)
        if votes > 0:
            counts[value] += votes
    if not counts:
        return float("nan")
    best_value = min(counts)
    best_count = counts[best_value]
    for value in sorted(counts):
        if counts[value] > best_count:
            best_value, best_count = value, counts[value]
    return best_value


def _split_rpm(rpm: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    values = _as_matrix(rpm, "rpm")
    if values.shape[1] < 2:
        raise ValueError("rpm must include at least one feature column and y.hat.")
    return values[:, :-1], values[:, -1]


def _resolve_k(k: KValue, row_count: int) -> int:
    """R's k handling: ``"all"`` or an infinite k means every row; otherwise
    ``as.integer(k[1])`` (truncation; NA means every row) clamped to ``1..n``."""
    if isinstance(k, str):
        if k == "all":
            return row_count
        try:
            value = float(k)
        except ValueError:
            return row_count
    else:
        value = float(np.asarray(k, dtype=np.float64).ravel()[0])
    if not math.isfinite(value) or abs(value) >= 2.0**31:
        return row_count
    return max(1, min(int(value), row_count))


def _as_vector(x: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("dist_estimate must be 1D.")
    if values.size == 0:
        raise ValueError("dist_estimate must be non-empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError("dist_estimate must contain only finite values.")
    return values
