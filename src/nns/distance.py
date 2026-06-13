from __future__ import annotations

from collections import defaultdict
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

KValue = int | Literal["all"]


def nns_distance(
    rpm: NDArray[np.float64],
    dist_estimate: NDArray[np.float64],
    k: KValue = "all",
    class_: object | None = None,
) -> float:
    """Return R's NNS.distance prediction for one target row.

    ``rpm`` is a numeric matrix whose last column is R's ``y.hat`` column.
    """
    features, y_hat = _split_rpm(rpm)
    dest = _as_vector(dist_estimate)
    if dest.size != features.shape[1]:
        raise ValueError("dist_estimate length must match rpm feature column count.")

    scaled_features, scaled_dest = _rescale_joint(features, dest)
    distances = _distance_sum(scaled_features, scaled_dest, zero_eps=1e-10)
    indices = np.argsort(distances, kind="mergesort")
    k_value = _resolve_k(k, features.shape[0])
    selected = indices[:k_value]
    selected_distances = distances[selected]
    selected_y = y_hat[selected]

    if k_value == 1:
        return float(selected_y[0])

    weights = _combined_weights(selected_distances)
    if class_ is not None:
        return _weighted_mode(selected_y, weights)
    return float(np.dot(selected_y, weights))


def nns_distance_path_single_bulk(
    rpm: NDArray[np.float64],
    x_test: NDArray[np.float64],
    k: KValue,
    class_: object | None = None,
) -> NDArray[np.float64]:
    """Return R's NNS.distance.path.single.bulk predictions for many target rows.

    This mirrors the ``NNS_distance_path_single_parallel_cpp`` kernel, which
    differs from the single-point ``NNS.distance`` kernel in its lognormal rank
    weight (population sd of ranks instead of sample sd).
    """
    features, y_hat = _split_rpm(rpm)
    tests = _as_matrix(x_test, "x_test")
    if tests.shape[1] != features.shape[1]:
        raise ValueError("x_test column count must match rpm feature column count.")

    k_value = _resolve_k(k, features.shape[0])
    out = np.empty(tests.shape[0], dtype=np.float64)
    for row_index, dest in enumerate(tests):
        scaled_features, scaled_dest = _rescale_joint(features, dest)
        distances = _distance_sum(scaled_features, scaled_dest, zero_eps=1e-10)
        indices = np.argsort(distances, kind="mergesort")
        selected = indices[:k_value]
        selected_distances = distances[selected]
        selected_y = y_hat[selected]

        if k_value == 1:
            out[row_index] = float(selected_y[0])
            continue

        weights = _combined_weights(selected_distances, rank_sd_population=True)
        if class_ is not None:
            out[row_index] = _weighted_mode(selected_y, weights)
        else:
            out[row_index] = float(np.dot(selected_y, weights))
    return out


def nns_distance_bulk(
    rpm: NDArray[np.float64],
    x_test: NDArray[np.float64],
    k: KValue,
    class_: object | None = None,
) -> NDArray[np.float64]:
    """Return R's NNS.distance.bulk predictions for many target rows."""
    features, y_hat = _split_rpm(rpm)
    tests = _as_matrix(x_test, "x_test")
    if tests.shape[1] != features.shape[1]:
        raise ValueError("x_test column count must match rpm feature column count.")

    k_value = _resolve_k(k, features.shape[0])
    rpm_rows = _r_column_major_as_row_chunks(features)
    test_rows = _r_column_major_as_row_chunks(tests)
    diff = rpm_rows[np.newaxis, :, :] - test_rows[:, np.newaxis, :]
    distances = np.sum(diff * diff + np.abs(diff), axis=2)
    distances[distances == 0.0] = 1e-12
    order = np.argsort(distances, axis=1, kind="quicksort")[:, :k_value]

    predictions = np.empty(tests.shape[0], dtype=np.float64)
    for row_index, row_order in enumerate(order):
        row_distances = distances[row_index, row_order]
        row_y = y_hat[row_order]
        weights = 1.0 / row_distances
        predictions[row_index] = float(np.dot(row_y, weights) / np.sum(weights))
    return predictions


def _r_column_major_as_row_chunks(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return cast(NDArray[np.float64], np.ravel(values, order="F").reshape(values.shape, order="C"))


def _rescale_joint(
    features: NDArray[np.float64],
    dest: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    col_min = np.minimum(np.min(features, axis=0), dest)
    col_max = np.maximum(np.max(features, axis=0), dest)
    ranges = col_max - col_min
    scaled_features = np.zeros_like(features, dtype=np.float64)
    scaled_dest = np.zeros_like(dest, dtype=np.float64)
    nonzero = ranges != 0.0
    scaled_features[:, nonzero] = (features[:, nonzero] - col_min[nonzero]) / ranges[nonzero]
    scaled_dest[nonzero] = (dest[nonzero] - col_min[nonzero]) / ranges[nonzero]
    return scaled_features, scaled_dest


def _distance_sum(
    features: NDArray[np.float64],
    dest: NDArray[np.float64],
    zero_eps: float,
) -> NDArray[np.float64]:
    diff = features - dest[np.newaxis, :]
    distances = cast(NDArray[np.float64], np.sum(diff * diff + np.abs(diff), axis=1))
    distances[distances == 0.0] = zero_eps
    return distances


def _combined_weights(
    distances: NDArray[np.float64],
    *,
    rank_sd_population: bool = False,
) -> NDArray[np.float64]:
    from scipy import stats  # type: ignore[import-untyped]

    count = distances.size
    ranks = np.arange(1, count + 1, dtype=np.float64)

    uniform = np.full(count, 1.0 / count, dtype=np.float64)
    t_weights = _normalized(stats.t.pdf(distances, df=count))
    empirical = _normalized(
        np.divide(1.0, distances, out=np.zeros_like(distances), where=distances > 0.0)
    )
    exponential = _normalized(stats.expon.pdf(ranks, scale=1.0 / count))

    lognormal = np.zeros(count, dtype=np.float64)
    if count >= 2:
        # R's single-point NNS.distance kernel uses the sample sd of ranks; the
        # bulk path kernels use the population sd sqrt((k^2 - 1) / 12).
        if rank_sd_population:
            sd_ranks = float(np.sqrt((count * count - 1.0) / 12.0))
        else:
            sd_ranks = float(np.std(ranks, ddof=1))
        lognormal = np.abs(stats.lognorm.logpdf(ranks, s=sd_ranks, scale=1.0))[::-1]
        lognormal = _normalized(lognormal)

    power_law = _normalized(ranks**-2.0)

    normal = np.zeros(count, dtype=np.float64)
    sd_distances = float(np.std(distances, ddof=1))
    if np.isfinite(sd_distances) and sd_distances > 0.0:
        normal = _normalized(stats.norm.pdf(distances, loc=0.0, scale=sd_distances))

    rbf = np.zeros(count, dtype=np.float64)
    var_distances = float(np.var(distances, ddof=1))
    if np.isfinite(var_distances) and var_distances > 0.0:
        rbf = _normalized(np.exp(-distances / (2.0 * var_distances)))

    weights = uniform + t_weights + empirical + exponential + lognormal + power_law + normal + rbf
    total = float(np.sum(weights))
    if total > 0.0:
        return weights / total
    return uniform


def _normalized(values: NDArray[np.float64]) -> NDArray[np.float64]:
    clean = np.where(np.isfinite(values), values, 0.0)
    total = float(np.sum(clean))
    if total > 0.0:
        return clean / total
    return np.zeros_like(clean, dtype=np.float64)


def _weighted_mode(y: NDArray[np.float64], weights: NDArray[np.float64]) -> float:
    counts: defaultdict[float, int] = defaultdict(int)
    for value, weight in zip(y, weights, strict=True):
        count = int(np.ceil(100.0 * weight))
        if count > 0:
            counts[float(value)] += count
    if not counts:
        return float("nan")
    best_value, _ = max(counts.items(), key=lambda item: item[1])
    return best_value


def _split_rpm(rpm: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    values = _as_matrix(rpm, "rpm")
    if values.shape[1] < 2:
        raise ValueError("rpm must include at least one feature column and y.hat.")
    return values[:, :-1], values[:, -1]


def _resolve_k(k: KValue, row_count: int) -> int:
    if k == "all":
        return row_count
    k_value = int(k)
    if k_value < 1:
        raise ValueError("k must be >= 1.")
    return min(k_value, row_count)


def _as_vector(x: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("dist_estimate must be 1D.")
    if values.size == 0:
        raise ValueError("dist_estimate must be non-empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError("dist_estimate must contain only finite values.")
    return values


def _as_matrix(x: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"{name} must be 2D.")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")
    return values
