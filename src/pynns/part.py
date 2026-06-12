from __future__ import annotations

import math
from typing import Literal, TypeAlias, TypedDict, cast

import numpy as np
from numpy.typing import NDArray

from pynns.central_tendencies import _nearest_int_half_up_array, nns_mode
from pynns.dependence import _gravity

NoiseReduction: TypeAlias = Literal["off", "mean", "median", "mode", "mode_class"]


PartData = TypedDict(
    "PartData",
    {
        "x": NDArray[np.float64],
        "y": NDArray[np.float64],
        "quadrant": NDArray[np.str_],
        "prior.quadrant": NDArray[np.str_],
    },
)


class RegressionPoints(TypedDict):
    quadrant: NDArray[np.str_]
    x: NDArray[np.float64]
    y: NDArray[np.float64]


PartResult = TypedDict(
    "PartResult",
    {
        "order": int,
        "dt": PartData,
        "regression.points": RegressionPoints,
    },
)


def nns_part(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    type: str | None = None,
    order: int | None = None,
    obs_req: int | None = 8,
    min_obs_stop: bool = True,
    noise_reduction: NoiseReduction = "off",
) -> PartResult:
    """Return R's NNS.part partition map as NumPy arrays."""
    x_values, y_values = _as_pair(x, y)
    noise = _validate_noise_reduction(noise_reduction)
    if obs_req is None:
        obs_req = 8
    if obs_req < 0:
        raise ValueError("obs_req must be non-negative.")
    if order is None:
        max_order = max(math.ceil(math.log2(max(1, x_values.size))), 1)
    else:
        if isinstance(order, bool) or not isinstance(order, int):
            raise TypeError("order must be an integer or None.")
        max_order = order
        if max_order == 0:
            max_order = 1
        if max_order < 0:
            raise ValueError("order must be non-negative.")

    xonly = type is not None
    n = x_values.size
    floor_order = math.floor(math.log2(max(1, n)))
    quadrants = np.full(n, "q", dtype=f"<U{max_order + 1}")
    prior_quadrants = np.full(n, "pq", dtype=f"<U{max_order + 1}")
    depth = 0

    while True:
        if depth >= max_order:
            break
        if depth >= floor_order:
            break

        groups, inverse, counts = np.unique(quadrants, return_inverse=True, return_counts=True)
        split_group_ids = np.flatnonzero(counts > obs_req)
        if split_group_ids.size == 0:
            break

        center_x, center_y = _centers_for_groups(
            x_values,
            y_values,
            inverse,
            groups.size,
            split_group_ids,
            noise,
        )

        for group_id in split_group_ids:
            mask = inverse == group_id
            prior_quadrants[mask] = groups[group_id]
            cx = center_x[group_id]
            if xonly:
                low_x = np.isfinite(x_values[mask]) & np.isfinite(cx) & (x_values[mask] > cx)
                digits = np.where(low_x, "2", "1")
            else:
                cy = center_y[group_id]
                low_x = np.isfinite(x_values[mask]) & np.isfinite(cx) & (x_values[mask] <= cx)
                low_y = np.isfinite(y_values[mask]) & np.isfinite(cy) & (y_values[mask] <= cy)
                qn = 1 + low_x.astype(np.int64) + 2 * low_y.astype(np.int64)
                digits = qn.astype(str)
            quadrants[mask] = np.char.add(quadrants[mask], digits)

        depth += 1

        if min_obs_stop:
            _, post_counts = np.unique(quadrants, return_counts=True)
            if int(np.min(post_counts)) <= obs_req:
                break

    regression_points = _regression_points(x_values, y_values, prior_quadrants, noise)
    if _is_discrete_like_r(x_values):
        regression_points["x"] = _nearest_int_half_up_array(regression_points["x"])

    return {
        "order": depth,
        "dt": {
            "x": x_values.copy(),
            "y": y_values.copy(),
            "quadrant": quadrants.astype(str),
            "prior.quadrant": prior_quadrants.astype(str),
        },
        "regression.points": regression_points,
    }


def _centers_for_groups(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    inverse: NDArray[np.int64],
    n_groups: int,
    split_group_ids: NDArray[np.int64],
    noise: NoiseReduction,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    center_x = np.full(n_groups, np.nan, dtype=np.float64)
    center_y = np.full(n_groups, np.nan, dtype=np.float64)

    if noise == "mean":
        counts = np.bincount(inverse, minlength=n_groups).astype(np.float64)
        center_x[:] = np.bincount(inverse, weights=x, minlength=n_groups) / counts
        center_y[:] = np.bincount(inverse, weights=y, minlength=n_groups) / counts
        return center_x, center_y

    for group_id in split_group_ids:
        values_x = x[inverse == group_id]
        values_y = y[inverse == group_id]
        center_x[group_id] = _aggregate_x(values_x, noise)
        center_y[group_id] = _aggregate_y(values_y, noise)
    return center_x, center_y


def _regression_points(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    prior_quadrants: NDArray[np.str_],
    noise: NoiseReduction,
) -> RegressionPoints:
    groups = np.unique(prior_quadrants)
    out_x = np.empty(groups.size, dtype=np.float64)
    out_y = np.empty(groups.size, dtype=np.float64)
    for index, group in enumerate(groups):
        mask = prior_quadrants == group
        out_x[index] = _aggregate_x(x[mask], noise)
        out_y[index] = _aggregate_y(y[mask], noise)
    order = np.argsort(groups)
    return {
        "quadrant": groups[order].astype(str),
        "x": out_x[order],
        "y": out_y[order],
    }


def _aggregate_x(values: NDArray[np.float64], noise: NoiseReduction) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    if noise == "mean":
        return float(np.mean(finite))
    if noise == "median":
        return float(np.median(finite))
    if noise == "mode":
        return _mode(finite)
    return _gravity(finite)


def _aggregate_y(values: NDArray[np.float64], noise: NoiseReduction) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    if noise == "mean":
        return float(np.mean(finite))
    if noise == "median":
        return float(np.median(finite))
    if noise in {"mode", "mode_class"}:
        return _mode(finite)
    return _gravity(finite)


def _mode(values: NDArray[np.float64]) -> float:
    """Private compatibility wrapper for NNS_part's discrete mode path."""
    return float(nns_mode(values, discrete=True, multi=False))


def _is_discrete_like_r(values: NDArray[np.float64]) -> bool:
    finite = values[np.isfinite(values)]
    return bool(finite.size > 0 and np.all(finite == np.floor(finite)))


def _validate_noise_reduction(value: str) -> NoiseReduction:
    noise = value.lower()
    if noise not in {"off", "mean", "median", "mode", "mode_class"}:
        raise ValueError(
            "noise_reduction must be one of 'mean', 'median', 'mode', 'mode_class', 'off'."
        )
    return cast(NoiseReduction, noise)


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
