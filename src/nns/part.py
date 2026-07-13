from __future__ import annotations

import math
from typing import Any, Literal, TypeAlias, TypedDict, cast

import numpy as np
from numpy.typing import NDArray

from nns._helpers import _as_pair
from nns._native import native_fn
from nns.central_tendencies import _nearest_int_half_up_array, nns_mode
from nns.dependence import _gravity

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


def _part_pair(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Equal-length 1D pair with complete-case NA handling; infinities kept."""
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    if x_values.ndim != 1 or y_values.ndim != 1:
        raise ValueError("x and y must be 1D.")
    if x_values.size != y_values.size:
        raise ValueError("x and y must have the same length.")
    if x_values.size == 0:
        raise ValueError("x and y must be non-empty.")
    complete = ~(np.isnan(x_values) | np.isnan(y_values))
    if not np.all(complete):
        x_values = x_values[complete]
        y_values = y_values[complete]
    if x_values.size == 0:
        raise ValueError("x and y must contain at least one complete (non-missing) pair.")
    return x_values, y_values


def _part_max(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    type_value: str | None,
    noise: NoiseReduction,
) -> PartResult:
    """R NNS.part order='max': explicit maximum partition representation."""
    from nns.central_tendencies import nns_gravity, nns_mode

    if type_value is None:
        quadrants = np.asarray([f"q{i + 1}" for i in range(x.size)])
        result: PartResult = {
            "order": int(x.size),
            "dt": {
                "x": x.copy(),
                "y": y.copy(),
                "quadrant": quadrants,
                "prior.quadrant": np.full(x.size, "pq"),
            },
            "regression.points": {
                "x": x.copy(),
                "y": y.copy(),
                "quadrant": quadrants,
            },
        }
        return result

    def reducer(z: NDArray[np.float64]) -> float:
        if noise == "mean":
            return float(np.mean(z))
        if noise == "median":
            return float(np.median(z))
        if noise in ("mode", "mode_class"):
            return float(nns_mode(z, discrete=True, multi=False))
        return float(nns_gravity(z))

    unique_x = np.unique(x)
    y_by_x = np.asarray([reducer(y[x == v]) for v in unique_x], dtype=np.float64)
    match_id = np.searchsorted(unique_x, x) + 1
    result = {
        "order": int(unique_x.size),
        "dt": {
            "x": x.copy(),
            "y": y.copy(),
            "quadrant": np.asarray([f"q{i}" for i in match_id]),
            "prior.quadrant": np.full(x.size, "pq"),
        },
        "regression.points": {
            "x": unique_x,
            "y": y_by_x,
            "quadrant": np.asarray([f"q{i + 1}" for i in range(unique_x.size)]),
        },
    }
    return result


def nns_part(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    type: str | None = None,
    order: int | str | None = None,
    obs_req: int | None = 8,
    min_obs_stop: bool = True,
    noise_reduction: NoiseReduction = "off",
) -> PartResult:
    """Return R's NNS.part partition map as NumPy arrays.

    Mirrors the repaired R NNS.part: pairs with NA/NaN in either variable are
    dropped (complete-case handling keeps infinities), and ``order='max'``
    returns the explicit maximum partition representation without recursion.
    """
    x_values, y_values = _part_pair(x, y)
    noise = _validate_noise_reduction(noise_reduction)
    if obs_req is None:
        obs_req = 8
    if obs_req < 0:
        raise ValueError("obs_req must be non-negative.")

    if isinstance(order, str):
        if order.lower() != "max":
            raise ValueError("order must be None, 'max', or a positive integer.")
        return _part_max(x_values, y_values, type, noise)

    if order is None:
        max_order = max(math.ceil(math.log2(max(1, x_values.size))), 1)
    else:
        if isinstance(order, bool) or not isinstance(order, int):
            raise TypeError("order must be an integer, 'max', or None.")
        max_order = order
        if max_order == 0:
            max_order = 1
        if max_order < 0:
            raise ValueError("order must be non-negative.")

    xonly = type is not None
    n = x_values.size

    native_part = native_fn("nns_part_off")
    if native_part is not None and noise == "off":
        res = native_part(
            np.ascontiguousarray(x_values),
            np.ascontiguousarray(y_values),
            xonly,
            int(max_order),
            int(obs_req),
            bool(min_obs_stop),
        )
        return _part_result_from_native(res, x_values, y_values)

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


def _part_result_from_native(
    res: dict[str, Any],
    x_values: NDArray[np.float64],
    y_values: NDArray[np.float64],
) -> PartResult:
    """Rebuild the NNS.part payload from the native nns_part_off result."""
    dt = res["dt"]
    rp = res["regression.points"]
    return {
        "order": int(res["order"]),
        "dt": {
            "x": x_values.copy(),
            "y": y_values.copy(),
            "quadrant": np.asarray(dt["quadrant"], dtype=str),
            "prior.quadrant": np.asarray(dt["prior.quadrant"], dtype=str),
        },
        "regression.points": {
            "quadrant": np.asarray(rp["quadrant"], dtype=str),
            "x": np.asarray(rp["x"], dtype=np.float64),
            "y": np.asarray(rp["y"], dtype=np.float64),
        },
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
    # R mean()/median() propagate infinities; gravity/mode filter to finite
    # values internally, matching NNS.gravity / NNS.mode.
    if noise == "mean":
        return float(np.mean(values))
    if noise == "median":
        return float(np.median(values))
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    if noise == "mode":
        return _mode(finite)
    return _gravity(finite)


def _aggregate_y(values: NDArray[np.float64], noise: NoiseReduction) -> float:
    if noise == "mean":
        return float(np.mean(values))
    if noise == "median":
        return float(np.median(values))
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
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
