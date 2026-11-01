from __future__ import annotations

import math
import numbers
from typing import Any, Literal, TypeAlias, TypedDict, cast

import numpy as np
from numpy.typing import NDArray

from nns._native import native_fn
from nns._rmean import r_group_means_no_na, r_mean, r_mean_no_na
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
            return r_mean(z)
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
    ``type`` is ``None`` (XY quadrants) or ``"XONLY"``; ``order`` is ``None``,
    ``"max"`` or a positive integer and ``obs_req`` a non-negative integer
    (NumPy integers and integral floats are accepted). The returned ``order`` is
    the depth reached, at most ``floor(log2(n))``.
    """
    if type is not None and (not isinstance(type, str) or type.lower() != "xonly"):
        raise ValueError("[type] must be NULL or 'XONLY'.")
    x_values, y_values = _part_pair(x, y)
    noise = _validate_noise_reduction(noise_reduction)
    obs_req = _validate_obs_req(obs_req)

    if isinstance(order, str):
        if order.lower() != "max":
            raise ValueError("order must be None, 'max', or a positive integer.")
        return _part_max(x_values, y_values, type, noise)

    n = x_values.size
    floor_order = math.floor(math.log2(max(1, n)))
    if order is None:
        max_order = max(math.ceil(math.log2(max(1, n))), 1)
    else:
        max_order = _validate_order(order)
    # The partition never goes deeper than floor(log2(n)) (the loop below stops
    # there), so a larger requested order is equivalent to that depth.
    max_order = max(1, min(max_order, floor_order))

    xonly = type is not None

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
        native_result = _part_result_from_native(res, x_values, y_values)
        if _is_discrete_like_r(x_values):
            # The native kernel still rounds x half-up for integer-valued x, which
            # R removed (R fix 9328583); restore the cells' x centroids.
            native_result["regression.points"]["x"] = _prior_group_gravity(
                x_values,
                native_result["dt"]["prior.quadrant"],
                native_result["regression.points"]["quadrant"],
            )
        return native_result

    # Groups are tracked as dense integer ids whose numeric order equals the
    # lexicographic order of their quadrant labels: the current labels are the
    # leaves of the partition tree (prefix-free), so ordering children by
    # (parent id, appended digit) keeps the label order.
    group = np.zeros(n, dtype=np.intp)
    labels = ["q"]
    prior_group = np.full(n, -1, dtype=np.intp)  # index into prior_labels; -1 is "pq"
    prior_labels: list[str] = []
    depth = 0

    while depth < max_order and depth < floor_order:
        counts = np.bincount(group, minlength=len(labels))
        split = counts > obs_req
        split_group_ids = np.flatnonzero(split)
        if split_group_ids.size == 0:
            break

        center_x, center_y = _centers_for_groups(
            x_values,
            y_values,
            group,
            counts,
            split_group_ids,
            noise,
        )

        in_split = split[group]
        prior_index = np.full(len(labels), -1, dtype=np.intp)
        prior_index[split_group_ids] = len(prior_labels) + np.arange(split_group_ids.size)
        prior_labels.extend(labels[g] for g in split_group_ids.tolist())
        split_group = group[in_split]
        prior_group[in_split] = prior_index[split_group]

        xs = x_values[in_split]
        cx = center_x[split_group]
        if xonly:
            high_x = np.isfinite(xs) & np.isfinite(cx) & (xs > cx)
            digits = 1 + high_x.astype(np.intp)
        else:
            ys = y_values[in_split]
            cy = center_y[split_group]
            low_x = np.isfinite(xs) & np.isfinite(cx) & (xs <= cx)
            low_y = np.isfinite(ys) & np.isfinite(cy) & (ys <= cy)
            digits = 1 + low_x.astype(np.intp) + 2 * low_y.astype(np.intp)
        digit = np.zeros(n, dtype=np.intp)
        digit[in_split] = digits

        keys, group = np.unique(group * 5 + digit, return_inverse=True)
        group = group.reshape(-1)
        labels = [
            labels[key // 5] + (str(key % 5) if key % 5 else "") for key in keys.tolist()
        ]
        depth += 1

        if min_obs_stop and int(np.min(np.bincount(group))) <= obs_req:
            break

    all_prior_labels = [*prior_labels, "pq"]
    prior_group[prior_group < 0] = len(prior_labels)
    regression_points = _regression_points(
        x_values, y_values, prior_group, all_prior_labels, noise
    )

    return {
        "order": depth,
        "dt": {
            "x": x_values.copy(),
            "y": y_values.copy(),
            "quadrant": np.asarray(labels)[group],
            "prior.quadrant": np.asarray(all_prior_labels)[prior_group],
        },
        "regression.points": regression_points,
    }


def _validate_order(order: object) -> int:
    """R: [order] must be NULL, 'max', or a positive integer (integral doubles allowed)."""
    if isinstance(order, (bool, np.bool_)):
        raise TypeError("order must be an integer, 'max', or None.")
    if isinstance(order, numbers.Integral):
        value = int(order)
    elif isinstance(order, numbers.Real):
        as_float = float(order)
        if not math.isfinite(as_float) or as_float != math.floor(as_float):
            raise ValueError("order must be None, 'max', or a positive integer.")
        value = int(as_float)
    else:
        raise TypeError("order must be an integer, 'max', or None.")
    if value < 1:
        raise ValueError("order must be None, 'max', or a positive integer.")
    return value


def _validate_obs_req(obs_req: object) -> int:
    """R: [obs.req] must be a nonnegative integer (NULL means 8)."""
    if obs_req is None:
        return 8
    if isinstance(obs_req, (bool, np.bool_)):
        raise TypeError("obs_req must be a non-negative integer.")
    if isinstance(obs_req, numbers.Integral):
        value = int(obs_req)
    elif isinstance(obs_req, numbers.Real):
        as_float = float(obs_req)
        if not math.isfinite(as_float) or as_float != math.floor(as_float):
            raise ValueError("obs_req must be a non-negative integer.")
        value = int(as_float)
    else:
        raise TypeError("obs_req must be a non-negative integer.")
    if value < 0:
        raise ValueError("obs_req must be non-negative.")
    return value


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


def _group_slices(group: NDArray[np.intp], counts: NDArray[np.intp]) -> tuple[
    NDArray[np.intp], NDArray[np.intp]
]:
    """Stable ordering of the points by group and each group's start offset.

    ``values[order[start:start + count]]`` equals ``values[group == g]`` element for
    element (same values, same order), so per-group reductions are bit-identical
    to the boolean-mask formulation without its O(n * groups) cost.
    """
    order = np.argsort(group, kind="stable")
    starts = np.zeros(counts.size, dtype=np.intp)
    np.cumsum(counts[:-1], out=starts[1:])
    return order, starts


def _group_medians(
    values: NDArray[np.float64],
    group: NDArray[np.intp],
    counts: NDArray[np.intp],
    group_ids: NDArray[np.intp],
) -> NDArray[np.float64]:
    """np.median of each listed group (odd: middle value; even: mean of the middle two)."""
    order = np.lexsort((values, group))
    sorted_values = values[order]
    starts = np.zeros(counts.size, dtype=np.intp)
    np.cumsum(counts[:-1], out=starts[1:])
    m = counts[group_ids]
    lo = starts[group_ids] + (m - 1) // 2
    hi = starts[group_ids] + m // 2
    with np.errstate(invalid="ignore"):
        medians: NDArray[np.float64] = (sorted_values[lo] + sorted_values[hi]) / 2.0
    return medians


def _centers_for_groups(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    group: NDArray[np.intp],
    counts: NDArray[np.intp],
    split_group_ids: NDArray[np.intp],
    noise: NoiseReduction,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    n_groups = counts.size
    center_x = np.full(n_groups, np.nan, dtype=np.float64)
    center_y = np.full(n_groups, np.nan, dtype=np.float64)

    if noise == "mean":
        # NNS_part_cpp's mean_no_na: long-double sums in row order.
        center_x[:] = r_group_means_no_na(x, group, n_groups)
        center_y[:] = r_group_means_no_na(y, group, n_groups)
        return center_x, center_y

    if noise == "median":
        center_x[split_group_ids] = _group_medians(x, group, counts, split_group_ids)
        center_y[split_group_ids] = _group_medians(y, group, counts, split_group_ids)
        return center_x, center_y

    order, starts = _group_slices(group, counts)
    for group_id in split_group_ids.tolist():
        idx = order[starts[group_id] : starts[group_id] + counts[group_id]]
        center_x[group_id] = _aggregate_x(x[idx], noise)
        center_y[group_id] = _aggregate_y(y[idx], noise)
    return center_x, center_y


def _regression_points(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    prior_group: NDArray[np.intp],
    prior_labels: list[str],
    noise: NoiseReduction,
) -> RegressionPoints:
    counts = np.bincount(prior_group, minlength=len(prior_labels))
    present = np.flatnonzero(counts > 0)
    if noise == "median":
        out_x = _group_medians(x, prior_group, counts, present)
        out_y = _group_medians(y, prior_group, counts, present)
    else:
        order, starts = _group_slices(prior_group, counts)
        out_x = np.empty(present.size, dtype=np.float64)
        out_y = np.empty(present.size, dtype=np.float64)
        for index, group_id in enumerate(present.tolist()):
            idx = order[starts[group_id] : starts[group_id] + counts[group_id]]
            out_x[index] = _aggregate_x(x[idx], noise)
            out_y[index] = _aggregate_y(y[idx], noise)
    names = np.asarray([prior_labels[g] for g in present.tolist()])
    label_order = np.argsort(names, kind="stable")
    return {
        "quadrant": names[label_order],
        "x": out_x[label_order],
        "y": out_y[label_order],
    }


def _aggregate_x(values: NDArray[np.float64], noise: NoiseReduction) -> float:
    # R mean()/median() propagate infinities; gravity/mode filter to finite
    # values internally, matching NNS.gravity / NNS.mode.
    if noise == "mean":
        return r_mean_no_na(values)
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
        return r_mean_no_na(values)
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


def _prior_group_gravity(
    x: NDArray[np.float64],
    prior_quadrant: NDArray[np.str_],
    labels: NDArray[np.str_],
) -> NDArray[np.float64]:
    """Gravity of ``x`` within each ``prior_quadrant`` group, in ``labels`` order."""
    unique, inverse = np.unique(prior_quadrant, return_inverse=True)
    position = {str(label): index for index, label in enumerate(unique.tolist())}
    order = np.argsort(inverse, kind="stable")
    counts = np.bincount(inverse, minlength=unique.size)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    out = np.empty(labels.size, dtype=np.float64)
    for index, label in enumerate(labels.tolist()):
        group = position[str(label)]
        out[index] = _aggregate_x(x[order[starts[group] : starts[group] + counts[group]]], "off")
    return out


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
