from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeAlias, cast
from warnings import warn

import numpy as np
from numpy.typing import NDArray

_CACHE_PATH = Path(__file__).with_name("_r_cache.json")
_LOCK_PATH = _CACHE_PATH.with_suffix(".lock")
_SCHEMA_VERSION = 1
_NNS_VERSION = "12.1"

JsonValue: TypeAlias = None | str | float | list["JsonValue"] | dict[str, "JsonValue"]
RValue: TypeAlias = (
    None | float | str | list[str | None] | NDArray[np.float64] | dict[str, "RValue"]
)
Cache: TypeAlias = dict[str, JsonValue]

_CACHE: Cache | None = None
_CACHE_REFRESH = False


def nns(function: str, *args: Any) -> RValue:
    key = _cache_key(function, args)
    cache, refresh = _cache_state()

    if key in cache:
        return _decode(cache[key])

    if _offline():
        raise RuntimeError(
            f"R cache miss for NNS::{function} with key {key}. "
            "Run without CI/NNS_R_CACHE_ONLY/PYNNS_R_CACHE_ONLY/"
            f"NNS_OFFLINE/PYNNS_OFFLINE to populate {_CACHE_PATH}."
        )

    return _uncached_nns(function, args, key, refresh)


def nns_sd_cluster_dendrogram(
    data: list[list[float]],
    degree: int,
    type: str,
    min_cluster: int,
) -> RValue:
    args = {
        "data": data,
        "degree": degree,
        "type": type,
        "min_cluster": min_cluster,
    }
    key = _cache_key("NNS.SD.cluster.dendrogram", (args,))
    cache, refresh = _cache_state()
    if key in cache:
        return _decode(cache[key])
    if _offline():
        raise RuntimeError(f"R cache miss for NNS.SD.cluster.dendrogram with key {key}.")
    with _cache_lock():
        disk_cache, disk_refresh = _read_cache_from_disk()
        if refresh or disk_refresh:
            disk_cache = {}
        if key in disk_cache:
            return _decode(disk_cache[key])
        result = _call_r_sd_cluster_dendrogram(args)
        disk_cache[key] = _encode(result)
        _write_cache(disk_cache)
        return result


def nns_stack_numeric(
    x: list[list[float]],
    y: list[float],
    x_test: list[list[float]],
    *,
    cv_size: float,
    folds: int,
    method: list[int],
    order: int | str | None,
    stack: bool,
    dim_red_method: str | list[float],
    ts_test: int | None = None,
    pred_int: float | None = None,
    type: str | None = None,
    class_levels: Sequence[object] | None = None,
    balance: bool = False,
    seed: int | None = None,
) -> RValue:
    args = {
        "x": x,
        "y": y,
        "x_test": x_test,
        "cv_size": cv_size,
        "folds": folds,
        "method": method,
        "order": order,
        "stack": stack,
        "dim_red_method": dim_red_method,
        "ts_test": ts_test,
        "pred_int": pred_int,
        "type": type,
        "class_levels": class_levels,
    }
    if balance:
        args["balance"] = True
    if seed is not None:
        args["seed"] = seed
    key = _cache_key("NNS.stack.numeric", (args,))
    cache, refresh = _cache_state()
    if key in cache:
        return _decode(cache[key])
    if _offline():
        raise RuntimeError(f"R cache miss for NNS.stack.numeric with key {key}.")
    with _cache_lock():
        disk_cache, disk_refresh = _read_cache_from_disk()
        if refresh or disk_refresh:
            disk_cache = {}
        if key in disk_cache:
            return _decode(disk_cache[key])
        result = _call_r_stack_numeric(args)
        disk_cache[key] = _encode(result)
        _write_cache(disk_cache)
        return result


def nns_boost_numeric(
    x: list[list[float]],
    y: list[float],
    x_test: list[list[float]],
    *,
    learner_trials: int,
    cv_size: float,
    depth: int | str | None,
    features_only: bool,
    pred_int: float | None = None,
    type: str | None = None,
    class_levels: Sequence[object] | None = None,
    balance: bool = False,
    ts_test: int | None = None,
    epochs: int | None = None,
    seed: int | None = None,
) -> RValue:
    args = {
        "x": x,
        "y": y,
        "x_test": x_test,
        "learner_trials": learner_trials,
        "cv_size": cv_size,
        "depth": depth,
        "features_only": features_only,
        "pred_int": pred_int,
        "type": type,
        "class_levels": class_levels,
        "ts_test": ts_test,
        "epochs": epochs,
    }
    if balance:
        args["balance"] = True
    if seed is not None:
        args["seed"] = seed
    key = _cache_key("NNS.boost.numeric", (args,))
    cache, refresh = _cache_state()
    if key in cache:
        return _decode(cache[key])
    if _offline():
        raise RuntimeError(f"R cache miss for NNS.boost.numeric with key {key}.")
    with _cache_lock():
        disk_cache, disk_refresh = _read_cache_from_disk()
        if refresh or disk_refresh:
            disk_cache = {}
        if key in disk_cache:
            return _decode(disk_cache[key])
        result = _call_r_boost_numeric(args)
        disk_cache[key] = _encode(result)
        _write_cache(disk_cache)
        return result


def nns_boost_factor_predictor(
    x_factor: list[str],
    x_numeric: list[float],
    y: list[float],
    x_test_factor: list[str],
    x_test_numeric: list[float],
    *,
    levels: Sequence[object],
    learner_trials: int,
    cv_size: float,
    depth: int | str | None,
    features_only: bool,
) -> RValue:
    args = {
        "x_factor": x_factor,
        "x_numeric": x_numeric,
        "y": y,
        "x_test_factor": x_test_factor,
        "x_test_numeric": x_test_numeric,
        "levels": levels,
        "learner_trials": learner_trials,
        "cv_size": cv_size,
        "depth": depth,
        "features_only": features_only,
    }
    key = _cache_key("NNS.boost.factor_predictor", (args,))
    cache, refresh = _cache_state()
    if key in cache:
        return _decode(cache[key])
    if _offline():
        raise RuntimeError(f"R cache miss for NNS.boost.factor_predictor with key {key}.")
    with _cache_lock():
        disk_cache, disk_refresh = _read_cache_from_disk()
        if refresh or disk_refresh:
            disk_cache = {}
        if key in disk_cache:
            return _decode(disk_cache[key])
        result = _call_r_boost_factor_predictor(args)
        disk_cache[key] = _encode(result)
        _write_cache(disk_cache)
        return result


def nns_boost_multi_factor_predictor(
    x_first: list[str],
    x_numeric: list[float],
    x_second: list[str],
    y: list[float],
    x_test_first: list[str],
    x_test_numeric: list[float],
    x_test_second: list[str],
    *,
    first_levels: Sequence[object],
    second_levels: Sequence[object],
    learner_trials: int,
    cv_size: float,
    depth: int | str | None,
    features_only: bool,
) -> RValue:
    args = {
        "x_first": x_first,
        "x_numeric": x_numeric,
        "x_second": x_second,
        "y": y,
        "x_test_first": x_test_first,
        "x_test_numeric": x_test_numeric,
        "x_test_second": x_test_second,
        "first_levels": first_levels,
        "second_levels": second_levels,
        "learner_trials": learner_trials,
        "cv_size": cv_size,
        "depth": depth,
        "features_only": features_only,
    }
    key = _cache_key("NNS.boost.multi_factor_predictor.positional.v1", (args,))
    cache, refresh = _cache_state()
    if key in cache:
        return _decode(cache[key])
    if _offline():
        raise RuntimeError(f"R cache miss for NNS.boost.multi_factor_predictor with key {key}.")
    with _cache_lock():
        disk_cache, disk_refresh = _read_cache_from_disk()
        if refresh or disk_refresh:
            disk_cache = {}
        if key in disk_cache:
            return _decode(disk_cache[key])
        result = _call_r_boost_multi_factor_predictor(args)
        disk_cache[key] = _encode(result)
        _write_cache(disk_cache)
        return result


def nns_reg_factor_predictor(
    x: list[str],
    y: list[float],
    point_est: list[str] | None,
    *,
    levels: Sequence[object],
    order: int | str | None = None,
) -> RValue:
    args = {
        "x": x,
        "y": y,
        "point_est": point_est,
        "levels": levels,
        "order": order,
    }
    key = _cache_key("NNS.reg.factor_predictor.v2", (args,))
    cache, refresh = _cache_state()
    if key in cache:
        return _decode(cache[key])
    if _offline():
        raise RuntimeError(f"R cache miss for NNS.reg.factor_predictor with key {key}.")
    with _cache_lock():
        disk_cache, disk_refresh = _read_cache_from_disk()
        if refresh or disk_refresh:
            disk_cache = {}
        if key in disk_cache:
            return _decode(disk_cache[key])
        result = _call_r_reg_factor_predictor(args)
        disk_cache[key] = _encode(result)
        _write_cache(disk_cache)
        return result


def nns_reg_factor_dimred(
    x: list[str],
    z: list[float],
    y: list[float],
    point_factor: list[str],
    point_z: list[float],
    *,
    levels: Sequence[object],
    dim_red_method: str | list[float],
) -> RValue:
    args = {
        "x": x,
        "z": z,
        "y": y,
        "point_factor": point_factor,
        "point_z": point_z,
        "levels": levels,
        "dim_red_method": dim_red_method,
    }
    key = _cache_key("NNS.reg.factor_dimred", (args,))
    cache, refresh = _cache_state()
    if key in cache:
        return _decode(cache[key])
    if _offline():
        raise RuntimeError(f"R cache miss for NNS.reg.factor_dimred with key {key}.")
    with _cache_lock():
        disk_cache, disk_refresh = _read_cache_from_disk()
        if refresh or disk_refresh:
            disk_cache = {}
        if key in disk_cache:
            return _decode(disk_cache[key])
        result = _call_r_reg_factor_dimred(args)
        disk_cache[key] = _encode(result)
        _write_cache(disk_cache)
        return result


def nns_stack_factor_predictor(
    x: list[str],
    y: list[float],
    x_test: list[str],
    *,
    levels: Sequence[object],
    cv_size: float,
    folds: int,
    method: list[int],
    order: int | str | None,
    stack: bool,
    dim_red_method: str | list[float],
) -> RValue:
    args = {
        "x": x,
        "y": y,
        "x_test": x_test,
        "levels": levels,
        "cv_size": cv_size,
        "folds": folds,
        "method": method,
        "order": order,
        "stack": stack,
        "dim_red_method": dim_red_method,
    }
    key = _cache_key("NNS.stack.factor_predictor.v2", (args,))
    cache, refresh = _cache_state()
    if key in cache:
        return _decode(cache[key])
    if _offline():
        raise RuntimeError(f"R cache miss for NNS.stack.factor_predictor with key {key}.")
    with _cache_lock():
        disk_cache, disk_refresh = _read_cache_from_disk()
        if refresh or disk_refresh:
            disk_cache = {}
        if key in disk_cache:
            return _decode(disk_cache[key])
        result = _call_r_stack_factor_predictor(args)
        disk_cache[key] = _encode(result)
        _write_cache(disk_cache)
        return result


def nns_stack_mixed_factor_predictor(
    x: list[str],
    z: list[float],
    y: list[float],
    x_test: list[str],
    z_test: list[float],
    *,
    levels: Sequence[object],
    cv_size: float,
    folds: int,
    method: list[int],
    order: int | str | None,
    stack: bool,
    dim_red_method: str | list[float],
) -> RValue:
    args = {
        "x": x,
        "z": z,
        "y": y,
        "x_test": x_test,
        "z_test": z_test,
        "levels": levels,
        "cv_size": cv_size,
        "folds": folds,
        "method": method,
        "order": order,
        "stack": stack,
        "dim_red_method": dim_red_method,
    }
    key = _cache_key("NNS.stack.mixed_factor_predictor.v1", (args,))
    cache, refresh = _cache_state()
    if key in cache:
        return _decode(cache[key])
    if _offline():
        raise RuntimeError(f"R cache miss for NNS.stack.mixed_factor_predictor with key {key}.")
    with _cache_lock():
        disk_cache, disk_refresh = _read_cache_from_disk()
        if refresh or disk_refresh:
            disk_cache = {}
        if key in disk_cache:
            return _decode(disk_cache[key])
        result = _call_r_stack_mixed_factor_predictor(args)
        disk_cache[key] = _encode(result)
        _write_cache(disk_cache)
        return result


def nns_meboot_diagnostics(
    x: list[float],
    *,
    rho: float,
    reps: int = 2,
    drift: bool = True,
    trim: float = 0.1,
    xmin: float | None = None,
    xmax: float | None = None,
    sym: bool = False,
    scl_adjustment: bool = False,
    seed: int = 1,
) -> RValue:
    args = {
        "x": x,
        "rho": rho,
        "reps": reps,
        "drift": drift,
        "trim": trim,
        "xmin": xmin,
        "xmax": xmax,
        "sym": sym,
        "scl_adjustment": scl_adjustment,
        "seed": seed,
    }
    key = _cache_key("NNS.meboot.diagnostics", (args,))
    cache, refresh = _cache_state()
    if key in cache:
        return _decode(cache[key])
    if _offline():
        raise RuntimeError(f"R cache miss for NNS.meboot.diagnostics with key {key}.")
    with _cache_lock():
        disk_cache, disk_refresh = _read_cache_from_disk()
        if refresh or disk_refresh:
            disk_cache = {}
        if key in disk_cache:
            return _decode(disk_cache[key])
        result = _call_r_meboot_diagnostics(args)
        disk_cache[key] = _encode(result)
        _write_cache(disk_cache)
        return result


def nns_meboot_stat_summary(
    x: list[float],
    *,
    rho: float,
    reps: int = 100,
    seed: int = 1,
) -> RValue:
    args = {"x": x, "rho": rho, "reps": reps, "seed": seed}
    key = _cache_key("NNS.meboot.stat_summary", (args,))
    cache, refresh = _cache_state()
    if key in cache:
        return _decode(cache[key])
    if _offline():
        raise RuntimeError(f"R cache miss for NNS.meboot.stat_summary with key {key}.")
    with _cache_lock():
        disk_cache, disk_refresh = _read_cache_from_disk()
        if refresh or disk_refresh:
            disk_cache = {}
        if key in disk_cache:
            return _decode(disk_cache[key])
        result = _call_r_meboot_stat_summary(args)
        disk_cache[key] = _encode(result)
        _write_cache(disk_cache)
        return result


def nns_mc_grid(
    *,
    lower_rho: float,
    upper_rho: float,
    by: float,
    exp: float,
) -> RValue:
    args = {"lower_rho": lower_rho, "upper_rho": upper_rho, "by": by, "exp": exp}
    key = _cache_key("NNS.MC.grid", (args,))
    cache, refresh = _cache_state()
    if key in cache:
        return _decode(cache[key])
    if _offline():
        raise RuntimeError(f"R cache miss for NNS.MC.grid with key {key}.")
    with _cache_lock():
        disk_cache, disk_refresh = _read_cache_from_disk()
        if refresh or disk_refresh:
            disk_cache = {}
        if key in disk_cache:
            return _decode(disk_cache[key])
        result = _call_r_mc_grid(args)
        disk_cache[key] = _encode(result)
        _write_cache(disk_cache)
        return result


def nns_mc_stat_summary(
    x: list[float],
    *,
    reps: int,
    lower_rho: float,
    upper_rho: float,
    by: float,
    seed: int,
) -> RValue:
    args = {
        "x": x,
        "reps": reps,
        "lower_rho": lower_rho,
        "upper_rho": upper_rho,
        "by": by,
        "seed": seed,
    }
    key = _cache_key("NNS.MC.stat_summary", (args,))
    cache, refresh = _cache_state()
    if key in cache:
        return _decode(cache[key])
    if _offline():
        raise RuntimeError(f"R cache miss for NNS.MC.stat_summary with key {key}.")
    with _cache_lock():
        disk_cache, disk_refresh = _read_cache_from_disk()
        if refresh or disk_refresh:
            disk_cache = {}
        if key in disk_cache:
            return _decode(disk_cache[key])
        result = _call_r_mc_stat_summary(args)
        disk_cache[key] = _encode(result)
        _write_cache(disk_cache)
        return result


def nns_anova_custom(payload: dict[str, Any]) -> RValue:
    key = _cache_key("NNS.ANOVA.custom", (payload,))
    cache, refresh = _cache_state()
    if key in cache:
        return _decode(cache[key])
    if _offline():
        raise RuntimeError(f"R cache miss for NNS.ANOVA.custom with key {key}.")
    with _cache_lock():
        disk_cache, disk_refresh = _read_cache_from_disk()
        if refresh or disk_refresh:
            disk_cache = {}
        if key in disk_cache:
            return _decode(disk_cache[key])
        result = _call_r_anova_custom(payload)
        disk_cache[key] = _encode(result)
        _write_cache(disk_cache)
        return result


def nns_distance_bulk_custom(
    rpm: dict[str, list[float]],
    x_test: dict[str, list[float]],
    k: int | str,
    class_: object | None = None,
) -> RValue:
    args = {"rpm": rpm, "x_test": x_test, "k": k, "class": class_}
    key = _cache_key("NNS.distance.bulk.custom", (args,))
    cache, refresh = _cache_state()
    if key in cache:
        return _decode(cache[key])
    if _offline():
        raise RuntimeError(f"R cache miss for NNS.distance.bulk.custom with key {key}.")
    with _cache_lock():
        disk_cache, disk_refresh = _read_cache_from_disk()
        if refresh or disk_refresh:
            disk_cache = {}
        if key in disk_cache:
            return _decode(disk_cache[key])
        result = _call_r_distance_bulk_custom(args)
        disk_cache[key] = _encode(result)
        _write_cache(disk_cache)
        return result


def nns_diff_custom(name: str, point: float) -> RValue:
    args = {"name": name, "point": point}
    key = _cache_key("NNS.diff.custom", (args,))
    cache, refresh = _cache_state()
    if key in cache:
        return _decode(cache[key])
    if _offline():
        raise RuntimeError(f"R cache miss for NNS.diff.custom with key {key}.")
    with _cache_lock():
        disk_cache, disk_refresh = _read_cache_from_disk()
        if refresh or disk_refresh:
            disk_cache = {}
        if key in disk_cache:
            return _decode(disk_cache[key])
        result = _call_r_diff_custom(args)
        disk_cache[key] = _encode(result)
        _write_cache(disk_cache)
        return result


def dy_dx_overall(x: Sequence[float], y: Sequence[float]) -> RValue:
    args = {"x": x, "y": y}
    key = _cache_key("dy.dx.overall", (args,))
    cache, refresh = _cache_state()
    if key in cache:
        return _decode(cache[key])
    if _offline():
        raise RuntimeError(f"R cache miss for dy.dx.overall with key {key}.")
    with _cache_lock():
        disk_cache, disk_refresh = _read_cache_from_disk()
        if refresh or disk_refresh:
            disk_cache = {}
        if key in disk_cache:
            return _decode(disk_cache[key])
        result = _call_r_dy_dx_overall(args)
        disk_cache[key] = _encode(result)
        _write_cache(disk_cache)
        return result


def factor_dummy_custom(
    values: Sequence[object],
    levels: Sequence[object],
    *,
    full_rank: bool,
) -> RValue:
    args = {"values": values, "levels": levels, "full_rank": full_rank}
    key = _cache_key("factor_2_dummy.custom", (args,))
    cache, refresh = _cache_state()
    if key in cache:
        return _decode(cache[key])
    if _offline():
        raise RuntimeError(f"R cache miss for factor_2_dummy.custom with key {key}.")
    with _cache_lock():
        disk_cache, disk_refresh = _read_cache_from_disk()
        if refresh or disk_refresh:
            disk_cache = {}
        if key in disk_cache:
            return _decode(disk_cache[key])
        result = _call_r_factor_dummy(args)
        disk_cache[key] = _encode(result)
        _write_cache(disk_cache)
        return result


def dy_dx_numeric(x: Sequence[float], y: Sequence[float], eval_point: Sequence[float]) -> RValue:
    args = {"x": list(x), "y": list(y), "eval_point": list(eval_point)}
    key = _cache_key("dy.dx.numeric", (args,))
    cache, refresh = _cache_state()
    if key in cache:
        return _decode(cache[key])
    if _offline():
        raise RuntimeError(f"R cache miss for dy.dx.numeric with key {key}.")
    with _cache_lock():
        disk_cache, disk_refresh = _read_cache_from_disk()
        if refresh or disk_refresh:
            disk_cache = {}
        if key in disk_cache:
            return _decode(disk_cache[key])
        result = _call_r_dy_dx_numeric(args)
        disk_cache[key] = _encode(result)
        _write_cache(disk_cache)
        return result


def dy_d_scalar(
    x: Sequence[Sequence[float]],
    y: Sequence[float],
    wrt: int,
    eval_points: str,
) -> RValue:
    args = {"x": x, "y": y, "wrt": wrt, "eval_points": eval_points}
    key = _cache_key("dy.d.scalar", (args,))
    cache, refresh = _cache_state()
    if key in cache:
        return _decode(cache[key])
    if _offline():
        raise RuntimeError(f"R cache miss for dy.d.scalar with key {key}.")
    with _cache_lock():
        disk_cache, disk_refresh = _read_cache_from_disk()
        if refresh or disk_refresh:
            disk_cache = {}
        if key in disk_cache:
            return _decode(disk_cache[key])
        result = _call_r_dy_d_scalar(args)
        disk_cache[key] = _encode(result)
        _write_cache(disk_cache)
        return result


def dy_d_scalar_mixed(
    x: Sequence[Sequence[float]],
    y: Sequence[float],
    wrt: int,
    eval_points: object,
) -> RValue:
    args = {"x": x, "y": y, "wrt": wrt, "eval_points": eval_points}
    key = _cache_key("dy.d.scalar.mixed.v1", (args,))
    cache, refresh = _cache_state()
    if key in cache:
        return _decode(cache[key])
    if _offline():
        raise RuntimeError(f"R cache miss for dy.d.scalar.mixed with key {key}.")
    with _cache_lock():
        disk_cache, disk_refresh = _read_cache_from_disk()
        if refresh or disk_refresh:
            disk_cache = {}
        if key in disk_cache:
            return _decode(disk_cache[key])
        result = _call_r_dy_d_scalar_mixed(args)
        disk_cache[key] = _encode(result)
        _write_cache(disk_cache)
        return result


def nns_arma_pred_int(
    variable: list[float],
    *,
    h: int,
    seasonal_factor: int | list[int] | bool,
    method: str,
    pred_int: float,
    seed: int,
) -> RValue:
    args = {
        "variable": variable,
        "h": h,
        "seasonal_factor": seasonal_factor,
        "method": method,
        "pred_int": pred_int,
        "seed": seed,
    }
    key = _cache_key("NNS.ARMA.pred_int", (args,))
    cache, refresh = _cache_state()
    if key in cache:
        return _decode(cache[key])
    if _offline():
        raise RuntimeError(f"R cache miss for NNS.ARMA.pred_int with key {key}.")
    with _cache_lock():
        disk_cache, disk_refresh = _read_cache_from_disk()
        if refresh or disk_refresh:
            disk_cache = {}
        if key in disk_cache:
            return _decode(disk_cache[key])
        result = _call_r_arma_pred_int(args)
        disk_cache[key] = _encode(result)
        _write_cache(disk_cache)
        return result


def nns_arma_optim_custom(
    variable: list[float],
    *,
    h: int | None = None,
    training_set: int | None = None,
    seasonal_factor: list[int],
    lin_only: bool = False,
    pred_int: float | None = 0.95,
) -> RValue:
    args = {
        "variable": variable,
        "h": h,
        "training_set": training_set,
        "seasonal_factor": seasonal_factor,
        "lin_only": lin_only,
        "pred_int": pred_int,
    }
    key = _cache_key("NNS.ARMA.optim.custom", (args,))
    cache, refresh = _cache_state()
    if key in cache:
        return _decode(cache[key])
    if _offline():
        raise RuntimeError(f"R cache miss for NNS.ARMA.optim.custom with key {key}.")
    with _cache_lock():
        disk_cache, disk_refresh = _read_cache_from_disk()
        if refresh or disk_refresh:
            disk_cache = {}
        if key in disk_cache:
            return _decode(disk_cache[key])
        result = _call_r_arma_optim_custom(args)
        disk_cache[key] = _encode(result)
        _write_cache(disk_cache)
        return result


def nns_cdf_custom(
    variable: list[float] | list[list[float]],
    *,
    degree: float = 0.0,
    target: float | list[float] | None = None,
    type: str = "CDF",
    names: Sequence[str] | None = None,
) -> RValue:
    args = {"variable": variable, "degree": degree, "target": target, "type": type, "names": names}
    key = _cache_key("NNS.CDF.custom", (args,))
    cache, refresh = _cache_state()
    if key in cache:
        return _decode(cache[key])
    if _offline():
        raise RuntimeError(f"R cache miss for NNS.CDF.custom with key {key}.")
    with _cache_lock():
        disk_cache, disk_refresh = _read_cache_from_disk()
        if refresh or disk_refresh:
            disk_cache = {}
        if key in disk_cache:
            return _decode(disk_cache[key])
        result = _call_r_cdf_custom(args)
        disk_cache[key] = _encode(result)
        _write_cache(disk_cache)
        return result


def _uncached_nns(
    function: str,
    args: tuple[Any, ...],
    key: str,
    refresh: bool,
) -> RValue:
    global _CACHE, _CACHE_REFRESH
    with _cache_lock():
        disk_cache, disk_refresh = _read_cache_from_disk()
        if refresh or disk_refresh:
            disk_cache = {}
            _CACHE_REFRESH = False
        if key in disk_cache:
            _CACHE = disk_cache
            return _decode(disk_cache[key])

        result = _call_r(function, args)
        disk_cache[key] = _encode(result)
        _write_cache(disk_cache)
        _CACHE = disk_cache
        return result


def _cache_key(function: str, args: tuple[Any, ...]) -> str:
    payload = json.dumps(
        {"function": function, "args": args},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_state() -> tuple[Cache, bool]:
    global _CACHE, _CACHE_REFRESH
    if _CACHE is None:
        _CACHE, _CACHE_REFRESH = _read_cache_from_disk()
    return _CACHE, _CACHE_REFRESH


def _read_cache_from_disk() -> tuple[Cache, bool]:
    if not _CACHE_PATH.exists():
        return {}, False

    cache = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    if not isinstance(cache, dict) or cache.get("schema_version") != _SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported R cache schema in {_CACHE_PATH}.")

    nns_version = cache.get("nns_version")
    if nns_version != _NNS_VERSION:
        warn(
            f"R cache was built for NNS {nns_version}; "
            f"expected {_NNS_VERSION}. Refreshing entries.",
            RuntimeWarning,
            stacklevel=2,
        )
        return {}, True

    entries = cache.get("entries")
    if not isinstance(entries, dict):
        raise RuntimeError(f"Invalid R cache entries in {_CACHE_PATH}.")
    return cast(Cache, entries), False


def _write_cache(entries: Cache) -> None:
    payload = {
        "nns_version": _NNS_VERSION,
        "schema_version": _SCHEMA_VERSION,
        "entries": entries,
    }
    tmp_path = _CACHE_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(_CACHE_PATH)


@contextmanager
def _cache_lock() -> Iterator[None]:
    _LOCK_PATH.touch(exist_ok=True)
    with _LOCK_PATH.open("r+") as lock_file:
        if os.name == "posix":
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "posix":
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _offline() -> bool:
    return (
        os.environ.get("CI") == "true"
        or os.environ.get("NNS_R_CACHE_ONLY") == "1"
        or os.environ.get("PYNNS_R_CACHE_ONLY") == "1"
        or os.environ.get("NNS_OFFLINE") == "1"
        or os.environ.get("PYNNS_OFFLINE") == "1"
    )


def _call_r(function: str, args: tuple[Any, ...]) -> RValue:
    if not function.replace(".", "").replace("_", "").isalnum():
        raise ValueError(f"Unsupported NNS function name: {function!r}")

    script = (
        "library(NNS)\n"
        "args <- jsonlite::fromJSON(paste(readLines('stdin'), collapse = '\\n'))\n"
        f"result <- do.call(getFromNamespace('{function}', 'NNS'), args)\n"
        "encode <- function(x) {\n"
        "  if (is.matrix(x)) {\n"
        "    return(unname(lapply(seq_len(nrow(x)), function(i) as.numeric(x[i, ]))))\n"
        "  }\n"
        "  if (is.list(x)) return(lapply(x, encode))\n"
        "  if (is.character(x)) return(as.character(x))\n"
        "  as.numeric(x)\n"
        "}\n"
        "cat(jsonlite::toJSON(encode(result), auto_unbox = TRUE, digits = NA))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        input=json.dumps(args),
        text=True,
    )
    return _decode(json.loads(completed.stdout))


def _call_r_stack_numeric(args: dict[str, Any]) -> RValue:
    script = (
        "library(NNS)\n"
        "args <- jsonlite::fromJSON(paste(readLines('stdin'), collapse = '\\n'), "
        "simplifyVector = FALSE)\n"
        "mat <- function(z) do.call(rbind, lapply(z, as.numeric))\n"
        "order_arg <- args$order\n"
        "if (length(order_arg) == 0) order_arg <- NULL\n"
        "dim_arg <- args$dim_red_method\n"
        "if (is.list(dim_arg)) dim_arg <- as.numeric(unlist(dim_arg))\n"
        "ts_arg <- args$ts_test\n"
        "if (length(ts_arg) == 0) ts_arg <- NULL else ts_arg <- as.integer(ts_arg)\n"
        "pred_arg <- args$pred_int\n"
        "if (length(pred_arg) == 0) pred_arg <- NULL else pred_arg <- as.numeric(pred_arg)\n"
        "type_arg <- args$type\n"
        "if (length(type_arg) == 0) type_arg <- NULL else type_arg <- as.character(type_arg)\n"
        "levels_arg <- args$class_levels\n"
        "if (length(levels_arg) == 0) levels_arg <- NULL else "
        "levels_arg <- as.character(unlist(levels_arg))\n"
        "dv <- unlist(args$y)\n"
        "if (!is.null(levels_arg)) dv <- factor(as.character(dv), levels = levels_arg) "
        "else dv <- as.numeric(dv)\n"
        "seed_arg <- args$seed\n"
        "if (length(seed_arg) != 0) set.seed(as.integer(seed_arg))\n"
        "result <- NNS::NNS.stack("
        "mat(args$x), dv, IVs.test = mat(args$x_test), "
        "CV.size = as.numeric(args$cv_size), folds = as.integer(args$folds), "
        "method = as.numeric(unlist(args$method)), order = order_arg, "
        "stack = isTRUE(as.logical(unlist(args$stack))), "
        "dim.red.method = dim_arg, pred.int = pred_arg, ts.test = ts_arg, "
        "type = type_arg, balance = isTRUE(as.logical(unlist(args$balance))), "
        "status = FALSE, ncores = 1)\n"
        "encode <- function(x) {\n"
        "  if (is.null(x)) return(NULL)\n"
        "  if (is.matrix(x)) {\n"
        "    return(unname(lapply(seq_len(nrow(x)), function(i) as.numeric(x[i, ]))))\n"
        "  }\n"
        "  if (is.list(x)) return(lapply(x, encode))\n"
        "  if (is.character(x)) return(as.character(x))\n"
        "  as.numeric(x)\n"
        "}\n"
        "cat(jsonlite::toJSON(encode(result), auto_unbox = TRUE, digits = NA, null = 'null'))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        input=json.dumps(args),
        text=True,
    )
    return _decode(json.loads(completed.stdout))


def _call_r_sd_cluster_dendrogram(args: dict[str, Any]) -> RValue:
    script = (
        "library(NNS)\n"
        "args <- jsonlite::fromJSON(paste(readLines('stdin'), collapse = '\\n'), "
        "simplifyVector = FALSE)\n"
        "mat <- do.call(rbind, lapply(args$data, as.numeric))\n"
        "result <- NNS.SD.cluster(mat, degree = as.integer(args$degree), "
        "type = as.character(args$type), min_cluster = as.integer(args$min_cluster), "
        "dendrogram = TRUE)\n"
        "if (!is.null(result$Dendrogram)) result$Dendrogram$call <- "
        "deparse(result$Dendrogram$call)\n"
        "encode <- function(x) {\n"
        "  if (is.null(x)) return(NULL)\n"
        "  if (is.matrix(x)) {\n"
        "    return(unname(lapply(seq_len(nrow(x)), function(i) as.numeric(x[i, ]))))\n"
        "  }\n"
        "  if (is.list(x)) return(lapply(x, encode))\n"
        "  if (is.character(x)) return(as.character(x))\n"
        "  as.numeric(x)\n"
        "}\n"
        "cat(jsonlite::toJSON(encode(result), auto_unbox = TRUE, digits = NA))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        input=json.dumps(args),
        text=True,
    )
    return _decode(json.loads(completed.stdout))


def _call_r_reg_factor_predictor(args: dict[str, Any]) -> RValue:
    script = (
        "library(NNS)\n"
        "args <- jsonlite::fromJSON(paste(readLines('stdin'), collapse = '\\n'), "
        "simplifyVector = FALSE)\n"
        "point_arg <- args$point_est\n"
        "if (length(point_arg) == 0) point_arg <- NULL else "
        "point_arg <- factor(unlist(point_arg), levels = unlist(args$levels))\n"
        "order_arg <- args$order\n"
        "if (length(order_arg) == 0) order_arg <- NULL\n"
        "x <- factor(unlist(args$x), levels = unlist(args$levels))\n"
        "result <- NNS.reg(x, as.numeric(unlist(args$y)), factor.2.dummy = TRUE, "
        "order = order_arg, point.est = point_arg, plot = FALSE, "
        "residual.plot = FALSE, ncores = 1)\n"
        "encode <- function(x) {\n"
        "  if (is.data.frame(x) || data.table::is.data.table(x)) {\n"
        "    col_encode <- function(nm) {\n"
        "      z <- x[[nm]]\n"
        "      if (is.character(z)) return(as.character(z))\n"
        "      as.numeric(z)\n"
        "    }\n"
        "    return(stats::setNames(lapply(names(x), col_encode), names(x)))\n"
        "  }\n"
        "  if (is.matrix(x)) {\n"
        "    return(unname(lapply(seq_len(nrow(x)), function(i) as.numeric(x[i, ]))))\n"
        "  }\n"
        "  if (is.list(x)) return(lapply(x, encode))\n"
        "  if (is.character(x)) return(as.character(x))\n"
        "  as.numeric(x)\n"
        "}\n"
        "cat(jsonlite::toJSON(encode(result), auto_unbox = TRUE, digits = NA))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        input=json.dumps(args),
        text=True,
    )
    return _decode(json.loads(completed.stdout))


def _call_r_reg_factor_dimred(args: dict[str, Any]) -> RValue:
    script = (
        "library(NNS)\n"
        "args <- jsonlite::fromJSON(paste(readLines('stdin'), collapse = '\\n'), "
        "simplifyVector = FALSE)\n"
        "dim_arg <- args$dim_red_method\n"
        "if (is.list(dim_arg)) dim_arg <- as.numeric(unlist(dim_arg))\n"
        "x <- data.frame(cat = factor(unlist(args$x), levels = unlist(args$levels)), "
        "z = as.numeric(unlist(args$z)))\n"
        "point_arg <- data.frame(cat = factor(unlist(args$point_factor), "
        "levels = unlist(args$levels)), z = as.numeric(unlist(args$point_z)))\n"
        "result <- NNS.reg(x, as.numeric(unlist(args$y)), factor.2.dummy = TRUE, "
        "dim.red.method = dim_arg, point.est = point_arg, plot = FALSE, "
        "residual.plot = FALSE, ncores = 1)\n"
        "encode <- function(x) {\n"
        "  if (is.data.frame(x) || data.table::is.data.table(x)) {\n"
        "    col_encode <- function(nm) {\n"
        "      z <- x[[nm]]\n"
        "      if (is.character(z)) return(as.character(z))\n"
        "      as.numeric(z)\n"
        "    }\n"
        "    return(stats::setNames(lapply(names(x), col_encode), names(x)))\n"
        "  }\n"
        "  if (is.matrix(x)) {\n"
        "    return(unname(lapply(seq_len(nrow(x)), function(i) as.numeric(x[i, ]))))\n"
        "  }\n"
        "  if (is.list(x)) return(lapply(x, encode))\n"
        "  if (is.character(x)) return(as.character(x))\n"
        "  as.numeric(x)\n"
        "}\n"
        "cat(jsonlite::toJSON(encode(result), auto_unbox = TRUE, digits = NA))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        input=json.dumps(args),
        text=True,
    )
    return _decode(json.loads(completed.stdout))


def _call_r_stack_factor_predictor(args: dict[str, Any]) -> RValue:
    script = (
        "library(NNS)\n"
        "args <- jsonlite::fromJSON(paste(readLines('stdin'), collapse = '\\n'), "
        "simplifyVector = FALSE)\n"
        "order_arg <- args$order\n"
        "if (length(order_arg) == 0) order_arg <- NULL\n"
        "dim_arg <- args$dim_red_method\n"
        "if (is.list(dim_arg)) dim_arg <- as.numeric(unlist(dim_arg))\n"
        "x <- data.frame(x = factor(unlist(args$x), levels = unlist(args$levels)))\n"
        "x_test <- data.frame(x = factor(unlist(args$x_test), levels = unlist(args$levels)))\n"
        "result <- NNS.stack(x, as.numeric(unlist(args$y)), IVs.test = x_test, "
        "CV.size = as.numeric(args$cv_size), folds = as.integer(args$folds), "
        "method = as.numeric(unlist(args$method)), order = order_arg, "
        "stack = as.logical(args$stack), dim.red.method = dim_arg, status = FALSE, "
        "ncores = 1)\n"
        "encode <- function(x) {\n"
        "  if (length(x) == 0) return(NULL)\n"
        "  if (is.data.frame(x) || data.table::is.data.table(x)) {\n"
        "    col_encode <- function(nm) as.numeric(x[[nm]])\n"
        "    return(stats::setNames(lapply(names(x), col_encode), names(x)))\n"
        "  }\n"
        "  if (is.matrix(x)) {\n"
        "    return(unname(lapply(seq_len(nrow(x)), function(i) as.numeric(x[i, ]))))\n"
        "  }\n"
        "  if (is.list(x)) return(lapply(x, encode))\n"
        "  if (is.character(x)) return(as.character(x))\n"
        "  as.numeric(x)\n"
        "}\n"
        "cat(jsonlite::toJSON(encode(result), auto_unbox = TRUE, digits = NA))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        input=json.dumps(args),
        text=True,
        timeout=60,
    )
    return _decode(json.loads(completed.stdout))


def _call_r_stack_mixed_factor_predictor(args: dict[str, Any]) -> RValue:
    script = (
        "library(NNS)\n"
        "args <- jsonlite::fromJSON(paste(readLines('stdin'), collapse = '\\n'), "
        "simplifyVector = FALSE)\n"
        "order_arg <- args$order\n"
        "if (length(order_arg) == 0) order_arg <- NULL\n"
        "dim_arg <- args$dim_red_method\n"
        "if (is.list(dim_arg)) dim_arg <- as.numeric(unlist(dim_arg))\n"
        "x <- data.frame("
        "x = factor(unlist(args$x), levels = unlist(args$levels)), "
        "z = as.numeric(unlist(args$z)))\n"
        "x_test <- data.frame("
        "x = factor(unlist(args$x_test), levels = unlist(args$levels)), "
        "z = as.numeric(unlist(args$z_test)))\n"
        "result <- NNS.stack(x, as.numeric(unlist(args$y)), IVs.test = x_test, "
        "CV.size = as.numeric(args$cv_size), folds = as.integer(args$folds), "
        "method = as.numeric(unlist(args$method)), order = order_arg, "
        "stack = as.logical(args$stack), dim.red.method = dim_arg, status = FALSE, "
        "ncores = 1)\n"
        "encode <- function(x) {\n"
        "  if (length(x) == 0) return(NULL)\n"
        "  if (is.data.frame(x) || data.table::is.data.table(x)) {\n"
        "    col_encode <- function(nm) as.numeric(x[[nm]])\n"
        "    return(stats::setNames(lapply(names(x), col_encode), names(x)))\n"
        "  }\n"
        "  if (is.matrix(x)) {\n"
        "    return(unname(lapply(seq_len(nrow(x)), function(i) as.numeric(x[i, ]))))\n"
        "  }\n"
        "  if (is.list(x)) return(lapply(x, encode))\n"
        "  if (is.character(x)) return(as.character(x))\n"
        "  as.numeric(x)\n"
        "}\n"
        "cat(jsonlite::toJSON(encode(result), auto_unbox = TRUE, digits = NA))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        input=json.dumps(args),
        text=True,
        timeout=60,
    )
    return _decode(json.loads(completed.stdout))


def _call_r_boost_numeric(args: dict[str, Any]) -> RValue:
    script = (
        "library(NNS)\n"
        "args <- jsonlite::fromJSON(paste(readLines('stdin'), collapse = '\\n'), "
        "simplifyVector = FALSE)\n"
        "mat <- function(z) {\n"
        "  out <- do.call(rbind, lapply(z, as.numeric))\n"
        "  colnames(out) <- paste0('X', seq_len(ncol(out)))\n"
        "  out\n"
        "}\n"
        "depth_arg <- args$depth\n"
        "if (length(depth_arg) == 0) depth_arg <- NULL\n"
        "type_arg <- args$type\n"
        "if (length(type_arg) == 0) type_arg <- NULL else type_arg <- as.character(type_arg)\n"
        "levels_arg <- args$class_levels\n"
        "if (length(levels_arg) == 0) levels_arg <- NULL else "
        "levels_arg <- as.character(unlist(levels_arg))\n"
        "dv <- unlist(args$y)\n"
        "if (!is.null(levels_arg)) dv <- factor(as.character(dv), levels = levels_arg) "
        "else dv <- as.numeric(dv)\n"
        "seed_arg <- args$seed\n"
        "if (length(seed_arg) != 0) set.seed(as.integer(seed_arg))\n"
        "result <- NNS::NNS.boost("
        "mat(args$x), dv, IVs.test = mat(args$x_test), "
        "learner.trials = as.integer(args$learner_trials), "
        "CV.size = as.numeric(args$cv_size), depth = depth_arg, "
        "type = type_arg, "
        "ts.test = if (length(args$ts_test) == 0) NULL else as.integer(args$ts_test), "
        "epochs = if (length(args$epochs) == 0) NULL else as.integer(args$epochs), "
        "pred.int = if (length(args$pred_int) == 0) NULL else as.numeric(args$pred_int), "
        "features.only = isTRUE(as.logical(unlist(args$features_only))), "
        "feature.importance = FALSE, "
        "balance = isTRUE(as.logical(unlist(args$balance))), status = FALSE)\n"
        "encode <- function(x) {\n"
        "  if (is.null(x)) return(NULL)\n"
        "  if (is.matrix(x)) {\n"
        "    return(unname(lapply(seq_len(nrow(x)), function(i) as.numeric(x[i, ]))))\n"
        "  }\n"
        "  if (is.list(x)) return(lapply(x, encode))\n"
        "  if (is.character(x)) return(as.character(x))\n"
        "  as.numeric(x)\n"
        "}\n"
        "cat(jsonlite::toJSON(encode(result), auto_unbox = TRUE, digits = NA, null = 'null'))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        input=json.dumps(args),
        text=True,
    )
    return _decode(json.loads(completed.stdout))


def _call_r_boost_factor_predictor(args: dict[str, Any]) -> RValue:
    script = (
        "library(NNS)\n"
        "args <- jsonlite::fromJSON(paste(readLines('stdin'), collapse = '\\n'), "
        "simplifyVector = FALSE)\n"
        "depth_arg <- args$depth\n"
        "if (length(depth_arg) == 0) depth_arg <- NULL\n"
        "levels_arg <- as.character(unlist(args$levels))\n"
        "train <- data.frame("
        "F = factor(as.character(unlist(args$x_factor)), levels = levels_arg), "
        "Z = as.numeric(unlist(args$x_numeric)))\n"
        "test <- data.frame("
        "F = factor(as.character(unlist(args$x_test_factor)), levels = levels_arg), "
        "Z = as.numeric(unlist(args$x_test_numeric)))\n"
        "result <- NNS::NNS.boost("
        "train, as.numeric(unlist(args$y)), IVs.test = test, "
        "learner.trials = as.integer(args$learner_trials), "
        "CV.size = as.numeric(args$cv_size), depth = depth_arg, "
        "features.only = isTRUE(as.logical(unlist(args$features_only))), "
        "feature.importance = FALSE, status = FALSE)\n"
        "encode <- function(x) {\n"
        "  if (is.null(x)) return(NULL)\n"
        "  if (is.matrix(x)) {\n"
        "    return(unname(lapply(seq_len(nrow(x)), function(i) as.numeric(x[i, ]))))\n"
        "  }\n"
        "  if (is.list(x)) return(lapply(x, encode))\n"
        "  if (is.character(x)) return(as.character(x))\n"
        "  as.numeric(x)\n"
        "}\n"
        "cat(jsonlite::toJSON(encode(result), auto_unbox = TRUE, digits = NA, null = 'null'))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        input=json.dumps(args),
        text=True,
    )
    return _decode(json.loads(completed.stdout))


def _call_r_boost_multi_factor_predictor(args: dict[str, Any]) -> RValue:
    script = (
        "library(NNS)\n"
        "args <- jsonlite::fromJSON(paste(readLines('stdin'), collapse = '\\n'), "
        "simplifyVector = FALSE)\n"
        "depth_arg <- args$depth\n"
        "if (length(depth_arg) == 0) depth_arg <- NULL\n"
        "first_levels <- as.character(unlist(args$first_levels))\n"
        "second_levels <- as.character(unlist(args$second_levels))\n"
        "train <- data.frame("
        "X1 = factor(as.character(unlist(args$x_first)), levels = first_levels), "
        "X2 = as.numeric(unlist(args$x_numeric)), "
        "X3 = factor(as.character(unlist(args$x_second)), levels = second_levels))\n"
        "test <- data.frame("
        "X1 = factor(as.character(unlist(args$x_test_first)), levels = first_levels), "
        "X2 = as.numeric(unlist(args$x_test_numeric)), "
        "X3 = factor(as.character(unlist(args$x_test_second)), levels = second_levels))\n"
        "result <- NNS::NNS.boost("
        "train, as.numeric(unlist(args$y)), IVs.test = test, "
        "learner.trials = as.integer(args$learner_trials), "
        "CV.size = as.numeric(args$cv_size), depth = depth_arg, "
        "features.only = isTRUE(as.logical(unlist(args$features_only))), "
        "feature.importance = FALSE, status = FALSE)\n"
        "encode <- function(x) {\n"
        "  if (is.null(x)) return(NULL)\n"
        "  if (is.matrix(x)) {\n"
        "    return(unname(lapply(seq_len(nrow(x)), function(i) as.numeric(x[i, ]))))\n"
        "  }\n"
        "  if (is.list(x)) return(lapply(x, encode))\n"
        "  if (is.character(x)) return(as.character(x))\n"
        "  as.numeric(x)\n"
        "}\n"
        "cat(jsonlite::toJSON(encode(result), auto_unbox = TRUE, digits = NA, null = 'null'))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        input=json.dumps(args),
        text=True,
    )
    return _decode(json.loads(completed.stdout))


def _call_r_meboot_diagnostics(args: dict[str, Any]) -> RValue:
    script = (
        "library(NNS)\n"
        "args <- jsonlite::fromJSON(paste(readLines('stdin'), collapse = '\\n'), "
        "simplifyVector = FALSE)\n"
        "f <- get('FUN', envir = environment(NNS::NNS.meboot))\n"
        "set.seed(as.integer(args$seed))\n"
        "nullify <- function(v) if (length(v) == 0) NULL else as.numeric(v)\n"
        "result <- f("
        "x = as.numeric(unlist(args$x)), reps = as.integer(args$reps), "
        "rho = as.numeric(args$rho), drift = isTRUE(as.logical(args$drift)), "
        "trim = as.numeric(args$trim), xmin = nullify(args$xmin), xmax = nullify(args$xmax), "
        "expand.sd = FALSE, force.clt = FALSE, "
        "scl.adjustment = isTRUE(as.logical(args$scl_adjustment)), "
        "sym = isTRUE(as.logical(args$sym)))\n"
        "picked <- result[c('x','xx','z','dv','dvtrim','xmin','xmax','desintxb','ordxx','kappa')]\n"
        "encode <- function(x) {\n"
        "  if (is.null(x)) return(NULL)\n"
        "  if (is.list(x)) return(lapply(x, encode))\n"
        "  as.numeric(x)\n"
        "}\n"
        "cat(jsonlite::toJSON(encode(picked), auto_unbox = TRUE, digits = NA, null = 'null'))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        input=json.dumps(args),
        text=True,
    )
    return _decode(json.loads(completed.stdout))


def _call_r_meboot_stat_summary(args: dict[str, Any]) -> RValue:
    script = (
        "library(NNS)\n"
        "args <- jsonlite::fromJSON(paste(readLines('stdin'), collapse = '\\n'), "
        "simplifyVector = FALSE)\n"
        "f <- get('FUN', envir = environment(NNS::NNS.meboot))\n"
        "set.seed(as.integer(args$seed))\n"
        "result <- f(x = as.numeric(unlist(args$x)), reps = as.integer(args$reps), "
        "rho = as.numeric(args$rho))\n"
        "replicates <- result$replicates\n"
        "summary <- c(mean_ensemble = mean(result$ensemble), sd_ensemble = sd(result$ensemble), "
        "median_rep_means = median(colMeans(replicates)), "
        "median_rep_sds = median(apply(replicates, 2, sd)))\n"
        "cat(jsonlite::toJSON(as.numeric(summary), auto_unbox = TRUE, digits = NA))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        input=json.dumps(args),
        text=True,
    )
    return _decode(json.loads(completed.stdout))


def _call_r_mc_grid(args: dict[str, Any]) -> RValue:
    script = (
        "args <- jsonlite::fromJSON(paste(readLines('stdin'), collapse = '\\n'), "
        "simplifyVector = FALSE)\n"
        "rhos <- seq(as.numeric(args$lower_rho), as.numeric(args$upper_rho), "
        "as.numeric(args$by))\n"
        "neg_rhos <- abs(rhos[rhos <= 0])\n"
        "pos_rhos <- rhos[rhos > 0]\n"
        "exp_rhos <- rev(c((neg_rhos^as.numeric(args$exp)) * -1, "
        "pos_rhos^(1/as.numeric(args$exp))))\n"
        "result <- list(values = as.numeric(exp_rhos), names = paste0('rho = ', exp_rhos))\n"
        "cat(jsonlite::toJSON(result, auto_unbox = TRUE, digits = NA))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        input=json.dumps(args),
        text=True,
    )
    return _decode(json.loads(completed.stdout))


def _call_r_mc_stat_summary(args: dict[str, Any]) -> RValue:
    script = (
        "library(NNS)\n"
        "args <- jsonlite::fromJSON(paste(readLines('stdin'), collapse = '\\n'), "
        "simplifyVector = FALSE)\n"
        "set.seed(as.integer(args$seed))\n"
        "result <- NNS::NNS.MC("
        "x = as.numeric(unlist(args$x)), reps = as.integer(args$reps), "
        "lower_rho = as.numeric(args$lower_rho), upper_rho = as.numeric(args$upper_rho), "
        "by = as.numeric(args$by))\n"
        "replicates <- result$replicates\n"
        "block_sds <- vapply(replicates, function(m) median(apply(m, 2, sd)), numeric(1))\n"
        "summary <- c(mean_ensemble = mean(result$ensemble), sd_ensemble = sd(result$ensemble), "
        "median_block_sd = median(block_sds))\n"
        "cat(jsonlite::toJSON(as.numeric(summary), auto_unbox = TRUE, digits = NA))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        input=json.dumps(args),
        text=True,
    )
    return _decode(json.loads(completed.stdout))


def _call_r_anova_custom(args: dict[str, Any]) -> RValue:
    script = (
        "library(NNS)\n"
        "args <- jsonlite::fromJSON(paste(readLines('stdin'), collapse = '\\n'), "
        "simplifyVector = FALSE)\n"
        "if (args$mode == 'binary') {\n"
        "  ci <- if (isTRUE(args$robust)) 0.95 else NULL\n"
        "  result <- NNS::NNS.ANOVA(as.numeric(unlist(args$control)), "
        "as.numeric(unlist(args$treatment)), means.only = args$means_only, "
        "medians = args$medians, confidence.interval = ci, robust = args$robust, "
        "plot = FALSE)\n"
        "} else {\n"
        "  groups <- lapply(args$groups, function(x) as.numeric(unlist(x)))\n"
        "  result <- NNS::NNS.ANOVA(groups, means.only = args$means_only, "
        "medians = args$medians, confidence.interval = NULL, "
        "pairwise = args$pairwise, plot = FALSE)\n"
        "}\n"
        "encode <- function(x) {\n"
        "  if (is.null(x)) return(NULL)\n"
        "  if (is.matrix(x)) {\n"
        "    return(unname(lapply(seq_len(nrow(x)), function(i) as.numeric(x[i, ]))))\n"
        "  }\n"
        "  if (is.list(x)) return(lapply(x, encode))\n"
        "  if (is.character(x)) return(as.character(x))\n"
        "  as.numeric(x)\n"
        "}\n"
        "cat(jsonlite::toJSON(encode(result), auto_unbox = TRUE, digits = NA, null = 'null'))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        input=json.dumps(args),
        text=True,
    )
    return _decode(json.loads(completed.stdout))


def _call_r_distance_bulk_custom(args: dict[str, Any]) -> RValue:
    script = (
        "library(NNS)\n"
        "args <- jsonlite::fromJSON(paste(readLines('stdin'), collapse = '\\n'))\n"
        "rpm <- as.data.frame(args$rpm)\n"
        "x_test <- as.data.frame(args$x_test)\n"
        "class_arg <- args[['class']]\n"
        "if (length(class_arg) == 0) class_arg <- NULL\n"
        "result <- NNS:::NNS.distance.bulk(rpm, x_test, args$k, class = class_arg)\n"
        "cat(jsonlite::toJSON(as.numeric(result), auto_unbox = TRUE, digits = NA))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        input=json.dumps(args),
        text=True,
    )
    return _decode(json.loads(completed.stdout))


def _call_r_factor_dummy(args: dict[str, Any]) -> RValue:
    script = (
        "library(NNS)\n"
        "args <- jsonlite::fromJSON(paste(readLines('stdin'), collapse = '\\n'))\n"
        "x <- factor(unlist(args$values, use.names = FALSE), "
        "levels = unlist(args$levels, use.names = FALSE))\n"
        "fn <- if (isTRUE(args$full_rank)) getFromNamespace('factor_2_dummy_FR', 'NNS') "
        "else getFromNamespace('factor_2_dummy', 'NNS')\n"
        "result <- fn(x)\n"
        "if (is.null(dim(result))) {\n"
        "  out <- list(x = as.numeric(result))\n"
        "} else {\n"
        "  out <- setNames(lapply(seq_len(ncol(result)), "
        "function(i) as.numeric(result[, i])), colnames(result))\n"
        "}\n"
        "cat(jsonlite::toJSON(out, auto_unbox = TRUE, digits = NA))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        input=json.dumps(args),
        text=True,
    )
    return _decode(json.loads(completed.stdout))


def _call_r_diff_custom(args: dict[str, Any]) -> RValue:
    script = (
        "library(NNS)\n"
        "args <- jsonlite::fromJSON(paste(readLines('stdin'), collapse = '\\n'))\n"
        "f <- switch(args$name,\n"
        "  square = function(x) x^2,\n"
        "  sin = function(x) sin(x),\n"
        "  exp = function(x) exp(x),\n"
        "  constant = function(x) 5,\n"
        "  identity = function(x) x)\n"
        "result <- NNS::NNS.diff(f, args$point, plot = FALSE)\n"
        "payload <- as.numeric(result[, 1])\n"
        "names(payload) <- rownames(result)\n"
        "cat(jsonlite::toJSON(as.list(payload), auto_unbox = TRUE, digits = NA))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        input=json.dumps(args),
        text=True,
    )
    return _decode(json.loads(completed.stdout))


def _call_r_dy_dx_overall(args: dict[str, Any]) -> RValue:
    script = (
        "library(NNS)\n"
        "args <- jsonlite::fromJSON(paste(readLines('stdin'), collapse = '\\n'))\n"
        "result <- NNS::dy.dx(as.numeric(unlist(args$x)), as.numeric(unlist(args$y)), "
        "eval.point = 'overall')\n"
        "cat(jsonlite::toJSON(as.numeric(result), auto_unbox = TRUE, digits = NA))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        input=json.dumps(args),
        text=True,
    )
    return _decode(json.loads(completed.stdout))


def _call_r_dy_dx_numeric(args: dict[str, Any]) -> RValue:
    script = (
        "library(NNS)\n"
        "args <- jsonlite::fromJSON(paste(readLines('stdin'), collapse = '\\n'), "
        "simplifyVector = FALSE)\n"
        "result <- NNS::dy.dx(as.numeric(unlist(args$x)), as.numeric(unlist(args$y)), "
        "eval.point = as.numeric(unlist(args$eval_point)))\n"
        "out <- lapply(seq_along(result), function(i) as.numeric(result[[i]]))\n"
        "names(out) <- names(result)\n"
        "cat(jsonlite::toJSON(out, auto_unbox = TRUE, digits = NA))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        input=json.dumps(args),
        text=True,
    )
    return _decode(json.loads(completed.stdout))


def _call_r_dy_d_scalar(args: dict[str, Any]) -> RValue:
    script = (
        "library(NNS)\n"
        "args <- jsonlite::fromJSON(paste(readLines('stdin'), collapse = '\\n'))\n"
        "result <- NNS::dy.d_(as.data.frame(args$x), as.numeric(unlist(args$y)), "
        "wrt = as.integer(args$wrt), eval.point = args$eval_points)\n"
        "first <- result['First', ][[1]]\n"
        "second <- result['Second', ][[1]]\n"
        "out <- list(First = as.numeric(first), Second = as.numeric(second))\n"
        "cat(jsonlite::toJSON(out, auto_unbox = TRUE, digits = NA))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        input=json.dumps(args),
        text=True,
    )
    return _decode(json.loads(completed.stdout))


def _call_r_dy_d_scalar_mixed(args: dict[str, Any]) -> RValue:
    script = (
        "library(NNS)\n"
        "args <- jsonlite::fromJSON(paste(readLines('stdin'), collapse = '\\n'))\n"
        "eval_points <- args$eval_points\n"
        "if (is.data.frame(eval_points)) eval_points <- as.matrix(eval_points)\n"
        "result <- NNS::dy.d_(as.data.frame(args$x), as.numeric(unlist(args$y)), "
        "wrt = as.integer(args$wrt), eval.point = eval_points, mixed = TRUE, "
        "messages = FALSE)\n"
        "out <- list(First = as.numeric(result['First', ][[1]]), "
        "Second = as.numeric(result['Second', ][[1]]))\n"
        "if ('Mixed' %in% rownames(result)) out$Mixed <- as.numeric(result['Mixed', ][[1]])\n"
        "cat(jsonlite::toJSON(out, auto_unbox = TRUE, digits = NA))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        input=json.dumps(args),
        text=True,
    )
    return _decode(json.loads(completed.stdout))


def _call_r_arma_pred_int(args: dict[str, Any]) -> RValue:
    script = (
        "library(NNS)\n"
        "args <- jsonlite::fromJSON(paste(readLines('stdin'), collapse = '\\n'), "
        "simplifyVector = FALSE)\n"
        "seasonal <- args$seasonal_factor\n"
        "if (is.list(seasonal)) seasonal <- as.numeric(unlist(seasonal))\n"
        "set.seed(as.integer(args$seed))\n"
        "result <- NNS::NNS.ARMA("
        "as.numeric(unlist(args$variable)), h = as.integer(args$h), "
        "seasonal.factor = seasonal, method = args$method, "
        "pred.int = as.numeric(args$pred_int), plot = FALSE, seasonal.plot = FALSE)\n"
        "encode <- function(x) {\n"
        "  if (is.null(x)) return(NULL)\n"
        "  if (is.matrix(x) || is.data.frame(x)) {\n"
        "    out <- lapply(seq_along(x), function(i) as.numeric(x[[i]]))\n"
        "    names(out) <- names(x)\n"
        "    return(out)\n"
        "  }\n"
        "  if (is.list(x)) return(lapply(x, encode))\n"
        "  if (is.character(x)) return(as.character(x))\n"
        "  as.numeric(x)\n"
        "}\n"
        "cat(jsonlite::toJSON(encode(result), auto_unbox = TRUE, digits = NA, null = 'null'))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        input=json.dumps(args),
        text=True,
    )
    return _decode(json.loads(completed.stdout))


def _call_r_arma_optim_custom(args: dict[str, Any]) -> RValue:
    script = (
        "library(NNS)\n"
        "args <- jsonlite::fromJSON(paste(readLines('stdin'), collapse = '\\n'), "
        "simplifyVector = FALSE)\n"
        "h_arg <- if (is.null(args$h)) NULL else as.integer(args$h)\n"
        "training_arg <- if (is.null(args$training_set)) NULL else as.integer(args$training_set)\n"
        "pred_arg <- if (is.null(args$pred_int)) NULL else as.numeric(args$pred_int)\n"
        "result <- NNS::NNS.ARMA.optim("
        "as.numeric(unlist(args$variable)), h = h_arg, training.set = training_arg, "
        "seasonal.factor = as.integer(unlist(args$seasonal_factor)), "
        "lin.only = isTRUE(args$lin_only), pred.int = pred_arg, ncores = 1, "
        "print.trace = FALSE, plot = FALSE)\n"
        "encode <- function(x) {\n"
        "  if (is.null(x)) return(NULL)\n"
        "  if (is.matrix(x) || is.data.frame(x)) {\n"
        "    out <- lapply(seq_along(x), function(i) as.numeric(x[[i]]))\n"
        "    names(out) <- names(x)\n"
        "    return(out)\n"
        "  }\n"
        "  if (is.list(x)) return(lapply(x, encode))\n"
        "  if (is.character(x)) return(as.character(x))\n"
        "  if (is.logical(x)) return(as.numeric(x))\n"
        "  as.numeric(x)\n"
        "}\n"
        "cat(jsonlite::toJSON(encode(result), auto_unbox = TRUE, digits = NA, null = 'null'))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        input=json.dumps(args),
        text=True,
    )
    return _decode(json.loads(completed.stdout))


def _call_r_cdf_custom(args: dict[str, Any]) -> RValue:
    script = (
        "library(NNS)\n"
        "args <- jsonlite::fromJSON(paste(readLines('stdin'), collapse = '\\n'), "
        "simplifyVector = FALSE)\n"
        "variable <- args$variable\n"
        "if (is.list(variable) && length(variable) > 0 && is.list(variable[[1]])) {\n"
        "  variable <- do.call(rbind, lapply(variable, as.numeric))\n"
        "  if (!is.null(args$names)) colnames(variable) <- unlist(args$names)\n"
        "} else {\n"
        "  variable <- as.numeric(unlist(variable))\n"
        "}\n"
        "target <- args$target\n"
        "if (is.null(target)) {\n"
        "  result <- NNS::NNS.CDF(variable, degree = as.numeric(args$degree), "
        "type = args$type, plot = FALSE)\n"
        "} else {\n"
        "  target <- as.numeric(unlist(target))\n"
        "  result <- NNS::NNS.CDF(variable, degree = as.numeric(args$degree), "
        "target = target, type = args$type, plot = FALSE)\n"
        "}\n"
        "encode <- function(x) {\n"
        "  if (is.null(x)) return(NULL)\n"
        "  if (is.matrix(x) || is.data.frame(x)) {\n"
        "    out <- lapply(seq_along(x), function(i) as.numeric(x[[i]]))\n"
        "    names(out) <- names(x)\n"
        "    return(out)\n"
        "  }\n"
        "  if (is.list(x)) return(lapply(x, encode))\n"
        "  if (is.character(x)) return(as.character(x))\n"
        "  as.numeric(x)\n"
        "}\n"
        "cat(jsonlite::toJSON(encode(result), auto_unbox = TRUE, digits = NA, null = 'null'))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        input=json.dumps(args),
        text=True,
    )
    return _decode(json.loads(completed.stdout))


def _r_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("R_LIBS_USER", str(Path.home() / "R" / "library"))
    return env


def _decode(value: JsonValue) -> RValue:
    if value is None:
        return None
    if isinstance(value, dict):
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, str):
        if value == "NA":
            return float("nan")
        if value == "NaN":
            return float("nan")
        if value == "Inf":
            return float("inf")
        if value == "-Inf":
            return float("-inf")
        return value
    if isinstance(value, list) and all(isinstance(item, str) or item is None for item in value):
        return cast(list[str | None], [None if item is None else item for item in value])
    if isinstance(value, list):
        if any(isinstance(item, list | dict) for item in value):
            return np.asarray([_decode(item) for item in value], dtype=np.float64)
        converted: list[JsonValue] = []
        has_numeric_special = False
        for item in value:
            if item == "NA":
                converted.append(float("nan"))
                has_numeric_special = True
            elif item == "Inf":
                converted.append(float("inf"))
                has_numeric_special = True
            elif item == "-Inf":
                converted.append(float("-inf"))
                has_numeric_special = True
            else:
                converted.append(item)
        if has_numeric_special:
            return np.asarray(converted, dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def _encode(value: RValue) -> JsonValue:
    if value is None:
        return None
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, str):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, list):
        return cast(JsonValue, value)
    encoded = value.tolist()
    return cast(JsonValue, encoded)
