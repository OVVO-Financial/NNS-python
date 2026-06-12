from __future__ import annotations

import math
from collections import OrderedDict
from typing import SupportsInt, cast

import numpy as np
from numpy.typing import NDArray

SeasonalityResult = dict[str, object]
_CacheKey = tuple[bytes, tuple[int, ...], bool]
_CACHE_MAX_SIZE = 32
_CACHE: OrderedDict[_CacheKey, SeasonalityResult] = OrderedDict()


def nns_seas(
    variable: NDArray[np.float64],
    modulo: int | list[int] | NDArray[np.int64] | None = None,
    mod_only: bool = True,
    plot: bool = False,
) -> SeasonalityResult:
    """Seasonality test matching R's NNS.seas non-plotting path."""
    del plot
    values = _validate_variable(variable)
    modulo_values = None if modulo is None else _as_modulo(modulo)
    cache_key = _cache_key(values, modulo_values, mod_only)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    n = values.size
    if n < 5:
        result = _result(
            np.array([0], dtype=np.int64),
            np.array([0.0], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
        )
        _cache_put(cache_key, result)
        return _clone_result(result)

    mean_var = _mean_exact(values)
    use_cv = mean_var != 0.0
    exact_cv = abs(mean_var) <= 1e-12
    var_cov = (
        abs(_sample_sd_from_mean(values, mean_var, exact=True) / mean_var)
        if use_cv
        else abs(_acf1(values)) ** -1.0
    )
    if not np.isfinite(var_cov):
        var_cov = math.inf

    periods: list[int] = []
    covs: list[float] = []
    half_n = n // 2
    variable_1 = values[:-1]
    variable_2 = variable_1[:-1]
    if use_cv:
        for period in range(1, half_n + 1):
            component = values[::-period]
            t0 = _cv_stat(component, var_cov, exact_cv)
            if t0 > var_cov:
                continue
            component = variable_1[::-period]
            t1 = _cv_stat(component, var_cov, exact_cv)
            if t1 > var_cov:
                continue
            component = variable_2[::-period]
            t2 = _cv_stat(component, var_cov, exact_cv)
            if t2 <= var_cov:
                periods.append(period)
                covs.append((t0 + t1 + t2) / 3.0)
    else:
        for period in range(1, half_n + 1):
            t0 = _cv_or_fallback(_reverse_step(values, period), use_cv, var_cov, exact_cv)
            if t0 > var_cov:
                continue
            t1 = _cv_or_fallback(_reverse_step(variable_1, period), use_cv, var_cov, exact_cv)
            if t1 > var_cov:
                continue
            t2 = _cv_or_fallback(_reverse_step(variable_2, period), use_cv, var_cov, exact_cv)
            if t2 <= var_cov:
                periods.append(period)
                covs.append((t0 + t1 + t2) / 3.0)

    if periods:
        period_arr = np.asarray(periods, dtype=np.int64)
        coef_arr = np.asarray(covs, dtype=np.float64)
        var_arr = np.full(period_arr.size, var_cov, dtype=np.float64)
        period_arr, coef_arr, var_arr = _sort_periods(period_arr, coef_arr, var_arr)
    else:
        period_arr = np.array([1], dtype=np.int64)
        coef_arr = np.array([var_cov], dtype=np.float64)
        var_arr = np.array([var_cov], dtype=np.float64)

    if modulo is not None:
        if modulo_values is None:
            raise AssertionError("modulo_values unexpectedly missing")
        period_arr, coef_arr, var_arr = _apply_modulo(
            period_arr,
            coef_arr,
            var_arr,
            modulo_values,
            mod_only=mod_only,
            var_cov=var_cov,
        )

    period_arr, coef_arr, var_arr = _strict_cap(period_arr, coef_arr, var_arr, n, var_cov)
    result = _result(period_arr, coef_arr, var_arr)
    _cache_put(cache_key, result)
    return _clone_result(result)


def _validate_variable(variable: NDArray[np.float64]) -> NDArray[np.float64]:
    try:
        values = np.asarray(variable, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Variable must be numeric") from exc
    if values.ndim != 1:
        values = values.reshape(-1)
    if values.size == 0:
        raise ValueError("Variable must be numeric and non-empty")
    if np.any(np.isnan(values)):
        raise ValueError("You have some missing values, please address.")
    if np.any(np.isinf(values)):
        raise ValueError("Infinite values not allowed")
    return values


def _sample_sd(values: NDArray[np.float64]) -> float:
    if values.size < 2:
        return math.nan
    mean = _mean(values)
    return _sample_sd_from_mean(values, mean)


def _acf1(values: NDArray[np.float64]) -> float:
    n = values.size
    if n < 2:
        return math.nan
    mean = _mean(values)
    numerator = 0.0
    denom = 0.0
    for index in range(1, n):
        numerator += (float(values[index]) - mean) * (float(values[index - 1]) - mean)
    for value in values:
        delta = float(value) - mean
        denom += delta * delta
    if denom == 0.0:
        return math.nan
    return numerator / denom


def _cv_or_fallback(
    values: NDArray[np.float64],
    use_cv: bool,
    var_cov: float,
    exact_cv: bool,
) -> float:
    if values.size < 2:
        return var_cov
    if use_cv:
        mean = _mean_exact(values) if exact_cv else _mean(values)
        sd = _sample_sd_from_mean(values, mean, exact=exact_cv)
        stat = abs(sd / mean) if mean != 0.0 else math.inf
        if (
            not exact_cv
            and np.isfinite(stat)
            and abs(stat - var_cov) <= 1e-12 * max(1.0, abs(var_cov))
        ):
            mean = _mean_exact(values)
            sd = _sample_sd_from_mean(values, mean, exact=True)
            stat = abs(sd / mean) if mean != 0.0 else math.inf
    else:
        acf = _acf1(values)
        stat = abs(acf) ** -1.0
    if not np.isfinite(stat):
        return var_cov
    return float(stat)


def _cv_stat(values: NDArray[np.float64], var_cov: float, exact_cv: bool) -> float:
    if values.size < 2:
        return var_cov
    mean = _mean_exact(values) if exact_cv else _mean(values)
    sd = _sample_sd_from_mean(values, mean, exact=exact_cv)
    stat = abs(sd / mean) if mean != 0.0 else math.inf
    if not exact_cv and np.isfinite(stat) and abs(stat - var_cov) <= 1e-12 * max(1.0, abs(var_cov)):
        mean = _mean_exact(values)
        sd = _sample_sd_from_mean(values, mean, exact=True)
        stat = abs(sd / mean) if mean != 0.0 else math.inf
    if not np.isfinite(stat):
        return var_cov
    return float(stat)


def _mean(values: NDArray[np.float64]) -> float:
    if values.size >= 16:
        mean = float(np.sum(values)) / float(values.size)
        if abs(mean) > 1e-12:
            return mean
    total = 0.0
    for value in values:
        total += float(value)
    return total / float(values.size)


def _mean_exact(values: NDArray[np.float64]) -> float:
    total = 0.0
    for value in values:
        total += float(value)
    return total / float(values.size)


def _sample_sd_from_mean(values: NDArray[np.float64], mean: float, *, exact: bool = False) -> float:
    if not exact and values.size >= 16 and abs(mean) > 1e-12:
        ss = float(np.dot(values, values)) - float(values.size) * mean * mean
        if ss < 0.0:
            ss = 0.0
    else:
        ss = 0.0
        for value in values:
            delta = float(value) - mean
            ss += delta * delta
    return math.sqrt(ss / float(values.size - 1))


def _reverse_step(values: NDArray[np.float64], step: int) -> NDArray[np.float64]:
    return values[::-step]


def _sort_periods(
    periods: NDArray[np.int64],
    coef: NDArray[np.float64],
    var_cov: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    order = np.lexsort((periods, coef))
    return periods[order], coef[order], var_cov[order]


def _as_modulo(modulo: int | list[int] | NDArray[np.int64]) -> NDArray[np.int64]:
    values = np.asarray(modulo, dtype=np.int64).reshape(-1)
    return values


def _apply_modulo(
    periods: NDArray[np.int64],
    coef: NDArray[np.float64],
    var_arr: NDArray[np.float64],
    modulo: NDArray[np.int64],
    *,
    mod_only: bool,
    var_cov: float,
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    per_set: set[int] = set()
    for period in periods:
        for mod in modulo:
            m = int(mod)
            if m <= 0:
                continue
            remainder = int(period) % m
            minus = int(period) - remainder
            plus = int(period) + (m - remainder)
            if minus > 0:
                per_set.add(minus)
            if plus > 0:
                per_set.add(plus)

    if mod_only:
        current = {int(period) for period in periods}
        out_periods: list[int] = []
        out_coef: list[float] = []
        for period, cv in zip(periods, coef, strict=True):
            if int(period) in per_set:
                out_periods.append(int(period))
                out_coef.append(float(cv))
        for period in sorted(per_set):
            if period not in current:
                out_periods.append(period)
                out_coef.append(var_cov)
        if not out_periods:
            return (
                np.array([1], dtype=np.int64),
                np.array([var_cov], dtype=np.float64),
                np.array([var_cov], dtype=np.float64),
            )
    else:
        per_set.add(1)
        current = {int(period) for period in periods}
        out_periods = [int(period) for period in periods]
        out_coef = [float(cv) for cv in coef]
        for period in sorted(per_set):
            if period not in current:
                out_periods.append(period)
                out_coef.append(var_cov)

    period_arr = np.asarray(out_periods, dtype=np.int64)
    coef_arr = np.asarray(out_coef, dtype=np.float64)
    var_out = np.full(period_arr.size, var_cov, dtype=np.float64)
    return _sort_periods(period_arr, coef_arr, var_out)


def _strict_cap(
    periods: NDArray[np.int64],
    coef: NDArray[np.float64],
    var_arr: NDArray[np.float64],
    n: int,
    var_cov: float,
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    keep = periods < (n / 2.0)
    if np.any(keep):
        return _sort_periods(periods[keep], coef[keep], var_arr[keep])
    return (
        np.array([1], dtype=np.int64),
        np.array([var_cov], dtype=np.float64),
        np.array([var_cov], dtype=np.float64),
    )


def _result(
    periods: NDArray[np.int64],
    coef: NDArray[np.float64],
    var_cov: NDArray[np.float64],
) -> SeasonalityResult:
    return {
        "all.periods": {
            "Period": periods,
            "Coefficient.of.Variation": coef,
            "Variable.Coefficient.of.Variation": var_cov,
        },
        "best.period": int(periods[0]),
        "periods": periods.copy(),
    }


def _cache_key(
    values: NDArray[np.float64],
    modulo: NDArray[np.int64] | None,
    mod_only: bool,
) -> _CacheKey:
    modulo_tuple = () if modulo is None else tuple(int(value) for value in modulo)
    contiguous = np.ascontiguousarray(values, dtype=np.float64)
    return contiguous.tobytes(), modulo_tuple, bool(mod_only)


def _cache_get(key: _CacheKey) -> SeasonalityResult | None:
    result = _CACHE.get(key)
    if result is None:
        return None
    _CACHE.move_to_end(key)
    return _clone_result(result)


def _cache_put(key: _CacheKey, result: SeasonalityResult) -> None:
    _CACHE[key] = _clone_result(result)
    _CACHE.move_to_end(key)
    while len(_CACHE) > _CACHE_MAX_SIZE:
        _CACHE.popitem(last=False)


def _clone_result(result: SeasonalityResult) -> SeasonalityResult:
    table = result["all.periods"]
    if not isinstance(table, dict):
        raise TypeError("Invalid seasonality result cache payload.")
    cloned_table = {
        "Period": np.asarray(table["Period"]).copy(),
        "Coefficient.of.Variation": np.asarray(table["Coefficient.of.Variation"]).copy(),
        "Variable.Coefficient.of.Variation": np.asarray(
            table["Variable.Coefficient.of.Variation"]
        ).copy(),
    }
    return {
        "all.periods": cloned_table,
        "best.period": int(cast(SupportsInt, result["best.period"])),
        "periods": np.asarray(result["periods"]).copy(),
    }
