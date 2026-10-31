from __future__ import annotations

import itertools
import math
from collections import OrderedDict
from typing import SupportsInt, TypedDict, cast

import numpy as np
from numpy.typing import NDArray

SeasonalityTable = TypedDict(
    "SeasonalityTable",
    {
        "Period": NDArray[np.int64],
        "Coefficient.of.Variation": NDArray[np.float64],
        "Variable.Coefficient.of.Variation": NDArray[np.float64],
    },
)
"""Per-period seasonality diagnostics ordered by strength (R's ``$all.periods``)."""

SeasonalityResult = TypedDict(
    "SeasonalityResult",
    {
        "all.periods": SeasonalityTable,
        "best.period": int,
        "periods": NDArray[np.int64],
    },
)
"""``nns_seas`` result mirroring R's NNS.seas output names."""
_CacheKey = tuple[bytes, tuple[int, ...], bool]
_CACHE_MAX_SIZE = 32
_CACHE: OrderedDict[_CacheKey, SeasonalityResult] = OrderedDict()


def nns_seas(
    variable: NDArray[np.float64],
    modulo: int | list[int] | NDArray[np.int64] | None = None,
    mod_only: bool = True,
    plot: bool = False,
) -> SeasonalityResult:
    """Seasonality test matching R's NNS.seas; ``plot=True`` renders the figure."""
    result = _nns_seas_compute(variable, modulo=modulo, mod_only=mod_only)
    if plot:
        from nns.plotting.seasonality import plot_nns_seas

        plot_nns_seas(result)
    return result


def _nns_seas_compute(
    variable: NDArray[np.float64],
    *,
    modulo: int | list[int] | NDArray[np.int64] | None = None,
    mod_only: bool = True,
) -> SeasonalityResult:
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

    mean_var = _seq_mean(values)
    use_cv = mean_var != 0.0
    if use_cv:
        with np.errstate(divide="ignore", invalid="ignore"):
            var_cov = abs(_seq_sd(values, mean_var) / mean_var)
    else:
        var_cov = _inverse_abs(_acf1(values))
    if not np.isfinite(var_cov):
        var_cov = math.inf

    # R evaluates each period on the reverse-stride subsample of the series and of
    # the series without its last one and two observations.
    candidates = np.arange(1, n // 2 + 1, dtype=np.int64)
    t0 = _subsample_stats(values, candidates, use_cv, var_cov)
    t1 = _subsample_stats(values[:-1], candidates, use_cv, var_cov)
    t2 = _subsample_stats(values[:-2], candidates, use_cv, var_cov)
    passing = (t0 <= var_cov) & (t1 <= var_cov) & (t2 <= var_cov)
    periods = candidates[passing]
    covs = (t0[passing] + t1[passing] + t2[passing]) / 3.0

    if periods.size:
        period_arr = periods
        coef_arr = covs
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


def _seq_mean(values: NDArray[np.float64]) -> float:
    # R NNS_seas.cpp::vec_mean accumulates left to right; np.cumsum is sequential
    # too (np.sum would use pairwise summation and round differently).
    return float(np.cumsum(values)[-1]) / float(values.size)


def _seq_sd(values: NDArray[np.float64], mean: float) -> float:
    delta = values - mean
    return math.sqrt(float(np.cumsum(delta * delta)[-1]) / float(values.size - 1))


def _acf1(values: NDArray[np.float64]) -> float:
    # R NNS_seas.cpp::acf1 (lag-1 autocorrelation, sequential sums); NaN if undefined.
    if values.size < 2:
        return math.nan
    delta = values - _seq_mean(values)
    numerator = float(np.cumsum(delta[1:] * delta[:-1])[-1])
    denominator = float(np.cumsum(delta * delta)[-1])
    if denominator == 0.0:
        return math.nan
    return numerator / denominator


def _inverse_abs(value: float) -> float:
    # R's std::pow(fabs(a), -1.0): Inf for a == 0 (Python's 0.0 ** -1.0 raises) and
    # NaN for NaN. libm pow is kept on purpose: 1/x differs from it in the last bit
    # for about 0.1% of inputs.
    if math.isnan(value):
        return math.nan
    return math.inf if value == 0.0 else math.pow(abs(value), -1.0)


def _subsample_stats(
    series: NDArray[np.float64],
    periods: NDArray[np.int64],
    use_cv: bool,
    var_cov: float,
) -> NDArray[np.float64]:
    """R's cv_or_fallback on series[::-p] for every period p.

    Periods whose subsamples have the same length are evaluated together as rows of
    one gathered matrix. Row sums use np.cumsum, which accumulates sequentially like
    the C++ loops, so the statistics are bit-identical to R's.
    """
    m = series.size
    out = np.full(periods.size, var_cov, dtype=np.float64)
    if periods.size == 0 or m < 2:
        return out
    lengths = (m - 1) // periods + 1
    bounds = np.concatenate(([0], np.flatnonzero(np.diff(lengths)) + 1, [periods.size]))
    with np.errstate(divide="ignore", invalid="ignore"):
        for lo, hi in itertools.pairwise(bounds.tolist()):
            length = int(lengths[lo])
            if length < 2:
                continue
            steps = np.arange(length, dtype=np.int64)
            block = series[(m - 1) - periods[lo:hi, np.newaxis] * steps]
            mean = np.cumsum(block, axis=1)[:, -1] / float(length)
            delta = block - mean[:, np.newaxis]
            sum_sq = np.cumsum(delta * delta, axis=1)[:, -1]
            if use_cv:
                stat = np.abs(np.sqrt(sum_sq / float(length - 1)) / mean)
            else:
                cross = np.cumsum(delta[:, 1:] * delta[:, :-1], axis=1)[:, -1]
                acf = np.where(sum_sq == 0.0, math.nan, cross / sum_sq)
                stat = np.asarray([_inverse_abs(float(a)) for a in acf], dtype=np.float64)
            out[lo:hi] = np.where(np.isfinite(stat), stat, var_cov)
    return out


def _sort_periods(
    periods: NDArray[np.int64],
    coef: NDArray[np.float64],
    var_cov: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    # R sorts on the CV alone (std::sort, an insertion sort for <= 16 elements, so
    # ties keep their insertion order). A stable sort reproduces that order exactly
    # for small tables and stays deterministic for larger ones.
    order = np.argsort(coef, kind="stable")
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
    # R NNS_seas.cpp compares against the C++ integer division n/2, so for odd n the
    # period floor(n/2) (two points per subsample) is excluded as well.
    keep = periods < (n // 2)
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
    cloned_table: SeasonalityTable = {
        "Period": np.asarray(table["Period"], dtype=np.int64).copy(),
        "Coefficient.of.Variation": np.asarray(
            table["Coefficient.of.Variation"], dtype=np.float64
        ).copy(),
        "Variable.Coefficient.of.Variation": np.asarray(
            table["Variable.Coefficient.of.Variation"], dtype=np.float64
        ).copy(),
    }
    return {
        "all.periods": cloned_table,
        "best.period": int(cast(SupportsInt, result["best.period"])),
        "periods": np.asarray(result["periods"], dtype=np.int64).copy(),
    }
