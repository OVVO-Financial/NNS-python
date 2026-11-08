from __future__ import annotations

import math
from collections import OrderedDict
from typing import NotRequired, SupportsInt, TypedDict, cast

import numpy as np
from numpy.typing import NDArray

SeasonalityTable = TypedDict(
    "SeasonalityTable",
    {
        "Period": NDArray[np.int64],
        "Coefficient.of.Variation": NDArray[np.float64],
        "Variable.Coefficient.of.Variation": NDArray[np.float64],
        "p.value": NotRequired[NDArray[np.float64]],
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
_SD_TOL = math.sqrt(np.finfo(np.float64).eps)


def nns_seas(
    variable: NDArray[np.float64],
    modulo: int | list[int] | NDArray[np.int64] | None = None,
    mod_only: bool = True,
    plot: bool = False,
    *,
    detrend: bool = True,
    max_period: int | None = None,
    p_value: bool = False,
    nperm: int = 100,
    dispersion: str = "auto",
    random_seed: int | None = None,
) -> SeasonalityResult:
    """R 14.0 Gini-dispersion seasonal candidates and fundamental period.

    The R14 estimator changed its statistics (Gini rather than SD), trend
    adjustment and scan cap; it must not reuse the earlier R13 cached statistic.

    ``p_value=True`` adds R's permutation p-value of each period to
    ``all.periods`` (``"p.value"``, ``NaN`` for period 1): its dispersion ratio
    against the best ratio of each of ``nperm`` shuffled copies of the
    (detrended) series. ``periods`` then holds the periods with p-value at most
    0.05 (else 1) and ``best.period`` the first of them. With
    ``random_seed = s`` the shuffles are R's after ``set.seed(s)``.
    """
    from nns._rrng import r_rng
    from nns._seas_r14 import r14_seasonality

    result = cast(
        SeasonalityResult,
        r14_seasonality(
            variable, modulo, mod_only,
            detrend=detrend, max_period=max_period, dispersion=dispersion,
            p_value=p_value, nperm=nperm, rng=r_rng(random_seed) if p_value else None,
        ),
    )
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
    # Compare raw values: a repeated 0.1 need not centre to exactly 0.
    sd_var = 0.0 if bool(np.all(values == values[0])) else _seq_sd(values, mean_var)
    # CV is meaningless for a mean that is zero up to rounding; compare sds instead.
    use_cv = abs(mean_var) > _SD_TOL * sd_var
    var_cov = abs(sd_var / mean_var) if use_cv else sd_var

    candidates = np.arange(1, n // 2 + 1, dtype=np.int64)
    scores = np.array([_component_score(values, int(m), use_cv) for m in candidates])
    passing = scores < var_cov
    periods = candidates[passing]
    covs = scores[passing]

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


def _component_score(values: NDArray[np.float64], m: int, use_cv: bool) -> float:
    """Length-weighted mean CV (or sd) of the m component series ``values[j::m]``.

    As R's NNS_seas_cpp: a constant component (equal raw values) scores 0 and a near-zero component
    mean caps its CV at ``1 / _SD_TOL``. Column j of the padded matrix is component
    j; adding the rows one at a time keeps each column's sums in the C++ order, so
    the scores are bit-identical to R's. Only the last row has padding.
    """
    n = values.size
    width = -(-n // m)
    padded = np.zeros(m * width, dtype=np.float64)
    padded[:n] = values
    block = padded.reshape(width, m)
    lengths = ((n - 1 - np.arange(m)) // m + 1).astype(np.float64)
    total = np.zeros(m, dtype=np.float64)
    for row in block:
        total += row
    mean = total / lengths
    sum_sq = np.zeros(m, dtype=np.float64)
    same = np.ones(m, dtype=bool)
    for row in block[:-1]:
        sum_sq += (row - mean) ** 2
        same &= row == block[0]
    tail = lengths == width
    sum_sq[tail] += (block[-1, tail] - mean[tail]) ** 2
    same[tail] &= block[-1, tail] == block[0, tail]
    sd = np.where(same, 0.0, np.sqrt(sum_sq / (lengths - 1.0)))
    if use_cv:
        with np.errstate(invalid="ignore", divide="ignore"):
            score = np.where(sd == 0.0, 0.0, sd / np.maximum(np.abs(mean), _SD_TOL * sd))
    else:
        score = sd
    return float(np.cumsum(lengths * score)[-1]) / float(n)


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
