"""R 14.0 NNS.seas Gini-dispersion seasonality and fundamental period.

Behavioral oracle: OVVO-Financial/NNS@7b07a837, R/Seasonality_Test.R
and src/NNS_seas.cpp. This intentionally replaces the 13.3 standard-deviation
scoring and n/2 default scan; those are different statistical definitions.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from nns._helpers import _fast_lm
from nns._native import native_fn
from nns._rmean import r_cumsum, r_mean, r_sum
from nns._rrng import RRNG

F = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _as_float_series(values: F, dispersion: str, detrend: bool) -> F:
    x = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    if x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("Variable must be numeric, finite and non-empty")
    if dispersion not in {"auto", "absolute"}:
        raise ValueError("dispersion must be 'auto' or 'absolute'.")
    if dispersion == "absolute":
        x -= r_mean(x)
    n = int(x.size)
    if detrend and n >= 5:
        t = np.arange(1, n + 1, dtype=np.float64)
        _, slope = _fast_lm(t, x)
        if math.isfinite(slope) and slope != 0.0:
            center = t - np.mean(t)
            x -= slope * center
            _, slope2 = _fast_lm(t, x)
            residual = x - slope2 * center
            if np.max(np.abs(residual - r_mean(residual))) <= (
                8.0 * np.finfo(np.float64).eps * np.max(np.abs(values))
            ):
                x[:] = r_mean(x)
    return x


def _lag_scores(x: F, lags: Sequence[int] | IntArray, use_cv: bool) -> F:
    """R's ``seas_score`` for each lag (``NNS_seas_cpp``): the length-weighted
    mean over the lag's component series of the unbiased Gini mean difference,
    divided by the component mean when ``use_cv``.

    The native kernel is R's C++ loop itself, scored across threads. The NumPy
    fallback performs the same walk: one stable sort of the series, then per lag
    every component's terms in ascending order, summed sequentially from 0.0 as
    R's running ``acc`` and ``sum`` are.
    """
    lag_array = np.asarray(lags, dtype=np.int64)
    if lag_array.size == 0:
        return np.empty(0, dtype=np.float64)
    native = native_fn("seas_scores_r14")
    if native is not None:
        scores = native(
            np.ascontiguousarray(x, dtype=np.float64),
            np.ascontiguousarray(lag_array, dtype=np.int32),
            bool(use_cv),
        )
        return np.asarray(scores, dtype=np.float64)
    positions = np.argsort(x, kind="stable")
    ordered = np.asarray(x, dtype=np.float64)[positions]
    return np.array(
        [_lag_score_sorted(ordered, positions, int(lag), use_cv) for lag in lag_array],
        dtype=np.float64,
    )


def _lag_score_sorted(ordered: F, positions: IntArray, lag: int, use_cv: bool) -> float:
    """One lag of R's ``seas_score`` from the sorted series and its positions."""
    n = int(ordered.size)
    component = positions % lag
    lengths = (n - 1 - np.arange(lag)) // lag + 1
    # Group by component, keeping the ascending (stable) order within each one.
    key = component.astype(np.uint16) if lag <= np.iinfo(np.uint16).max else component
    grouped = np.argsort(key, kind="stable")
    owner = component[grouped]
    values = ordered[grouped]
    offsets = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    rank = np.arange(n) - offsets[owner]
    first = values[offsets]
    last = values[offsets + lengths - 1]
    terms = (2.0 * rank + 1.0 - lengths[owner].astype(np.float64)) * (values - first[owner])
    # Column 0 is R's 0.0 starting value; shorter rows end in zeros, which
    # leave a running sum that started from +0.0 unchanged.
    width = int(lengths[0]) + 1
    grid = np.zeros((lag, width), dtype=np.float64)
    grid[owner, rank + 1] = terms
    acc = np.cumsum(grid, axis=1)[:, -1]
    grid[:] = 0.0
    grid[owner, rank + 1] = values
    total = np.cumsum(grid, axis=1)[:, -1]
    k = lengths.astype(np.float64)
    varies = (lengths >= 2) & (first != last)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(varies, np.maximum(0.0, 2.0 * acc / (k * (k - 1.0))), 0.0)
        if use_cv:
            score = np.where(g == 0.0, 0.0, g / np.abs(total / k))
        else:
            score = g
    weighted = np.cumsum(np.concatenate(([0.0], k * score)))[-1]
    return float(weighted / n)


def _delete_one_scores(x: F, lag: int, use_cv: bool) -> F | None:
    """R .nns_seas_delete_one: paired, length-weighted jackknife scores."""
    n = int(x.size)
    output = np.empty(n, dtype=np.float64)
    sizes = np.array([x[j::lag].size for j in range(lag)], dtype=np.int64)
    if np.any(sizes < 3):
        return None
    component_full = np.zeros(lag, dtype=np.float64)
    component_del: list[F] = []
    # Compute each within-cell sum of |a-b| with sorted data in O(n log n).
    for j in range(lag):
        obs = np.asarray(x[j::lag], dtype=np.float64)
        k = obs.size
        order = np.argsort(obs, kind="stable")
        sorted_obs = obs[order]
        # R's cumsum() and sum() accumulate in long double.
        cs = r_cumsum(sorted_obs)
        total = float(cs[-1])
        ranks = np.arange(1, k + 1, dtype=np.float64)
        contributions = sorted_obs * (2.0 * ranks - k) + total - 2.0 * cs
        sum_contributions = r_sum(contributions)
        g = sum_contributions / (k * (k - 1.0))
        full = g / abs(total / k) if use_cv and g != 0.0 else g
        g_del = (sum_contributions - 2.0 * contributions) / ((k - 1.0) * (k - 2.0))
        if use_cv:
            denominator = np.abs((total - sorted_obs) / (k - 1.0))
            with np.errstate(divide="ignore", invalid="ignore"):
                del_values = g_del / denominator
        else:
            del_values = g_del
        del_values = np.where(np.isfinite(del_values) & (g_del != 0.0), del_values, 0.0)
        reordered = np.empty_like(del_values)
        reordered[order] = del_values
        component_full[j] = full
        component_del.append(reordered)
    weighted = r_sum(sizes * component_full)
    for j in range(lag):
        k = sizes[j]
        output[j::lag] = (weighted - k * component_full[j] + (k - 1.0) * component_del[j]) / (
            n - 1.0
        )
    return output


def _fundamental(x: F, periods: IntArray, scores: F, use_cv: bool) -> tuple[IntArray, F]:
    """R #129: smallest plausible divisor of strongest lag within 2 paired SE."""
    if periods.size < 2 or periods[0] <= 3:
        return periods, scores
    best = int(periods[0])
    denom = _delete_one_scores(x, best, use_cv)
    if denom is None:
        return periods, scores
    candidates: list[tuple[int, int]] = []
    for i, period in enumerate(periods):
        p = int(period)
        k = int(np.rint(best / p))
        if p < 2 or p >= best or k < 2:
            continue
        if best % p == 0 or (abs(best - k * p) <= (k + 1) / 2 and k + 1 < p):
            candidates.append((p, i))
    for period, idx in sorted(candidates):
        jack = _delete_one_scores(x, period, use_cv)
        if jack is None:
            continue
        diff = jack - denom
        centred = diff - r_mean(diff)
        se = math.sqrt((x.size - 1.0) / x.size * r_sum(centred * centred))
        if math.isfinite(se) and scores[idx] <= scores[0] + 2.0 * se:
            order = np.r_[idx, np.arange(periods.size)[np.arange(periods.size) != idx]]
            return periods[order], scores[order]
    return periods, scores


def r14_seasonality(
    values: F,
    modulo: int | Sequence[int] | IntArray | None,
    mod_only: bool,
    *,
    detrend: bool = True,
    max_period: int | None = None,
    dispersion: str = "auto",
    p_value: bool = False,
    nperm: int = 100,
    rng: RRNG | None = None,
) -> dict[str, object]:
    """R14 NNS.seas public result, with R's permutation p-values on request."""
    x = _as_float_series(values, dispersion, detrend)
    n = int(x.size)
    if max_period is not None and (max_period < 1 or not np.isfinite(max_period)):
        raise ValueError("max.period must be NULL or a single number >= 1.")
    if p_value and (not np.isfinite(nperm) or nperm < 1):
        raise ValueError("nperm must be a single number >= 1.")
    mp = None if max_period is None else int(max_period)

    def result(periods: IntArray, scores: F, cov: float) -> dict[str, object]:
        return {
            "all.periods": {
                "Period": periods,
                "Coefficient.of.Variation": scores,
                "Variable.Coefficient.of.Variation": np.full(periods.size, cov),
            },
            "best.period": int(periods[0]),
            "periods": periods.copy(),
        }

    if n < 5:
        return result(np.array([0], dtype=np.int64), np.array([0.0]), 0.0)

    use_cv = bool(np.min(x) > 0.0 or np.max(x) < 0.0)
    periods, scores, base = _seas_scan(x, modulo, mod_only, mp)
    periods, scores = _fundamental(x, periods, scores, use_cv)
    out = result(periods, scores, base)
    if p_value:
        if rng is None:
            raise ValueError("p_value = True needs a random stream.")
        _add_permutation_p_values(out, x, int(nperm), mp, rng)
    return out


def _add_permutation_p_values(
    out: dict[str, object], x: F, nperm: int, max_period: int | None, rng: RRNG
) -> None:
    """R NNS.seas(p.value = TRUE): each period's dispersion ratio against the best
    ratio of each of ``nperm`` shuffled copies (``sample(x)``) of the detrended
    series, scanned by ``NNS_seas_cpp`` with the observed table's out-of-scan
    periods added."""
    table = out["all.periods"]
    assert isinstance(table, dict)
    periods = np.asarray(table["Period"], dtype=np.int64)
    n = int(x.size)
    scan_max = min(n // 2, n // 5)
    if max_period is not None:
        scan_max = max(1, min(scan_max, max_period))
    extra = periods[periods > scan_max]
    null_best = np.empty(nperm, dtype=np.float64)
    for b in range(nperm):
        shuffled = rng.sample(x)
        if extra.size:
            p_null, s_null, base = _seas_scan(shuffled, extra, False, max_period)
        else:
            p_null, s_null, base = _seas_scan(shuffled, None, True, max_period)
        with np.errstate(divide="ignore", invalid="ignore"):
            null_best[b] = 1.0 if p_null[0] <= 1 else float(s_null[0] / base)
    with np.errstate(divide="ignore", invalid="ignore"):
        observed = np.asarray(table["Coefficient.of.Variation"], dtype=np.float64) / np.asarray(
            table["Variable.Coefficient.of.Variation"], dtype=np.float64
        )
    p_values = np.array(
        [(1.0 + float(np.count_nonzero(null_best <= ratio))) / (1.0 + nperm) for ratio in observed]
    )
    p_values[periods <= 1] = np.nan
    table["p.value"] = p_values
    significant = periods[~np.isnan(p_values) & (p_values <= 0.05)]
    chosen = significant.astype(np.int64) if significant.size else np.array([1], dtype=np.int64)
    out["periods"] = chosen
    out["best.period"] = int(chosen[0])


def _seas_scan(
    x: F,
    modulo: int | Sequence[int] | IntArray | None,
    mod_only: bool,
    max_period: int | None,
) -> tuple[IntArray, F, float]:
    """R's C++ ``NNS_seas_cpp``: candidate periods, their scores and the baseline."""
    n = int(x.size)
    if n < 5:
        return np.array([0], dtype=np.int64), np.array([0.0]), 0.0
    use_cv = bool(np.min(x) > 0.0 or np.max(x) < 0.0)
    max_lag = min(n // 2, n // 5)
    if max_period is not None:
        max_lag = max(1, min(max_lag, int(max_period)))
    scan = _lag_scores(x, np.arange(1, max_lag + 1), use_cv)
    base = float(scan[0])
    scored = {lag: float(score) for lag, score in zip(range(1, max_lag + 1), scan, strict=True)}
    current = {lag: score for lag, score in scored.items() if score < base}
    if not current:
        current = {1: base}

    if modulo is not None:
        mods = [int(modulo)] if isinstance(modulo, (int, np.integer)) else [int(v) for v in modulo]
        added: set[int] = set()
        for period in current:
            for m in mods:
                if m <= 0:
                    continue
                minus = period - period % m
                plus = period + (m - period % m)
                if minus > 0:
                    added.add(minus)
                if plus > 0:
                    added.add(plus)
        if mod_only:
            kept = {lag: score for lag, score in current.items() if lag in added}
            current = kept
        else:
            added.add(1)
        # R std::set iterates in ascending integer order. In a stable sort
        # over equal-VarCV injected rows, insertion order is user-visible.
        pending = [lag for lag in sorted(added) if lag not in current and lag <= n // 2]
        beyond = [lag for lag in pending if lag > max_lag]
        beyond_scores = dict(zip(beyond, _lag_scores(x, beyond, use_cv), strict=True))
        for lag in pending:
            # The scan already rejected in-range lags not in current;
            # R's added_cv gives them the baseline, not their worse score.
            current[lag] = min(float(beyond_scores[lag]), base) if lag > max_lag else base
        if not current:
            current = {1: base}
    current = {lag: score for lag, score in current.items() if lag < n // 2}
    if not current:
        current = {1: base}
    pairs = sorted(current.items(), key=lambda z: z[1])  # stable, R's order
    periods = np.asarray([lag for lag, _ in pairs], dtype=np.int64)
    scores = np.asarray([score for _, score in pairs], dtype=np.float64)
    return periods, scores, base
