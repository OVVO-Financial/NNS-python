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

F = NDArray[np.float64]
I = NDArray[np.int64]


def _as_float_series(values: F, dispersion: str, detrend: bool) -> F:
    x = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    if x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("Variable must be numeric, finite and non-empty")
    if dispersion not in {"auto", "absolute"}:
        raise ValueError("dispersion must be 'auto' or 'absolute'.")
    if dispersion == "absolute":
        x -= float(np.mean(x))
    n = int(x.size)
    if detrend and n >= 5:
        t = np.arange(1, n + 1, dtype=np.float64)
        _, slope = _fast_lm(t, x)
        if math.isfinite(slope) and slope != 0.0:
            center = t - np.mean(t)
            x -= slope * center
            _, slope2 = _fast_lm(t, x)
            residual = x - slope2 * center
            if np.max(np.abs(residual - np.mean(residual))) <= (
                8.0 * np.finfo(np.float64).eps * np.max(np.abs(values))
            ):
                x[:] = np.mean(x)
    return x


def _gini_component(values: F) -> float:
    """Unbiased Gini difference of observations, including tied observations."""
    n = int(values.size)
    if n < 2:
        return 0.0
    sorted_values = np.sort(values, kind="stable")
    if sorted_values[0] == sorted_values[-1]:
        return 0.0
    first = float(sorted_values[0])
    acc = 0.0
    for i, value in enumerate(sorted_values):
        acc += (2.0 * i + 1.0 - n) * (float(value) - first)
    return max(0.0, 2.0 * acc / (n * (n - 1.0)))


def _lag_score(x: F, lag: int, use_cv: bool) -> float:
    """Length-weighted component Gini dispersion / mean (when one-sided)."""
    n = x.size
    acc = 0.0
    for j in range(lag):
        values = x[j::lag]
        g = _gini_component(values)
        if use_cv and g != 0.0:
            score = g / abs(float(np.mean(values)))
        else:
            score = g
        acc += values.size * score
    return acc / n


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
        cs = np.cumsum(sorted_obs, dtype=np.float64)
        total = float(cs[-1])
        ranks = np.arange(1, k + 1, dtype=np.float64)
        contributions = sorted_obs * (2.0 * ranks - k) + total - 2.0 * cs
        sum_contributions = float(np.cumsum(contributions)[-1])
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
    weighted = float(np.sum(sizes * component_full))
    for j in range(lag):
        k = sizes[j]
        output[j::lag] = (
            weighted - k * component_full[j] + (k - 1.0) * component_del[j]
        ) / (n - 1.0)
    return output


def _fundamental(x: F, periods: I, scores: F, use_cv: bool) -> tuple[I, F]:
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
        se = math.sqrt((x.size - 1.0) / x.size * float(np.sum((diff - np.mean(diff))**2)))
        if math.isfinite(se) and scores[idx] <= scores[0] + 2.0 * se:
            order = np.r_[idx, np.arange(periods.size)[np.arange(periods.size) != idx]]
            return periods[order], scores[order]
    return periods, scores


def r14_seasonality(
    values: F,
    modulo: int | Sequence[int] | None,
    mod_only: bool,
    *,
    detrend: bool = True,
    max_period: int | None = None,
    dispersion: str = "auto",
) -> dict[str, object]:
    """R14 NNS.seas deterministic public result, except permutation p-values."""
    x = _as_float_series(values, dispersion, detrend)
    n = int(x.size)
    if max_period is not None and (max_period < 1 or not np.isfinite(max_period)):
        raise ValueError("max.period must be NULL or a single number >= 1.")

    def result(periods: I, scores: F, cov: float) -> dict[str, object]:
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
    max_lag = min(n // 2, n // 5)
    if max_period is not None:
        max_lag = max(1, min(max_lag, int(max_period)))
    base = _lag_score(x, 1, use_cv)
    scored = {lag: _lag_score(x, lag, use_cv) for lag in range(1, max_lag + 1)}
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
        for lag in added:
            if lag not in current and lag <= n // 2:
                if lag in scored:
                    score = scored[lag]
                else:
                    score = _lag_score(x, lag, use_cv)
                    if score >= base:
                        score = base
                current[lag] = score
        if not current:
            current = {1: base}
    current = {lag: score for lag, score in current.items() if lag < n // 2}
    if not current:
        current = {1: base}
    pairs = sorted(current.items(), key=lambda z: z[1])  # stable, R's order
    periods = np.asarray([lag for lag, _ in pairs], dtype=np.int64)
    scores = np.asarray([score for _, score in pairs], dtype=np.float64)
    periods, scores = _fundamental(x, periods, scores, use_cv)
    return result(periods, scores, base)
