from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pytest

from nns import nns_seas
from nns.seasonality import _SD_TOL, _component_score


def test_nns_seas_shapes_and_period_bounds() -> None:
    t = np.arange(1, 101, dtype=np.float64)
    values = np.sin(2.0 * np.pi * t / 12.0) + 0.05 * np.cos(t / 3.0)

    result = nns_seas(values)
    periods = cast(np.ndarray, result["periods"])
    table = cast(dict[str, Any], result["all.periods"])

    assert result["best.period"] == int(periods[0])
    assert table["Period"].shape == periods.shape
    assert table["Coefficient.of.Variation"].shape == periods.shape
    assert table["Variable.Coefficient.of.Variation"].shape == periods.shape
    assert np.all(periods >= 0)
    assert np.all(periods < values.size / 2.0)
    cv = table["Coefficient.of.Variation"]
    assert np.all(np.isfinite(cv) | np.isinf(cv))
    assert np.all(
        np.isfinite(table["Variable.Coefficient.of.Variation"])
        | np.isinf(table["Variable.Coefficient.of.Variation"])
    )


def test_nns_seas_short_series_zero_period_convention() -> None:
    result = nns_seas(np.array([1.0, 2.0, 3.0, 4.0]))

    assert result["best.period"] == 0
    np.testing.assert_array_equal(result["periods"], np.array([0]))
    np.testing.assert_array_equal(result["all.periods"]["Period"], np.array([0]))


def test_nns_seas_constant_series_reports_no_seasonality() -> None:
    # A lag is kept only when its CV is strictly below the series CV (0 here).
    result = nns_seas(np.full(20, 5.0))

    np.testing.assert_array_equal(result["periods"], np.array([1]))
    np.testing.assert_allclose(result["all.periods"]["Coefficient.of.Variation"], 0.0)
    np.testing.assert_allclose(result["all.periods"]["Variable.Coefficient.of.Variation"], 0.0)


def _reference_component_score(x: np.ndarray, m: int, use_cv: bool) -> float:
    # Element-by-element transcription of the R NNS_seas_cpp lag loop.
    n = x.size
    wsum = 0.0
    for j in range(m):
        length = (n - 1 - j) // m + 1
        mu = 0.0
        for i in range(j, n, m):
            mu += float(x[i])
        mu /= length
        ss = 0.0
        for i in range(j, n, m):
            d = float(x[i]) - mu
            ss += d * d
        sd = math.sqrt(ss / (length - 1))
        if use_cv:
            score = 0.0 if sd == 0.0 else sd / max(abs(mu), _SD_TOL * sd)
        else:
            score = sd
        wsum += length * score
    return wsum / n


def test_nns_seas_component_scores_are_bit_identical_to_sequential_loop() -> None:
    rng = np.random.default_rng(11)
    zero_mean = rng.normal(size=301)
    series = [
        rng.normal(10.0, 1.0, 257),
        rng.normal(0.0005, 0.01, 300),
        np.cumsum(rng.normal(size=199)),
        zero_mean - zero_mean.mean(),
        np.tile([0.0, 10.0, 0.0, 3.0], 25),
    ]
    for values in series:
        for m in range(1, values.size // 2 + 1):
            for use_cv in (True, False):
                assert _component_score(values, m, use_cv) == _reference_component_score(
                    values, m, use_cv
                )


def test_constant_series_and_components_compare_raw_values() -> None:
    # 0.1 repeated does not centre to exactly 0, so the sd was rounding noise.
    assert nns_seas(np.full(20, 0.1))["best.period"] == 1
    assert nns_seas(np.tile([0.1, 1.0], 10))["best.period"] == 2


def _r_seas_score(lag: int, ordered: np.ndarray, positions: np.ndarray, use_cv: bool) -> float:
    """R's seas_score (src/NNS_seas.cpp) transcribed line by line."""
    n = ordered.size
    rank = [0] * lag
    length = [(n - 1 - j) // lag + 1 for j in range(lag)]
    first = [0.0] * lag
    last = [0.0] * lag
    total = [0.0] * lag
    acc = [0.0] * lag
    for t in range(n):
        j = int(positions[t]) % lag
        v = float(ordered[t])
        if rank[j] == 0:
            first[j] = v
        last[j] = v
        acc[j] += (2.0 * rank[j] + 1.0 - length[j]) * (v - first[j])
        total[j] += v
        rank[j] += 1
    wsum = 0.0
    for j in range(lag):
        k = float(length[j])
        varies = length[j] >= 2 and first[j] != last[j]
        g = max(0.0, 2.0 * acc[j] / (k * (k - 1.0))) if varies else 0.0
        wsum += k * ((0.0 if g == 0.0 else g / abs(total[j] / k)) if use_cv else g)
    return wsum / n


@pytest.mark.parametrize("kind", ["continuous", "ties", "positive", "mixed"])
@pytest.mark.parametrize("n", [5, 37, 400])
def test_seasonality_lag_scores_match_r_loop_bitwise(kind: str, n: int) -> None:
    # The native scan (R's C++ itself) and the vectorised NumPy fallback both
    # reproduce R's per-lag walk to the last bit, at any thread count.
    import nns
    from nns._native import native_fn
    from nns._seas_r14 import _lag_score_sorted, _lag_scores

    rng = np.random.default_rng(n)
    x = rng.normal(size=n)
    if kind == "ties":
        x = np.round(x * 2.0)
    elif kind == "positive":
        x = np.abs(x) + 0.5
    elif kind == "mixed":
        x = np.round(rng.uniform(-3.0, 3.0, n), 1)
    use_cv = bool(x.min() > 0.0 or x.max() < 0.0)
    lags = np.arange(1, max(1, n // 5) + 1)
    positions = np.argsort(x, kind="stable")
    ordered = x[positions]
    expected = np.array([_r_seas_score(int(m), ordered, positions, use_cv) for m in lags])
    fallback = np.array([_lag_score_sorted(ordered, positions, int(m), use_cv) for m in lags])
    np.testing.assert_array_equal(fallback, expected)
    if native_fn("seas_scores_r14") is not None:
        for threads in (1, 3):
            nns.set_num_threads(threads)
            try:
                np.testing.assert_array_equal(_lag_scores(x, lags, use_cv), expected)
            finally:
                nns.set_num_threads(None)
