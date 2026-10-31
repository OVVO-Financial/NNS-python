from __future__ import annotations

import math
from typing import Any, cast

import numpy as np

from nns import nns_seas
from nns.seasonality import _subsample_stats


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


def test_nns_seas_constant_series_returns_zero_cv_periods() -> None:
    result = nns_seas(np.full(20, 5.0))

    np.testing.assert_array_equal(result["periods"], np.arange(1, 10))
    np.testing.assert_allclose(result["all.periods"]["Coefficient.of.Variation"], 0.0)
    np.testing.assert_allclose(result["all.periods"]["Variable.Coefficient.of.Variation"], 0.0)


def _reference_cv_or_fallback(x: np.ndarray, use_cv: bool, var_cov: float) -> float:
    # Element-by-element transcription of R NNS_seas.cpp::cv_or_fallback.
    n = x.size
    if n < 2:
        return var_cov
    total = 0.0
    for value in x:
        total += float(value)
    mean = total / n
    if use_cv:
        ss = 0.0
        for value in x:
            ss += (float(value) - mean) ** 2
        z = abs(math.sqrt(ss / (n - 1)) / mean) if mean != 0.0 else math.inf
    else:
        num = 0.0
        den = 0.0
        for t in range(1, n):
            num += (float(x[t]) - mean) * (float(x[t - 1]) - mean)
        for value in x:
            den += (float(value) - mean) ** 2
        if den == 0.0:
            return var_cov
        acf = num / den
        z = math.inf if acf == 0.0 else math.pow(abs(acf), -1.0)
    return z if math.isfinite(z) else var_cov


def test_nns_seas_subsample_stats_are_bit_identical_to_sequential_loop() -> None:
    rng = np.random.default_rng(11)
    zero_mean = rng.normal(size=301)
    series = [
        rng.normal(10.0, 1.0, 257),
        rng.normal(0.0005, 0.01, 300),
        np.cumsum(rng.normal(size=199)),
        zero_mean - zero_mean.mean(),
    ]
    for values in series:
        periods = np.arange(1, values.size // 2 + 1, dtype=np.int64)
        for use_cv in (True, False):
            var_cov = 3.0
            actual = _subsample_stats(values, periods, use_cv, var_cov)
            expected = np.array(
                [_reference_cv_or_fallback(values[::-p], use_cv, var_cov) for p in periods]
            )
            np.testing.assert_array_equal(actual, expected)
