from __future__ import annotations

import math
from typing import Any, cast

import numpy as np

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
