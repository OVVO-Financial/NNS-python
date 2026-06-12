from __future__ import annotations

from typing import Any, cast

import numpy as np

from pynns import nns_seas


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
