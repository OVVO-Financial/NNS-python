from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from _r import nns
from _tolerances import COMPOUND

from pynns import nns_seas


@pytest.mark.parity
@pytest.mark.parametrize("length", [1, 2, 4])
def test_nns_seas_short_series_matches_r(length: int) -> None:
    values = np.arange(1, length + 1, dtype=np.float64)

    expected = nns("NNS.seas", values.tolist(), None, True, False)
    actual = nns_seas(values)

    _assert_seas_matches(actual, expected)


@pytest.mark.parity
@pytest.mark.parametrize(
    "values",
    [
        np.full(20, 5.0),
        np.random.default_rng(123).normal(size=50),
        np.sin(2.0 * np.pi * np.arange(1, 71, dtype=np.float64) / 7.0),
        np.sin(2.0 * np.pi * np.arange(1, 61, dtype=np.float64) / 4.0),
        0.1 * np.arange(1, 121, dtype=np.float64)
        + np.sin(2.0 * np.pi * np.arange(1, 121, dtype=np.float64) / 12.0),
        np.tile(np.array([-1.0, 1.0]), 20),
    ],
)
def test_nns_seas_series_matches_r(values: np.ndarray) -> None:
    expected = nns("NNS.seas", values.tolist(), None, True, False)
    actual = nns_seas(values)

    _assert_seas_matches(actual, expected)


@pytest.mark.parity
@pytest.mark.parametrize("mod_only", [True, False])
def test_nns_seas_modulo_matches_r(mod_only: bool) -> None:
    values = np.sin(2.0 * np.pi * np.arange(1, 61, dtype=np.float64) / 7.0)
    modulo = [2, 3, 5, 7]

    expected = nns("NNS.seas", values.tolist(), modulo, mod_only, False)
    actual = nns_seas(values, modulo=modulo, mod_only=mod_only)

    _assert_seas_matches(actual, expected)


@pytest.mark.parity
@pytest.mark.parametrize("values", [np.array([1.0, np.nan, 3.0]), np.array([1.0, np.inf, 3.0])])
def test_nns_seas_non_finite_errors(values: np.ndarray) -> None:
    with pytest.raises(ValueError):
        nns_seas(values)


def _assert_seas_matches(actual: dict[str, object], expected: Any) -> None:
    assert isinstance(expected, dict)
    assert set(actual) == set(expected)
    assert int(cast(int, actual["best.period"])) == int(np.asarray(expected["best.period"]))
    np.testing.assert_array_equal(
        cast(np.ndarray, actual["periods"]),
        _array(expected["periods"]).astype(np.int64),
    )
    actual_periods = cast(dict[str, np.ndarray], actual["all.periods"])
    assert isinstance(actual_periods, dict)
    assert isinstance(expected["all.periods"], dict)
    np.testing.assert_array_equal(
        actual_periods["Period"],
        _array(expected["all.periods"]["Period"]).astype(np.int64),
    )
    for column in ("Coefficient.of.Variation", "Variable.Coefficient.of.Variation"):
        np.testing.assert_allclose(
            actual_periods[column],
            _array(expected["all.periods"][column]),
            atol=COMPOUND,
        )


def _array(value: object) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)
