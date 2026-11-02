from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from _r import nns
from _tolerances import COMPOUND

from nns import nns_seas


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


_AIR_PASSENGERS = np.array(
    [
        [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118],
        [115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140],
        [145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166],
        [171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194],
        [196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201],
        [204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229],
        [242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278],
        [284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306],
        [315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336],
        [340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337],
        [360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405],
        [417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432],
    ],
    dtype=np.float64,
).reshape(-1)


_ZERO_MEAN_SINE = np.sin(2.0 * np.pi * np.arange(1, 97) / 12.0)


@pytest.mark.parity
@pytest.mark.parametrize(
    ("values", "periods"),
    [
        # Every phase of each lag counts: period 12 was ranked 15th (best 50).
        (_AIR_PASSENGERS, [12, 13, 6]),
        # Lag 2 scores (4 cv(10, 11, 12, 13) + 4 cv(18, 21, 20, 23)) / 8.
        (np.array([10.0, 18.0, 11.0, 21.0, 12.0, 20.0, 13.0, 23.0]), [2]),
        # A constant zero phase scores 0 instead of 0/0.
        (np.tile([0.0, 10.0], 10), [2, 4, 6, 8]),
        # Mean zero up to rounding: standard deviations are compared.
        (_ZERO_MEAN_SINE - np.mean(_ZERO_MEAN_SINE), None),
    ],
)
def test_nns_seas_scores_every_component_series_like_r(
    values: np.ndarray, periods: list[int] | None
) -> None:
    expected = nns("NNS.seas", values.tolist(), None, True, False)
    actual = nns_seas(values)

    _assert_seas_matches(actual, expected)
    if periods is not None:
        assert sorted(cast(np.ndarray, actual["periods"]).tolist()) == sorted(periods)


@pytest.mark.parity
@pytest.mark.parametrize(
    "values",
    [
        np.array([1.0, 2.0, 3.0] * 4 + [1.0]),
        np.random.default_rng(4).normal(10.0, 1.0, 21),
        np.random.default_rng(4).normal(10.0, 1.0, 201),
    ],
)
def test_nns_seas_odd_length_cap_matches_r(values: np.ndarray) -> None:
    # R caps periods with the C++ integer division n/2, which drops floor(n/2) for odd n.
    expected = nns("NNS.seas", values.tolist(), None, True, False)
    actual = nns_seas(values)

    assert values.size // 2 not in cast(np.ndarray, actual["periods"])
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
@pytest.mark.parametrize("mod_only", [True, False])
@pytest.mark.parametrize(
    "values",
    [
        np.tile(np.array([-1.0, 1.0]), 5),
        np.tile(np.array([1.0, 2.0, 3.0]), 6),
    ],
)
def test_nns_seas_tied_periods_keep_r_order(values: np.ndarray, mod_only: bool) -> None:
    # Periods tied on CV (e.g. modulo-added rows at VarCV) keep R's insertion order
    # instead of being re-sorted by period, so best.period agrees with R.
    modulo = [7, 12]

    expected = nns("NNS.seas", values.tolist(), modulo, mod_only, False)
    actual = nns_seas(values, modulo=modulo, mod_only=mod_only)

    _assert_seas_matches(actual, expected)


@pytest.mark.parity
@pytest.mark.parametrize(
    "values",
    [
        np.array([2.0, 0.0, 0.0, -2.0, 0.0]),
        np.array([2.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0]),
    ],
)
def test_nns_seas_zero_mean_zero_autocorrelation_matches_r(values: np.ndarray) -> None:
    # Mean exactly 0 switches to the 1/|acf1| statistic; acf1 == 0 must give Inf
    # (R's pow(0, -1)) rather than raising ZeroDivisionError.
    expected = nns("NNS.seas", values.tolist(), None, True, False)
    actual = nns_seas(values)

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
