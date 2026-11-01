from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from _r import RValue, nns, nns_arma_optim_custom, nns_arma_pred_int
from _tolerances import COMPOUND

from nns import nns_arma, nns_arma_optim


@pytest.mark.parity
@pytest.mark.parametrize(
    ("name", "variable", "h", "seasonal_factor", "method", "training_set", "best_periods"),
    [
        ("known-linear", np.arange(1, 21, dtype=np.float64), 5, 4, "lin", None, 1),
        ("short-nonlin", np.arange(1, 7, dtype=np.float64), 3, 2, "nonlin", None, 1),
        ("multi-lag-linear", np.arange(1, 31, dtype=np.float64), 5, [3, 4], "lin", None, 1),
        (
            "explicit-nonlin",
            np.sin(np.arange(1, 41, dtype=np.float64) / 3.0) + 2.0,
            5,
            4,
            "nonlin",
            None,
            1,
        ),
        (
            "explicit-both",
            np.sin(np.arange(1, 41, dtype=np.float64) / 3.0) + 2.0,
            5,
            4,
            "both",
            None,
            1,
        ),
        ("means", np.arange(1, 21, dtype=np.float64), 5, 4, "means", None, 1),
        (
            "means-multi-lag",
            np.sin(np.arange(1, 61, dtype=np.float64) / 2.0) + 3.0,
            4,
            [6, 12],
            "means",
            None,
            1,
        ),
        (
            "auto-seasonal",
            np.sin(np.arange(1, 60, dtype=np.float64) / 3.0)
            + 0.1 * np.arange(1, 60, dtype=np.float64),
            5,
            True,
            "nonlin",
            None,
            1,
        ),
        (
            "all-seasonal-best2",
            np.sin(np.arange(1, 60, dtype=np.float64) / 3.0)
            + 0.1 * np.arange(1, 60, dtype=np.float64),
            5,
            False,
            "nonlin",
            None,
            2,
        ),
        (
            "training-set",
            np.sin(np.arange(1, 50, dtype=np.float64) / 3.0) + 2.0,
            5,
            4,
            "lin",
            30,
            1,
        ),
        ("constant-auto", np.full(20, 5.0), 3, True, "nonlin", None, 1),
        ("constant-explicit", np.full(20, 5.0), 3, 4, "lin", None, 1),
        ("negative-both", np.sin(np.arange(1, 31, dtype=np.float64)), 3, 5, "both", None, 1),
        (
            "negative-nonlin-sine",
            3.0 * np.sin(2.0 * np.pi * np.arange(1, 121, dtype=np.float64) / 12.0)
            + 0.3 * np.cos(np.arange(1, 121, dtype=np.float64)),
            12,
            12,
            "nonlin",
            None,
            1,
        ),
    ],
)
def test_nns_arma_matches_r(
    name: str,
    variable: np.ndarray,
    h: int,
    seasonal_factor: Any,
    method: str,
    training_set: int | None,
    best_periods: int | None,
) -> None:
    del name
    expected = nns(
        "NNS.ARMA",
        variable.tolist(),
        h,
        training_set,
        seasonal_factor,
        None,
        best_periods,
        None,
        True,
        False,
        method,
        False,
        False,
        False,
        False,
        None,
    )

    actual = nns_arma(
        variable,
        h=h,
        training_set=training_set,
        seasonal_factor=seasonal_factor,
        method=method,
        best_periods=best_periods,
    )

    np.testing.assert_allclose(actual, _array(expected), atol=COMPOUND, equal_nan=True)


@pytest.mark.parity
@pytest.mark.parametrize("method", ["lin", "nonlin", "both"])
@pytest.mark.parametrize(
    ("seasonal_factor", "weights"),
    [([6, 12], [1.0, 1.0]), ([6, 12], [3.0, 1.0]), (12, [0.5]), ([6, 12], "equal")],
)
def test_nns_arma_weights_are_relative_and_applied_once_like_r(
    method: str, seasonal_factor: Any, weights: Any
) -> None:
    # R fix 06ec89a: weights are applied once, rescaled to sum to 1, and "equal" is
    # accepted with numeric seasonal factors (c(1, 1) used to give 4x the forecast).
    variable = 10.0 + np.sin(np.arange(1, 73, dtype=np.float64) / 2.0) + 0.05 * np.arange(1, 73)
    expected = nns(
        "NNS.ARMA",
        variable.tolist(),
        3,
        None,
        seasonal_factor,
        weights,
        1,
        None,
        True,
        False,
        method,
        False,
        False,
        False,
        False,
        None,
    )
    actual = nns_arma(
        variable,
        h=3,
        seasonal_factor=seasonal_factor,
        method=method,
        weights=weights if isinstance(weights, str) else np.asarray(weights),
    )

    np.testing.assert_allclose(actual, _array(expected), atol=COMPOUND)


@pytest.mark.parametrize("h", [0, -1, 2.5, float("nan"), float("inf"), True, "3", [2, 3]])
def test_nns_arma_requires_a_positive_integer_h(h: object) -> None:
    # R fix 8aeddb8: h = 0 returned one forecast, h = 2.5 was truncated to 2 and
    # h = -1 failed with "invalid 'length' argument"; R now stops with this message.
    variable = 10.0 + np.sin(np.arange(1, 73, dtype=np.float64) / 2.0)
    with pytest.raises(ValueError, match=r"^\[h\] must be a positive integer\.$"):
        nns_arma(variable, h=h, seasonal_factor=12)  # type: ignore[arg-type]


def test_nns_arma_accepts_whole_number_h() -> None:
    variable = 10.0 + np.sin(np.arange(1, 73, dtype=np.float64) / 2.0)
    expected = nns_arma(variable, h=3, seasonal_factor=12)
    for h in (3.0, np.int64(3), np.float64(3.0)):
        np.testing.assert_array_equal(nns_arma(variable, h=h, seasonal_factor=12), expected)  # type: ignore[arg-type]


def test_nns_arma_rejects_bad_weights() -> None:
    variable = 10.0 + np.sin(np.arange(1, 73, dtype=np.float64) / 2.0)
    with pytest.raises(ValueError, match="one weight per lag"):
        nns_arma(variable, h=2, seasonal_factor=[6, 12], weights=np.array([1.0]))
    with pytest.raises(ValueError, match="must not sum to 0"):
        nns_arma(variable, h=2, seasonal_factor=[6, 12], weights=np.array([1.0, -1.0]))


_WHITE_NOISE = np.round(10.0 + np.random.default_rng(11).standard_normal(120), 6)
_TREND_SEASONAL = np.round(
    100.0
    + 2.0 * np.arange(1, 97, dtype=np.float64)
    + 15.0 * np.sin(2.0 * np.pi * np.arange(1, 97, dtype=np.float64) / 12.0),
    6,
)


@pytest.mark.parity
@pytest.mark.parametrize(
    ("variable", "h", "seasonal_factor", "best_periods", "modulo", "method", "dynamic"),
    [
        (_WHITE_NOISE, 4, True, 1, 7, "nonlin", False),
        (_WHITE_NOISE, 4, True, 1, 5, "lin", False),
        (np.full(60, 5.0), 3, True, 1, None, "nonlin", False),
        (np.full(60, 5.0), 3, 12, 1, None, "nonlin", False),
        (np.full(60, 5.0), 3, [12, 24], 1, None, "both", False),
        (np.full(60, 5.0), 3, [12, 24], 1, None, "means", False),
        (np.full(30, -2.0), 2, 7, 1, None, "nonlin", False),
        (_TREND_SEASONAL, 4, False, 2, 12, "nonlin", True),
        (_TREND_SEASONAL, 4, False, 2, None, "nonlin", True),
        (_TREND_SEASONAL, 3, True, None, None, "nonlin", False),
    ],
)
def test_nns_arma_degenerate_weights_match_r(
    variable: np.ndarray,
    h: int,
    seasonal_factor: Any,
    best_periods: int | None,
    modulo: int | None,
    method: str,
    dynamic: bool,
) -> None:
    # R fixes ed0a68c / 3b6a6cf: all-zero seasonal weights (no-seasonality sentinel,
    # modulo-added periods, constant series) and the NaN weights of a constant series
    # with a numeric seasonal_factor fall back to equal weights instead of forecasting
    # 0 or NaN; 7f941b4 / 5b82067: best period for best_periods=None and best_periods /
    # modulo when dynamic=True (Python already did both).
    expected = nns(
        "NNS.ARMA",
        variable.tolist(),
        h,
        None,
        seasonal_factor,
        None,
        best_periods,
        modulo,
        True,
        False,
        method,
        dynamic,
        False,
        False,
        False,
        None,
    )
    actual = nns_arma(
        variable,
        h=h,
        seasonal_factor=seasonal_factor,
        best_periods=best_periods,
        modulo=modulo,
        method=method,
        dynamic=dynamic,
    )

    np.testing.assert_allclose(actual, _array(expected), atol=COMPOUND)
    assert np.all(actual != 0.0)


@pytest.mark.parity
def test_nns_arma_dynamic_means_matches_r() -> None:
    variable = np.sin(np.arange(1, 40, dtype=np.float64) / 3.0) + 2.0

    expected = nns(
        "NNS.ARMA",
        variable.tolist(),
        3,
        None,
        False,
        None,
        1,
        None,
        True,
        False,
        "means",
        True,
        False,
        False,
        False,
        None,
    )
    actual = nns_arma(variable, h=3, seasonal_factor=False, method="means", dynamic=True)

    np.testing.assert_allclose(actual, _array(expected), atol=COMPOUND, equal_nan=True)


@pytest.mark.parity
@pytest.mark.stochastic
def test_nns_arma_pred_int_structure_matches_r() -> None:
    variable = np.sin(np.arange(1, 41, dtype=np.float64) / 3.0) + 2.0

    expected = _dict(
        nns_arma_pred_int(
            variable.tolist(),
            h=5,
            seasonal_factor=4,
            method="nonlin",
            pred_int=0.95,
            seed=123,
        )
    )
    actual = nns_arma(
        variable,
        h=5,
        seasonal_factor=4,
        method="nonlin",
        pred_int=0.95,
        random_seed=123,
    )
    deterministic = nns_arma(variable, h=5, seasonal_factor=4, method="nonlin")

    assert isinstance(actual, dict)
    assert set(actual) == set(expected)
    np.testing.assert_allclose(actual["Estimates"], expected["Estimates"], atol=COMPOUND)
    np.testing.assert_allclose(actual["Estimates"], deterministic, atol=COMPOUND)
    for value in actual.values():
        assert value.shape == (5,)


@pytest.mark.parity
@pytest.mark.stochastic
def test_nns_arma_pred_int_statistical_summary_is_close_to_r() -> None:
    variable = np.sin(np.arange(1, 41, dtype=np.float64) / 3.0) + 2.0

    expected = _dict(
        nns_arma_pred_int(
            variable.tolist(),
            h=5,
            seasonal_factor=[3, 4],
            method="lin",
            pred_int=0.95,
            seed=123,
        )
    )
    actual = nns_arma(
        variable,
        h=5,
        seasonal_factor=[3, 4],
        method="lin",
        pred_int=0.95,
        random_seed=123,
    )
    assert isinstance(actual, dict)

    expected_lower = expected["Lower 95% pred.int"]
    expected_upper = expected["Upper 95% pred.int"]
    actual_lower = actual["Lower 95% pred.int"]
    actual_upper = actual["Upper 95% pred.int"]
    expected_summary = np.array(
        [np.mean(expected_lower), np.mean(expected_upper), np.mean(expected_upper - expected_lower)]
    )
    actual_summary = np.array(
        [np.mean(actual_lower), np.mean(actual_upper), np.mean(actual_upper - actual_lower)]
    )

    np.testing.assert_allclose(actual_summary, expected_summary, rtol=0.6, atol=0.6)


@pytest.mark.parity
@pytest.mark.stochastic
def test_nns_arma_single_lag_lin_pred_int_matches_r() -> None:
    # R's vectorised single-lag "lin" branch returns intervals (it used to raise in Python).
    t = np.arange(1, 97, dtype=np.float64)
    variable = 100.0 + 2.0 * t + 10.0 * np.sin(2.0 * np.pi * t / 12.0) + 3.0 * np.cos(t)

    expected = _dict(
        nns_arma_pred_int(
            variable.tolist(),
            h=12,
            seasonal_factor=12,
            method="lin",
            pred_int=0.95,
            seed=123,
        )
    )
    actual = nns_arma(
        variable, h=12, seasonal_factor=12, method="lin", pred_int=0.95, random_seed=123
    )
    deterministic = nns_arma(variable, h=12, seasonal_factor=12, method="lin")

    assert isinstance(actual, dict)
    assert set(actual) == set(expected)
    np.testing.assert_allclose(actual["Estimates"], expected["Estimates"], atol=COMPOUND)
    np.testing.assert_allclose(actual["Estimates"], deterministic, atol=COMPOUND)
    lower = actual["Lower 95% pred.int"]
    upper = actual["Upper 95% pred.int"]
    assert np.all(lower <= actual["Estimates"])
    assert np.all(upper >= actual["Estimates"])
    np.testing.assert_allclose(
        [np.mean(lower), np.mean(upper)],
        [np.mean(expected["Lower 95% pred.int"]), np.mean(expected["Upper 95% pred.int"])],
        rtol=0.1,
    )


@pytest.mark.parity
@pytest.mark.parametrize(
    ("name", "h", "training_set", "lin_only"),
    [
        ("lin-only-oos", 3, None, True),
        ("default-internal", None, 32, False),
    ],
)
def test_nns_arma_optim_matches_r(
    name: str,
    h: int | None,
    training_set: int | None,
    lin_only: bool,
) -> None:
    del name
    variable = np.sin(np.arange(1, 41, dtype=np.float64) / 3.0) + 2.0
    variable = variable + 0.02 * np.arange(1, 41, dtype=np.float64)
    seasonal_factor = [3, 4, 5, 6, 7, 8]

    expected = _dict_any(
        nns_arma_optim_custom(
            variable.tolist(),
            h=h,
            training_set=training_set,
            seasonal_factor=seasonal_factor,
            lin_only=lin_only,
        )
    )
    actual = nns_arma_optim(
        variable,
        h=h,
        training_set=training_set,
        seasonal_factor=seasonal_factor,
        lin_only=lin_only,
        ncores=1,
        print_trace=False,
    )

    assert set(actual) == set(expected)
    assert actual["method"] == expected["method"]
    assert actual["weights"] is None
    assert expected["weights"] is None
    assert bool(actual["shrink"]) == bool(_array_any(expected["shrink"]))
    assert bool(actual["nns.regress"]) == bool(_array_any(expected["nns.regress"]))
    for key in [
        "periods",
        "obj.fn",
        "bias.shift",
        "errors",
        "results",
        "lower.pred.int",
        "upper.pred.int",
    ]:
        np.testing.assert_allclose(
            _array_any(actual[key]),
            _array_any(expected[key]),
            atol=COMPOUND,
            equal_nan=True,
        )


def _two_cycle_series(seed: int) -> np.ndarray:
    t = np.arange(1, 161, dtype=np.float64)
    noise = np.round(np.random.default_rng(seed).normal(0.0, 1.5, t.size), 8)
    return 50.0 + 5.0 * np.sin(2.0 * np.pi * t / 12.0) + 4.0 * np.sin(2.0 * np.pi * t / 7.0) + noise


_AIR_PASSENGERS_132 = np.array(
    [
        112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118, 115, 126, 141, 135, 125,
        149, 170, 170, 158, 133, 114, 140, 145, 150, 178, 163, 172, 178, 199, 199, 184, 162,
        146, 166, 171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194, 196, 196, 236,
        235, 229, 243, 264, 272, 237, 211, 180, 201, 204, 188, 235, 227, 234, 264, 302, 293,
        259, 229, 203, 229, 242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278, 284,
        277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306, 315, 301, 356, 348, 355, 422,
        465, 467, 404, 347, 305, 336, 340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310,
        337, 360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
    ],
    dtype=np.float64,
)


@pytest.mark.parity
@pytest.mark.parametrize(
    ("name", "variable", "h", "training_set", "seasonal_factor", "objective", "approximation"),
    [
        # 5202d54: "both" is scored with NNS.ARMA(method="both").
        ("both-scored", _AIR_PASSENGERS_132, None, 120, [6, 12, 18, 24], "min", True),
        # ce818bd: the equal-weights candidate is evaluated (and chosen here).
        ("equal-weights", _two_cycle_series(33), 10, None, [4, 6, 7, 12, 14, 24], "min", True),
        # dc5e399: the nns.regress decision is applied to the results (shrink TRUE).
        ("regress-max", _AIR_PASSENGERS_132, None, 120, [6, 12, 24], "max", True),
        # 5aae7c7: R now evaluates nonlin/both without the linear approximation.
        ("exhaustive", _AIR_PASSENGERS_132, None, 120, [6, 12, 18, 24], "min", False),
    ],
)
def test_nns_arma_optim_selection_matches_patched_r(
    name: str,
    variable: np.ndarray,
    h: int | None,
    training_set: int | None,
    seasonal_factor: list[int],
    objective: str,
    approximation: bool,
) -> None:
    del name
    expected = _dict_any(
        nns_arma_optim_custom(
            variable.tolist(),
            h=h,
            training_set=training_set,
            seasonal_factor=seasonal_factor,
            objective=objective,
            linear_approximation=approximation,
        )
    )
    actual = nns_arma_optim(
        variable,
        h=h,
        training_set=training_set,
        seasonal_factor=seasonal_factor,
        objective=objective,
        linear_approximation=approximation,
        print_trace=False,
    )

    assert actual["method"] == expected["method"]
    assert bool(actual["shrink"]) == bool(_array_any(expected["shrink"]))
    assert bool(actual["nns.regress"]) == bool(_array_any(expected["nns.regress"]))
    if expected["weights"] is None:
        assert actual["weights"] is None
    else:
        np.testing.assert_allclose(actual["weights"], _array_any(expected["weights"]))
    for key in ("periods", "obj.fn", "bias.shift", "errors", "results", "lower.pred.int"):
        np.testing.assert_allclose(
            _array_any(actual[key]), _array_any(expected[key]), atol=COMPOUND, equal_nan=True
        )


@pytest.mark.parity
def test_nns_arma_optim_without_pred_int_matches_r() -> None:
    # R 13.3 returns numeric(0) bounds for pred.int = NULL; Python used to raise TypeError.
    variable = np.sin(np.arange(1, 41, dtype=np.float64) / 3.0) + 2.0
    variable = variable + 0.02 * np.arange(1, 41, dtype=np.float64)
    seasonal_factor = [3, 4, 5, 6, 7, 8]

    expected = _dict_any(
        nns_arma_optim_custom(
            variable.tolist(), h=3, seasonal_factor=seasonal_factor, pred_int=None
        )
    )
    actual = nns_arma_optim(
        variable, h=3, seasonal_factor=seasonal_factor, pred_int=None, print_trace=False
    )

    for key in ("periods", "obj.fn", "bias.shift", "results"):
        np.testing.assert_allclose(
            _array_any(actual[key]), _array_any(expected[key]), atol=COMPOUND
        )
    for key in ("lower.pred.int", "upper.pred.int"):
        assert _array_any(actual[key]).size == 0
        assert _array_any(expected[key]).size == 0


def test_nns_arma_known_linear_check() -> None:
    result = nns_arma(np.arange(1, 21, dtype=np.float64), h=5, seasonal_factor=4, method="lin")

    np.testing.assert_allclose(result, np.array([21, 22, 23, 24, 25], dtype=np.float64))


def _array(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float64)
    if isinstance(value, list):
        return np.asarray([np.nan if item == "NaN" else item for item in value], dtype=np.float64)
    raise AssertionError(f"Unexpected R value type: {type(value)!r}")


def _dict(value: RValue) -> dict[str, np.ndarray]:
    if not isinstance(value, dict):
        raise AssertionError(f"Expected R dictionary, got {type(value)!r}")
    return {key: _array(item) for key, item in value.items()}


def _dict_any(value: RValue) -> dict[str, object]:
    if not isinstance(value, dict):
        raise AssertionError(f"Expected R dictionary, got {type(value)!r}")
    return dict(value)


def _array_any(value: object) -> np.ndarray:
    if value is None:
        raise AssertionError("Unexpected None value.")
    if isinstance(value, np.ndarray):
        return value.astype(np.float64)
    if isinstance(value, (float, int, np.floating, np.integer, bool, np.bool_)):
        return np.asarray(value, dtype=np.float64)
    if isinstance(value, list):
        return np.asarray(value, dtype=np.float64)
    raise AssertionError(f"Unexpected value type: {type(value)!r}")
