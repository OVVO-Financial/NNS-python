from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from _r import RValue, nns, nns_arma_optim_custom, nns_arma_pred_int
from _tolerances import COMPOUND

from pynns import nns_arma, nns_arma_optim


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
