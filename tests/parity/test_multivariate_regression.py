from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from _r import nns
from _tolerances import COMPOUND

from pynns import nns_m_reg, nns_reg
from pynns.part import NoiseReduction
from pynns.regression import Order


@pytest.mark.parity
@pytest.mark.parametrize("order", [None, 1, 2])
def test_nns_reg_multivariate_call_matches_r(order: int | None) -> None:
    x = np.linspace(-2.0, 2.0, 50)
    y = x * x + 0.1 * np.sin(np.arange(x.size))

    expected = nns(
        "NNS.reg",
        x.tolist(),
        y.tolist(),
        False,
        order,
        None,
        None,
        None,
        None,
        "top",
        True,
        False,
        False,
        True,
        None,
        0,
        None,
        False,
        "off",
        "L2",
        None,
        False,
        True,
    )
    actual = nns_reg(x, y, order=order, multivariate_call=True)
    expected_dict = cast(dict[str, Any], expected)

    assert set(actual) == set(expected_dict)
    np.testing.assert_allclose(actual["x"], _array(expected_dict["x"]), atol=COMPOUND)
    np.testing.assert_allclose(actual["y"], _array(expected_dict["y"]), atol=COMPOUND)


MREG_CASES = [
    (50, 2, "linear", None, None, None, False, "off"),
    (50, 3, "nonlinear", 1, 1, np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]), False, "off"),
    (200, 3, "mixed", 2, 2, None, False, "mean"),
    (200, 5, "linear", "max", None, None, False, "median"),
    (50, 2, "nonlinear", 1, 1, np.array([[0.0, 0.0], [3.0, 0.0]]), True, "off"),
]
MREG_CI_CASES = [
    (2, 0.8, None, None, None),
    (3, 0.95, None, 2, None),
    (2, 0.95, 1, 1, np.array([[0.0, 0.0], [3.0, 0.0]])),
    (3, 0.8, 2, 2, np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])),
]
MREG_CLASS_CASES = [
    (2, np.array([1, 1, 1, 2, 2, 2], dtype=np.float64), np.array([[1.5, 0.0], [4.5, 1.0]]), 1),
    (
        3,
        np.array([1, 1, 2, 2, 3, 3, 2, 1, 3], dtype=np.float64),
        np.array([[1.5, 0.0, 0.0], [5.5, -0.7, 0.4]]),
        2,
    ),
]


@pytest.mark.parity
@pytest.mark.parametrize(
    ("size", "n_cols", "relationship", "order", "n_best", "point_est", "point_only", "noise"),
    MREG_CASES,
)
def test_nns_m_reg_matches_r(
    rng: np.random.Generator,
    size: int,
    n_cols: int,
    relationship: str,
    order: int | str | None,
    n_best: int | str | None,
    point_est: np.ndarray | None,
    point_only: bool,
    noise: str,
) -> None:
    x, y = _dataset(size, n_cols, relationship, rng)
    if point_est is not None and point_est.shape[1] != n_cols:
        point_est = np.pad(
            point_est[:, : min(point_est.shape[1], n_cols)],
            ((0, 0), (0, n_cols - point_est.shape[1])),
        )

    expected = _r_nns_m_reg(x, y, order, n_best, point_est, point_only, noise)
    actual = nns_m_reg(
        x,
        y,
        order=cast(Order, order),
        n_best=n_best,
        point_est=point_est,
        point_only=point_only,
        noise_reduction=cast(NoiseReduction, noise),
        ncores=1,
    )

    _assert_m_reg_matches(actual, expected)


@pytest.mark.parity
@pytest.mark.parametrize(
    ("n_cols", "confidence_interval", "order", "n_best", "point_est"),
    MREG_CI_CASES,
)
def test_nns_m_reg_confidence_interval_matches_r(
    rng: np.random.Generator,
    n_cols: int,
    confidence_interval: float,
    order: int | None,
    n_best: int | None,
    point_est: np.ndarray | None,
) -> None:
    x, y = _dataset(50, n_cols, "mixed", rng)
    if point_est is not None and point_est.shape[1] != n_cols:
        point_est = np.pad(
            point_est[:, : min(point_est.shape[1], n_cols)],
            ((0, 0), (0, n_cols - point_est.shape[1])),
        )

    expected = _r_nns_m_reg(
        x,
        y,
        order,
        n_best,
        point_est,
        False,
        "off",
        confidence_interval=confidence_interval,
    )
    actual = nns_m_reg(
        x,
        y,
        order=order,
        n_best=n_best,
        point_est=point_est,
        confidence_interval=confidence_interval,
        ncores=1,
    )

    _assert_m_reg_matches(actual, expected)


@pytest.mark.parity
@pytest.mark.parametrize("n_best", [1, 2])
@pytest.mark.parametrize(("n_cols", "classes", "point_est", "order"), MREG_CLASS_CASES)
def test_nns_m_reg_classification_matches_r(
    n_cols: int,
    classes: np.ndarray,
    point_est: np.ndarray,
    order: int,
    n_best: int,
) -> None:
    x, _ = _dataset(classes.size, n_cols, "mixed", np.random.default_rng(123))

    expected = _r_nns_m_reg(
        x,
        classes,
        order,
        n_best,
        point_est,
        False,
        "off",
        type="class",
    )
    actual = nns_m_reg(
        x,
        classes,
        order=order,
        n_best=n_best,
        type="class",
        point_est=point_est,
        ncores=1,
    )

    _assert_m_reg_matches(actual, expected)


@pytest.mark.parity
@pytest.mark.parametrize("n_best", [1, 2])
@pytest.mark.parametrize(("n_cols", "classes", "point_est", "order"), MREG_CLASS_CASES)
def test_nns_m_reg_class_confidence_interval_matches_r(
    n_cols: int,
    classes: np.ndarray,
    point_est: np.ndarray,
    order: int,
    n_best: int,
) -> None:
    x, _ = _dataset(classes.size, n_cols, "mixed", np.random.default_rng(123))

    expected = _r_nns_m_reg(
        x,
        classes,
        order,
        n_best,
        point_est,
        False,
        "off",
        confidence_interval=0.95,
        type="class",
    )
    actual = nns_m_reg(
        x,
        classes,
        order=order,
        n_best=n_best,
        type="class",
        point_est=point_est,
        confidence_interval=0.95,
        ncores=1,
    )

    _assert_m_reg_matches(actual, expected)


@pytest.mark.parity
def test_nns_m_reg_factor_levels_return_numeric_codes() -> None:
    x, _ = _dataset(9, 3, "mixed", np.random.default_rng(321))
    labels = np.array(["B", "B", "A", "A", "C", "C", "A", "B", "C"])
    levels = ["A", "B", "C"]
    encoded = np.array([2, 2, 1, 1, 3, 3, 1, 2, 3], dtype=np.float64)
    point_est = x[:2]

    expected = _r_nns_m_reg(
        x,
        encoded,
        1,
        1,
        point_est,
        False,
        "off",
        type="class",
    )
    actual = nns_m_reg(
        x,
        labels,
        order=1,
        n_best=1,
        type="class",
        point_est=point_est,
        class_levels=levels,
    )

    _assert_m_reg_matches(actual, expected)


@pytest.mark.parity
def test_nns_m_reg_factor_levels_class_confidence_interval_matches_r() -> None:
    x, _ = _dataset(9, 3, "mixed", np.random.default_rng(321))
    labels = np.array(["B", "B", "A", "A", "C", "C", "A", "B", "C"])
    levels = ["A", "B", "C"]
    encoded = np.array([2, 2, 1, 1, 3, 3, 1, 2, 3], dtype=np.float64)
    point_est = x[:2]

    expected = _r_nns_m_reg(
        x,
        encoded,
        1,
        1,
        point_est,
        False,
        "off",
        confidence_interval=0.95,
        type="class",
    )
    actual = nns_m_reg(
        x,
        labels,
        order=1,
        n_best=1,
        type="class",
        point_est=point_est,
        confidence_interval=0.95,
        class_levels=levels,
    )

    _assert_m_reg_matches(actual, expected)


@pytest.mark.parity
def test_nns_reg_matrix_classification_dispatches_to_m_reg() -> None:
    x, _ = _dataset(9, 3, "mixed", np.random.default_rng(654))
    y = np.array([1, 1, 2, 2, 3, 3, 2, 1, 3], dtype=np.float64)
    point_est = np.array([[0.0, 0.0, 1.0], [1.5, 0.8, -0.2]])

    expected = _r_nns_m_reg(
        x,
        y,
        1,
        1,
        point_est,
        False,
        "mode_class",
        type="class",
    )
    actual = nns_reg(x, y, order=1, type="class", point_est=point_est)

    _assert_m_reg_matches(actual, expected)


def _r_nns_m_reg(
    x: np.ndarray,
    y: np.ndarray,
    order: int | str | None,
    n_best: int | str | None,
    point_est: np.ndarray | None,
    point_only: bool,
    noise: str,
    confidence_interval: float | None = None,
    type: str | None = None,
) -> Any:
    return nns(
        "NNS.M.reg",
        x.tolist(),
        y.tolist(),
        False,
        order,
        n_best,
        type,
        None if point_est is None else point_est.tolist(),
        point_only,
        False,
        False,
        None,
        noise,
        "L2",
        False,
        False,
        1,
        confidence_interval,
    )


def _assert_m_reg_matches(actual: dict[str, Any], expected: Any) -> None:
    assert isinstance(expected, dict)
    assert set(actual) == set(expected)
    for key in actual:
        if isinstance(actual[key], dict):
            assert isinstance(expected[key], dict)
            assert set(actual[key]) == set(expected[key])
            for column, values in actual[key].items():
                if column == "NNS.ID":
                    np.testing.assert_array_equal(
                        values.astype(str),
                        np.asarray(expected[key][column], dtype=str),
                    )
                else:
                    np.testing.assert_allclose(values, _array(expected[key][column]), atol=COMPOUND)
        elif actual[key] is None:
            assert _array(expected[key]).size == 0
        else:
            np.testing.assert_allclose(actual[key], _array(expected[key]), atol=COMPOUND)


def _dataset(
    size: int,
    n_cols: int,
    relationship: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    base = np.linspace(-2.0, 2.0, size)
    cols = [base]
    if n_cols > 1:
        cols.append(np.sin(base))
    for index in range(2, n_cols):
        cols.append(np.cos((index + 1) * base) + 0.01 * rng.normal(size=size))
    x = np.column_stack(cols)
    if relationship == "linear":
        beta = np.linspace(0.4, 1.0, n_cols)
        y = x @ beta + 0.01 * np.sin(np.arange(size))
    elif relationship == "nonlinear":
        y = x[:, 0] ** 2 + np.sin(x[:, 1])
    else:
        y = x[:, 0] + x[:, 1] ** 2 + 0.2 * x[:, -1]
    return x, y


def _array(value: object) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)
