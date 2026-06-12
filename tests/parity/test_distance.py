from __future__ import annotations

from typing import Literal

import numpy as np
import pytest
from _r import nns, nns_distance_bulk_custom
from _tolerances import EXACT

from nns import nns_distance, nns_distance_bulk


@pytest.mark.parity
@pytest.mark.parametrize("k", [1, 2, 3, "all"])
def test_nns_distance_matches_r(k: int | Literal["all"]) -> None:
    rpm, dest = _rpm_and_target()

    expected = nns("NNS.distance", _rpm_dict(rpm), dest.tolist(), k, None)
    assert isinstance(expected, np.ndarray)
    actual = nns_distance(rpm, dest, k=k)

    np.testing.assert_allclose(actual, float(expected), atol=EXACT)


@pytest.mark.parity
@pytest.mark.parametrize("k", [1, 2, 3, "all"])
@pytest.mark.parametrize("case", ["binary", "multiclass", "zero_distance", "noninteger"])
def test_nns_distance_class_matches_r(k: int | Literal["all"], case: str) -> None:
    rpm, dest = _class_rpm_and_target(case)

    expected = nns("NNS.distance", _class_rpm_dict(rpm), dest.tolist(), k, "class")
    assert isinstance(expected, np.ndarray)
    actual = nns_distance(rpm, dest, k=k, class_="class")

    np.testing.assert_allclose(actual, float(expected), atol=EXACT)


@pytest.mark.parity
def test_nns_distance_class_ties_keep_rpm_order() -> None:
    rpm = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 2.0],
            [1.0, 1.0, 3.0],
            [1.0, 1.0, 3.0],
        ],
        dtype=np.float64,
    )
    dest = np.array([1.0, 1.0], dtype=np.float64)

    expected = nns("NNS.distance", _class_rpm_dict(rpm), dest.tolist(), 1, "class")
    assert isinstance(expected, np.ndarray)
    actual = nns_distance(rpm, dest, k=1, class_="class")

    assert float(expected) == 2.0
    np.testing.assert_allclose(actual, float(expected), atol=EXACT)


@pytest.mark.parity
@pytest.mark.parametrize("k", [1, 2, "all"])
def test_nns_distance_bulk_matches_r(k: int | Literal["all"]) -> None:
    rpm, _ = _rpm_and_target()
    x_test = rpm[:4, :-1] + np.array([0.05, -0.03, 0.02])

    expected = _r_distance_bulk(rpm, x_test, k)
    actual = nns_distance_bulk(rpm, x_test, k=k)

    np.testing.assert_allclose(actual, expected, atol=EXACT)


@pytest.mark.parity
@pytest.mark.parametrize("k", [1, 2, "all"])
@pytest.mark.parametrize("case", ["binary", "multiclass", "zero_distance"])
def test_nns_distance_bulk_class_matches_installed_r(k: int | Literal["all"], case: str) -> None:
    rpm, dest = _class_rpm_and_target(case)
    x_test = np.vstack((dest, rpm[1, :-1] + np.array([0.02, -0.01])))

    expected = nns_distance_bulk_custom(
        _class_rpm_dict(rpm),
        {"x1": x_test[:, 0].tolist(), "x2": x_test[:, 1].tolist()},
        k,
        "class",
    )
    assert isinstance(expected, np.ndarray)
    actual = nns_distance_bulk(rpm, x_test, k=k, class_="class")

    np.testing.assert_allclose(actual, expected, atol=EXACT)


def _rpm_and_target() -> tuple[np.ndarray, np.ndarray]:
    row = np.arange(1, 13, dtype=np.float64)
    features = np.column_stack(
        (
            np.sin(row / 3.0) + 1.5,
            np.cos(row / 5.0) + 2.0,
            row / 10.0 + 0.5,
        )
    )
    y_hat = np.sin(row / 4.0) + row / 20.0
    return np.column_stack((features, y_hat)), np.array([1.25, 2.75, 1.4])


def _class_rpm_and_target(case: str) -> tuple[np.ndarray, np.ndarray]:
    features = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.2],
            [2.0, 0.8],
            [3.0, 1.0],
            [4.0, 1.7],
            [5.0, 2.2],
        ],
        dtype=np.float64,
    )
    if case == "binary":
        y_hat = np.array([1.0, 1.0, 2.0, 2.0, 2.0, 1.0])
        dest = np.array([2.6, 0.9])
    elif case == "multiclass":
        y_hat = np.array([1.0, 2.0, 3.0, 2.0, 3.0, 1.0])
        dest = np.array([3.5, 1.25])
    elif case == "zero_distance":
        y_hat = np.array([1.0, 1.0, 2.0, 3.0, 3.0, 2.0])
        dest = features[2].copy()
    elif case == "noninteger":
        y_hat = np.array([0.5, 0.5, 1.5, 1.5, 2.5, 2.5])
        dest = np.array([3.5, 1.25])
    else:
        raise ValueError(case)
    return np.column_stack((features, y_hat)), dest


def _rpm_dict(rpm: np.ndarray) -> dict[str, list[float]]:
    return {
        "x1": rpm[:, 0].tolist(),
        "x2": rpm[:, 1].tolist(),
        "x3": rpm[:, 2].tolist(),
        "y.hat": rpm[:, 3].tolist(),
    }


def _class_rpm_dict(rpm: np.ndarray) -> dict[str, list[float]]:
    return {
        "x1": rpm[:, 0].tolist(),
        "x2": rpm[:, 1].tolist(),
        "y.hat": rpm[:, 2].tolist(),
    }


def _r_distance_bulk(rpm: np.ndarray, x_test: np.ndarray, k: int | str) -> np.ndarray:
    expected = nns_distance_bulk_custom(
        _rpm_dict(rpm),
        {
            "x1": x_test[:, 0].tolist(),
            "x2": x_test[:, 1].tolist(),
            "x3": x_test[:, 2].tolist(),
        },
        k,
    )
    assert isinstance(expected, np.ndarray)
    return expected
