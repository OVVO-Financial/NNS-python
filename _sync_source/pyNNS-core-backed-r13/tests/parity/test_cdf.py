from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from _r import nns_cdf_custom
from _tolerances import COMPOUND, EXACT

from pynns import nns_cdf


@pytest.mark.parity
@pytest.mark.parametrize("degree", [0.0, 1.0, 2.0, 3.0])
def test_nns_cdf_univariate_simple_matches_r(degree: float) -> None:
    x = np.array([1.0, 2.0, 3.0])

    expected = cast(dict[str, object], nns_cdf_custom(x.tolist(), degree=degree))
    actual = nns_cdf(x, degree=degree)

    _assert_cdf_result(actual, expected)


@pytest.mark.parity
@pytest.mark.parametrize(
    "x",
    [
        np.array([1.0, 2.0, 2.0, 3.0]),
        np.repeat(5.0, 5),
        np.array([-2.0, -1.0, 0.0, 1.0, 2.0]),
    ],
)
@pytest.mark.parametrize("degree", [0.0, 1.0])
def test_nns_cdf_univariate_edge_vectors_match_r(x: np.ndarray, degree: float) -> None:
    expected = cast(dict[str, object], nns_cdf_custom(x.tolist(), degree=degree))
    actual = nns_cdf(x, degree=degree)

    _assert_cdf_result(actual, expected)


@pytest.mark.parity
@pytest.mark.parametrize("type_name", ["CDF", "survival", "hazard", "cumulative hazard"])
def test_nns_cdf_univariate_types_match_r(type_name: str) -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])

    expected = cast(dict[str, object], nns_cdf_custom(x.tolist(), degree=1.0, type=type_name))
    actual = nns_cdf(x, degree=1.0, type=type_name)

    _assert_cdf_result(actual, expected, atol=COMPOUND)


@pytest.mark.parity
@pytest.mark.parametrize("type_name", ["CDF", "survival", "hazard", "cumulative hazard"])
def test_nns_cdf_univariate_target_matches_r(type_name: str) -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])

    expected = cast(
        dict[str, object],
        nns_cdf_custom(x.tolist(), degree=1.0, target=2.5, type=type_name),
    )
    actual = nns_cdf(x, degree=1.0, target=2.5, type=type_name)

    _assert_cdf_result(actual, expected, atol=COMPOUND)


@pytest.mark.parity
def test_nns_cdf_univariate_installed_r_nan_quirk() -> None:
    actual = nns_cdf(np.array([1.0, np.nan, 2.0]), degree=0.0)
    function = cast(dict[str, np.ndarray], actual["Function"])

    np.testing.assert_allclose(function["x"], np.array([1.0, 2.0]), atol=EXACT)
    np.testing.assert_allclose(function["CDF"], np.array([1.0 / 3.0, 2.0 / 3.0]), atol=EXACT)


@pytest.mark.parity
def test_nns_cdf_univariate_installed_r_inf_quirk() -> None:
    actual = nns_cdf(np.array([1.0, np.inf, 2.0]), degree=0.0)
    function = cast(dict[str, np.ndarray], actual["Function"])

    np.testing.assert_allclose(function["x"], np.array([1.0, 2.0, np.inf]), atol=EXACT)
    np.testing.assert_allclose(
        function["CDF"],
        np.array([1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0]),
        atol=EXACT,
    )


@pytest.mark.parity
@pytest.mark.parametrize("target", [0.0, 5.0])
def test_nns_cdf_univariate_out_of_bounds_target_raises(target: float) -> None:
    with pytest.raises(ValueError, match="target out of bounds"):
        nns_cdf(np.array([1.0, 2.0, 3.0]), target=target)


@pytest.mark.parity
def test_nns_cdf_univariate_vector_target_raises() -> None:
    with pytest.raises(ValueError, match="target must be scalar"):
        nns_cdf(np.array([1.0, 2.0, 3.0]), target=np.array([1.0, 2.0]))


@pytest.mark.parity
@pytest.mark.parametrize("degree", [0.0, 1.0])
def test_nns_cdf_multivariate_cdf_matches_r(degree: float) -> None:
    matrix = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0]])

    expected = cast(dict[str, object], nns_cdf_custom(matrix.tolist(), degree=degree))
    actual = nns_cdf(matrix, degree=degree)

    _assert_cdf_result(actual, expected)


@pytest.mark.parity
@pytest.mark.parametrize("type_name", ["CDF", "survival", "hazard", "cumulative hazard"])
def test_nns_cdf_multivariate_types_and_target_match_r(type_name: str) -> None:
    matrix = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0]])

    expected = cast(
        dict[str, object],
        nns_cdf_custom(matrix.tolist(), degree=0.0, target=[2.0, 2.0], type=type_name),
    )
    actual = nns_cdf(matrix, degree=0.0, target=np.array([2.0, 2.0]), type=type_name)

    _assert_cdf_result(actual, expected, atol=COMPOUND)


@pytest.mark.parity
def test_nns_cdf_multivariate_names_match_r() -> None:
    matrix = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0]])

    expected = cast(dict[str, object], nns_cdf_custom(matrix.tolist(), names=["a", "b"]))
    actual = nns_cdf(matrix, names=["a", "b"])

    assert list(cast(dict[str, np.ndarray], actual["Function"])) == ["a", "b", "CDF"]
    _assert_cdf_result(actual, expected)


@pytest.mark.parity
def test_nns_cdf_multivariate_out_of_bounds_target_raises() -> None:
    matrix = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0]])

    with pytest.raises(ValueError, match="target out of bounds"):
        nns_cdf(matrix, target=np.array([0.0, 2.0]))


def _assert_cdf_result(
    actual: dict[str, object],
    expected: dict[str, object],
    *,
    atol: float = EXACT,
) -> None:
    assert list(actual) == ["Function", "target.value"]
    actual_function = cast(dict[str, np.ndarray], actual["Function"])
    expected_function = cast(dict[str, np.ndarray], expected["Function"])
    assert set(actual_function) == set(expected_function)
    for key, expected_column in expected_function.items():
        np.testing.assert_allclose(
            actual_function[key],
            _as_numeric(expected_column),
            atol=atol,
            equal_nan=True,
        )
    np.testing.assert_allclose(
        np.asarray(actual["target.value"], dtype=np.float64).reshape(-1),
        _as_numeric(expected["target.value"]).reshape(-1),
        atol=atol,
        equal_nan=True,
    )


def _as_numeric(value: object) -> np.ndarray:
    if isinstance(value, list):
        return np.asarray(
            [float("nan") if item == "NaN" else item for item in value],
            dtype=np.float64,
        )
    return np.asarray(value, dtype=np.float64)
