from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from _r import nns_cdf_custom
from _tolerances import COMPOUND, EXACT

from nns import nns_cdf


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
@pytest.mark.parametrize("degree", [0.0, 1.0])
def test_nns_cdf_univariate_drops_missing_values_like_r(degree: float) -> None:
    # R fix 41c3297: NNS.CDF drops NA first, so the CDF reaches 1 (13.3 capped it at
    # n_finite / n) and a target can be evaluated.
    x = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
    finite = x[~np.isnan(x)]

    expected = cast(
        dict[str, object], nns_cdf_custom(finite.tolist(), degree=degree, target=2.0)
    )
    actual = nns_cdf(x, degree=degree, target=2.0)

    _assert_cdf_result(actual, expected)
    assert cast(dict[str, np.ndarray], actual["Function"])["CDF"][-1] == 1.0


def test_nns_cdf_multivariate_drops_incomplete_rows() -> None:
    matrix = np.array([[1.0, 2.0], [2.0, np.nan], [3.0, 1.0], [4.0, 4.0]])
    actual = nns_cdf(matrix)
    function = cast(dict[str, np.ndarray], actual["Function"])

    np.testing.assert_array_equal(function["1"], [1.0, 3.0, 4.0])
    np.testing.assert_allclose(function["CDF"], [1.0 / 3.0, 1.0 / 3.0, 1.0], atol=EXACT)


@pytest.mark.parametrize(
    "variable",
    [np.array([1.0, np.inf, 2.0]), np.array([[1.0, 2.0], [2.0, -np.inf], [3.0, 1.0]])],
)
def test_nns_cdf_rejects_infinite_values(variable: np.ndarray) -> None:
    with pytest.raises(ValueError, match="finite"):
        nns_cdf(variable)


_HAZARD_TIE_RUNS = [
    # A run of 12 zeros: the +-5 finite-difference window of the middle points lies
    # inside the run (zero width), as in daily returns with zero-return days.
    [-2.0, -1.5, -1.0, -0.5, *([0.0] * 12), 0.5, 1.0, 1.5, 2.0, 3.0],
    # Discrete counts with long tie runs.
    [
        *[3.0, 1.0, 2.0, 4.0, 3.0, 2.0, 3.0, 5.0, 2.0, 3.0, 1.0, 3.0, 2.0, 4.0, 3.0, 2.0],
        *[6.0, 3.0, 2.0, 3.0, 4.0, 2.0, 3.0, 3.0, 0.0, 2.0, 3.0, 4.0, 3.0, 2.0, 3.0, 1.0],
        *[3.0, 2.0, 5.0, 3.0, 2.0, 3.0, 4.0, 3.0],
    ],
]


@pytest.mark.parity
@pytest.mark.parametrize("x", _HAZARD_TIE_RUNS)
@pytest.mark.parametrize("degree", [0.0, 1.0])
def test_nns_cdf_hazard_with_tie_runs_matches_r(x: list[float], degree: float) -> None:
    # R fix be02460: a zero-width window inside a tie run is widened to the
    # neighbouring distinct values (13.3 divided 0/0 and nns_reg stopped).
    expected = cast(
        dict[str, object],
        nns_cdf_custom(x, degree=degree, target=1.0, type="hazard"),
    )
    actual = nns_cdf(np.asarray(x), degree=degree, target=1.0, type="hazard")

    assert np.all(np.isfinite(cast(dict[str, np.ndarray], actual["Function"])["h(x)"]))
    _assert_cdf_result(actual, expected, atol=COMPOUND)


@pytest.mark.parity
def test_nns_cdf_hazard_constant_variable_raises() -> None:
    with pytest.raises(ValueError, match="at least two distinct values"):
        nns_cdf(np.repeat(2.0, 5), type="hazard")


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


def _trivariate_matrix() -> np.ndarray:
    rng = np.random.default_rng(7)
    z = rng.standard_normal((40, 3))
    base = z[:, :1]
    return np.round(np.hstack([base, 0.6 * base + 0.8 * z[:, 1:2], z[:, 2:3] - 0.5 * base]), 6)


@pytest.mark.parity
@pytest.mark.parametrize("degree", [0.0, 1.0])
@pytest.mark.parametrize("type_name", ["survival", "hazard", "cumulative hazard"])
def test_nns_cdf_trivariate_types_and_target_match_r(type_name: str, degree: float) -> None:
    # R fix 0f94f02: the joint survival for d >= 3 uses inclusion-exclusion over all
    # subsets, and target.value is the same function as the rows.
    matrix = _trivariate_matrix()
    target = [0.1, 0.0, -0.2]

    expected = cast(
        dict[str, object],
        nns_cdf_custom(matrix.tolist(), degree=degree, target=target, type=type_name),
    )
    actual = nns_cdf(matrix, degree=degree, target=np.array(target), type=type_name)

    _assert_cdf_result(actual, expected, atol=COMPOUND)


def test_nns_cdf_trivariate_degree0_survival_is_joint_exceedance() -> None:
    matrix = _trivariate_matrix()
    actual = nns_cdf(matrix, degree=0.0, type="survival", target=np.array([0.1, 0.0, -0.2]))
    function = cast(dict[str, np.ndarray], actual["Function"])

    brute = np.array([np.mean(np.all(matrix > row, axis=1)) for row in matrix])
    np.testing.assert_allclose(function["CDF"], brute, atol=1e-12)
    target_brute = np.mean(np.all(matrix > np.array([0.1, 0.0, -0.2]), axis=1))
    np.testing.assert_allclose(actual["target.value"], [target_brute], atol=1e-12)


def test_nns_cdf_multivariate_checks_every_target_coordinate() -> None:
    matrix = _trivariate_matrix()
    with pytest.raises(ValueError, match="target out of bounds"):
        nns_cdf(matrix, target=np.array([0.0, 0.0, 1e6]))
    with pytest.raises(ValueError, match="one value per column"):
        nns_cdf(matrix, target=np.array([0.0, 0.0]))


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
