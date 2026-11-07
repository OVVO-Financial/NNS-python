from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from nns import co_upm_nd, nns_cdf


def test_nns_cdf_return_keys_and_empty_target_value() -> None:
    result = nns_cdf(np.array([1.0, 2.0, 3.0]))

    assert list(result) == ["Function", "target.value"]
    assert np.asarray(result["target.value"]).size == 0


def test_nns_cdf_repeated_calls_are_deterministic() -> None:
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    first = nns_cdf(x, degree=1.0, type="cumulative hazard")
    second = nns_cdf(x, degree=1.0, type="cumulative hazard")

    first_function = cast(dict[str, np.ndarray], first["Function"])
    second_function = cast(dict[str, np.ndarray], second["Function"])
    for key in first_function:
        np.testing.assert_allclose(first_function[key], second_function[key])
    np.testing.assert_allclose(first["target.value"], second["target.value"])


def test_nns_cdf_finite_degree_zero_values_are_probabilities() -> None:
    x = np.array([3.0, 1.0, 2.0, 2.0])
    result = nns_cdf(x, degree=0.0)
    function = cast(dict[str, np.ndarray], result["Function"])

    assert np.all(function["CDF"] >= 0.0)
    assert np.all(function["CDF"] <= 1.0)


def test_nns_cdf_univariate_survival_is_one_minus_cdf_for_finite_values() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    cdf = cast(dict[str, np.ndarray], nns_cdf(x, degree=1.0)["Function"])
    survival = cast(dict[str, np.ndarray], nns_cdf(x, degree=1.0, type="survival")["Function"])

    np.testing.assert_allclose(survival["S(x)"], 1.0 - cdf["CDF"])


def test_nns_cdf_univariate_keeps_duplicate_sorted_rows() -> None:
    x = np.array([3.0, 2.0, 2.0, 1.0])
    function = cast(dict[str, np.ndarray], nns_cdf(x, degree=0.0)["Function"])

    np.testing.assert_allclose(function["x"], np.array([1.0, 2.0, 2.0, 3.0]))
    assert function["CDF"].shape == x.shape


def test_nns_cdf_multivariate_row_count_matches_input() -> None:
    matrix = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0], [4.0, 0.0]])
    function = cast(dict[str, np.ndarray], nns_cdf(matrix, degree=1.0)["Function"])

    assert function["CDF"].shape == (matrix.shape[0],)


def test_nns_cdf_invalid_type_raises() -> None:
    try:
        nns_cdf(np.array([1.0, 2.0, 3.0]), type="density")
    except ValueError as exc:
        assert "invalid type" in str(exc)
    else:
        raise AssertionError("expected invalid type to raise")


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        (["CDF", "b"], ["CDF.1", "b"]),
        (["a", "a"], ["a", "a.1"]),
        (["CDF", "CDF.1"], ["CDF.2", "CDF.1"]),
        (["a", "a", "a.1"], ["a", "a.2", "a.1"]),
    ],
)
def test_nns_cdf_multivariate_names_never_drop_a_coordinate(
    names: list[str], expected: list[str]
) -> None:
    matrix = np.array([[1.0, 2.0, 5.0], [2.0, 1.0, 4.0], [3.0, 3.0, 6.0]])[:, : len(names)]
    function = cast(dict[str, np.ndarray], nns_cdf(matrix, names=names)["Function"])

    assert list(function) == [*expected, "CDF"]
    for col, name in enumerate(expected):
        np.testing.assert_array_equal(function[name], matrix[:, col])


def test_nns_cdf_dataframe_column_named_cdf_keeps_its_coordinate() -> None:
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0]], columns=["CDF", "b"])
    function = cast(dict[str, np.ndarray], nns_cdf(frame)["Function"])

    np.testing.assert_array_equal(function["CDF.1"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(function["CDF"], [1 / 3, 1 / 3, 1.0])


def test_nns_cdf_multivariate_degree1_survival_equals_co_upm_nd() -> None:
    rng = np.random.default_rng(7)
    matrix = rng.standard_normal((30, 3))
    survival = cast(dict[str, np.ndarray], nns_cdf(matrix, degree=1.0, type="survival")["Function"])
    expected = [co_upm_nd(matrix, row, degree=1.0) for row in matrix]

    np.testing.assert_allclose(survival["CDF"], expected, atol=1e-12)


def test_degree_zero_survival_beyond_three_columns_is_the_joint_upper_share() -> None:
    from nns._rrng import RRNG

    # R: set.seed(9); x <- matrix(sample(0:4, 400, TRUE), 20, 20)
    x = (RRNG(9).sample_int_replace(5, 400) - 1).astype(np.float64).reshape(20, 20, order="F")
    share = np.array([np.mean(np.all(x > row, axis=1)) for row in x])

    result = nns_cdf(x, degree=0, type="cumulative hazard")
    hazard = cast(dict[str, np.ndarray], result["Function"])["CDF"]

    np.testing.assert_array_equal(hazard, np.maximum(-np.log(np.maximum(share, 1e-10)), 0.0))


_CDF_TYPES = ["CDF", "survival", "hazard", "cumulative hazard"]


def _cdf_target_data() -> list[np.ndarray]:
    rng = np.random.default_rng(11)
    return [np.arange(1.0, 21.0), np.sort(np.round(rng.lognormal(size=30), 2))]


def _cdf_curve(x: np.ndarray, degree: int, type_name: str) -> np.ndarray:
    function = cast(dict[str, np.ndarray], nns_cdf(x, degree, type=type_name)["Function"])
    return np.asarray(list(function.values())[1], dtype=np.float64)


def _cdf_at(x: np.ndarray, degree: int, type_name: str, target: float) -> float:
    value = nns_cdf(x, degree, target=target, type=type_name)["target.value"]
    return float(np.asarray(value, dtype=np.float64).reshape(-1)[0])


@pytest.mark.parametrize("type_name", _CDF_TYPES)
@pytest.mark.parametrize("degree", [0, 1])
def test_nns_cdf_target_value_equals_the_curve_at_each_data_point(
    type_name: str, degree: int
) -> None:
    # R NNS.CDF: hazard used the survival at the nearest x, and cumulative hazard
    # refit the curve through NNS.reg, so neither matched the returned curve.
    for x in _cdf_target_data():
        curve = _cdf_curve(x, degree, type_name)
        at = [_cdf_at(x, degree, type_name, float(t)) for t in np.sort(x)]
        np.testing.assert_allclose(at, curve, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("type_name", ["CDF", "survival", "cumulative hazard"])
@pytest.mark.parametrize("degree", [0, 1])
def test_nns_cdf_target_value_lies_between_neighbouring_curve_values(
    type_name: str, degree: int
) -> None:
    for x in _cdf_target_data():
        xs = np.unique(x)
        sorted_x = np.sort(x)
        first = np.r_[True, sorted_x[1:] != sorted_x[:-1]]
        curve = _cdf_curve(x, degree, type_name)[first]
        mid = (xs[1:] + xs[:-1]) / 2.0
        at = np.array([_cdf_at(x, degree, type_name, float(t)) for t in mid])
        lo = np.minimum(curve[1:], curve[:-1])
        hi = np.maximum(curve[1:], curve[:-1])
        assert np.all((at >= lo - 1e-12) & (at <= hi + 1e-12))


@pytest.mark.parametrize("degree", [0, 1])
def test_nns_cdf_hazard_target_value_between_data_points_is_clamped(degree: int) -> None:
    for x in _cdf_target_data():
        xs = np.unique(x)
        mid = (xs[1:] + xs[:-1]) / 2.0
        at = np.array([_cdf_at(x, degree, "hazard", float(t)) for t in mid])
        assert np.all(np.isfinite(at) & (at >= 0.0) & (at <= 1e6))


def test_nns_cdf_cumulative_hazard_at_a_target_is_minus_log_survival() -> None:
    x = np.arange(1.0, 21.0)
    assert _cdf_at(x, 0, "cumulative hazard", 19.0) == pytest.approx(-np.log(0.05), rel=1e-14)
    assert _cdf_at(x, 0, "cumulative hazard", 10.0) == pytest.approx(-np.log(0.5), rel=1e-14)
    at_max = _cdf_at(x, 0, "cumulative hazard", 20.0)
    assert np.isfinite(at_max)
    assert at_max == _cdf_curve(x, 0, "cumulative hazard")[19]
