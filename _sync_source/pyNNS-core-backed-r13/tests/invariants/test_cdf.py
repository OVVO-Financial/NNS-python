from __future__ import annotations

from typing import cast

import numpy as np

from pynns import nns_cdf


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
