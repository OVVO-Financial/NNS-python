from __future__ import annotations

from typing import Any, cast

import numpy as np

from pynns.var import _var_interpolate_and_extrapolate, _var_multivariate_stack_stage


def test_var_interpolate_and_extrapolate_shape_and_names() -> None:
    variables = np.column_stack(
        (
            np.arange(1.0, 21.0, dtype=float),
            np.arange(2.0, 41.0, 2.0, dtype=float),
        )
    )
    result = cast(
        dict[str, Any],
        _var_interpolate_and_extrapolate(variables, h=3, tau=2, names=["x1", "x2"]),
    )
    interpolated = cast(np.ndarray, result["interpolated_and_extrapolated"])
    univariate = cast(np.ndarray, result["univariate"])

    assert interpolated.shape == variables.shape
    assert result["names"] == ["x1", "x2"]
    assert univariate.shape == (3, 2)


def test_var_interpolate_and_extrapolate_h0_returns_only_interpolated() -> None:
    variables = np.array(
        [
            [1.0, np.nan],
            [np.nan, 5.0],
            [3.0, 6.0],
            [4.0, 7.0],
        ],
        dtype=float,
    )
    result = cast(
        dict[str, np.ndarray],
        _var_interpolate_and_extrapolate(variables, h=0, tau=1, names=["a", "b"]),
    )
    interpolated = result["interpolated_and_extrapolated"]

    assert "univariate" not in result
    assert interpolated.shape == variables.shape
    assert not np.isnan(interpolated).any()
    np.testing.assert_array_equal(result["names"], ["a", "b"])


def test_var_interpolate_and_extrapolate_is_deterministic_for_repeat_call() -> None:
    variables = np.array(
        [[1.0, 2.0], [2.0, np.nan], [4.0, 6.0], [5.0, 8.0], [6.0, 10.0]],
        dtype=float,
    )
    first = cast(
        dict[str, np.ndarray],
        _var_interpolate_and_extrapolate(variables, h=2, tau=1, names=["x1", "x2"]),
    )
    second = cast(
        dict[str, np.ndarray],
        _var_interpolate_and_extrapolate(variables, h=2, tau=1, names=["x1", "x2"]),
    )
    first_interpolated = first["interpolated_and_extrapolated"]
    second_interpolated = second["interpolated_and_extrapolated"]
    first_univariate = first["univariate"]
    second_univariate = second["univariate"]

    np.testing.assert_allclose(first_interpolated, second_interpolated)
    np.testing.assert_allclose(first_univariate, second_univariate, equal_nan=False)


def test_var_multivariate_stack_stage_shape_and_determinism() -> None:
    variables = np.column_stack(
        (
            np.arange(1.0, 21.0, dtype=float),
            np.arange(2.0, 41.0, 2.0, dtype=float),
        )
    )
    first = cast(
        dict[str, np.ndarray],
        _var_interpolate_and_extrapolate(variables, h=3, tau=2, names=["x1", "x2"]),
    )
    first_stage = _var_multivariate_stack_stage(
        first["interpolated_and_extrapolated"],
        first["univariate"],
        h=3,
        tau=2,
        names=["x1", "x2"],
        dim_red_method="cor",
    )
    second = cast(
        dict[str, np.ndarray],
        _var_interpolate_and_extrapolate(variables, h=3, tau=2, names=["x1", "x2"]),
    )
    second_stage = _var_multivariate_stack_stage(
        second["interpolated_and_extrapolated"],
        second["univariate"],
        h=3,
        tau=2,
        names=["x1", "x2"],
        dim_red_method="cor",
    )

    first_multivariate = cast(np.ndarray, first_stage["multivariate"])
    second_multivariate = cast(np.ndarray, second_stage["multivariate"])
    first_relevant = cast(np.ndarray, first_stage["relevant_variables"])
    second_relevant = cast(np.ndarray, second_stage["relevant_variables"])

    assert first_multivariate.shape == (3, 2)
    assert first_multivariate.shape == second_multivariate.shape
    assert first_relevant.shape == second_relevant.shape
    np.testing.assert_array_equal(first_multivariate, second_multivariate)
    assert np.array_equal(first_relevant, second_relevant)
