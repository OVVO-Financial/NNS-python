from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from _r import nns

from pynns.var import (
    _lag_mtx,
    _var_interpolate_and_extrapolate,
    _var_multivariate_stack_stage,
    nns_var,
)


def _to_matrix(result: object, names: list[str], key: str) -> np.ndarray:
    assert isinstance(result, dict)
    values = result[key]
    assert isinstance(values, dict)
    return np.column_stack([np.asarray(values[name], dtype=np.float64) for name in names])


def _json_safe_data(values: np.ndarray) -> list[list[float | None]]:
    return [[None if np.isnan(item) else float(item) for item in row] for row in values.tolist()]


def _relative_diagnostics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float | int]:
    actual_values = np.asarray(actual, dtype=np.float64)
    expected_values = np.asarray(expected, dtype=np.float64)
    diff = np.abs(actual_values - expected_values)
    finite = np.isfinite(diff)
    if not np.any(finite):
        return {
            "max_abs_diff": 0.0,
            "max_rel_pct_masked": 0.0,
            "p95_rel_pct_masked": 0.0,
            "median_rel_pct_masked": 0.0,
            "near_zero_reference": int(expected_values.size),
        }
    material = finite & (np.abs(expected_values) > 1e-8)
    rel_pct = np.zeros_like(diff, dtype=np.float64)
    rel_pct[material] = 100.0 * diff[material] / np.abs(expected_values[material])
    if np.any(material):
        material_rel = rel_pct[material]
        max_rel = float(np.max(material_rel))
        p95_rel = float(np.percentile(material_rel, 95))
        median_rel = float(np.median(material_rel))
    else:
        max_rel = 0.0
        p95_rel = 0.0
        median_rel = 0.0
    return {
        "max_abs_diff": float(np.max(diff[finite])),
        "max_rel_pct_masked": max_rel,
        "p95_rel_pct_masked": p95_rel,
        "median_rel_pct_masked": median_rel,
        "near_zero_reference": int(np.count_nonzero(finite & ~material)),
    }


def _assert_public_numeric_close(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    rel_pct: float = 1e-7,
    abs_tol: float = 1e-8,
) -> None:
    diagnostics = _relative_diagnostics(actual, expected)
    assert diagnostics["max_abs_diff"] <= abs_tol or diagnostics["p95_rel_pct_masked"] <= rel_pct
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=max(1e-8, rel_pct / 100.0),
        atol=abs_tol,
        equal_nan=True,
    )


def test_lag_mtx_scalar_tau_matches_reference_blocks() -> None:
    x = np.column_stack(
        (
            np.array([1, 2, 3, 4, 5], dtype=np.float64),
            np.array([6, 7, 8, 9, 10], dtype=np.float64),
        )
    )
    actual, names = _lag_mtx(x, 2, names=["a", "b"])

    expected = np.array(
        [
            [3, 8, 2, 1, 7, 6],
            [4, 9, 3, 2, 8, 7],
            [5, 10, 4, 3, 9, 8],
        ],
        dtype=np.float64,
    )
    expected_names = ["a_tau_0", "b_tau_0", "a_tau_1", "a_tau_2", "b_tau_1", "b_tau_2"]

    np.testing.assert_allclose(actual, expected)
    assert names == expected_names


def test_lag_mtx_nested_tau_keeps_requested_lags_plus_tau_zero() -> None:
    x = np.column_stack(
        (
            np.array([1, 2, 3, 4, 5], dtype=np.float64),
            np.array([6, 7, 8, 9, 10], dtype=np.float64),
        )
    )
    actual, names = _lag_mtx(x, ([1, 2], [1]), names=["a", "b"])

    expected_names = ["a_tau_0", "b_tau_0", "a_tau_1", "a_tau_2", "b_tau_1"]
    expected = np.array(
        [
            [3, 8, 2, 1, 7],
            [4, 9, 3, 2, 8],
            [5, 10, 4, 3, 9],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(actual, expected)
    assert names == expected_names


def _expected_var_reference(
    variables: np.ndarray,
    h: int,
    tau: int | list[int] | list[list[int]],
) -> dict[str, object]:
    result = cast(dict[str, Any], nns("NNS.VAR", _json_safe_data(variables), h, tau))
    names = list(result["interpolated_and_extrapolated"].keys()) if h > 0 else list(result.keys())
    if h > 0:
        interpolated_and_extrapolated = _to_matrix(result, names, "interpolated_and_extrapolated")
        univariate = _to_matrix(result, names, "univariate")
    else:
        interpolated_and_extrapolated = np.column_stack(
            [np.asarray(result[name], dtype=np.float64) for name in names]
        )
        univariate = None

    expected: dict[str, object] = {
        "interpolated_and_extrapolated": interpolated_and_extrapolated,
        "names": names,
    }
    if h > 0:
        expected["univariate"] = univariate
    return expected


def _to_relevant_matrix(result: dict[str, Any], key: str) -> np.ndarray:
    table = result[key]
    assert isinstance(table, dict)
    columns = list(table.keys())
    values = [
        np.asarray(table[name], dtype=object)
        if np.ndim(table[name]) != 0
        else np.array([table[name]], dtype=object)
        for name in columns
    ]
    max_length = max((value.size for value in values), default=0)
    matrix = np.full((max_length, len(columns)), None, dtype=object)
    for col, data in enumerate(values):
        matrix[: data.size, col] = data
    return matrix


def _expected_var_multivariate_reference(
    variables: np.ndarray,
    h: int,
    tau: int | list[int] | list[list[int]],
    dim_red_method: str,
) -> dict[str, Any]:
    result = cast(
        dict[str, Any],
        nns(
            "NNS.VAR",
            _json_safe_data(variables),
            h,
            tau,
            dim_red_method,
        ),
    )
    names = list(result["interpolated_and_extrapolated"].keys())
    assert isinstance(result["univariate"], dict)
    assert isinstance(result["multivariate"], dict)
    assert isinstance(result["relevant_variables"], dict)
    return {
        "interpolated_and_extrapolated": _to_matrix(result, names, "interpolated_and_extrapolated"),
        "univariate": _to_matrix(result, names, "univariate"),
        "multivariate": _to_matrix(result, names, "multivariate"),
        "ensemble": _to_matrix(result, names, "ensemble"),
        "relevant_variables": _to_relevant_matrix(
            result,
            "relevant_variables",
        ),
        "relevant_names": names,
    }


@pytest.mark.parametrize(
    ("name", "h"),
    [
        ("complete_finite", 3),
        ("interior_na", 3),
        ("trailing_na", 3),
        ("negative", 3),
    ],
)
def test_var_interpolate_and_extrapolate_matches_r(
    name: str,
    h: int,
) -> None:
    base = np.column_stack(
        (
            np.arange(-2.0, 18.0, 1.0, dtype=float),
            np.arange(1.0, 40.0, 2.0, dtype=float),
        )
    )
    if name == "interior_na":
        base = base.copy()
        base[4, 0] = np.nan
    elif name == "trailing_na":
        base = base.copy()
        base[19, 0] = np.nan
    elif name == "negative":
        base = -base

    expected_result = _expected_var_reference(base, h, 2)
    names = cast(list[str], expected_result["names"])
    actual_result = _var_interpolate_and_extrapolate(base, h, tau=2, names=names)
    actual_interpolated = cast(np.ndarray, actual_result["interpolated_and_extrapolated"])
    expected_interpolated = cast(
        np.ndarray,
        expected_result["interpolated_and_extrapolated"],
    )
    actual_univariate = cast(np.ndarray, actual_result["univariate"])
    expected_univariate = cast(np.ndarray, expected_result["univariate"])

    np.testing.assert_allclose(actual_interpolated, expected_interpolated, equal_nan=True)
    assert actual_result["names"] == expected_result["names"]

    assert "univariate" in actual_result
    assert "univariate" in expected_result
    assert isinstance(actual_result["univariate"], np.ndarray)
    assert isinstance(expected_result["univariate"], np.ndarray)
    np.testing.assert_allclose(actual_univariate, expected_univariate, equal_nan=True)


def test_var_interpolate_and_extrapolate_h0_matches_r() -> None:
    variables = np.column_stack(
        (
            np.arange(-2.0, 18.0, 1.0, dtype=float),
            np.arange(1.0, 40.0, 2.0, dtype=float),
        )
    )

    expected_result = _expected_var_reference(variables, 0, 2)
    actual_result = _var_interpolate_and_extrapolate(variables, 0, tau=2)
    actual_interpolated = cast(np.ndarray, actual_result["interpolated_and_extrapolated"])
    expected_interpolated = cast(
        np.ndarray,
        expected_result["interpolated_and_extrapolated"],
    )

    np.testing.assert_allclose(actual_interpolated, expected_interpolated, equal_nan=True)
    assert "univariate" not in actual_result


@pytest.mark.parametrize(
    ("name", "tau", "dim_red_method"),
    [
        ("complete", 2, "cor"),
        ("tau1", 1, "cor"),
        ("nested", ([1, 2], [1]), "cor"),
        ("dep", 2, "NNS.dep"),
        ("caus", 2, "NNS.caus"),
        ("all", 2, "all"),
    ],
)
def test_var_multivariate_stack_stage_matches_r(
    name: str,
    tau: int | list[int] | list[list[int]],
    dim_red_method: str,
) -> None:
    del name
    variables = np.column_stack(
        (
            np.arange(-2.0, 18.0, 1.0, dtype=float),
            np.arange(1.0, 40.0, 2.0, dtype=float),
        )
    )

    expected_result = _expected_var_multivariate_reference(variables, 3, tau, dim_red_method)
    names = cast(list[str], expected_result["relevant_names"])
    first_stage = _var_interpolate_and_extrapolate(variables, 3, tau=tau, names=names)
    actual_result = _var_multivariate_stack_stage(
        cast(np.ndarray, first_stage["interpolated_and_extrapolated"]),
        cast(np.ndarray, first_stage["univariate"]),
        h=3,
        tau=tau,
        names=names,
        dim_red_method=dim_red_method,
    )

    actual_multivariate = cast(np.ndarray, actual_result["multivariate"])
    actual_relevant = cast(np.ndarray, actual_result["relevant_variables"])
    expected_multivariate = cast(np.ndarray, expected_result["multivariate"])
    expected_relevant = cast(np.ndarray, expected_result["relevant_variables"])

    if dim_red_method in {"NNS.caus", "all"}:
        _assert_public_numeric_close(actual_multivariate, expected_multivariate, rel_pct=1.0)
    else:
        np.testing.assert_allclose(actual_multivariate, expected_multivariate, equal_nan=True)
    assert actual_relevant.shape == expected_relevant.shape
    assert actual_result["names"] == names
    assert np.array_equal(actual_relevant, expected_relevant)


@pytest.mark.parametrize(
    ("name", "tau"),
    [
        ("complete", 2),
        ("scalar_tau", 1),
        ("nested_tau", ([1, 2], [1])),
    ],
)
def test_public_nns_var_cor_matches_r(
    name: str,
    tau: int | list[int] | list[list[int]],
) -> None:
    del name
    variables = np.column_stack(
        (
            np.arange(-2.0, 18.0, 1.0, dtype=float),
            np.arange(1.0, 40.0, 2.0, dtype=float),
        )
    )

    expected_result = _expected_var_multivariate_reference(variables, 3, tau, "cor")
    actual_result = nns_var(variables, 3, tau=tau, dim_red_method="cor")

    assert set(actual_result) == {
        "interpolated_and_extrapolated",
        "relevant_variables",
        "univariate",
        "multivariate",
        "ensemble",
        "names",
    }
    assert actual_result["names"] == expected_result["relevant_names"]
    for key in ("interpolated_and_extrapolated", "univariate", "multivariate", "ensemble"):
        actual_values = cast(np.ndarray, actual_result[key])
        expected_values = cast(np.ndarray, expected_result[key])
        assert actual_values.shape == expected_values.shape
        assert np.all(np.isfinite(actual_values))
        _assert_public_numeric_close(actual_values, expected_values)
    assert np.array_equal(
        cast(np.ndarray, actual_result["relevant_variables"]),
        cast(np.ndarray, expected_result["relevant_variables"]),
    )


def test_public_nns_var_cor_handles_missing_values_like_r() -> None:
    variables = np.column_stack(
        (
            np.arange(-2.0, 18.0, 1.0, dtype=float),
            np.arange(1.0, 40.0, 2.0, dtype=float),
        )
    )
    variables[4, 0] = np.nan
    variables[-1, 1] = np.nan

    expected_result = _expected_var_multivariate_reference(variables, 3, 2, "cor")
    actual_result = nns_var(variables, 3, tau=2, dim_red_method="cor")

    for key in ("interpolated_and_extrapolated", "univariate", "multivariate", "ensemble"):
        _assert_public_numeric_close(
            cast(np.ndarray, actual_result[key]),
            cast(np.ndarray, expected_result[key]),
            abs_tol=1e-8,
        )
    assert np.array_equal(
        cast(np.ndarray, actual_result["relevant_variables"]),
        cast(np.ndarray, expected_result["relevant_variables"]),
    )


def test_public_nns_var_nns_dep_matches_r() -> None:
    variables = np.column_stack(
        (
            np.arange(-2.0, 18.0, 1.0, dtype=float),
            np.arange(1.0, 40.0, 2.0, dtype=float),
        )
    )

    expected_result = _expected_var_multivariate_reference(variables, 3, 2, "NNS.dep")
    actual_result = nns_var(variables, 3, tau=2, dim_red_method="NNS.dep")

    assert set(actual_result) == {
        "interpolated_and_extrapolated",
        "relevant_variables",
        "univariate",
        "multivariate",
        "ensemble",
        "names",
    }
    assert actual_result["names"] == expected_result["relevant_names"]
    for key in ("interpolated_and_extrapolated", "univariate", "multivariate", "ensemble"):
        actual_values = cast(np.ndarray, actual_result[key])
        expected_values = cast(np.ndarray, expected_result[key])
        assert actual_values.shape == expected_values.shape
        assert np.all(np.isfinite(actual_values))
        _assert_public_numeric_close(actual_values, expected_values)
    assert np.array_equal(
        cast(np.ndarray, actual_result["relevant_variables"]),
        cast(np.ndarray, expected_result["relevant_variables"]),
    )


def test_public_nns_var_nns_caus_matches_r() -> None:
    variables = np.column_stack(
        (
            np.arange(-2.0, 18.0, 1.0, dtype=float),
            np.arange(1.0, 40.0, 2.0, dtype=float),
        )
    )

    expected_result = _expected_var_multivariate_reference(variables, 3, 2, "NNS.caus")
    actual_result = nns_var(variables, 3, tau=2, dim_red_method="NNS.caus")

    assert set(actual_result) == {
        "interpolated_and_extrapolated",
        "relevant_variables",
        "univariate",
        "multivariate",
        "ensemble",
        "names",
    }
    assert actual_result["names"] == expected_result["relevant_names"]
    for key in ("interpolated_and_extrapolated", "univariate", "multivariate", "ensemble"):
        actual_values = cast(np.ndarray, actual_result[key])
        expected_values = cast(np.ndarray, expected_result[key])
        assert actual_values.shape == expected_values.shape
        assert np.all(np.isfinite(actual_values))
        _assert_public_numeric_close(actual_values, expected_values, rel_pct=1.0)
    assert np.array_equal(
        cast(np.ndarray, actual_result["relevant_variables"]),
        cast(np.ndarray, expected_result["relevant_variables"]),
    )


def test_public_nns_var_all_matches_r() -> None:
    variables = np.column_stack(
        (
            np.arange(-2.0, 18.0, 1.0, dtype=float),
            np.arange(1.0, 40.0, 2.0, dtype=float),
        )
    )

    expected_result = _expected_var_multivariate_reference(variables, 3, 2, "all")
    actual_result = nns_var(variables, 3, tau=2, dim_red_method="all")

    assert set(actual_result) == {
        "interpolated_and_extrapolated",
        "relevant_variables",
        "univariate",
        "multivariate",
        "ensemble",
        "names",
    }
    assert actual_result["names"] == expected_result["relevant_names"]
    for key in ("interpolated_and_extrapolated", "univariate", "multivariate", "ensemble"):
        actual_values = cast(np.ndarray, actual_result[key])
        expected_values = cast(np.ndarray, expected_result[key])
        assert actual_values.shape == expected_values.shape
        assert np.all(np.isfinite(actual_values))
        _assert_public_numeric_close(actual_values, expected_values, rel_pct=1.0)
    assert np.array_equal(
        cast(np.ndarray, actual_result["relevant_variables"]),
        cast(np.ndarray, expected_result["relevant_variables"]),
    )


def test_public_nns_var_h0_returns_normalized_interpolation_dict() -> None:
    variables = np.column_stack(
        (
            np.arange(-2.0, 18.0, 1.0, dtype=float),
            np.arange(1.0, 40.0, 2.0, dtype=float),
        )
    )

    expected_result = _expected_var_reference(variables, 0, 2)
    actual_result = nns_var(variables, 0, tau=2)

    assert set(actual_result) == {"interpolated_and_extrapolated", "names"}
    _assert_public_numeric_close(
        cast(np.ndarray, actual_result["interpolated_and_extrapolated"]),
        cast(np.ndarray, expected_result["interpolated_and_extrapolated"]),
    )
    assert actual_result["names"] == expected_result["names"]
