from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from _r import nns, nns_var_custom

from nns.var import (
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


# The perfectly linear 20-row fixtures are degenerate for the multivariate stage:
# their evenly spaced synthetic X* puts NNS.reg partition boundaries on data
# points, so the ~1e-16 differences between R's (long double) correlations and
# normalization and the port's move a regression point. Given R's exact X*, the
# port reproduces R's partition. NNS.caus also ties every lagged regressor there,
# so its selections are checked for validity. On these fixtures the multivariate
# estimates are checked for validity only; the driven-series tests below hold
# every method, missing values and both objectives to exact parity.
_DEGENERATE_CAUSATION = {"NNS.caus", "all"}


def _assert_degenerate_fixture_matches(
    actual: dict[str, Any], expected: dict[str, Any], names: list[str], tau: Any
) -> None:
    _assert_valid_relevant_variables(cast(np.ndarray, actual["relevant_variables"]), names, tau)
    for key in ("multivariate", "ensemble"):
        if key in actual:
            actual_values = cast(np.ndarray, actual[key])
            assert actual_values.shape == cast(np.ndarray, expected[key]).shape
            assert np.all(np.isfinite(actual_values))


def _assert_valid_relevant_variables(relevant: np.ndarray, names: list[str], tau: Any) -> None:
    from nns.var import _var_lag_names, _var_lags

    lagged, _ = _var_lag_names(names, _var_lags(tau, len(names))["lags"])
    for column, name in enumerate(names):
        selected = [value for value in relevant[:, column].tolist() if value is not None]
        assert selected
        assert set(selected) <= set(lagged)
        assert f"{name}_tau_0" not in selected


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

    assert actual_result["names"] == names
    if dim_red_method in _DEGENERATE_CAUSATION:
        _assert_degenerate_fixture_matches(actual_result, expected_result, names, tau)
        return
    np.testing.assert_allclose(actual_multivariate, expected_multivariate, equal_nan=True)
    assert actual_relevant.shape == expected_relevant.shape
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
        actual_result["relevant_variables"],
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

    for key in ("interpolated_and_extrapolated", "univariate"):
        _assert_public_numeric_close(
            cast(np.ndarray, actual_result[key]),
            cast(np.ndarray, expected_result[key]),
            abs_tol=1e-8,
        )
    assert np.array_equal(
        actual_result["relevant_variables"],
        cast(np.ndarray, expected_result["relevant_variables"]),
    )
    _assert_degenerate_fixture_matches(
        cast(dict[str, Any], actual_result), expected_result, actual_result["names"], 2
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
        actual_result["relevant_variables"],
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
        if key in {"interpolated_and_extrapolated", "univariate"}:
            _assert_public_numeric_close(actual_values, expected_values)
    _assert_degenerate_fixture_matches(
        cast(dict[str, Any], actual_result), expected_result, actual_result["names"], 2
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
        if key in {"interpolated_and_extrapolated", "univariate"}:
            _assert_public_numeric_close(actual_values, expected_values)
    _assert_degenerate_fixture_matches(
        cast(dict[str, Any], actual_result), expected_result, actual_result["names"], 2
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


_VAR_KEYS = ("interpolated_and_extrapolated", "univariate", "multivariate", "ensemble")


def _driven_variables(n: int = 60, seed: int = 21) -> np.ndarray:
    """x is a random walk, y is driven by lagged x, z is noisy seasonal."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    x = np.cumsum(rng.standard_normal(n))
    y = np.concatenate(([0.0], 0.8 * x[:-1])) + 0.3 * rng.standard_normal(n)
    z = np.sin(2.0 * np.pi * t / 12.0) + 0.2 * rng.standard_normal(n)
    return np.column_stack((x, y, z))


def _custom_var_reference(variables: np.ndarray, **kwargs: Any) -> dict[str, Any]:
    result = cast(dict[str, Any], nns_var_custom(_json_safe_data(variables), **kwargs))
    names = list(result["interpolated_and_extrapolated"].keys())
    expected: dict[str, Any] = {key: _to_matrix(result, names, key) for key in _VAR_KEYS}
    expected["relevant_variables"] = _to_relevant_matrix(result, "relevant_variables")
    expected["names"] = names
    return expected


def _assert_var_matches(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    assert actual["names"] == expected["names"]
    for key in _VAR_KEYS:
        actual_values = cast(np.ndarray, actual[key])
        expected_values = cast(np.ndarray, expected[key])
        assert actual_values.shape == expected_values.shape
        _assert_public_numeric_close(actual_values, expected_values)
    assert np.array_equal(
        cast(np.ndarray, actual["relevant_variables"]),
        cast(np.ndarray, expected["relevant_variables"]),
    )


@pytest.mark.parametrize("dim_red_method", ["cor", "NNS.dep", "NNS.caus", "all"])
def test_public_nns_var_driven_series_matches_r(dim_red_method: str) -> None:
    variables = _driven_variables()
    expected = _custom_var_reference(variables, h=4, tau=2, dim_red_method=dim_red_method)
    actual = nns_var(variables, 4, tau=2, dim_red_method=dim_red_method)

    _assert_var_matches(cast(dict[str, Any], actual), expected)
    # y is driven by lagged x: the multivariate estimate must be built on it.
    relevant_y = [value for value in actual["relevant_variables"][:, 1] if value is not None]
    assert "x1_tau_1" in relevant_y
    assert "x2_tau_0" not in relevant_y


@pytest.mark.parametrize(
    ("name", "tau"),
    [("vector", [1, 2, 2]), ("lists", [[1, 2], [1], [1, 3]])],
)
def test_public_nns_var_tau_forms_match_r(name: str, tau: Any) -> None:
    del name
    variables = _driven_variables()
    expected = _custom_var_reference(variables, h=3, tau=tau)
    actual = nns_var(variables, 3, tau=tau)
    _assert_var_matches(cast(dict[str, Any], actual), expected)


def test_public_nns_var_own_lag_weights_match_r() -> None:
    variables = _driven_variables()
    expected = _custom_var_reference(variables, h=3, tau=2, naive_weights=False)
    actual = nns_var(variables, 3, tau=2, naive_weights=False)
    _assert_var_matches(cast(dict[str, Any], actual), expected)


def test_public_nns_var_objective_max_matches_r() -> None:
    variables = _driven_variables()
    expected = _custom_var_reference(variables, h=3, tau=1, objective="max")
    actual = nns_var(variables, 3, tau=1, objective="max")
    _assert_var_matches(cast(dict[str, Any], actual), expected)


def test_public_nns_var_interior_and_trailing_missing_values_match_r() -> None:
    variables = _driven_variables()
    variables[[10, 11, 30], 0] = np.nan
    variables[-2:, 2] = np.nan
    expected = _custom_var_reference(variables, h=3, tau=2)
    actual = nns_var(variables, 3, tau=2)
    _assert_var_matches(cast(dict[str, Any], actual), expected)
