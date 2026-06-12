from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from _r import nns_stack_factor_predictor, nns_stack_mixed_factor_predictor, nns_stack_numeric
from _tolerances import COMPOUND

from pynns import nns_stack


@pytest.mark.parity
@pytest.mark.parametrize("method", [[1], [2], [1, 2]])
@pytest.mark.parametrize("stack", [True, False])
def test_nns_stack_numeric_matches_r(method: list[int], stack: bool) -> None:
    x = np.linspace(-2.0, 2.0, 30)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = x + np.sin(x) + 0.25 * np.cos(x)
    point = variable[:5]

    expected = nns_stack_numeric(
        variable.tolist(),
        y.tolist(),
        point.tolist(),
        cv_size=0.25,
        folds=2,
        method=method,
        order=None,
        stack=stack,
        dim_red_method="cor",
    )
    actual = nns_stack(
        variable,
        y,
        point,
        cv_size=0.25,
        folds=2,
        method=method,
        stack=stack,
        dim_red_method="cor",
    )

    _assert_stack_matches(actual, expected, exact_probability_threshold=False)


@pytest.mark.parity
def test_nns_stack_equal_dim_red_matches_r() -> None:
    x = np.linspace(-1.5, 1.5, 36)
    variable = np.column_stack((x, x**2, np.sin(x)))
    y = 0.5 * x + x**2 - 0.25 * np.sin(x)
    point = variable[::9]

    expected = nns_stack_numeric(
        variable.tolist(),
        y.tolist(),
        point.tolist(),
        cv_size=0.25,
        folds=2,
        method=[2],
        order=2,
        stack=False,
        dim_red_method="equal",
    )
    actual = nns_stack(
        variable,
        y,
        point,
        cv_size=0.25,
        folds=2,
        method=2,
        order=2,
        stack=False,
        dim_red_method="equal",
    )

    _assert_stack_matches(actual, expected, exact_probability_threshold=False)


@pytest.mark.parity
def test_nns_stack_factor_predictor_method1_matches_r() -> None:
    x = np.asarray(["b", "a", "b", "c", "a", "c", "b", "a"])
    y = np.asarray([2.0, 1.0, 3.0, 4.0, 1.5, 3.5, 2.5, 1.25])
    point = np.asarray(["a", "c", "b"])
    levels = ["a", "b", "c"]

    expected = nns_stack_factor_predictor(
        x.tolist(),
        y.tolist(),
        point.tolist(),
        levels=levels,
        cv_size=0.25,
        folds=1,
        method=[1],
        order=None,
        stack=True,
        dim_red_method="cor",
    )
    actual = nns_stack(
        x,
        y,
        point,
        factor_levels=levels,
        cv_size=0.25,
        folds=1,
        method=1,
        stack=True,
        dim_red_method="cor",
    )

    _assert_stack_matches(actual, expected, exact_probability_threshold=False)


@pytest.mark.parity
def test_nns_stack_factor_predictor_method2_factor_only_matches_r_fallback() -> None:
    x = np.asarray(["b", "a", "b", "c", "a", "c", "b", "a"])
    y = np.asarray([2.0, 1.0, 3.0, 4.0, 1.5, 3.5, 2.5, 1.25])
    point = np.asarray(["a", "c", "b"])
    levels = ["a", "b", "c"]

    expected = nns_stack_factor_predictor(
        x.tolist(),
        y.tolist(),
        point.tolist(),
        levels=levels,
        cv_size=0.25,
        folds=1,
        method=[2],
        order=None,
        stack=True,
        dim_red_method="cor",
    )
    actual = nns_stack(
        x,
        y,
        point,
        factor_levels=levels,
        cv_size=0.25,
        folds=1,
        method=2,
        stack=True,
        dim_red_method="cor",
    )

    _assert_stack_matches(actual, expected, exact_probability_threshold=False)


@pytest.mark.parity
def test_nns_stack_factor_predictor_method12_factor_only_matches_r_fallback() -> None:
    x = np.asarray(["b", "a", "b", "c", "a", "c", "b", "a"])
    y = np.asarray([2.0, 1.0, 3.0, 4.0, 1.5, 3.5, 2.5, 1.25])
    point = np.asarray(["a", "c", "b"])
    levels = ["a", "b", "c"]

    expected = nns_stack_factor_predictor(
        x.tolist(),
        y.tolist(),
        point.tolist(),
        levels=levels,
        cv_size=0.25,
        folds=1,
        method=[1, 2],
        order=None,
        stack=True,
        dim_red_method="cor",
    )
    actual = nns_stack(
        x,
        y,
        point,
        factor_levels=levels,
        cv_size=0.25,
        folds=1,
        method=(1, 2),
        stack=True,
        dim_red_method="cor",
    )

    _assert_stack_matches(actual, expected, exact_probability_threshold=False)


@pytest.mark.parity
def test_nns_stack_mixed_factor_predictor_method2_matches_r() -> None:
    x = np.asarray(["b", "a", "b", "c", "a", "c", "b", "a"], dtype=object)
    z = np.arange(1, x.size + 1, dtype=np.float64) / 10.0
    variable = np.column_stack((x, z.astype(object)))
    y = np.asarray([2.0, 1.0, 3.0, 4.0, 1.5, 3.5, 2.5, 1.25])
    point_factor = np.asarray(["a", "c", "b"], dtype=object)
    point_z = np.asarray([0.15, 0.55, 0.75], dtype=object)
    point = np.column_stack((point_factor, point_z))
    levels = ["a", "b", "c"]

    expected = nns_stack_mixed_factor_predictor(
        x.tolist(),
        z.tolist(),
        y.tolist(),
        point_factor.tolist(),
        [0.15, 0.55, 0.75],
        levels=levels,
        cv_size=0.25,
        folds=1,
        method=[2],
        order=None,
        stack=True,
        dim_red_method="cor",
    )
    actual = nns_stack(
        variable,
        y,
        point,
        factor_levels=(levels, None),
        cv_size=0.25,
        folds=1,
        method=2,
        stack=True,
        dim_red_method="cor",
    )

    _assert_stack_matches(actual, expected, exact_probability_threshold=False)


@pytest.mark.parity
def test_nns_stack_mixed_factor_predictor_method12_matches_r() -> None:
    x = np.asarray(["b", "a", "b", "c", "a", "c", "b", "a"], dtype=object)
    z = np.arange(1, x.size + 1, dtype=np.float64) / 10.0
    variable = np.column_stack((x, z.astype(object)))
    y = np.asarray([2.0, 1.0, 3.0, 4.0, 1.5, 3.5, 2.5, 1.25])
    point_factor = np.asarray(["a", "c", "b"], dtype=object)
    point_z = np.asarray([0.15, 0.55, 0.75], dtype=object)
    point = np.column_stack((point_factor, point_z))
    levels = ["a", "b", "c"]

    expected = nns_stack_mixed_factor_predictor(
        x.tolist(),
        z.tolist(),
        y.tolist(),
        point_factor.tolist(),
        [0.15, 0.55, 0.75],
        levels=levels,
        cv_size=0.25,
        folds=1,
        method=[1, 2],
        order=None,
        stack=True,
        dim_red_method="cor",
    )
    actual = nns_stack(
        variable,
        y,
        point,
        factor_levels=(levels, None),
        cv_size=0.25,
        folds=1,
        method=(1, 2),
        stack=True,
        dim_red_method="cor",
    )

    _assert_stack_matches(actual, expected, exact_probability_threshold=False)


@pytest.mark.parity
@pytest.mark.parametrize(
    ("method", "ts_test"),
    [([1], 5), ([1], 10), ([2], 5), ([2], 10), ([1, 2], 10)],
)
def test_nns_stack_ts_test_matches_r(method: list[int], ts_test: int) -> None:
    x = np.linspace(-2.0, 2.0, 40)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = x + np.sin(x) + 0.25 * np.cos(x)
    point = variable[:5]

    expected = nns_stack_numeric(
        variable.tolist(),
        y.tolist(),
        point.tolist(),
        cv_size=0.25,
        folds=1,
        method=method,
        order=None,
        stack=True,
        dim_red_method="cor",
        ts_test=ts_test,
    )
    actual = nns_stack(
        variable,
        y,
        point,
        cv_size=0.25,
        folds=1,
        method=method,
        stack=True,
        dim_red_method="cor",
        ts_test=ts_test,
    )

    _assert_stack_matches(actual, expected, exact_probability_threshold=False)


@pytest.mark.parity
def test_nns_stack_var_like_ts_test_matches_r() -> None:
    h = 5
    x = np.linspace(-2.0, 2.0, 40)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = x + np.sin(x) + 0.25 * np.cos(x)
    point = variable[-h:]
    ts_test = max(2 * h, int(0.2 * y.size))

    expected = nns_stack_numeric(
        variable.tolist(),
        y.tolist(),
        point.tolist(),
        cv_size=0.25,
        folds=1,
        method=[1, 2],
        order=None,
        stack=True,
        dim_red_method="cor",
        ts_test=ts_test,
    )
    actual = nns_stack(
        variable,
        y,
        point,
        cv_size=0.25,
        folds=1,
        method=(1, 2),
        stack=True,
        dim_red_method="cor",
        ts_test=ts_test,
    )

    _assert_stack_matches(actual, expected, exact_probability_threshold=False)


@pytest.mark.parity
@pytest.mark.parametrize("method", [[1], [2], [1, 2]])
def test_nns_stack_pred_int_matches_r(method: list[int]) -> None:
    x = np.linspace(-2.0, 2.0, 40)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = x + np.sin(x) + 0.25 * np.cos(x)
    point = variable[:5]

    expected = nns_stack_numeric(
        variable.tolist(),
        y.tolist(),
        point.tolist(),
        cv_size=0.25,
        folds=1,
        method=method,
        order=None,
        stack=True,
        dim_red_method="cor",
        pred_int=0.95,
    )
    actual = nns_stack(
        variable,
        y,
        point,
        cv_size=0.25,
        folds=1,
        method=method,
        stack=True,
        dim_red_method="cor",
        pred_int=0.95,
    )

    _assert_stack_matches(actual, expected, exact_probability_threshold=False)


@pytest.mark.parity
@pytest.mark.parametrize("method", [[1], [2], [1, 2]])
def test_nns_stack_binary_class_matches_r(method: list[int]) -> None:
    x = np.linspace(-2.0, 2.0, 36)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = np.where(x + np.sin(x) > 0.0, 2.0, 1.0)
    point = variable[::9]

    expected = nns_stack_numeric(
        variable.tolist(),
        y.tolist(),
        point.tolist(),
        cv_size=0.25,
        folds=1,
        method=method,
        order=None,
        stack=True,
        dim_red_method="cor",
        type="class",
    )
    actual = nns_stack(
        variable,
        y,
        point,
        cv_size=0.25,
        folds=1,
        method=method,
        stack=True,
        dim_red_method="cor",
        type="class",
    )

    _assert_stack_matches(actual, expected, exact_probability_threshold=False)


@pytest.mark.parity
@pytest.mark.parametrize("method", [[1], [2], [1, 2]])
def test_nns_stack_binary_class_pred_int_matches_r(method: list[int]) -> None:
    x = np.linspace(-2.0, 2.0, 36)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = np.where(x + np.sin(x) > 0.0, 2.0, 1.0)
    point = variable[::9]

    expected = nns_stack_numeric(
        variable.tolist(),
        y.tolist(),
        point.tolist(),
        cv_size=0.25,
        folds=1,
        method=method,
        order=None,
        stack=True,
        dim_red_method="cor",
        type="class",
        pred_int=0.95,
    )
    actual = nns_stack(
        variable,
        y,
        point,
        cv_size=0.25,
        folds=1,
        method=method,
        stack=True,
        dim_red_method="cor",
        type="class",
        pred_int=0.95,
    )

    _assert_stack_matches(actual, expected, exact_probability_threshold=False)
    assert isinstance(actual["pred.int"], dict)
    assert all(values.shape == actual["stack"].shape for values in actual["pred.int"].values())
    if method == [1, 2]:
        assert set(actual["pred.int"]) == set(actual["reg.pred.int"])
        for values in actual["pred.int"].values():
            np.testing.assert_allclose(values, np.round(values))


@pytest.mark.parity
@pytest.mark.parametrize("method", [[1], [2], [1, 2]])
def test_nns_stack_multiclass_matches_r(method: list[int]) -> None:
    x = np.linspace(-2.0, 2.0, 36)
    variable = np.column_stack((x, x**2, np.sin(x)))
    y = np.where(x < -0.5, 1.0, np.where(x > 0.75, 3.0, 2.0))
    point = variable[[0, 7, 18, 31]]

    expected = nns_stack_numeric(
        variable.tolist(),
        y.tolist(),
        point.tolist(),
        cv_size=0.25,
        folds=1,
        method=method,
        order=1,
        stack=True,
        dim_red_method="cor",
        type="class",
    )
    actual = nns_stack(
        variable,
        y,
        point,
        cv_size=0.25,
        folds=1,
        method=method,
        order=1,
        stack=True,
        dim_red_method="cor",
        type="class",
    )

    _assert_stack_matches(actual, expected, exact_probability_threshold=False)


@pytest.mark.parity
def test_nns_stack_factor_like_class_pred_int_matches_r() -> None:
    x = np.linspace(-2.0, 2.0, 30)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    labels = np.where(x < -0.5, "A", np.where(x > 0.75, "C", "B"))
    point = variable[::10]

    expected = nns_stack_numeric(
        variable.tolist(),
        labels.tolist(),
        point.tolist(),
        cv_size=0.25,
        folds=1,
        method=[1, 2],
        order=1,
        stack=True,
        dim_red_method="cor",
        type="class",
        class_levels=["A", "B", "C"],
        pred_int=0.95,
    )
    actual = nns_stack(
        variable,
        labels,
        point,
        cv_size=0.25,
        folds=1,
        method=(1, 2),
        order=1,
        stack=True,
        dim_red_method="cor",
        type="class",
        class_levels=["A", "B", "C"],
        pred_int=0.95,
    )

    _assert_stack_matches(actual, expected, exact_probability_threshold=False)


@pytest.mark.parity
def test_nns_stack_factor_like_class_matches_r() -> None:
    x = np.linspace(-2.0, 2.0, 30)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    labels = np.where(x < -0.5, "A", np.where(x > 0.75, "C", "B"))
    point = variable[::10]

    expected = nns_stack_numeric(
        variable.tolist(),
        labels.tolist(),
        point.tolist(),
        cv_size=0.25,
        folds=1,
        method=[1, 2],
        order=1,
        stack=True,
        dim_red_method="cor",
        type="class",
        class_levels=["A", "B", "C"],
    )
    actual = nns_stack(
        variable,
        labels,
        point,
        cv_size=0.25,
        folds=1,
        method=(1, 2),
        order=1,
        stack=True,
        dim_red_method="cor",
        type="class",
        class_levels=["A", "B", "C"],
    )

    _assert_stack_matches(actual, expected, exact_probability_threshold=False)


def test_nns_stack_raw_character_class_raises() -> None:
    x = np.linspace(-2.0, 2.0, 20)
    variable = np.column_stack((x, np.sin(x)))
    labels = np.where(x > 0.0, "B", "A")

    with pytest.raises(ValueError, match="class_levels"):
        nns_stack(variable, labels, variable[:3], type="class", cv_size=0.25, folds=1)


@pytest.mark.parity
@pytest.mark.stochastic
@pytest.mark.parametrize("method", [[1], [2], [1, 2]])
def test_nns_stack_balance_binary_class_matches_r_structure(method: list[int]) -> None:
    x = np.linspace(-2.0, 2.0, 48)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = np.where(x < 1.0, 1.0, 2.0)
    point = variable[[2, 12, 28, 42]]

    expected = nns_stack_numeric(
        variable.tolist(),
        y.tolist(),
        point.tolist(),
        cv_size=0.25,
        folds=1,
        method=method,
        order=None,
        stack=True,
        dim_red_method="cor",
        type="class",
        balance=True,
        seed=42,
    )
    actual = nns_stack(
        variable,
        y,
        point,
        cv_size=0.25,
        folds=1,
        method=method,
        stack=True,
        dim_red_method="cor",
        type="class",
        balance=True,
        random_seed=42,
    )

    _assert_stack_class_structure(actual, expected, point_rows=point.shape[0], classes=np.unique(y))


@pytest.mark.parity
@pytest.mark.stochastic
def test_nns_stack_balance_multiclass_and_factor_structure() -> None:
    x = np.linspace(-2.0, 2.0, 45)
    variable = np.column_stack((x, x**2, np.sin(x)))
    labels = np.where(x < -0.75, "A", np.where(x > 1.0, "C", "B"))
    point = variable[[0, 11, 30, 44]]

    expected = nns_stack_numeric(
        variable.tolist(),
        labels.tolist(),
        point.tolist(),
        cv_size=0.25,
        folds=1,
        method=[1, 2],
        order=1,
        stack=True,
        dim_red_method="cor",
        type="class",
        class_levels=["A", "B", "C"],
        balance=True,
        seed=7,
    )
    actual = nns_stack(
        variable,
        labels,
        point,
        cv_size=0.25,
        folds=1,
        method=(1, 2),
        order=1,
        stack=True,
        dim_red_method="cor",
        type="class",
        class_levels=["A", "B", "C"],
        balance=True,
        random_seed=7,
    )

    _assert_stack_class_structure(
        actual,
        expected,
        point_rows=point.shape[0],
        classes=np.array([1.0, 2.0, 3.0]),
    )


@pytest.mark.parity
@pytest.mark.stochastic
def test_nns_stack_balance_class_pred_int_matches_r_structure() -> None:
    x = np.linspace(-2.0, 2.0, 48)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = np.where(x < 1.0, 1.0, 2.0)
    point = variable[[2, 12, 28, 42]]

    expected = nns_stack_numeric(
        variable.tolist(),
        y.tolist(),
        point.tolist(),
        cv_size=0.25,
        folds=1,
        method=[1],
        order=None,
        stack=True,
        dim_red_method="cor",
        type="class",
        balance=True,
        seed=42,
        pred_int=0.95,
    )
    actual = nns_stack(
        variable,
        y,
        point,
        cv_size=0.25,
        folds=1,
        method=(1,),
        stack=True,
        dim_red_method="cor",
        type="class",
        balance=True,
        random_seed=42,
        pred_int=0.95,
    )

    _assert_stack_class_structure(
        actual,
        expected,
        point_rows=point.shape[0],
        classes=np.unique(y),
        expect_pred_int=True,
    )


@pytest.mark.parity
@pytest.mark.stochastic
def test_nns_stack_balance_type_none_forces_class_path() -> None:
    x = np.linspace(-2.0, 2.0, 40)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = np.where(x < 1.25, 1.0, 2.0)
    point = variable[:5]

    expected = nns_stack_numeric(
        variable.tolist(),
        y.tolist(),
        point.tolist(),
        cv_size=0.25,
        folds=1,
        method=[1],
        order=None,
        stack=True,
        dim_red_method="cor",
        type=None,
        balance=True,
        seed=9,
    )
    actual = nns_stack(
        variable,
        y,
        point,
        cv_size=0.25,
        folds=1,
        method=1,
        balance=True,
        random_seed=9,
    )

    _assert_stack_class_structure(
        actual,
        expected,
        point_rows=point.shape[0],
        classes=np.array([1.0, 2.0]),
    )


def test_nns_stack_balance_raw_character_class_raises() -> None:
    x = np.linspace(-2.0, 2.0, 20)
    variable = np.column_stack((x, np.sin(x)))
    labels = np.where(x > 0.0, "B", "A")

    with pytest.raises(ValueError, match="levels"):
        nns_stack(
            variable,
            labels,
            variable[:3],
            type="class",
            cv_size=0.25,
            folds=1,
            balance=True,
            random_seed=1,
        )


def _assert_stack_matches(
    actual: dict[str, Any],
    expected: Any,
    *,
    exact_probability_threshold: bool = True,
) -> None:
    assert isinstance(expected, dict)
    assert set(actual) == set(expected)
    for key in actual:
        if key == "probability.threshold" and not exact_probability_threshold:
            assert np.isfinite(float(actual[key]))
            assert 0.0 <= float(actual[key]) <= 1.0
            assert np.isfinite(float(_numeric(expected[key])))
            continue
        if actual[key] is None:
            assert expected[key] is None or expected[key] == {}
        elif isinstance(actual[key], dict):
            assert isinstance(expected[key], dict)
            assert set(actual[key]) == set(expected[key])
            for column, values in actual[key].items():
                np.testing.assert_allclose(
                    np.asarray(values, dtype=np.float64),
                    _numeric(expected[key][column]),
                    atol=COMPOUND,
                )
        else:
            np.testing.assert_allclose(
                np.asarray(actual[key], dtype=np.float64),
                _numeric(expected[key]),
                atol=COMPOUND,
            )


def _numeric(value: object) -> np.ndarray:
    if isinstance(value, str):
        if value == "NA":
            return np.asarray(np.nan, dtype=np.float64)
        if value == "Inf":
            return np.asarray(np.inf, dtype=np.float64)
        if value == "-Inf":
            return np.asarray(-np.inf, dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def _assert_stack_class_structure(
    actual: dict[str, Any],
    expected: Any,
    *,
    point_rows: int,
    classes: np.ndarray,
    expect_pred_int: bool = False,
) -> None:
    assert isinstance(expected, dict)
    assert set(actual) == set(expected)
    for key in ("reg", "dim.red", "stack"):
        actual_values = np.asarray(actual[key], dtype=np.float64)
        expected_values = _numeric(expected[key])
        if expected_values.shape == ():
            assert actual_values.shape == (point_rows,)
            assert np.all(np.isnan(actual_values))
            assert np.isnan(float(expected_values))
            continue
        assert actual_values.shape == expected_values.shape
        if actual_values.ndim > 0:
            assert actual_values.shape == (point_rows,)
            finite_actual = actual_values[np.isfinite(actual_values)]
            assert np.all(np.isin(finite_actual, classes))
            finite_expected = expected_values[np.isfinite(expected_values)]
            assert np.all(np.isin(finite_expected, classes))
    assert np.isfinite(float(actual["probability.threshold"]))
    assert np.isfinite(float(_numeric(expected["probability.threshold"])))
    for key in ("reg.pred.int", "dim.red.pred.int", "pred.int"):
        actual_pred_int = actual[key]
        expected_pred_int = expected[key]
        if not expect_pred_int or actual_pred_int is None:
            assert actual_pred_int is None
            assert expected_pred_int is None
            continue
        assert isinstance(actual_pred_int, dict)
        assert isinstance(expected_pred_int, dict)
        assert set(actual_pred_int) == set(expected_pred_int)
        for values in actual_pred_int.values():
            assert values.shape == (point_rows,)
            assert np.all(np.isfinite(values))
