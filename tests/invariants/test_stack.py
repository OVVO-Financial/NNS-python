from __future__ import annotations

import numpy as np
import pytest

from nns import nns_stack


def test_nns_stack_numeric_shapes_and_keys() -> None:
    x = np.linspace(-2.0, 2.0, 40)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = x + np.sin(x)
    point = variable[:7]

    result = nns_stack(variable, y, point, cv_size=0.25, folds=2, method=(1, 2))

    assert set(result) == {
        "OBJfn.reg",
        "NNS.reg.n.best",
        "probability.threshold",
        "OBJfn.dim.red",
        "NNS.dim.red.threshold",
        "reg",
        "reg.pred.int",
        "dim.red",
        "dim.red.pred.int",
        "stack",
        "pred.int",
        "weights",
        "class.levels",
    }
    assert result["reg"].shape == (7,)
    assert result["dim.red"].shape == (7,)
    assert result["stack"].shape == (7,)
    assert result["probability.threshold"] == pytest.approx(0.5)
    assert np.all(np.isfinite(result["stack"]))




def test_nns_stack_pred_int_falls_back_to_point_estimate_when_regression_drops_rows() -> None:
    x = np.array(
        [
            [6.0, 6.0, 6.0],
            [6.0, 1.0, 6.0],
            [6.0, 13.0, 6.0],
            [1.0, 6.0, 6.0],
            [6.0, 6.0, 6.0],
            [6.0, 6.0, 6.0],
            [6.0, 0.0, 6.0],
            [6.0, 6.0, 6.0],
            [6.0, 6.0, 0.0],
            [6.0, 6.0, 6.0],
            [6.0, 6.0, 6.0],
            [6.0, 6.0, 6.0],
            [6.0, 6.0, 6.0],
            [6.0, 6.0, 6.0],
            [6.0, 6.0, 6.0],
            [0.0, 0.5, 6.0],
            [6.0, 6.0, 6.0],
            [6.0, 0.0, 6.0],
            [6.0, 6.0, 6.0],
            [6.0, 6.0, 6.0],
            [6.0, 6.0, 6.0],
            [6.0, 6.0, 6.0],
            [6.0, 6.0, 6.0],
        ],
        dtype=np.float64,
    )
    y = 0.5 * x[:, 0] - 0.25 * x[:, 1]

    result = nns_stack(x, y, x[:3], cv_size=0.25, folds=1, method=(1, 2), pred_int=0.95)

    assert result["pred.int"] is not None
    assert all(values.shape == (3,) for values in result["pred.int"].values())


def test_nns_stack_classification_shapes_and_codes() -> None:
    x = np.linspace(-2.0, 2.0, 30)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = np.where(x < -0.5, 1.0, np.where(x > 0.75, 3.0, 2.0))
    point = variable[:6]

    first = nns_stack(variable, y, point, type="class", cv_size=0.25, folds=1, method=(1, 2))
    second = nns_stack(variable, y, point, type="class", cv_size=0.25, folds=1, method=(1, 2))

    assert first["stack"].shape == (6,)
    assert np.all(np.isin(first["stack"], np.unique(y)))
    np.testing.assert_allclose(first["stack"], second["stack"])
    assert first["pred.int"] is None


def test_nns_stack_factor_predictor_expands_train_and_test() -> None:
    x = np.asarray(["b", "a", "b", "c", "a", "c", "b", "a"])
    y = np.asarray([2.0, 1.0, 3.0, 4.0, 1.5, 3.5, 2.5, 1.25])

    result = nns_stack(
        x,
        y,
        np.asarray(["a", "c", "b"]),
        factor_levels=["a", "b", "c"],
        cv_size=0.25,
        folds=1,
        method=1,
    )

    assert result["reg"].shape == (3,)
    assert result["stack"].shape == (3,)
    assert np.all(np.isfinite(result["stack"]))


def test_nns_stack_factor_predictor_method2_factor_only_falls_back_to_method1() -> None:
    x = np.asarray(["b", "a", "b", "c", "a", "c", "b", "a"])
    y = np.asarray([2.0, 1.0, 3.0, 4.0, 1.5, 3.5, 2.5, 1.25])

    result = nns_stack(
        x,
        y,
        np.asarray(["a", "c", "b"]),
        factor_levels=["a", "b", "c"],
        cv_size=0.25,
        folds=1,
        method=2,
    )

    assert result["reg"].shape == (3,)
    # Method 2 fell back to Method 1, so dim.red is absent (R NA -> port None).
    assert result["dim.red"] is None
    np.testing.assert_allclose(result["stack"], result["reg"])


def test_nns_stack_factor_predictor_method12_factor_only_falls_back_to_method1() -> None:
    x = np.asarray(["b", "a", "b", "c", "a", "c", "b", "a"])
    y = np.asarray([2.0, 1.0, 3.0, 4.0, 1.5, 3.5, 2.5, 1.25])

    result = nns_stack(
        x,
        y,
        np.asarray(["a", "c", "b"]),
        factor_levels=["a", "b", "c"],
        cv_size=0.25,
        folds=1,
        method=(1, 2),
    )

    assert result["reg"].shape == (3,)
    # Method 2 fell back to Method 1, so dim.red is absent (R NA -> port None).
    assert result["dim.red"] is None
    np.testing.assert_allclose(result["stack"], result["reg"])


def test_nns_stack_mixed_factor_predictor_method12_remains_deferred() -> None:
    x = np.asarray(["b", "a", "b", "c", "a", "c", "b", "a"], dtype=object)
    numeric = np.linspace(-1.0, 1.0, x.size)
    y = numeric + np.where(x == "a", 0.0, np.where(x == "b", 0.5, 1.0))
    variable = np.column_stack((x, numeric.astype(object)))
    point = variable[:3]

    result = nns_stack(
        variable,
        y,
        point,
        factor_levels=(["a", "b", "c"], None),
        cv_size=0.25,
        folds=1,
        method=(1, 2),
    )

    assert result["stack"].shape == (point.shape[0],)
    assert np.all(np.isfinite(result["stack"]))


def test_nns_stack_mixed_factor_predictor_class_method12_supported() -> None:
    x = np.asarray(["b", "a", "b", "c", "a", "c", "b", "a"], dtype=object)
    numeric = np.linspace(-1.0, 1.0, x.size)
    y = np.where(x == "a", 1.0, np.where(x == "b", 2.0, 3.0))
    variable = np.column_stack((x, numeric.astype(object)))
    point = variable[:3]

    result = nns_stack(
        variable,
        y,
        point,
        factor_levels=(["a", "b", "c"], None),
        cv_size=0.25,
        folds=1,
        method=(1, 2),
        type="class",
    )

    assert result["stack"].shape == (point.shape[0],)
    assert np.all(np.isin(result["stack"], np.unique(y)))


@pytest.mark.stochastic
def test_nns_stack_mixed_factor_predictor_class_balance_method12_supported() -> None:
    x = np.asarray(["b", "a", "b", "c", "a", "c", "b", "a"], dtype=object)
    numeric = np.linspace(-1.0, 1.0, x.size)
    y = np.where(x == "a", 1.0, np.where(x == "b", 2.0, 3.0))
    variable = np.column_stack((x, numeric.astype(object)))
    point = variable[:3]

    result = nns_stack(
        variable,
        y,
        point,
        factor_levels=(["a", "b", "c"], None),
        cv_size=0.25,
        folds=1,
        method=(1, 2),
        type="class",
        balance=True,
        random_seed=13,
    )

    assert result["stack"].shape == (point.shape[0],)
    assert np.all(np.isin(result["stack"], np.unique(y)))


def test_nns_stack_class_pred_int_shapes_and_rounding() -> None:
    x = np.linspace(-2.0, 2.0, 20)
    variable = np.column_stack((x, np.sin(x)))
    y = np.where(x > 0.0, 2.0, 1.0)

    single = nns_stack(variable, y, variable[:5], type="class", method=1, pred_int=0.95)
    combined = nns_stack(variable, y, variable[:5], type="class", method=(1, 2), pred_int=0.95)

    assert single["pred.int"] is not None
    assert set(single["pred.int"]) == {"pred.int.neg", "pred.int.pos"}
    assert all(values.shape == (5,) for values in single["pred.int"].values())
    assert combined["pred.int"] is not None
    assert all(values.shape == (5,) for values in combined["pred.int"].values())
    for values in combined["pred.int"].values():
        np.testing.assert_allclose(values, np.round(values))


def test_nns_stack_mixed_factor_predictor_pred_int_shapes() -> None:
    x = np.asarray(["b", "a", "b", "c", "a", "c", "b", "a"], dtype=object)
    numeric = np.linspace(-1.0, 1.0, x.size)
    y = numeric + np.where(x == "a", 0.0, np.where(x == "b", 0.5, 1.0))
    variable = np.column_stack((x, numeric.astype(object)))
    point = variable[:3]

    result = nns_stack(
        variable,
        y,
        point,
        factor_levels=(["a", "b", "c"], None),
        cv_size=0.25,
        folds=1,
        method=(1, 2),
        pred_int=0.95,
    )

    assert result["pred.int"] is not None
    assert result["reg.pred.int"] is not None
    assert result["dim.red.pred.int"] is not None
    assert all(values.shape == point.shape[:1] for values in result["pred.int"].values())


@pytest.mark.stochastic
def test_nns_stack_balance_shape_codes_and_seed_determinism() -> None:
    x = np.linspace(-2.0, 2.0, 40)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = np.where(x < 1.0, 1.0, 2.0)
    point = variable[:6]

    first = nns_stack(
        variable,
        y,
        point,
        cv_size=0.25,
        folds=1,
        method=(1, 2),
        type="class",
        balance=True,
        random_seed=11,
    )
    second = nns_stack(
        variable,
        y,
        point,
        cv_size=0.25,
        folds=1,
        method=(1, 2),
        type="class",
        balance=True,
        random_seed=11,
    )

    assert first["stack"].shape == (6,)
    assert np.all(np.isin(first["stack"], np.unique(y)))
    np.testing.assert_allclose(first["stack"], second["stack"])
    assert first["pred.int"] is None


@pytest.mark.parametrize("method", [(1,), (2,), (1, 2)])
def test_nns_stack_pred_int_shapes(method: tuple[int, ...]) -> None:
    x = np.linspace(-2.0, 2.0, 40)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = x + np.sin(x)
    point = variable[:7]

    result = nns_stack(variable, y, point, cv_size=0.25, folds=1, method=method, pred_int=0.95)

    assert result["stack"].shape == (7,)
    assert result["pred.int"] is not None
    assert all(values.shape == (7,) for values in result["pred.int"].values())


@pytest.mark.parametrize("method", [(1,), (2,), (1, 2)])
def test_nns_stack_ts_test_shape_and_determinism(method: tuple[int, ...]) -> None:
    x = np.linspace(-2.0, 2.0, 40)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = x + np.sin(x)
    point = variable[:7]

    first = nns_stack(variable, y, point, cv_size=0.25, folds=1, method=method, ts_test=10)
    second = nns_stack(variable, y, point, cv_size=0.25, folds=1, method=method, ts_test=10)

    assert first["stack"].shape == (7,)
    assert set(first) == set(second)
    np.testing.assert_allclose(first["stack"], second["stack"])


@pytest.mark.parametrize("ts_test", [0, 41])
def test_nns_stack_invalid_ts_test_raises(ts_test: int) -> None:
    x = np.linspace(-2.0, 2.0, 40)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = x + np.sin(x)

    with pytest.raises(ValueError):
        nns_stack(variable, y, variable[:3], cv_size=0.25, folds=1, method=1, ts_test=ts_test)
