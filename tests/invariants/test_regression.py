from __future__ import annotations

import warnings
from decimal import Decimal

import numpy as np
import pytest

from nns import nns_reg, prepare_factor_predictors


def test_nns_reg_shapes_and_bounds() -> None:
    x = np.linspace(-2.0, 2.0, 100)
    y = np.sin(x)

    result = nns_reg(x, y, order=3, point_est=np.array([-3.0, 0.0, 3.0]))

    assert 0.0 <= result["R2"] <= 1.0
    assert result["SE"] >= 0.0
    assert result["Fitted.xy"]["x"].shape == x.shape
    assert result["Fitted.xy"]["y.hat"].shape == x.shape
    assert result["Point.est"].shape == (3,)
    assert result["derivative"]["Coefficient"].size == result["regression.points"]["x"].size - 1


def test_nns_reg_class_predicts_one_of_a_cells_tied_classes() -> None:
    # The cell x = 1:4 holds classes 1, 1, 3, 3; the mean of the tied modes made it class 2.
    x = np.arange(1.0, 9.0)
    y = np.array([1.0, 1.0, 3.0, 3.0, 2.0, 2.0, 2.0, 2.0])
    result = nns_reg(x, y, type="CLASS", order=2)
    assert set(result["Fitted.xy"]["y.hat"][:4].tolist()) <= {1.0, 3.0}
    np.testing.assert_array_equal(result["Fitted.xy"]["y.hat"], [1, 1, 1, 1, 2, 2, 2, 2])


def test_nns_reg_degenerate_run_derivative_without_warning() -> None:
    from nns._reg_engine import _derivative

    rp = {
        "x": np.array([0.0, 1e-320], dtype=np.float64),
        "y": np.array([0.0, 1.0], dtype=np.float64),
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        result = _derivative(rp)

    # R computes diff(y) / diff(x) silently; a subnormal run overflows to Inf
    # without raising, and an exactly-zero run yields a 0 coefficient.
    assert result["Coefficient"].shape == (1,)
    assert not np.isnan(result["Coefficient"][0])
    assert [w for w in caught if issubclass(w.category, RuntimeWarning)] == []


def test_nns_reg_order_max_is_perfect_fit() -> None:
    x = np.linspace(-2.0, 2.0, 50)
    y = x**2

    result = nns_reg(x, y, order="max")

    np.testing.assert_allclose(result["Fitted.xy"]["y.hat"], y, atol=1e-12)
    assert result["R2"] == pytest.approx(1.0)
    assert result["SE"] == pytest.approx(0.0)


def test_nns_reg_increasing_order_does_not_reduce_r2_for_smooth_curve() -> None:
    x = np.linspace(-2.0, 2.0, 200)
    y = np.sin(x)

    r1 = nns_reg(x, y, order=1)["R2"]
    r2 = nns_reg(x, y, order=2)["R2"]
    r3 = nns_reg(x, y, order=3)["R2"]

    assert r2 >= r1 - 1e-12
    assert r3 >= r2 - 1e-12


def test_nns_reg_dim_red_shapes_and_equation() -> None:
    x1 = np.linspace(-2.0, 2.0, 80)
    x = np.column_stack((x1, np.sin(x1), np.cos(x1)))
    y = x[:, 0] + x[:, 1] + 0.25 * x[:, 2]
    point_est = np.array([[0.0, 0.0, 1.0], [3.0, 0.0, 1.0]])

    result = nns_reg(x, y, dim_red_method="equal", point_est=point_est, point_only=True)

    # point_only abbreviates the output: diagnostics and fitted values are None.
    assert result["R2"] is None
    assert result["Fitted.xy"] is None
    assert result["x.star"]["x"].shape == y.shape
    assert len(result["equation"]["Variable"]) == x.shape[1] + 1
    assert np.asarray(result["equation"]["Coefficient"]).shape == (x.shape[1] + 1,)
    assert np.asarray(result["Point.est"]).shape == (2,)

def test_nns_reg_dim_red_multivariate_call_returns_regression_points() -> None:
    x1 = np.linspace(-2.0, 2.0, 30)
    x = np.column_stack((x1, np.sin(x1), np.cos(x1)))
    y = x[:, 0] + x[:, 1] + 0.25 * x[:, 2]

    result = nns_reg(x, y, dim_red_method="equal", multivariate_call=True)

    assert set(result) == {"x", "y"}
    assert result["x"].ndim == 1
    assert result["y"].ndim == 1
    assert result["x"].shape == result["y"].shape


def test_nns_reg_confidence_interval_shapes_and_row_drop() -> None:
    x = np.linspace(-2.0, 2.0, 50)
    y = np.sin(x)
    point_est = np.array([-3.0, -1.0, 0.0, 2.5])

    result = nns_reg(x, y, order=1, point_est=point_est, confidence_interval=0.95)

    assert result["Fitted.xy"]["conf.int.pos"].shape == x.shape
    assert result["Fitted.xy"]["conf.int.neg"].shape == x.shape
    assert result["Point.est"].shape == point_est.shape
    assert result["pred.int"] is not None
    assert set(result["pred.int"]) == {"pred.int.neg", "pred.int.pos"}
    # Repaired contract: one interval row per prediction point, including
    # points below the training range (previously dropped).
    assert result["pred.int"]["pred.int.neg"].shape == (4,)
    assert result["pred.int"]["pred.int.pos"].shape == (4,)
    assert np.all(result["pred.int"]["pred.int.neg"] <= result["pred.int"]["pred.int.pos"])

def test_nns_reg_confidence_interval_none_output_unchanged() -> None:
    x = np.linspace(-2.0, 2.0, 50)
    y = np.sin(x)

    result = nns_reg(x, y, order=1)

    assert "conf.int.pos" not in result["Fitted.xy"]
    assert "conf.int.neg" not in result["Fitted.xy"]
    assert result["pred.int"] is None


@pytest.mark.parametrize(
    "path",
    ["smooth", "smooth_confidence"],
)
def test_nns_reg_spline_eligible_smooth_paths_run(path: str) -> None:
    x = np.linspace(-2.0, 2.0, 20)
    y = np.sin(x)

    if path == "smooth":
        result = nns_reg(x, y, smooth=True)
    else:
        result = nns_reg(x, y, smooth=True, confidence_interval=0.95)

    assert result["Fitted.xy"]["y.hat"].shape == x.shape
    assert np.all(np.isfinite(result["Fitted.xy"]["y.hat"]))


def test_nns_reg_small_smooth_falls_back_to_piecewise_path() -> None:
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 1.0])
    point = np.array([1.5, 2.5])

    smoothed = nns_reg(x, y, point_est=point, smooth=True, confidence_interval=0.95)
    ordinary = nns_reg(x, y, point_est=point, confidence_interval=0.95)

    np.testing.assert_allclose(smoothed["Point.est"], ordinary["Point.est"])
    np.testing.assert_allclose(
        smoothed["regression.points"]["y"],
        ordinary["regression.points"]["y"],
    )
    assert smoothed["pred.int"] is not None


def test_nns_reg_order_max_smooth_applies_spline() -> None:
    x = np.linspace(-2.0, 2.0, 20)
    y = np.sin(x)
    point = np.array([-1.5, 0.0, 1.5])

    smoothed = nns_reg(x, y, order="max", point_est=point, smooth=True, confidence_interval=0.95)
    ordinary = nns_reg(x, y, order="max", point_est=point, confidence_interval=0.95)

    # The repaired engine applies the smoothing spline whenever there are at
    # least four regression points, order = "max" included; the regression
    # points themselves are unchanged.
    np.testing.assert_allclose(
        smoothed["regression.points"]["y"],
        ordinary["regression.points"]["y"],
    )
    assert np.all(np.isfinite(smoothed["Point.est"]))
    assert smoothed["pred.int"] is not None

@pytest.mark.parametrize(
    "kwargs",
    [
        {"smooth": True, "order": 2},
        {"smooth": True, "confidence_interval": 0.95},
    ],
)
def test_nns_reg_dimred_smooth_paths_run(kwargs: dict[str, object]) -> None:
    x1 = np.linspace(-2.0, 2.0, 20)
    x = np.column_stack((x1, np.sin(x1)))
    y = x[:, 0] + x[:, 1]

    result = nns_reg(x, y, dim_red_method="equal", **kwargs)

    assert result["Fitted.xy"]["y.hat"].shape == y.shape
    assert np.all(np.isfinite(result["Fitted.xy"]["y.hat"]))


def test_nns_reg_univariate_point_only_matches_regular_shape() -> None:
    x = np.linspace(-2.0, 2.0, 20)
    y = np.sin(x)

    result = nns_reg(x, y, point_only=True, point_est=np.array([-1.0, 0.0, 1.0]))

    assert result["Fitted.xy"] is None
    assert result["regression.points"]["x"].ndim == 1
    assert np.asarray(result["Point.est"]).shape == (3,)

def test_nns_reg_univariate_multi_column_point_est_rejected() -> None:
    x = np.linspace(-2.0, 2.0, 20)
    y = np.sin(x)

    # Repaired contract: a univariate model requires exactly one prediction
    # column; the silent column-major flattening was removed.
    with pytest.raises(ValueError, match="exactly"):
        nns_reg(x, y, point_est=np.array([[-1.0, 1.0], [0.0, 2.0]]))

def test_nns_reg_dimred_tau_validation() -> None:
    x1 = np.linspace(-2.0, 2.0, 30)
    x = np.column_stack((x1, np.sin(x1), np.cos(x1)))
    y = x[:, 0] + x[:, 1]

    ts_result = nns_reg(x, y, dim_red_method="NNS.caus", tau="ts")
    assert ts_result["x.star"]["x"].shape == y.shape

    # Repaired contract: tau must be NULL, 'cs', or 'ts'.
    with pytest.raises(ValueError, match="tau"):
        nns_reg(x, y, dim_red_method="NNS.caus", tau=3)

def test_nns_reg_classification_outputs_numeric_codes() -> None:
    x = np.linspace(0.0, 5.0, 6)
    y = np.array([1, 1, 1, 2, 2, 2], dtype=np.float64)

    result = nns_reg(x, y, type="CLASS", point_est=np.array([1.5, 4.5]))

    assert result["Prediction.Accuracy"] is not None
    assert set(result["Fitted.xy"]["y.hat"]).issubset(set(y))
    assert set(result["Point.est"]).issubset(set(y))


def test_nns_reg_class_confidence_interval_returns_nominal_prediction_sets_r14() -> None:
    x = np.linspace(0.0, 11.0, 12)
    y = np.array([1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2], dtype=np.float64)

    result = nns_reg(
        x,
        y,
        type="class",
        point_est=np.array([2.5, 6.5, 11.5]),
        confidence_interval=0.95,
    )

    # R14 Classification.R returns nominal class sets, not ordinal intervals.
    assert "conf.int.pos" not in result["Fitted.xy"]
    assert "conf.int.neg" not in result["Fitted.xy"]
    assert result["pred.int"] is None
    sets = result["Prediction.set"]
    assert len(sets) == 3
    assert all(len(row) > 0 and set(row).issubset(set(y)) for row in sets)
    point_probs = np.column_stack(list(result["Point.probabilities"].values()))
    np.testing.assert_allclose(np.sum(point_probs, axis=1), 1.0, atol=1e-12)
    for row, classes in zip(point_probs, sets, strict=True):
        included = [i for i, val in enumerate(result["class.values"]) if val in classes]
        assert float(np.sum(row[included])) >= 0.95 - 1e-12
    assert set(result["Point.est"]).issubset(set(y))


def test_nns_reg_raw_string_class_labels_supported() -> None:
    x = np.linspace(0.0, 5.0, 6)
    y = np.array(["A", "A", "A", "B", "B", "B"])

    # The repaired engine encodes categorical responses natively: predictions
    # are numeric codes 1..K with the label map in class.levels.
    result = nns_reg(x, y, type="class", point_est=np.array([0.5, 4.5]))
    assert result["class.levels"] == ["A", "B"]
    assert set(np.asarray(result["Point.est"]).tolist()).issubset({1.0, 2.0})

def test_nns_reg_factor_predictor_encoded_natively() -> None:
    x = np.array(["a", "b", "a", "b", "a", "b"])
    y = np.array([1.0, 2.0, 1.5, 2.5, 0.5, 1.8])

    # Training-fitted dummy encoding requires no factor_levels argument.
    result = nns_reg(x, y, factor_2_dummy=True, point_est=np.array(["a", "b"]))
    assert np.asarray(result["Point.est"]).shape == (2,)

def test_nns_reg_factor_predictor_expands_point_est_with_training_levels() -> None:
    x = np.array(["b", "a", "b", "c"])
    y = np.array([2.0, 1.0, 3.0, 4.0])

    result = nns_reg(
        x,
        y,
        factor_2_dummy=True,
        point_est=np.array(["a", "c"]),
    )

    rpm_columns = [key for key in result["RPM"] if key != "y.hat"]
    assert len(rpm_columns) == 3
    assert np.asarray(result["Point.est"]).shape == (2,)

    # Unseen prediction levels are rejected rather than silently encoded.
    with pytest.raises(ValueError, match="unseen"):
        nns_reg(x, y, factor_2_dummy=True, point_est=np.array(["d"]))

def test_nns_reg_factor_predictor_dimred_expands_before_projection() -> None:
    factor = np.array(["b", "a", "b", "c", "a", "c"], dtype=object)
    numeric = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=object)
    x = np.column_stack((factor, numeric))
    y = np.array([2.0, 1.0, 3.0, 4.0, 1.5, 4.5])

    result = nns_reg(
        x,
        y,
        factor_2_dummy=True,
        dim_red_method="equal",
        point_est=np.array([["a", 1.5], ["c", 3.5]], dtype=object),
    )

    # Three dummy columns plus the numeric column plus the denominator row.
    assert len(result["equation"]["Variable"]) == 5
    assert result["x.star"]["x"].shape == y.shape
    assert np.asarray(result["Point.est"]).shape == (2,)


def test_nns_reg_treats_decimal_predictors_as_numeric() -> None:
    x = np.array([Decimal(i) for i in range(1, 9)], dtype=object)
    y = 2.0 * np.arange(1.0, 9.0)

    actual = nns_reg(x, y, order="max", point_est=[2.5])["Point.est"]
    expected = nns_reg(x.astype(np.float64), y, order="max", point_est=[2.5])["Point.est"]

    np.testing.assert_array_equal(actual, expected)


def test_nns_reg_keeps_numeric_looking_strings_categorical() -> None:
    x = np.array([str(i) for i in range(1, 9)], dtype=object)
    y = 2.0 * np.arange(1.0, 9.0)

    with pytest.raises(ValueError, match="unseen level"):
        nns_reg(x, y, order="max", point_est=["2.5"])


def test_prepared_bool_factors_predict_like_direct_regression() -> None:
    x = np.tile([False, True], 50)
    y = np.arange(100.0)
    point = np.array([False, True])

    design = prepare_factor_predictors(x, point_est=point)
    direct = nns_reg(x, y, point_est=point)["Point.est"]
    # R14 flags a wholly categorical design and uses exact joint cells.
    # Pre-encoded numeric dummy columns must explicitly request order=max
    # to preserve the same cell model rather than a continuous partition.
    prepared = nns_reg(
        design.x, y, point_est=design.point_est, order="max"
    )["Point.est"]

    assert design.feature_names == ("x_FALSE.", "x_TRUE.")
    np.testing.assert_array_equal(prepared, direct)


def test_prepared_mixed_factors_predict_like_direct_regression() -> None:
    rng = np.random.default_rng(5)
    label = rng.choice(["a", "b", "c"], size=60)
    flag = rng.random(60) < 0.5
    score = np.round(rng.normal(size=60), 6)
    x = np.empty((60, 3), dtype=object)
    x[:, 0], x[:, 1], x[:, 2] = label, flag, score
    y = np.round(score + (label == "b") + 2.0 * flag, 6)
    point = np.array([["c", True, 0.25], ["a", False, -0.5]], dtype=object)

    design = prepare_factor_predictors(x, point_est=point)
    direct = nns_reg(x, y, point_est=point)["Point.est"]
    prepared = nns_reg(design.x, y, point_est=design.point_est)["Point.est"]

    assert design.feature_names == ("V1_a", "V1_b", "V1_c", "V2_FALSE.", "V2_TRUE.", "V3")
    np.testing.assert_array_equal(prepared, direct)


def test_prepared_names_rename_outputs_and_keep_point_columns_aligned() -> None:
    import pandas as pd

    x = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
    point = pd.DataFrame({"a": [2.5], "b": [25.0]})

    for names in (["b", "a"], ["left", "b"]):
        design = prepare_factor_predictors(x, point_est=point, names=names)
        assert design.feature_names == tuple(names)
        np.testing.assert_array_equal(design.x, x.to_numpy())
        np.testing.assert_array_equal(design.point_est, [[2.5, 25.0]])


def test_explicit_reg_overrides_binary_numeric_auto_classification_like_r14() -> None:
    x = np.arange(8.0)
    y = np.array([0., 0., 0., 1., 1., 1., 0., 1.])

    automatic = nns_reg(x, y)
    assert "Point.probabilities" in automatic  # automatic 0/1 nominal task

    forced = nns_reg(x, y, type="REG", order="max", point_est=[2.5])
    assert forced["class.levels"] is None
    assert "regression.points" in forced
    assert "Point.probabilities" not in forced
    assert forced["Prediction.Accuracy"] is None
    np.testing.assert_array_equal(forced["Fitted.xy"]["y"], y)

    for nonnumeric in (y.astype(bool), y.astype(str)):
        with pytest.raises(ValueError, match="requires a numeric response"):
            nns_reg(x, nonnumeric, type="REG")
