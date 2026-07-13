from __future__ import annotations

import warnings

import numpy as np
import pytest

from nns import nns_reg


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


def test_nns_reg_class_confidence_interval_outputs_rounded_pred_int_only() -> None:
    x = np.linspace(0.0, 11.0, 12)
    y = np.array([1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2], dtype=np.float64)

    result = nns_reg(
        x,
        y,
        type="class",
        point_est=np.array([2.5, 6.5, 11.5]),
        confidence_interval=0.95,
    )

    assert result["Fitted.xy"]["conf.int.pos"].shape == y.shape
    assert result["Fitted.xy"]["conf.int.neg"].shape == y.shape
    assert result["pred.int"] is not None
    assert set(result["pred.int"]) == {"pred.int.neg", "pred.int.pos"}
    for values in result["pred.int"].values():
        np.testing.assert_allclose(values, np.round(values))
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