from __future__ import annotations

import numpy as np
import pytest

from nns import nns_boost


def test_nns_boost_shapes_and_feature_weights() -> None:
    x = np.linspace(-2.0, 2.0, 30)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = x + np.sin(x)

    result = nns_boost(variable, y, variable[:6], cv_size=0.25, feature_importance=False)

    assert result["results"].shape == (6,)
    assert result["pred.int"] is None
    assert sum(result["feature.weights"].values()) == pytest.approx(1.0)
    assert len(result["feature.frequency"]) == len(result["feature.weights"])
    assert np.all(np.isfinite(result["results"]))


def test_nns_boost_class_shapes_and_codes() -> None:
    x = np.linspace(-2.0, 2.0, 30)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = np.where(x < -0.5, 1.0, np.where(x > 0.75, 3.0, 2.0))

    result = nns_boost(
        variable,
        y,
        variable[:6],
        cv_size=0.25,
        depth=1,
        type="class",
        feature_importance=False,
    )

    assert result["results"].shape == (6,)
    assert np.all(np.isin(result["results"], np.unique(y)))
    assert result["pred.int"] is None


def test_nns_boost_ts_test_shape_and_feature_weights() -> None:
    x = np.linspace(-2.0, 2.0, 20)
    variable = np.column_stack((x, np.sin(x)))
    y = x + np.sin(x)

    result = nns_boost(variable, y, variable[:5], ts_test=4, cv_size=0.25, feature_importance=False)

    assert result["results"].shape == (5,)
    assert result["pred.int"] is None
    assert sum(result["feature.weights"].values()) == pytest.approx(1.0)
    assert np.all(np.isfinite(result["results"]))


def test_nns_boost_factor_predictor_encodes_natively() -> None:
    # Factor predictors are one-hot encoded natively (no explicit levels needed).
    x = np.linspace(-2.0, 2.0, 20)
    labels = np.where(x > 0.0, "B", "A")
    # Object columns keep x numeric (a '<U' array would make every column a
    # factor, as a character matrix does in R).
    variable = np.column_stack((labels.astype(object), x.astype(object)))
    y = x + np.where(labels == "B", 1.0, 0.0)

    result = nns_boost(variable, y, variable[:3], cv_size=0.25, feature_importance=False)
    assert result["results"].shape == (3,)
    assert np.all(np.isfinite(result["results"]))


@pytest.mark.parametrize("features_only", [False, True])
def test_nns_boost_multiple_factor_predictors_are_positional(features_only: bool) -> None:
    x = np.linspace(-2.0, 2.0, 24)
    first = np.where(x < -0.5, "low", np.where(x > 0.75, "high", "mid"))
    second = np.where(np.sin(x) > 0.0, "up", "down")
    variable = np.column_stack((first.astype(object), x.astype(object), second.astype(object)))
    y = x + np.where(first == "low", 1.0, np.where(first == "mid", 2.0, 3.0)) * 0.25

    result = nns_boost(
        variable,
        y,
        variable[:4],
        cv_size=0.25,
        features_only=features_only,
        feature_importance=False,
        random_seed=1,
    )

    assert set(result) == (
        {"feature.weights", "feature.frequency"}
        if features_only
        else {
            "results",
            "pred.int",
            "feature.weights",
            "feature.frequency",
            "class.levels",
        }
    )
    assert sum(result["feature.weights"].values()) == pytest.approx(1.0)
    if not features_only:
        assert result["results"].shape == (4,)


def test_nns_boost_numeric_pred_int_shape() -> None:
    x = np.linspace(-2.0, 2.0, 20)
    variable = np.column_stack((x, np.sin(x)))
    y = x + np.sin(x)

    result = nns_boost(variable, y, variable[:5], pred_int=0.95, feature_importance=False)

    assert result["results"].shape == (5,)
    assert isinstance(result["pred.int"], dict)
    assert set(result["pred.int"]) == {"pred.int.neg", "pred.int.pos"}
    assert result["pred.int"]["pred.int.neg"].shape == result["results"].shape
    assert result["pred.int"]["pred.int.pos"].shape == result["results"].shape
    assert np.all(np.isfinite(result["pred.int"]["pred.int.neg"]))
    assert np.all(np.isfinite(result["pred.int"]["pred.int.pos"]))


def test_nns_boost_features_only_ignores_numeric_pred_int() -> None:
    x = np.linspace(-2.0, 2.0, 20)
    variable = np.column_stack((x, np.sin(x)))
    y = x + np.sin(x)

    result = nns_boost(
        variable,
        y,
        variable[:5],
        pred_int=0.95,
        features_only=True,
        feature_importance=False,
    )

    assert set(result) == {"feature.weights", "feature.frequency"}


def test_nns_boost_class_pred_int_shape() -> None:
    x = np.linspace(-2.0, 2.0, 20)
    variable = np.column_stack((x, np.sin(x)))
    y = np.where(x > 0.0, 2.0, 1.0)

    result = nns_boost(
        variable,
        y,
        variable[:5],
        type="class",
        pred_int=0.95,
        feature_importance=False,
    )

    assert result["results"].shape == (5,)
    assert isinstance(result["pred.int"], dict)
    assert set(result["pred.int"]) == {"pred.int.neg", "pred.int.pos"}
    assert result["pred.int"]["pred.int.neg"].shape == result["results"].shape
    assert result["pred.int"]["pred.int.pos"].shape == result["results"].shape


def test_nns_boost_stochastic_epoch_path_shape_and_seed_determinism() -> None:
    x = np.linspace(-2.0, 2.0, 20)
    variable = np.column_stack([np.sin((idx + 1) * x) for idx in range(11)])
    y = x + np.sin(x)

    first = nns_boost(
        variable,
        y,
        variable[:3],
        cv_size=0.25,
        learner_trials=5,
        epochs=5,
        random_seed=7,
        feature_importance=False,
    )
    second = nns_boost(
        variable,
        y,
        variable[:3],
        cv_size=0.25,
        learner_trials=5,
        epochs=5,
        random_seed=7,
        feature_importance=False,
    )

    assert first["results"].shape == (3,)
    assert first["pred.int"] is None
    assert sum(first["feature.weights"].values()) == pytest.approx(1.0)
    assert len(first["feature.frequency"]) >= 1
    assert np.all(np.isfinite(first["results"]))
    np.testing.assert_allclose(first["results"], second["results"])
    assert first["feature.frequency"] == second["feature.frequency"]


def test_nns_boost_stochastic_epoch_path_pred_int_shape() -> None:
    x = np.linspace(-2.0, 2.0, 20)
    variable = np.column_stack([np.sin((idx + 1) * x) for idx in range(11)])
    y = x + np.sin(x)

    result = nns_boost(
        variable,
        y,
        variable[:3],
        cv_size=0.25,
        learner_trials=5,
        epochs=5,
        pred_int=0.95,
        random_seed=8,
        feature_importance=False,
    )

    assert result["results"].shape == (3,)
    assert isinstance(result["pred.int"], dict)
    assert result["pred.int"]["pred.int.neg"].shape == result["results"].shape
    assert result["pred.int"]["pred.int.pos"].shape == result["results"].shape


def test_nns_boost_threshold_on_stochastic_epoch_path_matches_r() -> None:
    # [threshold] is a probability supplied to LPM.VaR over the learner-trial
    # objective distribution, as in R. threshold=1.0 selects the extreme
    # objective cutoff; the best trial always survives, so the run completes
    # instead of raising the old literal-cutoff error.
    x = np.linspace(-2.0, 2.0, 20)
    variable = np.column_stack([np.sin((idx + 1) * x) for idx in range(11)])
    y = x + np.sin(x)

    result = nns_boost(
        variable,
        y,
        variable[:3],
        cv_size=0.25,
        threshold=1.0,
        epochs=5,
        learner_trials=5,
        feature_importance=False,
    )

    assert result["results"].shape == (3,)
    assert np.all(np.isfinite(result["results"]))


@pytest.mark.stochastic
def test_nns_boost_balance_shape_codes_and_seed_determinism() -> None:
    x = np.linspace(-2.0, 2.0, 42)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = np.where(x < 1.0, 1.0, 2.0)

    first = nns_boost(
        variable,
        y,
        variable[:6],
        cv_size=0.25,
        depth=1,
        type="class",
        balance=True,
        random_seed=11,
        feature_importance=False,
    )
    second = nns_boost(
        variable,
        y,
        variable[:6],
        cv_size=0.25,
        depth=1,
        type="class",
        balance=True,
        random_seed=11,
        feature_importance=False,
    )

    assert first["results"].shape == (6,)
    assert np.all(np.isin(first["results"], np.unique(y)))
    np.testing.assert_allclose(first["results"], second["results"])
    assert first["feature.frequency"] == second["feature.frequency"]


def test_nns_boost_balance_does_not_enable_stochastic_epoch_path() -> None:
    x = np.linspace(-2.0, 2.0, 20)
    variable = np.column_stack([np.sin((idx + 1) * x) for idx in range(11)])
    y = np.where(x > 0.0, 2.0, 1.0)

    result = nns_boost(
        variable,
        y,
        variable[:3],
        type="class",
        balance=True,
        learner_trials=5,
        epochs=5,
        random_seed=1,
        feature_importance=False,
    )

    assert result["results"].shape == (3,)
    assert np.all(np.isin(result["results"], np.unique(y)))


def test_nns_boost_ts_test_stochastic_epoch_path_shape_and_seed_determinism() -> None:
    x = np.linspace(-2.0, 2.0, 24)
    variable = np.column_stack([np.sin((idx + 1) * x) for idx in range(11)])
    y = x + np.sin(x)

    first = nns_boost(
        variable,
        y,
        variable[:3],
        ts_test=4,
        learner_trials=5,
        epochs=5,
        random_seed=1,
        feature_importance=False,
    )
    second = nns_boost(
        variable,
        y,
        variable[:3],
        ts_test=4,
        learner_trials=5,
        epochs=5,
        random_seed=1,
        feature_importance=False,
    )

    assert first["results"].shape == (3,)
    assert first["pred.int"] is None
    assert sum(first["feature.weights"].values()) == pytest.approx(1.0)
    np.testing.assert_allclose(first["results"], second["results"])
    assert first["feature.frequency"] == second["feature.frequency"]




def test_nns_boost_feature_importance_warns_and_obj_fn_checked() -> None:
    rng = np.random.default_rng(23)
    x = rng.normal(size=(60, 2))
    y = x[:, 0] + rng.normal(scale=0.1, size=60)
    kwargs = {"epochs": 2, "learner_trials": 2, "folds": 2}
    with pytest.warns(UserWarning, match="feature_importance=True draws no figure"):
        nns_boost(x, y, x[:3], feature_importance=True, **kwargs)
    with pytest.raises(ValueError, match=r"\[obj.fn\] must return one numeric value"):
        nns_boost(x, y, x[:3], obj_fn=lambda p, a: (p - a) ** 2, **kwargs)
    with pytest.raises(ValueError, match="random_seed"):
        nns_boost(x, y, x[:3], seed=7, random_seed=5, **kwargs)
