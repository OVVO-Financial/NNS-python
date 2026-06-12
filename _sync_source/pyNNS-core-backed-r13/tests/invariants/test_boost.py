from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import pynns.boost as boost_module
from pynns import nns_boost


def test_nns_boost_shapes_and_feature_weights() -> None:
    x = np.linspace(-2.0, 2.0, 30)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = x + np.sin(x)

    result = nns_boost(variable, y, variable[:6], cv_size=0.25, feature_importance=False)

    assert result["results"].shape == (6,)
    assert result["pred.int"] is None
    assert np.sum(result["feature.weights"]) == pytest.approx(1.0)
    assert result["feature.frequency"].shape == result["feature.weights"].shape
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
    assert np.sum(result["feature.weights"]) == pytest.approx(1.0)
    assert np.all(np.isfinite(result["results"]))


def test_nns_boost_factor_predictor_requires_explicit_levels() -> None:
    x = np.linspace(-2.0, 2.0, 20)
    labels = np.where(x > 0.0, "B", "A")
    variable = np.column_stack((labels, x))
    y = x + np.where(labels == "B", 1.0, 0.0)

    with pytest.raises(ValueError, match="explicit factor_levels"):
        nns_boost(variable, y, variable[:3], cv_size=0.25, feature_importance=False)


@pytest.mark.parametrize("features_only", [False, True])
def test_nns_boost_multiple_factor_predictors_are_positional(features_only: bool) -> None:
    x = np.linspace(-2.0, 2.0, 24)
    first = np.where(x < -0.5, "low", np.where(x > 0.75, "high", "mid"))
    second = np.where(np.sin(x) > 0.0, "up", "down")
    variable = np.column_stack((first, x, second))
    y = x + np.where(first == "low", 1.0, np.where(first == "mid", 2.0, 3.0)) * 0.25

    result = nns_boost(
        variable,
        y,
        variable[:4],
        cv_size=0.25,
        factor_levels=(["low", "mid", "high"], None, ["down", "up"]),
        features_only=features_only,
        feature_importance=False,
        random_seed=1,
    )

    assert set(result) == (
        {"feature.weights", "feature.frequency"}
        if features_only
        else {"results", "pred.int", "feature.weights", "feature.frequency", "n.best"}
    )
    assert np.sum(result["feature.weights"]) == pytest.approx(1.0)
    if not features_only:
        assert result["results"].shape == (4,)


def test_nns_boost_numeric_pred_int_shape() -> None:
    x = np.linspace(-2.0, 2.0, 20)
    variable = np.column_stack((x, np.sin(x)))
    y = x + np.sin(x)

    result = nns_boost(variable, y, variable[:5], pred_int=0.95, feature_importance=False)

    assert result["results"].shape == (5,)
    assert isinstance(result["pred.int"], dict)
    assert set(result["pred.int"]) == {"lower.pred.int", "upper.pred.int"}
    assert result["pred.int"]["lower.pred.int"].shape == result["results"].shape
    assert result["pred.int"]["upper.pred.int"].shape == result["results"].shape
    assert np.all(np.isfinite(result["pred.int"]["lower.pred.int"]))
    assert np.all(np.isfinite(result["pred.int"]["upper.pred.int"]))


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
    assert set(result["pred.int"]) == {"lower.pred.int", "upper.pred.int"}
    assert result["pred.int"]["lower.pred.int"].shape == result["results"].shape
    assert result["pred.int"]["upper.pred.int"].shape == result["results"].shape


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
    assert np.sum(first["feature.weights"]) == pytest.approx(1.0)
    assert first["feature.frequency"].size >= 1
    assert np.all(np.isfinite(first["results"]))
    np.testing.assert_allclose(first["results"], second["results"])
    np.testing.assert_allclose(first["feature.frequency"], second["feature.frequency"])


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
    assert result["pred.int"]["lower.pred.int"].shape == result["results"].shape
    assert result["pred.int"]["upper.pred.int"].shape == result["results"].shape


def test_nns_boost_threshold_does_not_enable_stochastic_epoch_path() -> None:
    x = np.linspace(-2.0, 2.0, 20)
    variable = np.column_stack([np.sin((idx + 1) * x) for idx in range(11)])
    y = x + np.sin(x)

    with pytest.raises(
        NotImplementedError,
        match="threshold on the n_features > 10 stochastic epoch path",
    ):
        nns_boost(
            variable,
            y,
            variable[:3],
            cv_size=0.25,
            threshold=1.0,
            feature_importance=False,
        )


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
    np.testing.assert_allclose(first["feature.frequency"], second["feature.frequency"])


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
    assert np.sum(first["feature.weights"]) == pytest.approx(1.0)
    np.testing.assert_allclose(first["results"], second["results"])
    np.testing.assert_allclose(first["feature.frequency"], second["feature.frequency"])


def test_nns_boost_balance_retries_ordinary_fit_error(monkeypatch: pytest.MonkeyPatch) -> None:
    x = np.linspace(-2.0, 2.0, 24)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = np.where(x < 1.0, 1.0, 2.0)
    original = boost_module._nns_boost_core
    calls = {"count": 0}

    def fail_first(*args: Any, **kwargs: Any) -> dict[str, object]:
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("ordinary fit failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(boost_module, "_nns_boost_core", fail_first)

    with pytest.warns(RuntimeWarning, match="retrying with balance = False"):
        result = nns_boost(
            variable,
            y,
            variable[:4],
            type="class",
            balance=True,
            cv_size=0.25,
            depth=1,
            random_seed=2,
            feature_importance=False,
        )

    assert calls["count"] == 2
    assert result["results"].shape == (4,)
    assert np.all(np.isin(result["results"], np.unique(y)))
