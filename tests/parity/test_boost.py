from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from _r import nns_boost_factor_predictor, nns_boost_multi_factor_predictor, nns_boost_numeric
from _tolerances import COMPOUND

from nns import nns_boost
from nns.boost import _accuracy, _all_feature_sets, _learner_scores, _sse


@pytest.mark.parity
@pytest.mark.parametrize("depth", [None, 1, 2])
def test_nns_boost_numeric_matches_r(depth: int | None) -> None:
    x = np.linspace(-2.0, 2.0, 30)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = x + np.sin(x) + 0.25 * np.cos(x)
    point = variable[:5]

    expected = nns_boost_numeric(
        variable.tolist(),
        y.tolist(),
        point.tolist(),
        learner_trials=10,
        cv_size=0.25,
        depth=depth,
        features_only=False,
    )
    actual = nns_boost(
        variable,
        y,
        point,
        learner_trials=10,
        cv_size=0.25,
        depth=depth,
        feature_importance=False,
    )

    _assert_boost_matches(actual, expected)


@pytest.mark.parity
def test_nns_boost_ivs_test_none_matches_r() -> None:
    x = np.linspace(-2.0, 2.0, 24)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = x + np.sin(x) + 0.25 * np.cos(x)

    expected = nns_boost_numeric(
        variable.tolist(),
        y.tolist(),
        variable.tolist(),
        learner_trials=10,
        cv_size=0.25,
        depth=None,
        features_only=False,
    )
    # random_seed is pinned for determinism. The deterministic feature-set path
    # still draws from the CV-split RNG for iterations above n_rows/4, so an
    # unseeded call left this assertion theoretically seed-sensitive even though
    # the boosted result is empirically seed-invariant here (see
    # test_nns_boost_ivs_test_none_is_seed_invariant). Pinning the seed removes
    # any residual flakiness without altering the matched values.
    actual = nns_boost(
        variable,
        y,
        learner_trials=10,
        cv_size=0.25,
        feature_importance=False,
        random_seed=4,
    )

    _assert_boost_matches(actual, expected)


@pytest.mark.parity
def test_nns_boost_ivs_test_none_is_seed_invariant() -> None:
    # Regression guard for the previously reported cache-parity failure: the
    # depth=None / feature_importance=False boosted result must be identical
    # across seeds (and an unseeded call), so the parity comparison cannot be
    # destabilised by RNG draws on the CV-split path.
    x = np.linspace(-2.0, 2.0, 24)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = x + np.sin(x) + 0.25 * np.cos(x)

    baseline = np.asarray(
        nns_boost(
            variable,
            y,
            learner_trials=10,
            cv_size=0.25,
            feature_importance=False,
        )["results"],
        dtype=np.float64,
    )
    for seed in (None, 0, 1, 4, 42, 1234):
        result = np.asarray(
            nns_boost(
                variable,
                y,
                learner_trials=10,
                cv_size=0.25,
                feature_importance=False,
                random_seed=seed,
            )["results"],
            dtype=np.float64,
        )
        np.testing.assert_array_equal(result, baseline)


@pytest.mark.parity
def test_nns_boost_deterministic_wider_feature_set_matches_r() -> None:
    x = np.linspace(-1.0, 1.0, 24)
    variable = np.column_stack((x, 2.0 * x, np.sin(x), np.cos(x)))
    y = x + 0.2 * np.sin(x)

    expected = nns_boost_numeric(
        variable.tolist(),
        y.tolist(),
        variable[:4].tolist(),
        learner_trials=100,
        cv_size=0.25,
        depth=None,
        features_only=False,
    )
    actual = nns_boost(
        variable,
        y,
        variable[:4],
        learner_trials=100,
        cv_size=0.25,
        feature_importance=False,
        random_seed=4,
    )

    _assert_boost_matches(actual, expected)


def test_nns_boost_depth_1_learner_scores_match_r() -> None:
    x = np.linspace(-2.0, 2.0, 30)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = x + np.sin(x) + 0.25 * np.cos(x)

    scores = _learner_scores(
        variable,
        y,
        _all_feature_sets(3),
        depth=1,
        cv_size=0.25,
        objective_fn=_sse,
        rng=np.random.default_rng(42),
    )

    np.testing.assert_allclose(
        scores,
        np.array(
            [
                0.40188115414190784,
                0.78285682511456822,
                35.935805501222866,
                1.0515124585617981,
                16.21790224565731,
                17.371713848236109,
                9.018035482048569,
            ]
        ),
        atol=COMPOUND,
    )


@pytest.mark.parity
def test_nns_boost_features_only_matches_r() -> None:
    x = np.linspace(-2.0, 2.0, 30)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = x + np.sin(x) + 0.25 * np.cos(x)

    expected = nns_boost_numeric(
        variable.tolist(),
        y.tolist(),
        variable[:5].tolist(),
        learner_trials=10,
        cv_size=0.25,
        depth=None,
        features_only=True,
    )
    actual = nns_boost(
        variable,
        y,
        variable[:5],
        learner_trials=10,
        cv_size=0.25,
        features_only=True,
        feature_importance=False,
    )

    _assert_boost_matches(actual, expected)


@pytest.mark.parity
@pytest.mark.parametrize("ts_test", [3, 5, 8])
def test_nns_boost_ts_test_deterministic_matches_r(ts_test: int) -> None:
    x = np.linspace(-2.0, 2.0, 24)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = x + np.sin(x) + 0.25 * np.cos(x)

    expected = nns_boost_numeric(
        variable.tolist(),
        y.tolist(),
        variable[:4].tolist(),
        learner_trials=10,
        cv_size=0.25,
        depth=None,
        features_only=False,
        ts_test=ts_test,
    )
    actual = nns_boost(
        variable,
        y,
        variable[:4],
        learner_trials=10,
        cv_size=0.25,
        ts_test=ts_test,
        feature_importance=False,
    )

    _assert_boost_matches(actual, expected)


@pytest.mark.parity
def test_nns_boost_ts_test_features_only_matches_r() -> None:
    x = np.linspace(-2.0, 2.0, 24)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = x + np.sin(x) + 0.25 * np.cos(x)

    expected = nns_boost_numeric(
        variable.tolist(),
        y.tolist(),
        variable[:4].tolist(),
        learner_trials=10,
        cv_size=0.25,
        depth=None,
        features_only=True,
        ts_test=5,
    )
    actual = nns_boost(
        variable,
        y,
        variable[:4],
        learner_trials=10,
        cv_size=0.25,
        features_only=True,
        ts_test=5,
        feature_importance=False,
    )

    _assert_boost_matches(actual, expected)


@pytest.mark.stochastic
def test_nns_boost_stochastic_epoch_path_matches_r_structure() -> None:
    x = np.linspace(-2.0, 2.0, 64)
    variable = np.column_stack([np.sin((idx + 1) * x) for idx in range(11)])
    y = x + np.sin(x)

    expected = nns_boost_numeric(
        variable.tolist(),
        y.tolist(),
        variable[:3].tolist(),
        learner_trials=4,
        epochs=4,
        cv_size=0.25,
        depth=None,
        features_only=False,
    )
    actual = nns_boost(
        variable,
        y,
        variable[:3],
        learner_trials=4,
        epochs=4,
        cv_size=0.25,
        random_seed=4,
        feature_importance=False,
    )

    assert set(actual) == set(cast(dict[str, object], expected))
    assert (
        np.asarray(actual["results"], dtype=np.float64).shape
        == np.asarray(
            cast(dict[str, object], expected)["results"],
            dtype=np.float64,
        ).shape
    )
    assert np.asarray(actual["feature.weights"], dtype=np.float64).ndim == 1
    assert np.asarray(actual["feature.frequency"], dtype=np.float64).ndim == 1
    assert actual["pred.int"] is None


@pytest.mark.stochastic
def test_nns_boost_stochastic_epoch_ts_test_matches_r_structure() -> None:
    x = np.linspace(-2.0, 2.0, 64)
    variable = np.column_stack([np.sin((idx + 1) * x) for idx in range(11)])
    y = x + np.sin(x)

    expected = nns_boost_numeric(
        variable.tolist(),
        y.tolist(),
        variable[:3].tolist(),
        learner_trials=4,
        epochs=4,
        cv_size=0.25,
        depth=None,
        features_only=False,
        ts_test=5,
    )
    actual = nns_boost(
        variable,
        y,
        variable[:3],
        learner_trials=4,
        epochs=4,
        cv_size=0.25,
        ts_test=5,
        random_seed=5,
        feature_importance=False,
    )

    assert set(actual) == set(cast(dict[str, object], expected))
    assert (
        np.asarray(actual["results"], dtype=np.float64).shape
        == np.asarray(
            cast(dict[str, object], expected)["results"],
            dtype=np.float64,
        ).shape
    )
    assert np.asarray(actual["feature.weights"], dtype=np.float64).ndim == 1
    assert np.asarray(actual["feature.frequency"], dtype=np.float64).ndim == 1
    assert actual["pred.int"] is None


@pytest.mark.parity
def test_nns_boost_factor_predictor_matches_r() -> None:
    x = np.linspace(-2.0, 2.0, 24)
    labels = np.where(x < -0.5, "low", np.where(x > 0.75, "high", "mid"))
    y = x + np.where(labels == "low", 1.0, np.where(labels == "mid", 2.0, 3.0)) * 0.25
    variable = np.column_stack((labels, x))

    expected = nns_boost_factor_predictor(
        labels.tolist(),
        x.tolist(),
        y.tolist(),
        labels[:5].tolist(),
        x[:5].tolist(),
        levels=["low", "mid", "high"],
        learner_trials=10,
        cv_size=0.25,
        depth=None,
        features_only=False,
    )
    actual = nns_boost(
        variable,
        y,
        variable[:5],
        learner_trials=10,
        cv_size=0.25,
        factor_levels=(["low", "mid", "high"], None),
        feature_importance=False,
    )

    _assert_boost_matches(actual, expected)


@pytest.mark.parity
def test_nns_boost_factor_predictor_features_only_matches_r() -> None:
    x = np.linspace(-2.0, 2.0, 24)
    labels = np.where(x < -0.5, "low", np.where(x > 0.75, "high", "mid"))
    y = x + np.where(labels == "low", 1.0, np.where(labels == "mid", 2.0, 3.0)) * 0.25
    variable = np.column_stack((labels, x))

    expected = nns_boost_factor_predictor(
        labels.tolist(),
        x.tolist(),
        y.tolist(),
        labels[:5].tolist(),
        x[:5].tolist(),
        levels=["low", "mid", "high"],
        learner_trials=10,
        cv_size=0.25,
        depth=None,
        features_only=True,
    )
    actual = nns_boost(
        variable,
        y,
        variable[:5],
        learner_trials=10,
        cv_size=0.25,
        factor_levels=(["low", "mid", "high"], None),
        features_only=True,
        feature_importance=False,
    )

    _assert_boost_matches(actual, expected)


@pytest.mark.parity
@pytest.mark.parametrize("features_only", [False, True])
def test_nns_boost_multiple_factor_predictors_match_r_positional(
    features_only: bool,
) -> None:
    x = np.linspace(-2.0, 2.0, 24)
    first = np.where(x < -0.5, "low", np.where(x > 0.75, "high", "mid"))
    second = np.where(np.sin(x) > 0.0, "up", "down")
    y = (
        x
        + np.where(first == "low", 1.0, np.where(first == "mid", 2.0, 3.0)) * 0.25
        + np.where(second == "up", 0.1, -0.1)
    )
    variable = np.column_stack((first, x.astype(object), second))

    expected = nns_boost_multi_factor_predictor(
        first.tolist(),
        x.tolist(),
        second.tolist(),
        y.tolist(),
        first[:4].tolist(),
        x[:4].tolist(),
        second[:4].tolist(),
        first_levels=["low", "mid", "high"],
        second_levels=["down", "up"],
        learner_trials=10,
        cv_size=0.25,
        depth=None,
        features_only=features_only,
    )
    actual = nns_boost(
        variable,
        y,
        variable[:4],
        learner_trials=10,
        cv_size=0.25,
        factor_levels=(["low", "mid", "high"], None, ["down", "up"]),
        features_only=features_only,
        feature_importance=False,
        random_seed=1,
    )

    _assert_boost_matches(actual, expected)


@pytest.mark.parity
@pytest.mark.parametrize(("depth", "pred_int"), [(1, 0.95), (2, 0.8)])
def test_nns_boost_numeric_pred_int_matches_r(depth: int, pred_int: float) -> None:
    x = np.linspace(-2.0, 2.0, 40)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = 1.0 + 0.8 * x + 0.5 * np.sin(x) - 0.2 * np.cos(x)
    point = variable[30:40]

    expected = nns_boost_numeric(
        variable.tolist(),
        y.tolist(),
        point.tolist(),
        learner_trials=10,
        cv_size=0.25,
        depth=depth,
        features_only=False,
        pred_int=pred_int,
    )
    actual = nns_boost(
        variable,
        y,
        point,
        learner_trials=10,
        cv_size=0.25,
        depth=depth,
        pred_int=pred_int,
        feature_importance=False,
    )

    _assert_boost_matches(actual, expected)
    assert isinstance(actual["pred.int"], dict)
    assert set(actual["pred.int"]) == {"lower.pred.int", "upper.pred.int"}
    assert actual["pred.int"]["lower.pred.int"].shape == actual["results"].shape
    assert actual["pred.int"]["upper.pred.int"].shape == actual["results"].shape


@pytest.mark.parity
def test_nns_boost_features_only_ignores_pred_int_like_r() -> None:
    x = np.linspace(-2.0, 2.0, 30)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = x + np.sin(x) + 0.25 * np.cos(x)

    expected = nns_boost_numeric(
        variable.tolist(),
        y.tolist(),
        variable[:5].tolist(),
        learner_trials=10,
        cv_size=0.25,
        depth=None,
        features_only=True,
        pred_int=0.95,
    )
    actual = nns_boost(
        variable,
        y,
        variable[:5],
        learner_trials=10,
        cv_size=0.25,
        features_only=True,
        pred_int=0.95,
        feature_importance=False,
    )

    assert set(actual) == {"feature.weights", "feature.frequency"}
    _assert_boost_matches(actual, expected)


@pytest.mark.parity
@pytest.mark.parametrize("depth", [None, 1, 2])
def test_nns_boost_binary_class_matches_r(depth: int | None) -> None:
    x = np.linspace(-2.0, 2.0, 30)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = np.where(x + np.sin(x) > 0.0, 2.0, 1.0)
    point = variable[:5]

    expected = nns_boost_numeric(
        variable.tolist(),
        y.tolist(),
        point.tolist(),
        learner_trials=10,
        cv_size=0.25,
        depth=depth,
        features_only=False,
        type="class",
    )
    actual = nns_boost(
        variable,
        y,
        point,
        learner_trials=10,
        cv_size=0.25,
        depth=depth,
        type="class",
        feature_importance=False,
    )

    _assert_boost_matches(actual, expected)


@pytest.mark.parity
@pytest.mark.parametrize("depth", [1, 2])
def test_nns_boost_binary_class_pred_int_matches_r(depth: int) -> None:
    x = np.linspace(-2.0, 2.0, 30)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = np.where(x + np.sin(x) > 0.0, 2.0, 1.0)
    point = variable[:5]

    expected = nns_boost_numeric(
        variable.tolist(),
        y.tolist(),
        point.tolist(),
        learner_trials=10,
        cv_size=0.25,
        depth=depth,
        features_only=False,
        type="class",
        pred_int=0.95,
    )
    actual = nns_boost(
        variable,
        y,
        point,
        learner_trials=10,
        cv_size=0.25,
        depth=depth,
        type="class",
        pred_int=0.95,
        feature_importance=False,
    )

    _assert_boost_matches(actual, expected)
    assert isinstance(actual["pred.int"], dict)
    assert set(actual["pred.int"]) == {"lower.pred.int", "upper.pred.int"}
    assert all(values.shape == actual["results"].shape for values in actual["pred.int"].values())


@pytest.mark.parity
@pytest.mark.parametrize("depth", [1, 2])
def test_nns_boost_multiclass_matches_r(depth: int) -> None:
    x = np.linspace(-2.0, 2.0, 30)
    variable = np.column_stack((x, x**2, np.sin(x)))
    y = np.where(x < -0.5, 1.0, np.where(x > 0.75, 3.0, 2.0))

    expected = nns_boost_numeric(
        variable.tolist(),
        y.tolist(),
        variable[:5].tolist(),
        learner_trials=10,
        cv_size=0.25,
        depth=depth,
        features_only=False,
        type="class",
    )
    actual = nns_boost(
        variable,
        y,
        variable[:5],
        learner_trials=10,
        cv_size=0.25,
        depth=depth,
        type="class",
        feature_importance=False,
    )

    _assert_boost_matches(actual, expected)


@pytest.mark.parity
def test_nns_boost_features_only_ignores_class_pred_int_like_r() -> None:
    x = np.linspace(-2.0, 2.0, 30)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = np.where(x + np.sin(x) > 0.0, 2.0, 1.0)

    expected = nns_boost_numeric(
        variable.tolist(),
        y.tolist(),
        variable[:5].tolist(),
        learner_trials=10,
        cv_size=0.25,
        depth=1,
        features_only=True,
        type="class",
        pred_int=0.95,
    )
    actual = nns_boost(
        variable,
        y,
        variable[:5],
        learner_trials=10,
        cv_size=0.25,
        depth=1,
        features_only=True,
        type="class",
        pred_int=0.95,
        feature_importance=False,
    )

    assert set(actual) == {"feature.weights", "feature.frequency"}
    _assert_boost_matches(actual, expected)


@pytest.mark.parity
def test_nns_boost_factor_like_class_matches_r() -> None:
    x = np.linspace(-2.0, 2.0, 30)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    labels = np.where(x < -0.5, "A", np.where(x > 0.75, "C", "B"))

    expected = nns_boost_numeric(
        variable.tolist(),
        labels.tolist(),
        variable[:5].tolist(),
        learner_trials=10,
        cv_size=0.25,
        depth=1,
        features_only=False,
        type="class",
        class_levels=["A", "B", "C"],
    )
    actual = nns_boost(
        variable,
        labels,
        variable[:5],
        learner_trials=10,
        cv_size=0.25,
        depth=1,
        type="class",
        class_levels=["A", "B", "C"],
        feature_importance=False,
    )

    _assert_boost_matches(actual, expected)


@pytest.mark.parity
def test_nns_boost_class_stable_metadata_matches_r_when_n_best_is_structural() -> None:
    x = np.linspace(-2.0, 2.0, 30)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = np.where(x + np.sin(x) > 0.0, 2.0, 1.0)

    expected = nns_boost_numeric(
        variable.tolist(),
        y.tolist(),
        variable[:5].tolist(),
        learner_trials=10,
        cv_size=0.25,
        depth=1,
        features_only=False,
        type="class",
    )
    actual = nns_boost(
        variable,
        y,
        variable[:5],
        learner_trials=10,
        cv_size=0.25,
        depth=1,
        type="class",
        feature_importance=False,
    )

    assert isinstance(expected, dict)
    expected_dict = cast(dict[str, Any], expected)
    np.testing.assert_allclose(actual["results"], expected_dict["results"], atol=COMPOUND)
    np.testing.assert_allclose(
        actual["feature.weights"],
        expected_dict["feature.weights"],
        atol=COMPOUND,
    )
    np.testing.assert_allclose(
        actual["feature.frequency"],
        expected_dict["feature.frequency"],
        atol=COMPOUND,
    )


@pytest.mark.parity
def test_nns_boost_class_features_only_matches_r() -> None:
    x = np.linspace(-2.0, 2.0, 30)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = np.where(x + np.sin(x) > 0.0, 2.0, 1.0)

    expected = nns_boost_numeric(
        variable.tolist(),
        y.tolist(),
        variable[:5].tolist(),
        learner_trials=10,
        cv_size=0.25,
        depth=1,
        features_only=True,
        type="class",
    )
    actual = nns_boost(
        variable,
        y,
        variable[:5],
        learner_trials=10,
        cv_size=0.25,
        depth=1,
        type="class",
        features_only=True,
        feature_importance=False,
    )

    _assert_boost_matches(actual, expected)


@pytest.mark.parity
@pytest.mark.stochastic
@pytest.mark.parametrize("depth", [1, 2])
def test_nns_boost_balance_binary_class_matches_r_structure(depth: int) -> None:
    x = np.linspace(-2.0, 2.0, 50)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = np.where(x < 1.0, 1.0, 2.0)
    point = variable[:10]

    expected = nns_boost_numeric(
        variable.tolist(),
        y.tolist(),
        point.tolist(),
        learner_trials=10,
        cv_size=0.25,
        depth=depth,
        features_only=False,
        type="class",
        balance=True,
        seed=42,
    )
    actual = nns_boost(
        variable,
        y,
        point,
        learner_trials=10,
        cv_size=0.25,
        depth=depth,
        type="class",
        balance=True,
        random_seed=42,
        feature_importance=False,
    )

    _assert_boost_class_structure(actual, expected, point_rows=point.shape[0], classes=np.unique(y))


@pytest.mark.parity
@pytest.mark.stochastic
def test_nns_boost_balance_multiclass_and_factor_structure() -> None:
    x = np.linspace(-2.0, 2.0, 48)
    variable = np.column_stack((x, x**2, np.sin(x)))
    labels = np.where(x < -0.75, "A", np.where(x > 1.0, "C", "B"))
    point = variable[:8]

    expected = nns_boost_numeric(
        variable.tolist(),
        labels.tolist(),
        point.tolist(),
        learner_trials=10,
        cv_size=0.25,
        depth=1,
        features_only=False,
        type="class",
        class_levels=["A", "B", "C"],
        balance=True,
        seed=7,
    )
    actual = nns_boost(
        variable,
        labels,
        point,
        learner_trials=10,
        cv_size=0.25,
        depth=1,
        type="class",
        class_levels=["A", "B", "C"],
        balance=True,
        random_seed=7,
        feature_importance=False,
    )

    _assert_boost_class_structure(
        actual,
        expected,
        point_rows=point.shape[0],
        classes=np.array([1.0, 2.0, 3.0]),
    )


@pytest.mark.parity
@pytest.mark.stochastic
def test_nns_boost_balance_class_pred_int_matches_r_structure() -> None:
    x = np.linspace(-2.0, 2.0, 48)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = np.where(x < 1.0, 1.0, 2.0)
    point = variable[:8]

    expected = nns_boost_numeric(
        variable.tolist(),
        y.tolist(),
        point.tolist(),
        learner_trials=10,
        cv_size=0.25,
        depth=1,
        features_only=False,
        type="class",
        balance=True,
        seed=42,
        pred_int=0.95,
    )
    actual = nns_boost(
        variable,
        y,
        point,
        learner_trials=10,
        cv_size=0.25,
        depth=1,
        type="class",
        balance=True,
        random_seed=42,
        pred_int=0.95,
        feature_importance=False,
    )

    _assert_boost_class_structure(
        actual,
        expected,
        point_rows=point.shape[0],
        classes=np.unique(y),
        expect_pred_int=True,
    )


@pytest.mark.parity
@pytest.mark.stochastic
def test_nns_boost_balance_type_none_forces_class_path() -> None:
    x = np.linspace(-2.0, 2.0, 42)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = np.where(x < 1.25, 1.0, 2.0)
    point = variable[:6]

    expected = nns_boost_numeric(
        variable.tolist(),
        y.tolist(),
        point.tolist(),
        learner_trials=10,
        cv_size=0.25,
        depth=1,
        features_only=False,
        type=None,
        balance=True,
        seed=9,
    )
    actual = nns_boost(
        variable,
        y,
        point,
        learner_trials=10,
        cv_size=0.25,
        depth=1,
        balance=True,
        random_seed=9,
        feature_importance=False,
    )

    _assert_boost_class_structure(
        actual,
        expected,
        point_rows=point.shape[0],
        classes=np.array([1.0, 2.0]),
    )


def test_nns_boost_balance_raw_character_class_raises() -> None:
    x = np.linspace(-2.0, 2.0, 20)
    variable = np.column_stack((x, np.sin(x)))
    labels = np.where(x > 0.0, "B", "A")

    with pytest.raises(ValueError, match="levels"):
        nns_boost(
            variable,
            labels,
            variable[:3],
            type="class",
            balance=True,
            random_seed=1,
        )


def test_nns_boost_depth_1_class_learner_scores_match_r() -> None:
    x = np.linspace(-2.0, 2.0, 30)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = np.where(x + np.sin(x) > 0.0, 2.0, 1.0)

    scores = _learner_scores(
        variable,
        y,
        _all_feature_sets(3),
        depth=1,
        cv_size=0.25,
        objective_fn=_accuracy,
        rng=np.random.default_rng(42),
        type_value="class",
    )

    np.testing.assert_allclose(
        scores,
        np.array(
            [
                0.71428571428571430,
                0.85714285714285710,
                0.42857142857142855,
                0.85714285714285710,
                0.42857142857142855,
                0.71428571428571430,
                0.57142857142857140,
            ]
        ),
        atol=COMPOUND,
    )


def test_nns_boost_raw_character_class_raises() -> None:
    x = np.linspace(-2.0, 2.0, 20)
    variable = np.column_stack((x, np.sin(x)))
    labels = np.where(x > 0.0, "B", "A")

    with pytest.raises(ValueError, match="class_levels"):
        nns_boost(variable, labels, variable[:3], type="class", cv_size=0.25)


def _assert_boost_matches(actual: dict[str, Any], expected: Any) -> None:
    assert isinstance(expected, dict)
    assert set(actual) == set(expected)
    for key in actual:
        _assert_nested_numeric_close(actual[key], expected[key])


def _assert_nested_numeric_close(actual: Any, expected: Any) -> None:
    if actual is None:
        assert expected is None
        return
    if isinstance(actual, dict):
        assert isinstance(expected, dict)
        assert set(actual) == set(expected)
        for key in actual:
            _assert_nested_numeric_close(actual[key], expected[key])
        return
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        atol=COMPOUND,
    )


def _assert_boost_class_structure(
    actual: dict[str, Any],
    expected: Any,
    *,
    point_rows: int,
    classes: np.ndarray,
    expect_pred_int: bool = False,
) -> None:
    assert isinstance(expected, dict)
    assert set(actual) == set(expected)
    actual_results = np.asarray(actual["results"], dtype=np.float64)
    expected_results = np.asarray(expected["results"], dtype=np.float64)
    assert actual_results.shape == expected_results.shape == (point_rows,)
    assert np.all(np.isin(actual_results[np.isfinite(actual_results)], classes))
    assert np.all(np.isin(expected_results[np.isfinite(expected_results)], classes))
    if expect_pred_int:
        assert isinstance(actual["pred.int"], dict)
        assert isinstance(expected["pred.int"], dict)
        assert set(actual["pred.int"]) == set(expected["pred.int"])
        for values in actual["pred.int"].values():
            assert values.shape == (point_rows,)
            assert np.all(np.isfinite(values))
    else:
        assert actual["pred.int"] is None
        assert expected["pred.int"] is None
    assert np.asarray(actual["feature.weights"], dtype=np.float64).ndim == 1
    assert np.asarray(expected["feature.weights"], dtype=np.float64).ndim == 1
    assert np.asarray(actual["feature.frequency"], dtype=np.float64).ndim == 1
    assert np.asarray(expected["feature.frequency"], dtype=np.float64).ndim == 1
