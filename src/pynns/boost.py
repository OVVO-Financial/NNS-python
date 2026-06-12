from __future__ import annotations

import itertools
import math
import warnings
from collections.abc import Callable, Sequence
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from pynns.categorical import _balance_class_training, _dense_factor_codes, encode_factor_codes
from pynns.dependence import _gravity
from pynns.regression import (
    Order,
    _normalize_type,
    _prepare_y_values,
    _r_minmax_columns,
    _round_clamp_classes,
    nns_reg,
)
from pynns.stack import nns_stack

Objective = Literal["min", "max"]
BoostResult = dict[str, Any]


def nns_boost(
    ivs_train: NDArray[np.float64],
    dv_train: NDArray[np.float64],
    ivs_test: NDArray[np.float64] | None = None,
    *,
    type: str | None = None,
    depth: Order = None,
    learner_trials: int = 100,
    epochs: int | None = None,
    cv_size: float | None = None,
    balance: bool = False,
    ts_test: int | None = None,
    threshold: float | None = None,
    obj_fn: Callable[[NDArray[np.float64], NDArray[np.float64]], float] | None = None,
    objective: Objective = "min",
    extreme: bool = False,
    features_only: bool = False,
    feature_importance: bool = True,
    pred_int: float | None = None,
    status: bool = False,
    random_seed: int | None = None,
    class_levels: list[object] | None = None,
    factor_levels: Sequence[object] | Sequence[Sequence[object] | None] | None = None,
) -> BoostResult:
    """Deterministic NNS.boost port using real NNS.reg and NNS.stack internals."""
    del status
    type_value = _normalize_type(type)
    if balance:
        type_value = "class"
    x_input: NDArray[Any] | NDArray[np.float64] = np.asarray(ivs_train)
    x_test_input: NDArray[Any] | NDArray[np.float64] | None = (
        None if ivs_test is None else np.asarray(ivs_test)
    )
    if factor_levels is not None:
        x_input, x_test_input = _encode_factor_predictors(
            x_input,
            x_test_input,
            factor_levels=factor_levels,
        )
    elif x_input.dtype.kind in {"U", "S", "O"} or (
        x_test_input is not None and x_test_input.dtype.kind in {"U", "S", "O"}
    ):
        raise ValueError("string/object predictor values require explicit factor_levels.")

    x_train = _as_matrix(x_input, "ivs_train")
    x_test = (
        x_train.copy() if x_test_input is None else _as_point_matrix(x_test_input, x_train.shape[1])
    )
    ts_test_value = None if ts_test is None else int(ts_test)
    if ts_test_value is not None and ts_test_value <= 0:
        raise ValueError("ts_test must be a positive integer.")
    if balance:
        y_train, class_codes = _dense_factor_codes(dv_train, levels=class_levels)
    elif type_value == "class":
        y_train, _ = _prepare_y_values(
            dv_train,
            type_value=type_value,
            class_levels=class_levels,
        )
        class_codes = np.unique(y_train[np.isfinite(y_train)])
    else:
        y_train = _as_vector(dv_train, "dv_train")
        class_codes = np.empty(0, dtype=np.float64)
    if x_train.shape[0] != y_train.size:
        raise ValueError("ivs_train and dv_train must have the same row count.")
    if x_train.shape[1] > 10 and threshold is not None:
        raise NotImplementedError(
            "nns_boost threshold on the n_features > 10 stochastic epoch path is deferred "
            "because installed R errors before constructing test.features."
        )
    rng = np.random.default_rng(random_seed)
    if balance:
        x_train, y_train = _balance_class_training(
            x_train,
            y_train,
            classes=class_codes,
            rng=rng,
        )

    try:
        return _nns_boost_core(
            x_train,
            y_train,
            x_test,
            type_value=type_value,
            depth=depth,
            learner_trials=learner_trials,
            cv_size=cv_size,
            threshold=threshold,
            obj_fn=obj_fn,
            objective=objective,
            extreme=extreme,
            features_only=features_only,
            feature_importance=feature_importance,
            pred_int=pred_int,
            ts_test=ts_test_value,
            epochs=epochs,
            rng=rng,
        )
    except NotImplementedError:
        raise
    except Exception:
        if not balance:
            raise
        warnings.warn(
            "[retry] First attempt failed; retrying with balance = False",
            RuntimeWarning,
            stacklevel=2,
        )
        return nns_boost(
            ivs_train,
            dv_train,
            ivs_test,
            type=type,
            depth=depth,
            learner_trials=learner_trials,
            epochs=epochs,
            cv_size=cv_size,
            balance=False,
            ts_test=ts_test,
            threshold=threshold,
            obj_fn=obj_fn,
            objective=objective,
            extreme=extreme,
            features_only=features_only,
            feature_importance=feature_importance,
            pred_int=pred_int,
            random_seed=random_seed,
            class_levels=class_levels,
            factor_levels=factor_levels,
        )


def _nns_boost_core(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    x_test: NDArray[np.float64],
    *,
    type_value: str | None,
    depth: Order,
    learner_trials: int,
    cv_size: float | None,
    threshold: float | None,
    obj_fn: Callable[[NDArray[np.float64], NDArray[np.float64]], float] | None,
    objective: Objective,
    extreme: bool,
    features_only: bool,
    feature_importance: bool,
    pred_int: float | None,
    ts_test: int | None,
    epochs: int | None,
    rng: np.random.Generator,
) -> BoostResult:
    objective_l = objective.lower()
    if objective_l not in {"min", "max"}:
        raise ValueError("objective must be 'min' or 'max'.")
    objective_value: Objective = "min" if objective_l == "min" else "max"
    if type_value == "class" and obj_fn is None:
        objective_value = "max"
        objective_fn: Callable[[NDArray[np.float64], NDArray[np.float64]], float] = _accuracy
    else:
        objective_fn = _sse if obj_fn is None else obj_fn

    n_rows, n_cols = x_train.shape
    feature_sets = _all_feature_sets(n_cols)
    deterministic = (len(feature_sets) < n_rows) or n_cols <= 10
    if deterministic:
        trial_sets = feature_sets
        learner_trials = len(trial_sets)
    else:
        learner_trials = min(learner_trials, len(feature_sets))
        trial_sets = [_random_feature_set(n_cols, rng, min_size=2) for _ in range(learner_trials)]

    if threshold is None:
        cv_fraction = 0.25 if cv_size is None else float(cv_size)
        scores = _learner_scores(
            x_train,
            y_train,
            trial_sets,
            depth=depth,
            cv_size=cv_fraction,
            objective_fn=objective_fn,
            rng=rng,
            type_value=type_value,
            ts_test=ts_test,
        )
    else:
        scores = np.asarray([threshold], dtype=np.float64)

    threshold_value = _threshold(scores, objective_value, extreme)
    if threshold is None:
        keepers = _keeper_sets(trial_sets, scores, threshold_value, objective_value, extreme)
    else:
        keepers = trial_sets
    if not deterministic and threshold is None:
        epoch_count = 2 * n_rows if epochs is None else int(epochs)
        if epoch_count < 1:
            raise ValueError("epochs must be >= 1.")
        keepers = _epoch_keeper_sets(
            x_train,
            y_train,
            keepers,
            threshold_value,
            objective_value,
            depth=depth,
            cv_size=0.25 if cv_size is None else float(cv_size),
            objective_fn=objective_fn,
            rng=rng,
            type_value=type_value,
            epochs=epoch_count,
            ts_test=ts_test,
        )
    if not keepers:
        if threshold is not None:
            if objective_value == "min":
                raise ValueError("Please increase threshold.")
            raise ValueError("Please reduce threshold.")
        best_index = int(np.nanargmin(scores) if objective_value == "min" else np.nanargmax(scores))
        keepers = [trial_sets[best_index]]

    counts = _feature_counts(keepers, n_cols)
    if np.sum(counts) == 0.0:
        counts[:] = 1.0
    weights = counts / float(np.sum(counts))
    order_idx = np.flatnonzero(counts > 0.0)
    if features_only or feature_importance:
        order_idx = order_idx[np.argsort(-counts[order_idx], kind="mergesort")]

    if features_only:
        return {
            "feature.weights": weights[order_idx],
            "feature.frequency": counts[order_idx],
        }

    coef = weights.copy()
    xstar_fit = nns_reg(
        x_train,
        y_train,
        dim_red_method=coef,
        order=depth,
        point_only=False,
    )
    xstar_train = np.asarray(xstar_fit["x.star"]["x"], dtype=np.float64)
    xstar_train = _fill_nan_with_gravity(xstar_train)
    xstar_test = _project_xstar(x_train, x_test, coef)
    xstar_test = _fill_nan_with_gravity(xstar_test)

    final_fit = nns_stack(
        np.column_stack((xstar_train, xstar_train)),
        y_train,
        np.column_stack((xstar_test, xstar_test)),
        method=1,
        objective=objective_value,
        cv_size=0.25 if cv_size is None else cv_size,
        type=type_value,
        pred_int=pred_int,
    )
    estimates = np.asarray(final_fit["stack"], dtype=np.float64)
    if estimates.size == 0 or np.any(np.isnan(estimates)):
        estimates = np.asarray(final_fit["reg"], dtype=np.float64)
    estimates = _fill_nan_with_gravity(estimates)
    if type_value == "class":
        estimates = _round_clamp_classes(estimates, y_train)

    return {
        "results": estimates,
        "pred.int": final_fit["pred.int"],
        "feature.weights": weights[order_idx],
        "feature.frequency": counts[order_idx],
        "n.best": final_fit["NNS.reg.n.best"],
    }


def _all_feature_sets(n_cols: int) -> list[tuple[int, ...]]:
    return [
        combo
        for size in range(1, n_cols + 1)
        for combo in itertools.combinations(range(n_cols), size)
    ]


def _encode_factor_predictors(
    x: NDArray[Any],
    x_test: NDArray[Any] | None,
    *,
    factor_levels: Sequence[object] | Sequence[Sequence[object] | None],
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    x_array = np.asarray(x)
    test_array = None if x_test is None else np.asarray(x_test)
    if x_array.ndim == 1:
        combined = (
            x_array.reshape(-1)
            if test_array is None
            else np.concatenate((x_array.reshape(-1), test_array.reshape(-1)))
        )
        encoded = _encode_factor_column(combined, _boost_levels_for_column(factor_levels, 0, 1))
        train = encoded[: x_array.shape[0]]
        test = None if test_array is None else encoded[x_array.shape[0] :]
        return train.reshape(-1, 1), None if test is None else test.reshape(-1, 1)

    if x_array.ndim != 2:
        raise ValueError("ivs_train must be a vector or 2D matrix.")
    if test_array is not None:
        if test_array.ndim == 1:
            test_array = test_array.reshape(1, -1)
        if test_array.ndim != 2 or test_array.shape[1] != x_array.shape[1]:
            raise ValueError("ivs_test must have the same column count as ivs_train.")
    train_columns: list[NDArray[np.float64]] = []
    test_columns: list[NDArray[np.float64]] = []
    for col in range(x_array.shape[1]):
        column = x_array[:, col]
        combined = column if test_array is None else np.concatenate((column, test_array[:, col]))
        encoded = _encode_factor_column(
            combined,
            _boost_levels_for_column(factor_levels, col, x_array.shape[1]),
        )
        train_columns.append(encoded[: x_array.shape[0]])
        if test_array is not None:
            test_columns.append(encoded[x_array.shape[0] :])
    train_matrix = np.column_stack(train_columns)
    test_matrix = None if test_array is None else np.column_stack(test_columns)
    return train_matrix, test_matrix


def _encode_factor_column(
    values: NDArray[Any],
    levels: Sequence[object] | None,
) -> NDArray[np.float64]:
    if levels is None:
        return np.asarray(values, dtype=np.float64).reshape(-1)
    codes, _ = encode_factor_codes(values, levels=levels)
    return codes


def _boost_levels_for_column(
    factor_levels: Sequence[object] | Sequence[Sequence[object] | None],
    column: int,
    n_cols: int,
) -> Sequence[object] | None:
    if n_cols == 1:
        return factor_levels
    if column >= len(factor_levels):
        raise ValueError("factor_levels must provide levels for every predictor column.")
    return cast(Sequence[Sequence[object] | None], factor_levels)[column]


def _boost_factor_column_count(
    factor_levels: Sequence[object] | Sequence[Sequence[object] | None],
    n_cols: int,
) -> int:
    if n_cols == 1:
        return 1
    levels_by_column = cast(Sequence[Sequence[object] | None], factor_levels)
    return sum(levels is not None for levels in levels_by_column)


def _random_feature_set(
    n_cols: int,
    rng: np.random.Generator,
    *,
    min_size: int,
) -> tuple[int, ...]:
    low = min(min_size, n_cols)
    size = int(rng.integers(low, n_cols + 1))
    return tuple(sorted(rng.choice(n_cols, size=size, replace=False).astype(int).tolist()))


def _boost_cv_split(
    n_rows: int,
    iteration: int,
    cv_size: float,
    rng: np.random.Generator,
    ts_test: int | None = None,
    epoch_split: bool = False,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    if ts_test is not None:
        if ts_test >= n_rows:
            raise ValueError("ts_test must be smaller than the training row count.")
        if epoch_split:
            start = n_rows - (2 * ts_test) - 1
            if start < 0:
                raise ValueError("ts_test leaves too few epoch training rows.")
            test_idx = np.arange(start, n_rows, dtype=np.int64)
            mask = np.ones(n_rows, dtype=bool)
            mask[test_idx] = False
            return np.flatnonzero(mask).astype(np.int64), test_idx
        test_idx = np.arange(0, n_rows - ts_test, dtype=np.int64)
        train_idx = np.arange(n_rows - ts_test, n_rows, dtype=np.int64)
        return train_idx, test_idx

    test_count = max(1, int(cv_size * n_rows))
    if iteration <= n_rows / 4.0:
        one_based = np.linspace(iteration, n_rows, test_count).astype(np.int64)
        test_idx = np.clip(one_based - 1, 0, n_rows - 1)
    else:
        test_idx = rng.choice(n_rows, size=test_count, replace=False).astype(np.int64)
    mask = np.ones(n_rows, dtype=bool)
    mask[np.unique(test_idx)] = False
    return np.flatnonzero(mask).astype(np.int64), test_idx


def _learner_scores(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    feature_sets: list[tuple[int, ...]],
    *,
    depth: Order,
    cv_size: float,
    objective_fn: Callable[[NDArray[np.float64], NDArray[np.float64]], float],
    rng: np.random.Generator,
    type_value: str | None = None,
    ts_test: int | None = None,
) -> NDArray[np.float64]:
    scores = np.empty(len(feature_sets), dtype=np.float64)
    for idx, features in enumerate(feature_sets, start=1):
        scores[idx - 1] = _learner_score(
            x_train,
            y_train,
            features,
            iteration=idx,
            depth=depth,
            cv_size=cv_size,
            objective_fn=objective_fn,
            rng=rng,
            type_value=type_value,
            ts_test=ts_test,
        )
    return scores


def _learner_score(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    features: tuple[int, ...],
    *,
    iteration: int,
    depth: Order,
    cv_size: float,
    objective_fn: Callable[[NDArray[np.float64], NDArray[np.float64]], float],
    rng: np.random.Generator,
    type_value: str | None,
    ts_test: int | None,
    epoch_split: bool = False,
) -> float:
    train_idx, test_idx = _boost_cv_split(
        y_train.size,
        iteration,
        cv_size,
        rng,
        ts_test,
        epoch_split=epoch_split,
    )
    aug_x, aug_y = _augmented_training(x_train[train_idx], y_train[train_idx])
    train_subset = aug_x[:, features]
    point_subset = x_train[test_idx][:, features]
    if len(features) == 1:
        predicted = nns_reg(
            train_subset.reshape(-1),
            aug_y,
            point_est=point_subset.reshape(-1),
            order=depth,
            point_only=False,
            type=type_value,
        )["Point.est"]
    else:
        predicted = nns_reg(
            train_subset,
            aug_y,
            point_est=point_subset,
            dim_red_method="equal",
            order=depth,
            point_only=False,
            type=type_value,
        )["Point.est"]
    pred = _fill_nan_with_gravity(np.asarray(predicted, dtype=np.float64))
    if type_value == "class":
        pred = _round_clamp_classes(pred, y_train)
    return objective_fn(pred, y_train[test_idx])


def _augmented_training(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    joined = np.column_stack((y, x))
    five = np.column_stack([_fivenum(joined[:, col]) for col in range(joined.shape[1])])
    aug = np.vstack((five[:, 1:], x))
    aug_y = np.concatenate((five[:, 0], y))
    return aug, aug_y


def _fivenum(values: NDArray[np.float64]) -> NDArray[np.float64]:
    sorted_values = np.sort(np.asarray(values, dtype=np.float64)[np.isfinite(values)])
    n = sorted_values.size
    if n == 0:
        return np.full(5, np.nan, dtype=np.float64)
    n4 = math.floor((n + 3) / 2.0) / 2.0
    positions = np.array([1.0, n4, (n + 1) / 2.0, n + 1.0 - n4, float(n)])
    lower = np.floor(positions).astype(np.int64) - 1
    upper = np.ceil(positions).astype(np.int64) - 1
    return np.asarray(0.5 * (sorted_values[lower] + sorted_values[upper]), dtype=np.float64)


def _threshold(scores: NDArray[np.float64], objective: Objective, extreme: bool) -> float:
    clean = scores[np.isfinite(scores)]
    if clean.size == 0:
        return math.nan
    if extreme:
        return float(np.max(clean) if objective == "max" else np.min(clean))
    five = _fivenum(clean)
    return float(five[3] if objective == "max" else five[1])


def _keeper_sets(
    feature_sets: list[tuple[int, ...]],
    scores: NDArray[np.float64],
    threshold: float,
    objective: Objective,
    extreme: bool,
) -> list[tuple[int, ...]]:
    if extreme:
        target = float(np.nanmax(scores) if objective == "max" else np.nanmin(scores))
        return [feature_sets[int(np.flatnonzero(scores == target)[0])]]
    keepers: list[tuple[int, ...]] = []
    for features, score in zip(feature_sets, scores, strict=True):
        if objective == "max" and score >= threshold:
            keepers.append(features)
        if objective == "min" and score <= threshold:
            keepers.append(features)
    return keepers


def _epoch_keeper_sets(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    survivor_sets: list[tuple[int, ...]],
    threshold: float,
    objective: Objective,
    *,
    depth: Order,
    cv_size: float,
    objective_fn: Callable[[NDArray[np.float64], NDArray[np.float64]], float],
    rng: np.random.Generator,
    type_value: str | None,
    epochs: int,
    ts_test: int | None,
) -> list[tuple[int, ...]]:
    pool = _weighted_feature_pool(survivor_sets, x_train.shape[1])
    if pool.size == 0:
        return []
    keepers: list[tuple[int, ...]] = []
    for epoch in range(1, epochs + 1):
        size = int(rng.integers(1, x_train.shape[1] + 1))
        features = tuple(sorted(np.unique(rng.choice(pool, size=size, replace=True)).tolist()))
        score = _learner_score(
            x_train,
            y_train,
            features,
            iteration=epoch,
            depth=depth,
            cv_size=cv_size,
            objective_fn=objective_fn,
            rng=rng,
            type_value=type_value,
            ts_test=ts_test,
            epoch_split=True,
        )
        if not np.isfinite(score):
            score = 0.99 * threshold if objective == "max" else 1.01 * threshold
        if objective == "max" and score >= threshold:
            keepers.append(features)
        if objective == "min" and score <= threshold:
            keepers.append(features)
    return keepers


def _weighted_feature_pool(
    feature_sets: list[tuple[int, ...]],
    n_cols: int,
) -> NDArray[np.int64]:
    counts = _feature_counts(feature_sets, n_cols)
    positive = counts[counts > 0.0]
    if positive.size == 0:
        return np.empty(0, dtype=np.int64)
    scaled = counts / float(np.min(positive))
    repeats = np.where(scaled % 1.0 < 0.5, np.floor(scaled), np.ceil(scaled)).astype(np.int64)
    return np.repeat(np.arange(n_cols, dtype=np.int64), repeats)


def _feature_counts(feature_sets: list[tuple[int, ...]], n_cols: int) -> NDArray[np.float64]:
    counts = np.zeros(n_cols, dtype=np.float64)
    for features in feature_sets:
        for feature in features:
            counts[feature] += 1.0
    return counts


def _project_xstar(
    x_train: NDArray[np.float64],
    x_test: NDArray[np.float64],
    coef: NDArray[np.float64],
) -> NDArray[np.float64]:
    active = int(np.sum(np.abs(coef) > 0.0))
    if active == 0:
        active = 1
    joint = np.vstack((x_test, x_train))
    norm = _r_minmax_columns(joint, zero_guard=True)
    return np.asarray(norm[: x_test.shape[0]] @ coef / active, dtype=np.float64)


def _fill_nan_with_gravity(values: NDArray[np.float64]) -> NDArray[np.float64]:
    out = np.asarray(values, dtype=np.float64).copy()
    if np.any(np.isnan(out)):
        finite = out[np.isfinite(out)]
        fill = _gravity(finite) if finite.size else 0.0
        out[np.isnan(out)] = fill
    return out


def _sse(predicted: NDArray[np.float64], actual: NDArray[np.float64]) -> float:
    return float(np.sum((predicted - actual) ** 2))


def _accuracy(predicted: NDArray[np.float64], actual: NDArray[np.float64]) -> float:
    return float(np.mean(predicted == actual))


def _as_matrix(x: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    values = np.asarray(x, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty numeric vector or matrix.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")
    return values


def _as_point_matrix(x: NDArray[np.float64], n_cols: int) -> NDArray[np.float64]:
    values = np.asarray(x, dtype=np.float64)
    if values.ndim == 1:
        if n_cols == 1:
            values = values.reshape(-1, 1)
        else:
            values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != n_cols:
        raise ValueError("ivs_test must have the same column count as ivs_train.")
    if not np.all(np.isfinite(values)):
        raise ValueError("ivs_test must contain only finite values.")
    return values


def _as_vector(x: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    values = np.asarray(x, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")
    return values
