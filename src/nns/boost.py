"""NNS.boost port (reconciled R/Boost.R, NNS 13.2).

Python port of the reconciled R/Boost.R: learner trials fit genuine
multivariate NNS.reg feature subsets against fresh holdouts, the public
``threshold`` is a probability supplied to LPM.VaR over the learner-trial
objective distribution (never a literal score cutoff), epochs re-test only
the surviving combinations (also under exhaustive enumeration), and the
final estimate replicates the original keeper predictors by their relative
epoch frequencies into one multivariate ``nns_stack(method=1, stack=False)``
call. ``depth`` is passed to the regression engine as ``order``.

All random draws (holdouts, balancing, subset sampling, epoch scheduling)
replicate R's Mersenne-Twister stream via ``RRNG``, so a seeded run matches
R bit-for-bit.
"""

from __future__ import annotations

import itertools
import math
import warnings
from collections.abc import Callable
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from nns._reg_engine import _validate_dist, nns_reg_engine
from nns._rrng import RRNG
from nns.central_tendencies import nns_gravity
from nns.stack import nns_stack
from nns.var import lpm_var

Objective = Literal["min", "max"]

BoostResult = dict[str, Any]
"""R-style NNS.boost result (results, pred.int, feature.weights,
feature.frequency)."""


def _scalar_logical(value: Any, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"[{name}] must be TRUE or FALSE.")
    return bool(value)


def _column_is_numeric(column: NDArray[Any]) -> bool:
    if column.dtype.kind in {"f", "i", "u"}:
        return True
    try:
        np.asarray(column, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return True


def _scalar_integer(
    value: Any, name: str, minimum: int = 0, allow_null: bool = False
) -> int | None:
    if allow_null and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer, float, np.floating)):
        raise ValueError(f"[{name}] must be an integer >= {minimum}.")
    v = float(value)
    if not math.isfinite(v) or v < minimum or v != math.floor(v):
        raise ValueError(f"[{name}] must be an integer >= {minimum}.")
    return int(v)


def nns_boost(
    ivs_train: NDArray[Any],
    dv_train: NDArray[Any],
    ivs_test: NDArray[Any] | None = None,
    *,
    type: str | None = None,
    depth: Any = None,
    learner_trials: int = 100,
    epochs: int | None = None,
    cv_size: float | None = None,
    balance: bool = False,
    ts_test: int | None = None,
    threshold: float | None = None,
    obj_fn: Callable[[NDArray[np.float64], NDArray[np.float64]], float] | None = None,
    objective: Objective = "min",
    dist: str | None = None,
    extreme: bool = False,
    features_only: bool = False,
    feature_importance: bool = False,
    pred_int: float | None = None,
    status: bool = False,
    seed: int | None = 123,
    random_seed: int | None = None,
    class_levels: list[Any] | None = None,
    folds: int = 5,
    # Passed to the delegated NNS.stack final stage.
    ncores: int | None = None,
) -> BoostResult:
    del feature_importance
    # random_seed is an accepted alias for seed (matches nns_stack).
    if random_seed is not None and seed == 123:
        seed = random_seed

    balance = _scalar_logical(balance, "balance")
    extreme = _scalar_logical(extreme, "extreme")
    features_only = _scalar_logical(features_only, "features.only")
    status = _scalar_logical(status, "status")

    # Reproduce R's Mersenne-Twister stream (set.seed(seed)) so the CV split,
    # balance resampling, and random feature subsets match R exactly.
    seed_value = (
        _scalar_integer(seed, "seed", minimum=0)
        if seed is not None
        else int(np.random.SeedSequence().generate_state(1)[0])
    )
    rng = RRNG(seed_value)

    if not isinstance(objective, str) or objective.lower() not in {"min", "max"}:
        raise ValueError("[objective] must be exactly 'min' or 'max'.")
    objective_value: Objective = cast(Objective, objective.lower())
    dist_value = _validate_dist(dist)

    if type is not None:
        if not isinstance(type, str) or type.lower() != "class":
            raise ValueError("[type] must be NULL or 'CLASS'.")
        type = "class"
    if balance and type is None:
        warnings.warn("type = 'CLASS' selected because balance = TRUE.", stacklevel=2)
        type = "class"

    depth_value: Any = depth
    if depth_value is not None:
        if isinstance(depth_value, str):
            if depth_value.lower() != "max":
                raise ValueError("[depth] must be NULL, a positive integer, or 'max'.")
            depth_value = "max"
        else:
            depth_value = _scalar_integer(depth_value, "depth", minimum=1)

    learner_trials_value = cast(int, _scalar_integer(learner_trials, "learner.trials", minimum=1))
    epochs_value = _scalar_integer(epochs, "epochs", minimum=0, allow_null=True)
    folds_value = cast(int, _scalar_integer(folds, "folds", minimum=1))

    if cv_size is not None:
        cv_size = float(cv_size)
        if not math.isfinite(cv_size) or cv_size <= 0 or cv_size >= 1:
            raise ValueError("[CV.size] must be a finite scalar strictly between 0 and 1.")

    ts_test_value = _scalar_integer(ts_test, "ts.test", minimum=1, allow_null=True)

    if threshold is not None:
        threshold = float(threshold)
        if not math.isfinite(threshold):
            raise ValueError("[threshold] must be a finite numeric scalar or NULL.")

    if pred_int is not None:
        pred_int = float(pred_int)
        if not math.isfinite(pred_int) or pred_int <= 0 or pred_int >= 1:
            raise ValueError("[pred.int] must be a finite scalar strictly between 0 and 1.")

    # ----------------------------------------------------------------------
    # Data / response coding
    # ----------------------------------------------------------------------
    x = np.asarray(ivs_train)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2 or x.shape[1] < 1:
        raise ValueError("[IVs.train] must contain at least one predictor.")

    dv = np.asarray(dv_train)
    if dv.ndim == 2 and dv.shape[1] == 1:
        dv = dv[:, 0]
    if dv.ndim != 1:
        raise ValueError("[DV.train] must contain exactly one response column.")
    if dv.size != x.shape[0]:
        raise ValueError("nrow(IVs.train) must equal length(DV.train).")
    if dv.size < 4:
        raise ValueError("NNS.boost requires at least four training observations.")

    response_categorical = dv.dtype.kind in {"U", "S", "O", "b"}
    response_was_numeric = not response_categorical
    if response_was_numeric:
        dv_num = dv.astype(np.float64)
        if np.any(np.isnan(dv_num)):
            raise ValueError("[DV.train] contains missing values.")
        if np.any(~np.isfinite(dv_num)):
            raise ValueError("[DV.train] contains non-finite values.")

    if response_categorical and type is None:
        type = "class"
    is_class = type == "class"

    class_values: list[Any] | None = None
    if is_class:
        if response_was_numeric:
            class_values = sorted(set(dv.astype(np.float64).tolist()))
            y = np.asarray(
                [class_values.index(v) + 1 for v in dv.astype(np.float64).tolist()],
                dtype=np.float64,
            )
        else:
            observed = set(str(v) for v in dv.tolist())
            if class_levels is not None:
                levels = [str(level) for level in class_levels]
                unseen = observed - set(levels)
                if unseen:
                    raise ValueError(
                        "[DV.train] contains levels absent from class_levels: "
                        + ", ".join(sorted(unseen))
                    )
            else:
                levels = sorted(observed)
            class_values = levels
            y = np.asarray(
                [levels.index(str(v)) + 1 for v in dv.tolist()], dtype=np.float64
            )
        if np.unique(y).size < 2:
            raise ValueError("Classification requires at least two response classes.")
        if obj_fn is None:
            obj_fn = lambda predicted, actual: float(np.mean(predicted == actual))  # noqa: E731
            objective_value = "max"
    else:
        if response_categorical:
            raise ValueError("A nonnumeric response requires type = 'CLASS'.")
        y = dv.astype(np.float64)
        if obj_fn is None:
            obj_fn = lambda predicted, actual: float(np.sum((predicted - actual) ** 2))  # noqa: E731

    if ivs_test is None:
        z = x.copy()
    else:
        z = np.asarray(ivs_test)
        if z.ndim == 0:
            z = z.reshape(1, 1)
        if z.ndim == 1:
            if x.shape[1] == 1:
                z = z.reshape(-1, 1)
            elif z.size == x.shape[1]:
                z = z.reshape(1, -1)
            else:
                raise ValueError(
                    "A vector [IVs.test] must contain one complete test row, "
                    "unless the training data have one predictor."
                )
        if z.shape[1] != x.shape[1]:
            raise ValueError(
                "[IVs.test] must have the same number of predictors as [IVs.train]."
            )

    n_obs = x.shape[0]
    n_features = x.shape[1]

    if ts_test_value is not None and ts_test_value >= n_obs:
        raise ValueError(
            "[ts.test] must be smaller than the number of training observations."
        )

    if epochs_value is None:
        epochs_value = 2 * n_obs
    cv_fraction = float(rng.runif(0.2, 1.0 / 3.0)) if cv_size is None else cv_size

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    def score(predicted: NDArray[np.float64], actual: NDArray[np.float64]) -> float:
        if predicted.size != actual.size:
            raise ValueError(
                "The objective received predicted and actual vectors of different lengths."
            )
        value = float(cast(Callable[..., float], obj_fn)(predicted, actual))
        return value if math.isfinite(value) else float("nan")

    def central_value(v: NDArray[np.float64], classification: bool) -> float:
        finite = v[np.isfinite(v)]
        if finite.size == 0:
            return float("nan")
        out = float(nns_gravity(finite, discrete=classification))
        if not math.isfinite(out):
            out = float(np.mean(finite))
        return out

    def sanitize_predictions(
        predicted: NDArray[np.float64], fallback_y: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        predicted = np.asarray(predicted, dtype=np.float64).reshape(-1).copy()
        bad = ~np.isfinite(predicted)
        if np.any(bad):
            replacement = central_value(predicted[~bad], is_class)
            if not math.isfinite(replacement):
                replacement = central_value(fallback_y, is_class)
            if not math.isfinite(replacement):
                raise ValueError("NNS.reg returned no finite predictions.")
            predicted[bad] = replacement
        if is_class:
            predicted = np.clip(predicted, y.min(), y.max())
            lo = np.floor(predicted)
            predicted = np.where(predicted - lo < 0.5, lo, np.ceil(predicted))
        return predicted

    def has_all_classes(train_y: NDArray[np.float64]) -> bool:
        return not is_class or set(np.unique(train_y).tolist()) == set(np.unique(y).tolist())

    def random_validation_index() -> NDArray[np.int64]:
        size = max(1, min(n_obs - 1, round(cv_fraction * n_obs)))
        for _ in range(200):
            idx = np.sort(rng.sample_int(n_obs, size, replace=False) - 1).astype(np.int64)
            mask = np.ones(n_obs, dtype=bool)
            mask[idx] = False
            if has_all_classes(y[mask]):
                return idx
        raise ValueError(
            "Unable to create a validation split retaining every class in training. "
            "Reduce CV.size or provide more observations per class."
        )

    if ts_test_value is not None:
        validation_index = np.arange(n_obs - ts_test_value, n_obs, dtype=np.int64)
        mask = np.ones(n_obs, dtype=bool)
        mask[validation_index] = False
        if not has_all_classes(y[mask]):
            raise ValueError(
                "The chronological training prefix does not contain every response class."
            )
    else:
        validation_index = random_validation_index()

    def balance_training(
        train_x: NDArray[np.float64], train_y: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if not balance:
            return train_x, train_y
        values = np.unique(train_y)
        groups = [np.flatnonzero(train_y == v) for v in values]
        if len(groups) < 2 or any(g.size == 0 for g in groups):
            raise ValueError(
                "Balancing requires at least two non-empty classes in the fitting split."
            )
        smallest = min(g.size for g in groups)
        largest = max(g.size for g in groups)
        down = np.concatenate([rng.sample(g, smallest, replace=False) for g in groups])
        up = np.concatenate([rng.sample(g, largest, replace=True) for g in groups])
        idx = np.sort(np.concatenate([down, up]))
        return train_x[idx], train_y[idx]

    def fit_subset(
        feature_index: tuple[int, ...],
        train_index: NDArray[np.int64],
        test_index: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        cols = list(feature_index)
        # Preserve categorical columns as-is; nns_reg_engine one-hot encodes them
        # (factor.2.dummy) per learner exactly as R's NNS.boost does.
        train_x = x[np.ix_(train_index, cols)]
        train_y = y[train_index]
        test_x = x[np.ix_(test_index, cols)]
        bx, by = balance_training(train_x, train_y)
        # Every learner trial fits a genuine multivariate NNS.reg on the
        # sampled feature subset (R: dim.red.method = NULL); subsets are never
        # collapsed to an equal-weight synthetic X* before scoring.
        fit = nns_reg_engine(
            bx,
            by,
            point_est=test_x,
            dim_red_method=None,
            order=depth_value,
            type="CLASS" if is_class else None,
            point_only=True,
            dist=dist_value,
        )
        return sanitize_predictions(
            np.asarray(fit["Point.est"], dtype=np.float64), by
        )

    # ----------------------------------------------------------------------
    # Learner feature subsets
    # ----------------------------------------------------------------------
    total_sets = 2**n_features - 1 if n_features <= 30 else float("inf")
    exhaustive = math.isfinite(total_sets) and total_sets <= learner_trials_value

    test_features: list[tuple[int, ...]]
    if exhaustive:
        test_features = [
            combo
            for k in range(1, n_features + 1)
            for combo in itertools.combinations(range(n_features), k)
        ]
    else:
        target = (
            min(learner_trials_value, int(total_sets))
            if math.isfinite(total_sets)
            else learner_trials_value
        )
        seen: set[tuple[int, ...]] = set()
        test_features = []
        attempts = 0
        max_attempts = max(1000, target * 200)
        while len(test_features) < target and attempts < max_attempts:
            attempts += 1
            k = int(rng.sample_int(n_features, 1)[0])
            candidate = tuple(sorted((rng.sample_int(n_features, k, replace=False) - 1).tolist()))
            if candidate not in seen:
                seen.add(candidate)
                test_features.append(candidate)
        if len(test_features) < target:
            warnings.warn(
                "Fewer unique feature subsets were generated than requested.",
                stacklevel=2,
            )

    all_index = np.arange(n_obs, dtype=np.int64)

    learner_results = np.full(len(test_features), np.nan)
    for i, subset in enumerate(test_features):
        if status:
            print(
                f"Current Threshold Iterations Remaining = {len(test_features) - i - 1}",
                end="\r",
            )
        # Cross-sectional learner trials draw a fresh validation holdout per
        # trial (matching the epoch-stage resampling and R's RNG stream);
        # time-series trials keep the chronological terminal block.
        if ts_test_value is None:
            trial_validation_index = random_validation_index()
        else:
            trial_validation_index = validation_index
        trial_train_index = np.setdiff1d(all_index, trial_validation_index)
        predicted = fit_subset(subset, trial_train_index, trial_validation_index)
        learner_results[i] = score(predicted, y[trial_validation_index])

    finite_results = np.flatnonzero(np.isfinite(learner_results))
    if finite_results.size == 0:
        raise ValueError("No learner trial produced a finite objective value.")

    # The public [threshold] argument is a probability supplied to LPM.VaR over
    # the learner-trial objective distribution; the objective-score cutoff is
    # the distinct value LPM.VaR returns. Neither variable overwrites the other.
    threshold_probability = threshold
    if threshold_probability is None:
        threshold_probability = 0.80 if objective_value == "max" else 0.20
    if extreme:
        threshold_probability = 1.0 if objective_value == "max" else 0.0
    learner_threshold = float(
        lpm_var(threshold_probability, 1.0, learner_results[finite_results])
    )

    if status:
        print(
            f"\nLearner threshold probability = {threshold_probability:.2f}; "
            f"objective cutoff = {learner_threshold:.6f}"
        )

    passes = np.isfinite(learner_results) & (
        learner_results >= learner_threshold
        if objective_value == "max"
        else learner_results <= learner_threshold
    )
    reduced_test_features = [test_features[i] for i in np.flatnonzero(passes)]

    def best_trial_subset() -> list[tuple[int, ...]]:
        if objective_value == "min":
            idx = finite_results[int(np.argmin(learner_results[finite_results]))]
        else:
            idx = finite_results[int(np.argmax(learner_results[finite_results]))]
        return [test_features[int(idx)]]

    if not reduced_test_features:
        # Defensive: LPM.VaR returns a value inside the observed range, so the
        # best trial always passes; guard anyway for degenerate distributions.
        reduced_test_features = best_trial_subset()

    # ----------------------------------------------------------------------
    # Epoch stability stage: re-test only the surviving combinations, on a
    # fresh holdout per cross-sectional epoch (chronological expanding-window
    # blocks for time series). Exhaustive learner trials do not disable this
    # stage. RNG draw order mirrors R: survivor permutation first, then the
    # time-series split permutation, then one holdout draw per epoch.
    # ----------------------------------------------------------------------
    keeper_features: list[tuple[int, ...]]
    if epochs_value > 0:
        survivor_count = len(reduced_test_features)
        epoch_survivor_id = np.resize(
            np.arange(survivor_count, dtype=np.int64), epochs_value
        )
        if epoch_survivor_id.size > 1:
            epoch_survivor_id = rng.sample(
                epoch_survivor_id, epoch_survivor_id.size, replace=False
            )

        chronological_splits: list[tuple[NDArray[np.int64], NDArray[np.int64]]] = []
        epoch_split_id: NDArray[np.int64] | None = None
        if ts_test_value is not None:
            possible_blocks = (n_obs - 1) // ts_test_value
            for b in range(1, possible_blocks + 1):
                start = n_obs - b * ts_test_value
                validation = np.arange(start, start + ts_test_value, dtype=np.int64)
                training = np.arange(0, start, dtype=np.int64)
                if training.size >= 3 and has_all_classes(y[training]):
                    chronological_splits.append((training, validation))
            if not chronological_splits:
                raise ValueError(
                    "No chronological epoch split retained enough training "
                    "observations and every response class."
                )
            epoch_split_id = np.resize(
                np.arange(len(chronological_splits), dtype=np.int64), epochs_value
            )
            if epoch_split_id.size > 1:
                epoch_split_id = rng.sample(
                    epoch_split_id, epoch_split_id.size, replace=False
                )

        keeper_features = []
        for j in range(epochs_value):
            if status:
                print(f"% of epochs = {(j + 1) / epochs_value:.3f}", end="\r")
            features_j = reduced_test_features[int(epoch_survivor_id[j])]

            if ts_test_value is None:
                epoch_validation_index = random_validation_index()
                epoch_train_index = np.setdiff1d(all_index, epoch_validation_index)
            else:
                assert epoch_split_id is not None
                epoch_train_index, epoch_validation_index = chronological_splits[
                    int(epoch_split_id[j])
                ]

            predicted = fit_subset(features_j, epoch_train_index, epoch_validation_index)
            new_result = score(predicted, y[epoch_validation_index])
            ok = math.isfinite(new_result) and (
                new_result >= learner_threshold
                if objective_value == "max"
                else new_result <= learner_threshold
            )
            if ok:
                keeper_features.append(features_j)
    else:
        keeper_features = reduced_test_features

    if not keeper_features:
        warnings.warn(
            "No feature combination re-passed the objective cutoff during epochs; "
            "using the best learner-trial combination. Consider a lower "
            "[threshold] probability.",
            stacklevel=2,
        )
        keeper_features = best_trial_subset()

    counts = np.zeros(n_features, dtype=np.int64)
    for subset in keeper_features:
        for j in subset:
            counts[j] += 1
    active = np.flatnonzero(counts > 0)
    order_idx = active[np.argsort(-counts[active], kind="stable")]
    feature_names = [f"x{j + 1}" for j in range(n_features)]
    plot_table = {feature_names[j]: int(counts[j]) for j in order_idx}
    total_counts = float(sum(plot_table.values()))
    feature_weights_named = {k: v / total_counts for k, v in plot_table.items()}

    if features_only:
        return {
            "feature.weights": feature_weights_named,
            "feature.frequency": plot_table,
        }

    if status:
        print("\nGenerating Final Estimate")

    # ----------------------------------------------------------------------
    # Final estimate: replicate the original keeper predictors by their
    # relative epoch frequencies and fit one multivariate Method 1 NNS.stack.
    #
    # The historical scaling rule converts positive keeper-feature counts into
    # integer replication factors, so a more stable feature contributes
    # proportionally more columns. No synthetic scalar X* is constructed and
    # the frequencies never pass through dim_red_method: method=(1,) with
    # stack=False keeps the final estimator a multivariate NNS.reg whose
    # n.best is selected by NNS.stack's cross-validation.
    # ----------------------------------------------------------------------
    frequency_values = np.asarray([plot_table[k] for k in plot_table], dtype=np.float64)
    relative_frequency = frequency_values / frequency_values.min()
    replication_count = np.maximum(1, np.round(relative_frequency).astype(np.int64))

    name_to_index = {name: j for j, name in enumerate(feature_names)}
    replicated_cols = [
        name_to_index[name]
        for name, count in zip(plot_table, replication_count.tolist(), strict=True)
        for _ in range(count)
    ]
    replicated_train = x[:, replicated_cols]
    replicated_test = z[:, replicated_cols]

    final_stack = nns_stack(
        replicated_train,
        y,
        replicated_test,
        type="CLASS" if is_class else None,
        obj_fn=obj_fn,
        objective=objective_value,
        optimize_threshold=False,
        dist=dist_value,
        cv_size=cv_size,
        balance=False,
        ts_test=ts_test_value,
        folds=folds_value,
        order=depth_value,
        method=(1,),
        stack=False,
        pred_int=pred_int,
        status=status,
        ncores=1,
        seed=seed_value,
    )

    results_raw = final_stack["reg"]
    pred_int_output = final_stack["reg.pred.int"]

    estimates_code = sanitize_predictions(
        np.asarray(results_raw, dtype=np.float64), y
    )
    pred_int_out = pred_int_output

    if is_class:
        n_classes = len(cast(list[Any], class_values))
        codes = np.clip(np.round(estimates_code), 1, n_classes).astype(np.int64)
        if response_was_numeric:
            estimates: NDArray[Any] = np.asarray(
                [cast(list[Any], class_values)[c - 1] for c in codes.tolist()]
            )
        else:
            estimates = codes.astype(np.float64)
        if pred_int_out is not None:
            decoded = {}
            for key, v in pred_int_out.items():
                code = np.clip(
                    np.round(np.asarray(v, dtype=np.float64)), 1, n_classes
                ).astype(np.int64)
                if response_was_numeric:
                    decoded[key] = np.asarray(
                        [cast(list[Any], class_values)[c - 1] for c in code.tolist()]
                    )
                else:
                    decoded[key] = code.astype(np.float64)
            pred_int_out = decoded
    else:
        estimates = estimates_code

    return {
        "results": estimates,
        "pred.int": pred_int_out,
        "feature.weights": feature_weights_named,
        "feature.frequency": plot_table,
        "class.levels": class_values if is_class else None,
    }
