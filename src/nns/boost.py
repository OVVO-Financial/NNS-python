"""Repaired NNS.boost port.

Python port of the audited R/Boost.R (NNS 13.1 Beta tip): learner trials on
feature subsets scored against a holdout, survivor-weighted epoch sampling,
a frequency-weighted synthetic X* final design, local out-of-fold n.best
selection, and strict argument validation. ``depth`` is passed to the
regression engine as ``order``.

Deviation from R, by design: random subset generation, holdout draws, and
balancing use a local ``numpy.random.Generator`` (``seed``), so randomized
runs match R distributionally rather than bit-for-bit. With ``ts_test`` set
and exhaustive learner trials (2^p - 1 <= learner_trials), the entire run
is deterministic and matches R exactly.
"""

from __future__ import annotations

import itertools
import math
import warnings
from typing import Any, Callable, Literal, cast

import numpy as np
from numpy.typing import NDArray

from nns._reg_engine import nns_reg_engine
from nns._rrng import RRNG
from nns.central_tendencies import nns_gravity

Objective = Literal["min", "max"]

BoostResult = dict[str, Any]
"""R-style NNS.boost result (results, pred.int, feature.weights,
feature.frequency)."""


def _scalar_logical(value: Any, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"[{name}] must be TRUE or FALSE.")
    return bool(value)


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
    extreme: bool = False,
    features_only: bool = False,
    feature_importance: bool = False,
    pred_int: float | None = None,
    status: bool = False,
    seed: int | None = 123,
    random_seed: int | None = None,
    # Legacy compatibility (ignored):
    ncores: int | None = None,
) -> BoostResult:
    del ncores, feature_importance
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
            levels = sorted(set(str(v) for v in dv.tolist()))
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
        size = max(1, min(n_obs - 1, int(round(cv_fraction * n_obs))))
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
        train_x = x[np.ix_(train_index, cols)].astype(np.float64)
        train_y = y[train_index]
        test_x = x[np.ix_(test_index, cols)].astype(np.float64)
        bx, by = balance_training(train_x, train_y)
        fit = nns_reg_engine(
            bx,
            by,
            point_est=test_x,
            dim_red_method="equal" if bx.shape[1] > 1 else None,
            order=depth_value,
            type="CLASS" if is_class else None,
            point_only=True,
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
    train_index = np.setdiff1d(all_index, validation_index)
    actual = y[validation_index]

    results = np.full(len(test_features), np.nan)
    for i, subset in enumerate(test_features):
        if status:
            print(
                f"Current Threshold Iterations Remaining = {len(test_features) - i - 1}",
                end="\r",
            )
        predicted = fit_subset(subset, train_index, validation_index)
        results[i] = score(predicted, actual)

    finite_results = np.flatnonzero(np.isfinite(results))
    if finite_results.size == 0:
        raise ValueError("No learner trial produced a finite objective value.")

    supplied_threshold = threshold is not None
    if not supplied_threshold:
        if extreme:
            threshold = (
                float(results[finite_results].max())
                if objective_value == "max"
                else float(results[finite_results].min())
            )
        else:
            threshold = float(
                np.quantile(
                    results[finite_results],
                    0.75 if objective_value == "max" else 0.25,
                    method="averaged_inverted_cdf",
                )
            )
    if status:
        print(f"\nLearner Accuracy Threshold = {threshold:.4g}")

    threshold_f = cast(float, threshold)
    passes = np.isfinite(results) & (
        results >= threshold_f if objective_value == "max" else results <= threshold_f
    )
    reduced_test_features = [test_features[i] for i in np.flatnonzero(passes)]

    def best_trial_subset() -> list[tuple[int, ...]]:
        if objective_value == "min":
            idx = finite_results[int(np.argmin(results[finite_results]))]
        else:
            idx = finite_results[int(np.argmax(results[finite_results]))]
        return [test_features[int(idx)]]

    if not reduced_test_features:
        if supplied_threshold:
            direction = "increase" if objective_value == "min" else "reduce"
            raise ValueError(f"No learner subset met [threshold]; {direction} the threshold.")
        reduced_test_features = best_trial_subset()

    feature_count = np.zeros(n_features, dtype=np.float64)
    for subset in reduced_test_features:
        for j in subset:
            feature_count[j] += 1
    feature_prob = feature_count + max(1.0, feature_count.sum()) * 1e-12
    feature_prob = feature_prob / feature_prob.sum()

    # ----------------------------------------------------------------------
    # Weighted epoch stage
    # ----------------------------------------------------------------------
    keeper_features: list[tuple[int, ...]]
    if not exhaustive and epochs_value > 0:
        keeper_features = []
        for j in range(epochs_value):
            if status:
                print(f"% of epochs = {(j + 1) / epochs_value:.3f}", end="\r")
            k = int(rng.sample_int(n_features, 1)[0])
            subset = tuple(
                sorted(
                    (rng.sample_int_prob_noreplace(n_features, k, feature_prob) - 1).tolist()
                )
            )
            predicted = fit_subset(subset, train_index, validation_index)
            new_result = score(predicted, actual)
            ok = math.isfinite(new_result) and (
                new_result >= threshold_f
                if objective_value == "max"
                else new_result <= threshold_f
            )
            if ok:
                keeper_features.append(subset)
    else:
        keeper_features = reduced_test_features

    if not keeper_features:
        if supplied_threshold:
            direction = "increase" if objective_value == "min" else "reduce"
            raise ValueError(f"No epoch subset met [threshold]; {direction} the threshold.")
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
    # Frequency-weighted synthetic X*
    # ----------------------------------------------------------------------
    design_train = x.astype(np.float64)
    design_test = z.astype(np.float64)
    if np.any(~np.isfinite(design_train)):
        raise ValueError("[IVs.train] contains missing or non-finite values.")
    if np.any(~np.isfinite(design_test)):
        raise ValueError("[IVs.test] contains missing or non-finite values.")

    tmin = design_train.min(axis=0)
    tmax = design_train.max(axis=0)
    trange = tmax - tmin
    trange = np.where(~np.isfinite(trange) | (trange == 0), 1.0, trange)
    train_norm = (design_train - tmin) / trange
    test_norm = (design_test - tmin) / trange

    coef_design = np.zeros(n_features, dtype=np.float64)
    for name, w in feature_weights_named.items():
        coef_design[feature_names.index(name)] = w
    if not np.any(coef_design > 0):
        raise ValueError("No positive feature weights were available for the final estimate.")

    xstar_train = train_norm @ coef_design
    xstar_test = test_norm @ coef_design
    if np.any(~np.isfinite(xstar_train)) or np.any(~np.isfinite(xstar_test)):
        raise ValueError("The final synthetic predictor contains non-finite values.")

    def xstar_frame(v: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.column_stack((v, v))

    # ----------------------------------------------------------------------
    # Local n.best selection
    # ----------------------------------------------------------------------
    if ts_test_value is not None:
        final_splits = [validation_index]
    else:
        final_splits = [random_validation_index() for _ in range(5)]

    minimum_train_size = min(n_obs - idx.size for idx in final_splits)
    k_small = max(1, int(math.floor(math.sqrt(minimum_train_size))))
    k_candidates = list(dict.fromkeys(list(range(1, k_small + 1)) + [minimum_train_size]))
    k_scores = np.full(len(k_candidates), np.nan)

    for ki, k_value in enumerate(k_candidates):
        split_scores = np.full(len(final_splits), np.nan)
        for b, test_idx in enumerate(final_splits):
            tr_idx = np.setdiff1d(all_index, test_idx)
            bx, by = balance_training(xstar_frame(xstar_train[tr_idx]), y[tr_idx])
            fit = nns_reg_engine(
                bx,
                by,
                point_est=xstar_frame(xstar_train[test_idx]),
                n_best=min(k_value, bx.shape[0]),
                order=depth_value,
                type="CLASS" if is_class else None,
                factor_2_dummy=False,
                dist="L2",
                point_only=True,
            )
            pred = sanitize_predictions(np.asarray(fit["Point.est"], dtype=np.float64), by)
            split_scores[b] = score(pred, y[test_idx])
        finite_scores = split_scores[np.isfinite(split_scores)]
        if finite_scores.size:
            k_scores[ki] = float(np.mean(finite_scores))
        if status:
            print(
                f"Current NNS.reg(..., n.best = {k_value}) | mean eval(obj.fn) = "
                f"{k_scores[ki]:.6g} | Iterations Remaining = {len(k_candidates) - ki - 1}"
            )

    finite_k = np.flatnonzero(np.isfinite(k_scores))
    if finite_k.size == 0:
        raise ValueError("No n.best candidate produced a finite objective value.")
    if objective_value == "min":
        best_k = k_candidates[int(finite_k[int(np.argmin(k_scores[finite_k]))])]
    else:
        best_k = k_candidates[int(finite_k[int(np.argmax(k_scores[finite_k]))])]

    # ----------------------------------------------------------------------
    # Final fit
    # ----------------------------------------------------------------------
    bx, by = balance_training(xstar_frame(xstar_train), y)
    final_fit = nns_reg_engine(
        bx,
        by,
        point_est=xstar_frame(xstar_test),
        n_best=min(best_k, bx.shape[0]),
        order=depth_value,
        type="CLASS" if is_class else None,
        factor_2_dummy=False,
        dist="L2",
        point_only=False,
        confidence_interval=pred_int,
    )
    estimates_code = sanitize_predictions(
        np.asarray(final_fit["Point.est"], dtype=np.float64), by
    )
    pred_int_out = final_fit["pred.int"]

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
    }
