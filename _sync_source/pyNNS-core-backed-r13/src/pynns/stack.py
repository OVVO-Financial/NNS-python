from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from pynns.categorical import _balance_class_training, _dense_factor_codes
from pynns.central_tendencies import nns_mode
from pynns.dependence import _gravity
from pynns.regression import (
    Order,
    _expand_factor_predictors,
    _normalize_type,
    _prepare_y_values,
    _r_minmax_columns,
    _round_clamp_classes,
    nns_reg,
)

Objective = Literal["min", "max"]
Method = int | Sequence[int]
StackResult = dict[str, Any]


def nns_stack(
    ivs_train: NDArray[np.float64],
    dv_train: NDArray[np.float64],
    ivs_test: NDArray[np.float64] | None = None,
    *,
    type: str | None = None,
    obj_fn: Callable[[NDArray[np.float64], NDArray[np.float64]], float] | None = None,
    objective: Objective = "min",
    optimize_threshold: bool = True,
    dist: str = "L2",
    cv_size: float | None = None,
    balance: bool = False,
    ts_test: int | None = None,
    folds: int = 5,
    order: Order = None,
    method: Method = (1, 2),
    stack: bool = True,
    dim_red_method: object = "cor",
    pred_int: float | None = None,
    status: bool = False,
    ncores: int | None = None,
    class_levels: list[object] | None = None,
    factor_levels: Sequence[object] | Sequence[Sequence[object] | None] | None = None,
    random_seed: int | None = None,
) -> StackResult:
    """Port of R's deterministic numeric/classification NNS.stack orchestration."""
    del optimize_threshold, status, ncores
    type_value = _normalize_type(type)
    if balance:
        type_value = "class"
    methods = _methods(method)
    x_input: NDArray[Any] | NDArray[np.float64] = np.asarray(ivs_train)
    x_test_input: NDArray[Any] | NDArray[np.float64] | None = (
        None if ivs_test is None else np.asarray(ivs_test)
    )
    all_factor_predictors = False
    if factor_levels is not None and 2 in methods:
        all_factor_predictors = _all_predictors_are_factor(x_input, factor_levels)
        if all_factor_predictors:
            methods = (1,)
    mixed_factor = False
    raw_columns = 0
    if factor_levels is not None and not all_factor_predictors:
        mixed_factor = any(level is not None for level in factor_levels)
        if mixed_factor:
            raw_columns = x_input.shape[1] if x_input.ndim > 1 else 1
    if factor_levels is not None:
        x_input, x_test_input = _expand_factor_predictors(
            ivs_train,
            ivs_test,
            factor_levels=factor_levels,
        )

    x_train = _as_matrix(x_input, "ivs_train")
    if balance:
        y_train, class_codes = _dense_factor_codes(dv_train, levels=class_levels)
    elif type_value == "class":
        y_train, _ = _prepare_y_values(dv_train, type_value=type_value, class_levels=class_levels)
        class_codes = np.unique(y_train[np.isfinite(y_train)])
    else:
        y_train = _as_vector(dv_train, "dv_train")
        class_codes = np.empty(0, dtype=np.float64)
    if x_train.shape[0] != y_train.size:
        raise ValueError("ivs_train and dv_train must have the same row count.")
    x_test = (
        x_train.copy() if x_test_input is None else _as_point_matrix(x_test_input, x_train.shape[1])
    )
    if balance:
        rng = np.random.default_rng(random_seed)
        x_train, y_train = _balance_class_training(
            x_train,
            y_train,
            classes=class_codes,
            rng=rng,
        )
    objective_l = objective.lower()
    if objective_l not in {"min", "max"}:
        raise ValueError("objective must be 'min' or 'max'.")
    objective_value = cast(Objective, objective_l)
    if type_value == "class" and obj_fn is None:
        objective_value = "max"
        objective_fn: Callable[[NDArray[np.float64], NDArray[np.float64]], float] = _accuracy
    else:
        objective_fn = _sse if obj_fn is None else obj_fn

    if x_train.shape[1] == 1:
        methods = (1,)
        order = None

    cv_fraction = 0.25 if cv_size is None else float(cv_size)
    if not 0.0 < cv_fraction <= 1.0:
        raise ValueError("cv_size must be in (0, 1].")
    if folds < 1:
        raise ValueError("folds must be >= 1.")
    ts_test_value = None if ts_test is None else int(ts_test)

    method2_state = _evaluate_method2(
        x_train,
        y_train,
        x_test,
        methods=methods,
        mixed_factor=mixed_factor,
        raw_columns=raw_columns,
        objective=objective_value,
        objective_fn=objective_fn,
        cv_size=cv_fraction,
        folds=folds,
        order=order,
        stack=stack,
        dim_red_method=dim_red_method,
        dist=dist,
        ts_test=ts_test_value,
        pred_int=pred_int,
        type_value=type_value,
    )
    method1_state = _evaluate_method1(
        x_train,
        y_train,
        x_test,
        methods=methods,
        mixed_factor=mixed_factor,
        raw_columns=raw_columns,
        objective=objective_value,
        objective_fn=objective_fn,
        cv_size=cv_fraction,
        folds=folds,
        order=order,
        stack=stack,
        dim_red_method=dim_red_method,
        dist=dist,
        method2_state=method2_state,
        ts_test=ts_test_value,
        pred_int=pred_int,
        type_value=type_value,
    )

    reg = method1_state.prediction
    dimred = method2_state.prediction
    reg_obj = method1_state.objective
    dimred_obj = method2_state.objective

    estimates: NDArray[np.float64]
    if methods == (1, 2):
        reg_clean, dimred_clean = _fill_pairwise_na(reg, dimred)
        weights = _stack_weights(reg_obj, dimred_obj, methods, objective_value)
        estimates = weights[0] * reg_clean + weights[1] * dimred_clean
        stacked_pred_int = _combine_prediction_intervals(
            method1_state.pred_int,
            method2_state.pred_int,
            weights,
        )
    elif methods == (1,):
        estimates = reg
        stacked_pred_int = method1_state.pred_int
    else:
        estimates = dimred
        stacked_pred_int = method2_state.pred_int
    probability_threshold = _probability_threshold(
        method1_state.class_threshold,
        method2_state.class_threshold,
        type_value=type_value,
    )
    if type_value == "class":
        estimates = _class_threshold_round(estimates, probability_threshold, y_train)
        if methods == (1, 2):
            stacked_pred_int = _round_class_prediction_intervals(stacked_pred_int)

    return {
        "OBJfn.reg": reg_obj,
        "NNS.reg.n.best": method1_state.parameter,
        "probability.threshold": probability_threshold,
        "OBJfn.dim.red": dimred_obj,
        "NNS.dim.red.threshold": method2_state.parameter,
        "reg": reg,
        "reg.pred.int": method1_state.pred_int,
        "dim.red": dimred,
        "dim.red.pred.int": method2_state.pred_int,
        "stack": estimates,
        "pred.int": stacked_pred_int,
    }


class _MethodState:
    def __init__(
        self,
        prediction: NDArray[np.float64],
        objective: float,
        parameter: float,
        train_star: NDArray[np.float64] | None = None,
        test_star: NDArray[np.float64] | None = None,
        relevant_vars: NDArray[np.int64] | None = None,
        pred_int: dict[str, NDArray[np.float64]] | None = None,
        class_threshold: float | None = None,
    ) -> None:
        self.prediction = prediction
        self.objective = objective
        self.parameter = parameter
        self.train_star = train_star
        self.test_star = test_star
        self.relevant_vars = relevant_vars
        self.pred_int = pred_int
        self.class_threshold = class_threshold


def _evaluate_method2(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    x_test: NDArray[np.float64],
    *,
    methods: tuple[int, ...],
    mixed_factor: bool,
    raw_columns: int,
    objective: Objective,
    objective_fn: Callable[[NDArray[np.float64], NDArray[np.float64]], float],
    cv_size: float,
    folds: int,
    order: Order,
    stack: bool,
    dim_red_method: object,
    dist: str,
    ts_test: int | None,
    pred_int: float | None,
    type_value: str | None,
) -> _MethodState:
    n_rows, n_cols = x_train.shape
    if 2 not in methods or n_cols <= 1:
        obj = math.inf if objective == "min" else -math.inf
        return _MethodState(np.full(x_test.shape[0], np.nan), obj, math.nan)

    thresholds: list[float] = []
    fold_scores: list[float] = []
    threshold_results: list[float] = []
    train_star: NDArray[np.float64] | None = None
    test_star: NDArray[np.float64] | None = None
    relevant_vars = np.arange(n_cols, dtype=np.int64)

    for fold in range(1, folds + 1):
        train_idx, test_idx = _cv_split(n_rows, fold, cv_size, ts_test)
        cv_x_train = x_train[train_idx]
        cv_y_train = y_train[train_idx]
        cv_x_test = x_train[test_idx]
        cv_y_test = y_train[test_idx]

        cutoffs = _threshold_grid(cv_x_train, cv_y_train, dim_red_method, order, dist)
        scores = np.empty(cutoffs.size, dtype=np.float64)
        class_thresholds = np.empty(cutoffs.size, dtype=np.float64)
        for idx, cutoff in enumerate(cutoffs):
            predicted = _reg_point_est(
                cv_x_train,
                cv_y_train,
                cv_x_test,
                order=order,
                dim_red_method=dim_red_method,
                threshold=float(cutoff),
                dist=dist,
            )
            predicted = _fill_nan_with_gravity(predicted)
            if type_value == "class":
                class_thresholds[idx] = _classification_threshold(predicted, cv_y_test)
                predicted = _class_threshold_round(predicted, class_thresholds[idx], cv_y_train)
                threshold_results.append(float(class_thresholds[idx]))
            else:
                class_thresholds[idx] = math.nan
            scores[idx] = objective_fn(predicted, cv_y_test)
        best_index = int(np.nanargmin(scores) if objective == "min" else np.nanargmax(scores))
        best_threshold = float(cutoffs[best_index])
        thresholds.append(best_threshold)
        fold_scores.append(float(scores[best_index]))

        if stack and methods == (1, 2):
            fit = nns_reg(
                cv_x_train,
                cv_y_train,
                point_est=cv_x_test,
                dim_red_method=dim_red_method,
                threshold=best_threshold,
                order=order,
                dist=dist,
                point_only=False,
            )
            train_star = cast(dict[str, NDArray[np.float64]], fit["x.star"])["x"]
            test_star = _xstar_for_points(
                fit,
                cv_x_train,
                cv_x_test,
                mixed_factor=mixed_factor,
                raw_columns=raw_columns,
            )

    final_threshold = _threshold_mode(thresholds)
    final_class_threshold = (
        _threshold_mode(threshold_results) if type_value == "class" else math.nan
    )
    final_fit = nns_reg(
        x_train,
        y_train,
        point_est=x_test,
        dim_red_method=dim_red_method,
        threshold=final_threshold,
        order=order,
        dist=dist,
        point_only=False,
        confidence_interval=pred_int,
        type=type_value,
    )
    fitted = cast(dict[str, NDArray[np.float64]], final_fit["Fitted.xy"])
    fitted_yhat = fitted["y.hat"]
    prediction = _as_prediction(final_fit["Point.est"], x_test.shape[0])
    if type_value == "class":
        if not np.isfinite(final_class_threshold):
            final_class_threshold = _classification_threshold(fitted_yhat, y_train)
        fitted_yhat = _class_threshold_round(fitted_yhat, final_class_threshold, y_train)
        prediction = _class_threshold_round(prediction, final_class_threshold, y_train)
    final_obj = objective_fn(fitted_yhat, fitted["y"])
    final_pred_int = cast(dict[str, NDArray[np.float64]] | None, final_fit["pred.int"])
    final_pred_int = _prediction_interval_or_point_estimate(final_pred_int, prediction)

    if stack and methods == (1, 2):
        train_star = cast(dict[str, NDArray[np.float64]], final_fit["x.star"])["x"]
        test_star = _xstar_for_points(
            final_fit,
            x_train,
            x_test,
            mixed_factor=mixed_factor,
            raw_columns=raw_columns,
        )
    equation = cast(dict[str, NDArray[np.float64]], final_fit["equation"])
    coef = equation["Coefficient"][:-1]
    relevant_vars = np.flatnonzero(coef > 0.0).astype(np.int64)
    if relevant_vars.size == 0:
        relevant_vars = np.arange(n_cols, dtype=np.int64)

    return _MethodState(
        prediction=prediction,
        objective=final_obj,
        parameter=final_threshold,
        train_star=train_star,
        test_star=test_star,
        relevant_vars=relevant_vars,
        pred_int=final_pred_int,
        class_threshold=final_class_threshold if type_value == "class" else None,
    )


def _evaluate_method1(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    x_test: NDArray[np.float64],
    *,
    methods: tuple[int, ...],
    mixed_factor: bool,
    raw_columns: int,
    objective: Objective,
    objective_fn: Callable[[NDArray[np.float64], NDArray[np.float64]], float],
    cv_size: float,
    folds: int,
    order: Order,
    stack: bool,
    dim_red_method: object,
    dist: str,
    method2_state: _MethodState,
    ts_test: int | None,
    pred_int: float | None,
    type_value: str | None,
) -> _MethodState:
    if 1 not in methods:
        obj = math.inf if objective == "min" else -math.inf
        return _MethodState(np.full(x_test.shape[0], np.nan), obj, math.nan)

    n_rows = x_train.shape[0]
    l_value = max(1, math.floor(math.sqrt(n_rows)))
    k_candidates = [*list(range(1, l_value + 1)), n_rows]
    best_ks: list[int] = []
    fold_scores: list[float] = []
    threshold_results: list[float] = []

    for fold in range(1, folds + 1):
        train_idx, test_idx = _cv_split(n_rows, fold, cv_size, ts_test)
        cv_x_train = x_train[train_idx]
        cv_y_train = y_train[train_idx]
        cv_x_test = x_train[test_idx]
        cv_y_test = y_train[test_idx]

        if stack and methods == (1, 2) and method2_state.train_star is not None:
            fold_train_star, fold_test_star = _fold_xstar(
                cv_x_train,
                cv_y_train,
                cv_x_test,
                cv_y_test,
                mixed_factor=mixed_factor,
                raw_columns=raw_columns,
                objective=objective,
                objective_fn=objective_fn,
                order=order,
                dim_red_method=dim_red_method,
                dist=dist,
                type_value=type_value,
            )
            cv_x_train = np.column_stack((fold_train_star, fold_train_star))
            cv_x_test = np.column_stack((fold_test_star, fold_test_star))
        elif method2_state.relevant_vars is not None and method2_state.relevant_vars.size:
            cv_x_train = cv_x_train[:, method2_state.relevant_vars]
            cv_x_test = cv_x_test[:, method2_state.relevant_vars]

        setup = nns_reg(
            cv_x_train,
            cv_y_train,
            point_est=cv_x_test,
            n_best=1,
            order=order,
            dist=dist,
            point_only=False,
            type=type_value,
        )
        fitted = cast(dict[str, NDArray[np.float64]], setup["Fitted.xy"])
        yhat_vec = fitted["y.hat"]
        setup_prediction = _as_prediction(setup["Point.est"], cv_x_test.shape[0])
        path_predictions = _distance_path_predictions(
            cv_x_train,
            yhat_vec,
            cv_x_test,
            min(l_value, cv_x_train.shape[0]),
        )
        all_prediction = _distance_bulk_prediction(
            cv_x_train,
            yhat_vec,
            cv_x_test,
            min(n_rows, cv_x_train.shape[0]),
        )

        scores: list[float] = []
        tested_ks: list[int] = []
        class_thresholds: list[float] = []
        for k_value in k_candidates:
            if k_value == 1:
                predicted = setup_prediction
                if type_value == "class" and np.any(np.isnan(predicted)):
                    predicted = predicted.copy()
                    predicted[np.isnan(predicted)] = float(np.nanmean(predicted))
            elif k_value <= path_predictions.shape[1]:
                predicted = path_predictions[:, k_value - 1]
            else:
                predicted = all_prediction
            if type_value == "class":
                threshold_value = _classification_threshold(
                    predicted,
                    cv_y_test,
                    tie="first" if k_value == 1 else "median",
                )
                predicted = _class_threshold_round(predicted, threshold_value, cv_y_train)
                threshold_results.append(threshold_value)
            else:
                threshold_value = math.nan
            score = objective_fn(predicted, cv_y_test)
            scores.append(float(score))
            tested_ks.append(k_value)
            class_thresholds.append(threshold_value)
            if len(scores) > 3:
                if objective == "min" and scores[-1] >= scores[-2] and scores[-1] >= scores[-3]:
                    break
                if objective == "max" and scores[-1] <= scores[-2] and scores[-1] <= scores[-3]:
                    break
        scores_arr = np.asarray(scores, dtype=np.float64)
        best_index = int(
            np.nanargmin(scores_arr) if objective == "min" else np.nanargmax(scores_arr)
        )
        best_ks.append(tested_ks[best_index])
        fold_scores.append(float(scores_arr[best_index]))

    best_k = int(_round_k_mode(np.asarray(best_ks, dtype=np.float64)))
    final_class_threshold = (
        _threshold_mode(threshold_results) if type_value == "class" else math.nan
    )

    if stack and methods == (1, 2) and method2_state.train_star is not None:
        if method2_state.test_star is None:
            raise RuntimeError("stacked Method 1 requires Method 2 test projections.")
        full_x_train = np.column_stack((method2_state.train_star, method2_state.train_star))
        full_x_test = np.column_stack((method2_state.test_star, method2_state.test_star))
    elif method2_state.relevant_vars is not None and method2_state.relevant_vars.size:
        full_x_train = x_train[:, method2_state.relevant_vars]
        full_x_test = x_test[:, method2_state.relevant_vars]
    else:
        full_x_train = x_train
        full_x_test = x_test

    final_fit = nns_reg(
        full_x_train,
        y_train,
        point_est=full_x_test,
        n_best=best_k,
        order=order,
        dist=dist,
        point_only=False,
        confidence_interval=pred_int,
        type=type_value,
    )
    fitted = cast(dict[str, NDArray[np.float64]], final_fit["Fitted.xy"])
    prediction = _as_prediction(final_fit["Point.est"], x_test.shape[0])
    fitted_yhat = fitted["y.hat"]
    if type_value == "class":
        if not np.isfinite(final_class_threshold):
            final_class_threshold = _classification_threshold(fitted_yhat, y_train)
        fitted_yhat = _class_threshold_round(fitted_yhat, final_class_threshold, y_train)
        prediction = _class_threshold_round(prediction, final_class_threshold, y_train)
    final_obj = objective_fn(fitted_yhat, fitted["y"])
    final_pred_int = cast(dict[str, NDArray[np.float64]] | None, final_fit["pred.int"])
    final_pred_int = _prediction_interval_or_point_estimate(final_pred_int, prediction)
    return _MethodState(
        prediction=prediction,
        objective=final_obj,
        parameter=float(best_k),
        pred_int=final_pred_int,
        class_threshold=final_class_threshold if type_value == "class" else None,
    )


def _fold_xstar(
    cv_x_train: NDArray[np.float64],
    cv_y_train: NDArray[np.float64],
    cv_x_test: NDArray[np.float64],
    cv_y_test: NDArray[np.float64],
    *,
    mixed_factor: bool,
    raw_columns: int,
    objective: Objective,
    objective_fn: Callable[[NDArray[np.float64], NDArray[np.float64]], float],
    order: Order,
    dim_red_method: object,
    dist: str,
    type_value: str | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    cutoffs = _threshold_grid(cv_x_train, cv_y_train, dim_red_method, order, dist)
    scores = np.empty(cutoffs.size, dtype=np.float64)
    for idx, cutoff in enumerate(cutoffs):
        predicted = _reg_point_est(
            cv_x_train,
            cv_y_train,
            cv_x_test,
            order=order,
            dim_red_method=dim_red_method,
            threshold=float(cutoff),
            dist=dist,
        )
        predicted = _fill_nan_with_gravity(predicted)
        if type_value == "class":
            threshold = _classification_threshold(predicted, cv_y_test)
            predicted = _class_threshold_round(predicted, threshold, cv_y_train)
        scores[idx] = objective_fn(predicted, cv_y_test)
    best_index = int(np.nanargmin(scores) if objective == "min" else np.nanargmax(scores))
    fit = nns_reg(
        cv_x_train,
        cv_y_train,
        point_est=cv_x_test,
        dim_red_method=dim_red_method,
        threshold=float(cutoffs[best_index]),
        order=order,
        dist=dist,
        point_only=False,
    )
    return cast(dict[str, NDArray[np.float64]], fit["x.star"])["x"], _xstar_for_points(
        fit,
        cv_x_train,
        cv_x_test,
        mixed_factor=mixed_factor,
        raw_columns=raw_columns,
    )


def _distance_path_predictions(
    features: NDArray[np.float64],
    yhat: NDArray[np.float64],
    x_test: NDArray[np.float64],
    kmax: int,
) -> NDArray[np.float64]:
    if kmax < 1:
        return np.empty((x_test.shape[0], 0), dtype=np.float64)
    dist = _stack_distances(features, x_test)
    order = np.argsort(dist, axis=1, kind="quicksort")[:, :kmax]
    sorted_dist = np.take_along_axis(dist, order, axis=1)
    sorted_y = yhat[order]
    with np.errstate(divide="ignore", over="ignore"):
        weights = 1.0 / sorted_dist
    with np.errstate(over="ignore", invalid="ignore"):
        csum_weights = np.cumsum(weights, axis=1)
        csum_y = np.cumsum(weights * sorted_y, axis=1)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        return np.asarray(csum_y / csum_weights, dtype=np.float64)


def _distance_bulk_prediction(
    features: NDArray[np.float64],
    yhat: NDArray[np.float64],
    x_test: NDArray[np.float64],
    k: int,
) -> NDArray[np.float64]:
    dist = _stack_distances(features, x_test)
    order = np.argsort(dist, axis=1, kind="quicksort")[:, :k]
    row_dist = np.take_along_axis(dist, order, axis=1)
    row_y = yhat[order]
    with np.errstate(divide="ignore", over="ignore"):
        weights = 1.0 / row_dist
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        return np.asarray(
            np.sum(weights * row_y, axis=1) / np.sum(weights, axis=1),
            dtype=np.float64,
        )


def _stack_distances(
    features: NDArray[np.float64],
    x_test: NDArray[np.float64],
) -> NDArray[np.float64]:
    rpm_rows = np.ravel(features, order="F").reshape(features.shape, order="C")
    test_rows = np.ravel(x_test, order="F").reshape(x_test.shape, order="C")
    diff = rpm_rows[np.newaxis, :, :] - test_rows[:, np.newaxis, :]
    distances = np.sum(diff * diff + np.abs(diff), axis=2)
    distances[distances == 0.0] = 1e-12
    return np.asarray(distances, dtype=np.float64)


def _threshold_grid(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    dim_red_method: object,
    order: Order,
    dist: str,
) -> NDArray[np.float64]:
    if isinstance(dim_red_method, str) and dim_red_method.lower() == "cor":
        scores = _spearman_scores(x, y)
    elif isinstance(dim_red_method, str) and dim_red_method.lower() == "equal":
        return np.array([0.0], dtype=np.float64)
    else:
        fit = nns_reg(
            x,
            y,
            dim_red_method=dim_red_method,
            order=order,
            dist=dist,
            point_only=True,
        )
        equation = cast(dict[str, NDArray[np.float64]], fit["equation"])
        scores = np.abs(np.round(equation["Coefficient"][:-1], 2))
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores[(scores < 1.0) & (scores >= 0.0)]
    scores[~np.isfinite(scores)] = 0.0
    unique = np.unique(scores)[::-1]
    if unique.size > 0:
        unique = unique[1:]
    if unique.size == 0:
        unique = np.array([0.0], dtype=np.float64)
    if x.shape[1] == 2:
        unique = np.unique(np.concatenate((unique, np.array([0.0]))))
    return unique.astype(np.float64)


def _reg_point_est(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    point_est: NDArray[np.float64],
    *,
    order: Order,
    dim_red_method: object | None = None,
    threshold: float = 0.0,
    n_best: int | None = None,
    dist: str,
) -> NDArray[np.float64]:
    result = nns_reg(
        x,
        y,
        point_est=point_est,
        dim_red_method=dim_red_method,
        threshold=threshold,
        order=order,
        n_best=n_best,
        dist=dist,
        point_only=True,
    )
    return _as_prediction(result["Point.est"], point_est.shape[0])


def _xstar_for_points(
    fit: dict[str, Any],
    train_x: NDArray[np.float64],
    test_x: NDArray[np.float64],
    *,
    mixed_factor: bool,
    raw_columns: int,
) -> NDArray[np.float64]:
    equation = cast(dict[str, NDArray[np.float64]], fit["equation"])
    coef = np.asarray(equation["Coefficient"][:-1], dtype=np.float64)
    active = int(np.sum(np.abs(coef) > 0.0))
    if active == 0:
        active = 1
    if mixed_factor and raw_columns and coef.size != raw_columns:
        fallback = cast(dict[str, NDArray[np.float64]], fit["x.star"])["x"]
        return np.full(test_x.shape[0], float(np.mean(fallback)), dtype=np.float64)
    if coef.size != test_x.shape[1]:
        fallback = cast(dict[str, NDArray[np.float64]], fit["x.star"])["x"]
        return np.full(test_x.shape[0], float(np.mean(fallback)), dtype=np.float64)
    joint = np.vstack((test_x, train_x))
    norm = _r_minmax_columns(joint, zero_guard=True)
    out = np.asarray(norm[: test_x.shape[0]] @ coef / active, dtype=np.float64)
    return _fill_nan_with_gravity(out)


def _cv_split(
    n_rows: int,
    fold: int,
    cv_size: float,
    ts_test: int | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    if ts_test is not None:
        if ts_test < 1 or ts_test > n_rows:
            raise ValueError("ts_test must be in [1, n_rows].")
        test_idx = np.arange(0, n_rows - ts_test, dtype=np.int64)
        train_idx = np.arange(n_rows - ts_test, n_rows, dtype=np.int64)
        if train_idx.size < 2:
            raise ValueError("ts_test leaves too few training rows.")
        return train_idx, test_idx

    test_count = int(cv_size * n_rows)
    if test_count < 1:
        test_count = 1
    one_based = np.linspace(fold, n_rows, test_count).astype(np.int64)
    test_idx = np.clip(one_based - 1, 0, n_rows - 1)
    mask = np.ones(n_rows, dtype=bool)
    mask[np.unique(test_idx)] = False
    train_idx = np.flatnonzero(mask).astype(np.int64)
    if train_idx.size == 0:
        raise ValueError("cv_size leaves no training rows.")
    return train_idx, test_idx.astype(np.int64)


def _spearman_scores(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    y_rank = _rank_average(y)
    scores = np.empty(x.shape[1], dtype=np.float64)
    for col in range(x.shape[1]):
        scores[col] = abs(round(_pearson(_rank_average(x[:, col]), y_rank), 2))
    scores[~np.isfinite(scores)] = 0.0
    return scores


def _rank_average(values: NDArray[np.float64]) -> NDArray[np.float64]:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def _pearson(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    x_centered = x - float(np.mean(x))
    y_centered = y - float(np.mean(y))
    denom = math.sqrt(float(np.sum(x_centered**2) * np.sum(y_centered**2)))
    if denom == 0.0:
        return 0.0
    return float(np.sum(x_centered * y_centered) / denom)


def _threshold_mode(values: list[float]) -> float:
    if not values:
        return math.nan
    unique, counts = np.unique(np.asarray(values, dtype=np.float64), return_counts=True)
    tied = unique[counts == int(np.max(counts))]
    out = _gravity(tied)
    return 0.0 if not np.isfinite(out) else float(out)


def _round_k_mode(values: NDArray[np.float64]) -> int:
    mode_value = float(nns_mode(values, discrete=True))
    return int(math.floor(mode_value) if mode_value % 1.0 < 0.5 else math.ceil(mode_value))


def _stack_weights(
    reg_obj: float,
    dimred_obj: float,
    methods: tuple[int, ...],
    objective: Objective,
) -> NDArray[np.float64]:
    values = np.array([reg_obj, dimred_obj], dtype=np.float64)
    values[values == 0.0] = 1e-10
    if objective == "min":
        with np.errstate(divide="ignore"):
            weights = np.maximum(1e-10, 1.0 / (values**2))
    else:
        weights = np.maximum(1e-10, values**2)
    mask = np.array([1 in methods, 2 in methods], dtype=bool)
    weights[~mask] = 0.0
    weights[~np.isfinite(weights)] = 0.0
    total = float(np.sum(weights))
    if total > 0.0:
        return weights / total
    return np.array([0.5, 0.5], dtype=np.float64)


def _combine_prediction_intervals(
    left: dict[str, NDArray[np.float64]] | None,
    right: dict[str, NDArray[np.float64]] | None,
    weights: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]] | None:
    if left is None and right is None:
        return None
    if left is None:
        return right
    if right is None:
        return left
    left_values = list(left.values())
    right_values = list(right.values())
    if len(left_values) != len(right_values):
        raise ValueError("Cannot combine prediction intervals with different column counts.")
    return {
        key: weights[0] * left_values[index] + weights[1] * right_values[index]
        for index, key in enumerate(left)
    }


def _prediction_interval_or_point_estimate(
    pred_int: dict[str, NDArray[np.float64]] | None,
    prediction: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]] | None:
    if pred_int is None:
        return None
    expected = prediction.shape
    return {
        key: values if values.shape == expected else prediction.copy()
        for key, values in pred_int.items()
    }


def _round_class_prediction_intervals(
    pred_int: dict[str, NDArray[np.float64]] | None,
) -> dict[str, NDArray[np.float64]] | None:
    if pred_int is None:
        return None
    return {
        key: np.where(values % 1.0 < 0.5, np.floor(values), np.ceil(values)).astype(np.float64)
        for key, values in pred_int.items()
    }


def _fill_pairwise_na(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    a = left.copy()
    b = right.copy()
    a[np.isnan(a)] = b[np.isnan(a)]
    b[np.isnan(b)] = a[np.isnan(b)]
    return a, b


def _fill_nan_with_gravity(values: NDArray[np.float64]) -> NDArray[np.float64]:
    out = np.asarray(values, dtype=np.float64).copy()
    if np.any(np.isnan(out)):
        finite = out[np.isfinite(out)]
        fill = _gravity(finite) if finite.size else 0.0
        out[np.isnan(out)] = fill
    return out


def _as_prediction(value: object, length: int) -> NDArray[np.float64]:
    if value is None:
        return np.full(length, np.nan, dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return np.full(length, np.nan, dtype=np.float64)
    return arr


def _sse(predicted: NDArray[np.float64], actual: NDArray[np.float64]) -> float:
    return float(np.sum((predicted - actual) ** 2))


def _accuracy(predicted: NDArray[np.float64], actual: NDArray[np.float64]) -> float:
    return float(np.mean(np.asarray(predicted, dtype=np.float64) == actual))


def _classification_threshold(
    predicted: NDArray[np.float64],
    actual: NDArray[np.float64],
    *,
    tie: Literal["first", "median"] = "median",
) -> float:
    values = np.asarray(predicted, dtype=np.float64)
    if np.unique(values).size == 1:
        return 0.01 if tie == "first" else 0.5
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    scores = np.empty(grid.size, dtype=np.float64)
    for index, threshold in enumerate(grid):
        rounded = np.where(values % 1.0 < threshold, np.floor(values), np.ceil(values))
        scores[index] = np.mean(rounded == actual)
    best = np.flatnonzero(scores == float(np.max(scores)))
    return float(grid[int(best[0] if tie == "first" else np.median(best))])


def _class_threshold_round(
    values: NDArray[np.float64],
    threshold: float,
    y_train: NDArray[np.float64],
) -> NDArray[np.float64]:
    threshold_value = 0.5 if not np.isfinite(threshold) else float(threshold)
    rounded = np.where(values % 1.0 < threshold_value, np.floor(values), np.ceil(values))
    return _round_clamp_classes(rounded, y_train)


def _probability_threshold(
    method1: float | None,
    method2: float | None,
    *,
    type_value: str | None,
) -> float:
    if type_value != "class":
        return 0.5
    values = np.asarray(
        [value for value in (method1, method2) if value is not None and np.isfinite(value)],
        dtype=np.float64,
    )
    if values.size == 0:
        return 0.5
    return float(np.mean(values))


def _methods(method: Method) -> tuple[int, ...]:
    values: tuple[int, ...]
    if isinstance(method, int):
        values = (method,)
    else:
        values = tuple(int(item) for item in method)
    values = tuple(sorted(values))
    if not values or any(item not in {1, 2} for item in values):
        raise ValueError("method must contain 1, 2, or both.")
    return values


def _all_predictors_are_factor(
    x: NDArray[Any],
    factor_levels: Sequence[object] | Sequence[Sequence[object] | None],
) -> bool:
    if x.ndim <= 1:
        return True
    levels_by_column = cast(Sequence[Sequence[object] | None], factor_levels)
    if len(levels_by_column) < x.shape[1]:
        raise ValueError("factor_levels must provide levels for every predictor column.")
    return all(levels_by_column[col] is not None for col in range(x.shape[1]))


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
