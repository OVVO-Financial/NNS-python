from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from nns._stack_method1 import CandidateId, Method1FoldPredictions, select_method1_candidate
from nns.distance import nns_distance_path_single_bulk


def evaluate_method1(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    x_test: NDArray[np.float64],
    **kwargs: Any,
) -> Any:
    """Repaired Method 1 implementation installed into :mod:`nns.stack`.

    Candidate selection is based on complete pooled OOF predictions. Local
    candidates are bounded by ``floor(sqrt(n))`` and ``ALL`` is evaluated as a
    separate single-k limit condition. Genuine univariate designs use the
    univariate production estimator directly and do not invent a k path.
    """

    from nns import stack as s

    methods = kwargs["methods"]
    objective = kwargs["objective"]
    objective_fn = kwargs["objective_fn"]
    cv_size = kwargs["cv_size"]
    folds = kwargs["folds"]
    order = kwargs["order"]
    stack_enabled = kwargs["stack"]
    dim_red_method = kwargs["dim_red_method"]
    dist = kwargs["dist"]
    method2_state = kwargs["method2_state"]
    ts_test = kwargs["ts_test"]
    pred_int = kwargs["pred_int"]
    type_value = kwargs["type_value"]
    mixed_factor = kwargs["mixed_factor"]
    raw_columns = kwargs["raw_columns"]
    status = kwargs.get("status", False)
    ncores = kwargs.get("ncores", 1)

    if 1 not in methods:
        obj = math.inf if objective == "min" else -math.inf
        return s._MethodState(np.full(x_test.shape[0], np.nan), obj, math.nan)

    n_rows = x_train.shape[0]
    local_candidates = tuple(range(1, max(1, math.floor(math.sqrt(n_rows))) + 1))
    fold_predictions: list[Method1FoldPredictions] = []

    for fold in range(1, folds + 1):
        train_idx, valid_idx = s._cv_split(n_rows, fold, cv_size, ts_test)
        fold_x_train = x_train[train_idx]
        fold_y_train = y_train[train_idx]
        fold_x_valid = x_train[valid_idx]
        fold_y_valid = y_train[valid_idx]

        if stack_enabled and methods == (1, 2) and method2_state.train_star is not None:
            fold_train_star, fold_valid_star = s._fold_xstar(
                fold_x_train,
                fold_y_train,
                fold_x_valid,
                fold_y_valid,
                mixed_factor=mixed_factor,
                raw_columns=raw_columns,
                objective=objective,
                objective_fn=objective_fn,
                order=order,
                dim_red_method=dim_red_method,
                dist=dist,
                type_value=type_value,
            )
            fold_x_train = np.column_stack((fold_train_star, fold_train_star))
            fold_x_valid = np.column_stack((fold_valid_star, fold_valid_star))
        elif method2_state.relevant_vars is not None and method2_state.relevant_vars.size:
            fold_x_train = fold_x_train[:, method2_state.relevant_vars]
            fold_x_valid = fold_x_valid[:, method2_state.relevant_vars]

        if fold_x_train.shape[1] == 1:
            direct = s.nns_reg(
                fold_x_train[:, 0],
                fold_y_train,
                point_est=fold_x_valid[:, 0],
                order=None,
                dist=dist,
                point_only=True,
                type=type_value,
                ncores=ncores,
            )
            prediction = s._as_prediction(direct["Point.est"], valid_idx.size)
            fold_predictions.append(
                Method1FoldPredictions(valid_idx, {1: prediction, "all": prediction.copy()})
            )
            continue

        model = s._mreg_prepare_model(
            fold_x_train,
            fold_y_train,
            order=order,
            noise_reduction="mode_class" if type_value == "class" else "off",
            is_class=type_value == "class",
        )
        local_counts = [max(1, min(k, model.rpm.shape[0])) for k in local_candidates]
        local_path = s._mreg_predict_path(
            model,
            fold_x_valid,
            k_values=local_counts,
            is_class=type_value == "class",
        )
        predictions: dict[CandidateId, NDArray[np.float64]] = {
            k: np.asarray(local_path[count], dtype=np.float64).copy()
            for k, count in zip(local_candidates, local_counts, strict=True)
        }
        class_arg = "class" if type_value == "class" else None
        predictions["all"] = np.asarray(
            nns_distance_path_single_bulk(
                model.rpm,
                fold_x_valid,
                model.rpm.shape[0],
                class_arg,
            ),
            dtype=np.float64,
        )
        fold_predictions.append(Method1FoldPredictions(valid_idx, predictions))
        if status:
            print(f"Method 1 fold {fold}/{folds} complete")

    if x_train.shape[1] == 1 and not (
        stack_enabled and methods == (1, 2) and method2_state.train_star is not None
    ):
        selected: CandidateId = 1
        pooled_score = math.nan
        pooled_threshold = math.nan
        # The selector is still used for coverage validation; ALL is an alias
        # only inside this compatibility layer and cannot change the winner.
        def univariate_eval(
            candidate: CandidateId,
            predicted: NDArray[np.float64],
            actual: NDArray[np.float64],
        ) -> tuple[float, float]:
            threshold = (
                s._classification_threshold(predicted, actual, tie="first")
                if type_value == "class"
                else math.nan
            )
            evaluated = (
                s._class_threshold_round(predicted, threshold, y_train)
                if type_value == "class"
                else predicted
            )
            return float(objective_fn(evaluated, actual)), threshold

        selection = select_method1_candidate(
            n_obs=n_rows,
            actual=y_train,
            folds=fold_predictions,
            local_candidates=[1],
            evaluator=univariate_eval,
            objective=objective,
        )
        pooled_score = selection.scores[1]
        pooled_threshold = selection.thresholds[1]
    else:
        def evaluator(
            candidate: CandidateId,
            predicted: NDArray[np.float64],
            actual: NDArray[np.float64],
        ) -> tuple[float, float]:
            if type_value != "class":
                return float(objective_fn(predicted, actual)), math.nan
            threshold = s._classification_threshold(
                predicted,
                actual,
                tie="first" if candidate == 1 else "median",
            )
            evaluated = s._class_threshold_round(predicted, threshold, y_train)
            return float(objective_fn(evaluated, actual)), threshold

        selection = select_method1_candidate(
            n_obs=n_rows,
            actual=y_train,
            folds=fold_predictions,
            local_candidates=local_candidates,
            evaluator=evaluator,
            objective=objective,
        )
        selected = selection.selected
        pooled_score = selection.selected_score
        pooled_threshold = selection.selected_threshold

    if stack_enabled and methods == (1, 2) and method2_state.train_star is not None:
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

    univariate = full_x_train.shape[1] == 1
    final_n_best: int | str | None = None if univariate else selected
    final_x_train: NDArray[np.float64] = (
        full_x_train[:, 0] if univariate else full_x_train
    )
    final_x_test: NDArray[np.float64] = full_x_test[:, 0] if univariate else full_x_test
    final_fit = s.nns_reg(
        final_x_train,
        y_train,
        point_est=final_x_test,
        n_best=final_n_best,
        order=None if univariate else order,
        dist=dist,
        point_only=False,
        confidence_interval=pred_int,
        type=type_value,
        ncores=ncores,
    )
    fitted = cast(dict[str, NDArray[np.float64]], final_fit["Fitted.xy"])
    prediction = s._as_prediction(final_fit["Point.est"], x_test.shape[0])
    fitted_yhat = fitted["y.hat"]
    final_threshold = pooled_threshold
    if type_value == "class":
        if not np.isfinite(final_threshold):
            final_threshold = s._classification_threshold(fitted_yhat, y_train)
        fitted_yhat = s._class_threshold_round(fitted_yhat, final_threshold, y_train)
        prediction = s._class_threshold_round(prediction, final_threshold, y_train)
    final_obj = objective_fn(fitted_yhat, fitted["y"])
    final_pred_int = cast(dict[str, NDArray[np.float64]] | None, final_fit["pred.int"])
    final_pred_int = s._prediction_interval_or_point_estimate(final_pred_int, prediction)

    if selected == "all" and not univariate:
        full_model = s._mreg_prepare_model(
            full_x_train,
            y_train,
            order=order,
            noise_reduction="mode_class" if type_value == "class" else "off",
            is_class=type_value == "class",
        )
        public_parameter = float(full_model.rpm.shape[0])
    else:
        public_parameter = float(1 if univariate else selected)

    if status:
        print(
            f"Method 1 selected {selected!r} from pooled OOF score "
            f"{pooled_score:.6g}"
        )
    return s._MethodState(
        prediction=prediction,
        objective=float(final_obj),
        parameter=public_parameter,
        pred_int=final_pred_int,
        class_threshold=final_threshold if type_value == "class" else None,
    )
