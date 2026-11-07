"""NNS 14.0 classification stack: probability-vector CV, never coded regression.

Source of truth: OVVO-Financial/NNS R/Classification_Stack.R at
7b07a837c130ba7220dbe16c584221508757d5ae.  R's method-1 neighbour
counts and method-2 class-indicator dimensions are selected from out-of-fold
probability matrices.  Synthetic class indices describe feature engineering
only; using them as ordinal target values changes the model.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from nns._class_r14 import classify as classify_r14
from nns._reg_engine import _dependence, _dimred_coefficients

F = NDArray[np.float64]
Index = NDArray[np.int64]


def classify_stack_r14(
    *,
    x: F,
    z: F,
    y: F,
    class_values: list[Any] | None,
    response_categorical: bool,
    predictor_levels: list[list[str] | None],
    methods: list[int],
    order: Any,
    dist: str,
    dim_red_method: Any,
    splits: list[dict[str, Index]],
    balance_indices: Callable[[F], Index],
    numeric_design: Callable[..., dict[str, Any]],
    score: Callable[[F, F], float],
    stack: bool,
    objective: str,
    optimize_threshold: bool,
    pred_int: float | None,
    status: bool,
    tau: Any,
    return_fit: bool = False,
) -> dict[str, Any]:
    """Direct translation of R .nns_stack_classify's CV and final return."""
    values = np.unique(y)
    class_count = values.size
    original = class_values if class_values is not None else values.tolist()
    labels = [
        str(v) if not isinstance(v, (float, np.floating)) else f"{float(v):.15g}" for v in original
    ]
    if class_count != len(original):
        raise ValueError("The encoded response and class levels disagree.")
    first_seen = list(dict.fromkeys(y.tolist()))
    tie_order = np.array([int(np.flatnonzero(values == v)[0]) for v in first_seen])
    n = y.size

    def decode(codes: F) -> F:
        codes = np.asarray(codes, dtype=np.float64)
        if response_categorical:
            return codes
        return np.asarray(original, dtype=np.float64)[codes.astype(np.int64) - 1]

    def encode(train: F, test: F) -> dict[str, Any]:
        design = numeric_design(train, test, scale=False)
        return {
            "x": np.asarray(design["train"], dtype=np.float64),
            "point_est": np.asarray(design["test"], dtype=np.float64),
            "encoded_names": design["names"],
            "metadata": [{"categorical": level is not None} for level in predictor_levels],
        }

    full = encode(x, z)
    p = full["x"].shape[1]
    balanced = [balance_indices(y[s["train"]]) for s in splits]
    final_rows = balance_indices(y)

    def choose(probabilities: F, threshold: float = 0.5) -> F:
        if class_count == 2:
            return np.where(
                probabilities[:, tie_order[1]] > threshold,
                values[tie_order[1]],
                values[tie_order[0]],
            )
        return np.asarray(
            values[tie_order[np.argmax(probabilities[:, tie_order], axis=1)]], dtype=np.float64
        )

    def evaluate(probabilities: F, actual: F) -> tuple[float, float]:
        thresholds = (
            np.arange(1, 100, dtype=np.float64) / 100.0
            if class_count == 2 and optimize_threshold
            else np.asarray([0.5])
        )
        losses = np.asarray(
            [score(decode(choose(probabilities, float(t))), decode(actual)) for t in thresholds],
            dtype=np.float64,
        )
        usable = np.flatnonzero(np.isfinite(losses))
        if usable.size == 0:
            raise ValueError("No classification threshold has a finite objective.")
        best = losses[usable].min() if objective == "min" else losses[usable].max()
        tied = usable[losses[usable] == best]
        selected = int(tied[np.argmin(np.abs(thresholds[tied] - 0.5))])
        return float(losses[selected]), float(thresholds[selected])

    def design(
        train: F, response: F, test: F, component: int, select_count: int | None = None
    ) -> dict[str, Any]:
        encoded = encode(train, test)
        matrix = encoded["x"]
        if component == 1 and stack and select_count is not None and select_count < matrix.shape[1]:
            ind = np.asarray(response[:, None] == values[None, :], dtype=np.float64)
            strengths = np.asarray(
                [
                    max(0.0, *(_dependence(matrix[:, j], ind[:, c]) for c in range(class_count)))
                    for j in range(matrix.shape[1])
                ]
            )
            keep = np.argsort(-strengths, kind="stable")[:select_count]
            encoded["x"] = matrix[:, keep]
            encoded["point_est"] = encoded["point_est"][:, keep]
            encoded["encoded_names"] = [encoded["encoded_names"][int(j)] for j in keep]
        return encoded

    def fit(
        train: F,
        response: F,
        test: F,
        component: int,
        candidate: Any,
        select_count: int | None = None,
    ) -> dict[str, Any]:
        encoded = design(train, response, test, component, select_count)
        return classify_r14(
            encoded,
            {"y": response, "class_values": values, "class_levels": labels},
            order=order,
            n_best=candidate if component == 1 else None,
            dist=dist,
            dim_red_method=dim_red_method if component == 2 else None,
            tau=tau,
            threshold=0.0,
            point_only=True,
            confidence_interval=None,
            dim_count=candidate if component == 2 else None,
        )

    def as_probabilities(fitted: dict[str, Any]) -> F:
        prob = fitted.get("Point.probabilities")
        if prob is None:
            raise ValueError("R14 class fit did not produce probability vectors.")
        return np.column_stack([np.asarray(prob[label], dtype=np.float64) for label in labels])

    def record(total: F, count: Index, candidate: Any) -> dict[str, Any]:
        mask = count > 0
        if not np.any(mask):
            raise ValueError("The classification candidate has no held-out predictions.")
        probability = np.full((n, class_count), np.nan, dtype=np.float64)
        probability[mask] = total[mask] / count[mask, None]
        quality, threshold = evaluate(probability[mask], y[mask])
        return {
            "score": quality,
            "threshold": threshold,
            "candidate": candidate,
            "probability": probability,
            "valid": mask,
        }

    def tune(
        component: int, candidates: list[Any], select_count: int | None = None
    ) -> dict[str, Any]:
        candidates_results = []
        for candidate in candidates:
            total = np.zeros((n, class_count), dtype=np.float64)
            count = np.zeros(n, dtype=np.int64)
            for i, split in enumerate(splits):
                train = split["train"][balanced[i]]
                validation = split["validation"]
                fitted = fit(x[train], y[train], x[validation], component, candidate, select_count)
                total[validation] += as_probabilities(fitted)
                count[validation] += 1
            candidates_results.append(record(total, count, candidate))
        qualities = np.asarray([r["score"] for r in candidates_results])
        choice = int(np.argmin(qualities) if objective == "min" else np.argmax(qualities))
        return candidates_results[choice]

    if status:
        print("Selecting NNS class-frequency models on held-out probability vectors")
    dim_fit = tune(2, list(range(1, p + 1))) if 2 in methods else None
    reg_fit = None
    if 1 in methods:
        candidates = ["all"] if p == 1 else [*range(1, max(1, math.floor(math.sqrt(n))) + 1), "all"]
        reg_fit = tune(1, candidates, None if dim_fit is None else int(dim_fit["candidate"]))

    weights = {"reg": float(reg_fit is not None), "dim.red": float(dim_fit is not None)}
    if reg_fit is None:
        assert dim_fit is not None
        threshold = float(dim_fit["threshold"])
    else:
        threshold = float(reg_fit["threshold"])
    if reg_fit is not None and dim_fit is not None:
        scores = np.asarray([reg_fit["score"], dim_fit["score"]], dtype=np.float64)
        scores = np.where(scores == 0.0, 1e-10, scores)
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            w = 1.0 / scores**2 if objective == "min" else scores**2
        w = np.maximum(w, 1e-10)
        w[~np.isfinite(w)] = 0.0
        w = w / w.sum() if w.sum() > 0.0 else np.array([0.5, 0.5])
        weights = {"reg": float(w[0]), "dim.red": float(w[1])}
        valid = reg_fit["valid"] & dim_fit["valid"]
        combined = (
            weights["reg"] * reg_fit["probability"][valid]
            + weights["dim.red"] * dim_fit["probability"][valid]
        )
        _, threshold = evaluate(combined, y[valid])

    dim_cutoff: float | F = float("nan")
    if dim_fit is not None:
        selected_count = int(dim_fit["candidate"])
        cutoffs = []
        for label in values:
            coefficients = _dimred_coefficients(
                full["x"], np.asarray(y == label, dtype=np.float64), dim_red_method, tau, 0.0
            )
            if selected_count >= coefficients.size:
                cutoffs.append(0.0)
            else:
                ordered = np.sort(np.abs(coefficients))[::-1]
                cutoffs.append(float(ordered[selected_count - 1]))
        cutoffs_array = np.asarray(cutoffs, dtype=np.float64)
        dim_cutoff = (
            float(np.min(cutoffs_array))
            if np.allclose(cutoffs_array, cutoffs_array[0])
            else cutoffs_array
        )

    def final(component: int, fitted: dict[str, Any] | None) -> dict[str, Any] | None:
        if fitted is None:
            return None
        return fit(
            x[final_rows],
            y[final_rows],
            z,
            component,
            fitted["candidate"],
            None if dim_fit is None else int(dim_fit["candidate"]),
        )

    reg_final = final(1, reg_fit)
    dim_final = final(2, dim_fit)
    rp = None if reg_final is None else as_probabilities(reg_final)
    dp = None if dim_final is None else as_probabilities(dim_final)
    probability = (
        dp if rp is None else rp if dp is None else weights["reg"] * rp + weights["dim.red"] * dp
    )
    if probability is None:
        raise ValueError("No classification stack component was selected.")

    def prediction_set(prob: F | None) -> list[list[float]] | None:
        if prob is None or pred_int is None:
            return None
        sets: list[list[float]] = []
        for row in prob:
            order_idx = tie_order[np.argsort(-row[tie_order], kind="stable")]
            take = min(
                int(np.searchsorted(np.cumsum(row[order_idx]), pred_int - 1e-12)) + 1, class_count
            )
            sets.append(decode(values[order_idx[:take]]).tolist())
        return sets

    def interval(prob: F | None) -> dict[str, F] | None:
        sets = prediction_set(prob)
        if sets is None or class_count != 2:
            return None
        return {
            "pred.int.neg": np.asarray([min(s) for s in sets], dtype=np.float64),
            "pred.int.pos": np.asarray([max(s) for s in sets], dtype=np.float64),
        }

    def output(prob: F | None, candidate: dict[str, Any] | None) -> F | None:
        # An unavailable classification component is NULL/NA in R, not a
        # one-element prediction vector. It has no pointwise predictions.
        if prob is None or candidate is None:
            return None
        return decode(choose(prob, float(candidate["threshold"])))

    selected: list[str] | None = None
    equations: Any = None
    if dim_final is not None:
        equations = dim_final.get("equation")
        if isinstance(equations, dict):
            selected = []
            for eq in equations.values():
                if eq is None:
                    continue
                columns = eq.get("Variable", [])
                coefs = eq.get("Coefficient", [])
                selected.extend(
                    str(v)
                    for v, c in zip(columns, coefs, strict=True)
                    if v != "DENOMINATOR" and float(c) != 0.0
                )
            selected = list(dict.fromkeys(selected))

    k = float("nan")
    if reg_fit is not None:
        assert reg_final is not None
        k = (
            float(reg_final["n.best"])
            if reg_fit["candidate"] == "all"
            else float(reg_fit["candidate"])
        )
    result = {
        "OBJfn.reg": float("nan") if reg_fit is None else float(reg_fit["score"]),
        "NNS.reg.n.best": k,
        "probability.threshold": float(threshold),
        "OBJfn.dim.red": float("nan") if dim_fit is None else float(dim_fit["score"]),
        "NNS.dim.red.threshold": dim_cutoff,
        "NNS.dim.red.count": float("nan") if dim_fit is None else float(dim_fit["candidate"]),
        "NNS.dim.red.variables": selected,
        "NNS.dim.red.coefficients": equations,
        "reg": output(rp, reg_fit),
        "reg.pred.int": interval(rp),
        "dim.red": output(dp, dim_fit),
        "dim.red.pred.int": interval(dp),
        "stack": decode(choose(probability, float(threshold))),
        "pred.int": interval(probability),
        "Prediction.set": prediction_set(probability),
        "probabilities": probability,
        "reg.probabilities": rp,
        "dim.red.probabilities": dp,
        "weights": weights,
        "class.levels": original,
    }
    if return_fit:
        # The inference API can reconstruct a fitted stack from the original
        # data; never confuse the numeric-regression fit state with class paths.
        result["_fit"] = {"r14_class": True}
    return result
