from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

CandidateId: TypeAlias = int | Literal["all"]
Objective: TypeAlias = Literal["min", "max"]
CandidateEvaluator: TypeAlias = Callable[
    [NDArray[np.float64], NDArray[np.float64]], tuple[float, float]
]


@dataclass(frozen=True)
class Method1FoldPredictions:
    """Predictions for one CV fold keyed by conceptual Method 1 candidate."""

    validation_idx: NDArray[np.int64]
    predictions: Mapping[CandidateId, NDArray[np.float64]]


@dataclass(frozen=True)
class Method1Selection:
    """Complete pooled-OOF Method 1 candidate selection result."""

    selected: CandidateId
    selected_score: float
    selected_threshold: float
    pooled_predictions: Mapping[CandidateId, NDArray[np.float64]]
    count_vectors: Mapping[CandidateId, NDArray[np.int64]]
    scores: Mapping[CandidateId, float]
    thresholds: Mapping[CandidateId, float]
    eligible: tuple[CandidateId, ...]
    excluded: tuple[CandidateId, ...]
    stopping_k: int | None


def select_method1_candidate(
    *,
    n_obs: int,
    actual: NDArray[np.float64],
    folds: Sequence[Method1FoldPredictions],
    local_candidates: Sequence[int],
    evaluator: CandidateEvaluator,
    objective: Objective,
) -> Method1Selection:
    """Select Method 1 using complete pooled OOF predictions.

    The repaired R semantics are deliberately candidate-global:

    * each conceptual local ``k`` is accumulated across every fold;
    * candidate 1 defines the required OOF coverage pattern;
    * partial/empty candidates are excluded before scoring;
    * diminishing-returns stopping is applied only to complete pooled scores;
    * ``ALL`` is always scored separately and is never subject to local stopping.
    """

    if n_obs < 1:
        raise ValueError("n_obs must be positive.")
    actual_values = np.asarray(actual, dtype=np.float64).reshape(-1)
    if actual_values.size != n_obs:
        raise ValueError("actual must contain n_obs values.")
    if objective not in {"min", "max"}:
        raise ValueError("objective must be 'min' or 'max'.")

    local_ids = tuple(dict.fromkeys(int(k) for k in local_candidates if int(k) >= 1))
    if not local_ids or local_ids[0] != 1:
        raise ValueError("local_candidates must begin with candidate 1.")
    candidate_ids: tuple[CandidateId, ...] = (*local_ids, "all")

    sums = {candidate: np.zeros(n_obs, dtype=np.float64) for candidate in candidate_ids}
    counts = {candidate: np.zeros(n_obs, dtype=np.int64) for candidate in candidate_ids}

    for fold in folds:
        validation_idx = np.asarray(fold.validation_idx, dtype=np.int64).reshape(-1)
        if np.any(validation_idx < 0) or np.any(validation_idx >= n_obs):
            raise ValueError("fold validation indices are outside [0, n_obs).")
        if np.unique(validation_idx).size != validation_idx.size:
            raise ValueError("fold validation indices must be unique.")

        for candidate, raw_prediction in fold.predictions.items():
            if candidate not in sums:
                continue
            prediction = np.asarray(raw_prediction, dtype=np.float64).reshape(-1)
            if prediction.size != validation_idx.size:
                raise ValueError(
                    f"candidate {candidate!r} prediction length does not match validation indices."
                )
            finite = np.isfinite(prediction)
            if not np.any(finite):
                continue
            rows = validation_idx[finite]
            sums[candidate][rows] += prediction[finite]
            counts[candidate][rows] += 1

    reference_count = counts[1]
    if not np.any(reference_count > 0):
        raise ValueError("candidate 1 has no OOF coverage.")

    pooled: dict[CandidateId, NDArray[np.float64]] = {}
    scores: dict[CandidateId, float] = {}
    thresholds: dict[CandidateId, float] = {}
    complete: dict[CandidateId, bool] = {}

    for candidate in candidate_ids:
        count = counts[candidate]
        raw = np.full(n_obs, np.nan, dtype=np.float64)
        covered = count > 0
        raw[covered] = sums[candidate][covered] / count[covered]
        pooled[candidate] = raw

        same_coverage = np.array_equal(count, reference_count)
        valid = covered & np.isfinite(raw) & np.isfinite(actual_values)
        if not same_coverage or not np.any(valid):
            complete[candidate] = False
            continue

        score, threshold = evaluator(raw[valid], actual_values[valid])
        score_value = float(score)
        threshold_value = float(threshold)
        if not math.isfinite(score_value):
            complete[candidate] = False
            continue
        complete[candidate] = True
        scores[candidate] = score_value
        thresholds[candidate] = threshold_value

    evaluated_local: list[int] = []
    stopping_k: int | None = None
    for candidate in local_ids:
        if not complete.get(candidate, False):
            break
        evaluated_local.append(candidate)
        if len(evaluated_local) < 4:
            continue
        recent = [scores[k] for k in evaluated_local[-3:]]
        stop = (
            recent[2] >= recent[1] and recent[2] >= recent[0]
            if objective == "min"
            else recent[2] <= recent[1] and recent[2] <= recent[0]
        )
        if stop:
            stopping_k = candidate
            break

    eligible: list[CandidateId] = list(evaluated_local)
    if complete.get("all", False):
        eligible.append("all")
    if not eligible:
        raise ValueError("no Method 1 candidate has complete finite OOF coverage.")

    if objective == "min":
        best_score = min(scores[candidate] for candidate in eligible)
    else:
        best_score = max(scores[candidate] for candidate in eligible)
    selected = next(candidate for candidate in eligible if scores[candidate] == best_score)

    excluded = tuple(candidate for candidate in candidate_ids if candidate not in eligible)
    return Method1Selection(
        selected=selected,
        selected_score=float(scores[selected]),
        selected_threshold=float(thresholds[selected]),
        pooled_predictions=pooled,
        count_vectors=counts,
        scores=scores,
        thresholds=thresholds,
        eligible=tuple(eligible),
        excluded=excluded,
        stopping_k=stopping_k,
    )
