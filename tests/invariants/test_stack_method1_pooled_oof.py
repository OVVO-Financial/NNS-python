from __future__ import annotations

import numpy as np
import pytest

from nns._stack_method1 import CandidateId, Method1FoldPredictions, select_method1_candidate


def _sse(
    _candidate: CandidateId,
    predicted: np.ndarray,
    actual: np.ndarray,
) -> tuple[float, float]:
    return float(np.sum((predicted - actual) ** 2)), 0.5


def test_method1_requires_identical_complete_oof_coverage() -> None:
    actual = np.arange(6, dtype=np.float64)
    folds = [
        Method1FoldPredictions(
            validation_idx=np.array([0, 1, 2], dtype=np.int64),
            predictions={
                1: np.array([0.0, 1.0, 2.0]),
                2: np.array([0.0, 1.0, 2.0]),
                3: np.array([0.0, 1.0, 2.0]),
                "all": np.array([0.0, 1.0, 2.0]),
            },
        ),
        Method1FoldPredictions(
            validation_idx=np.array([3, 4, 5], dtype=np.int64),
            predictions={
                1: np.array([3.0, 4.0, 5.0]),
                2: np.array([3.0, 4.0, 5.0]),
                "all": np.array([3.0, 4.0, 5.0]),
            },
        ),
    ]

    result = select_method1_candidate(
        n_obs=6,
        actual=actual,
        folds=folds,
        local_candidates=[1, 2, 3],
        evaluator=_sse,
        objective="min",
    )

    assert 3 in result.excluded
    assert 3 not in result.scores
    np.testing.assert_array_equal(result.count_vectors[1], np.ones(6, dtype=np.int64))
    np.testing.assert_array_equal(result.count_vectors[2], result.count_vectors[1])
    assert not np.array_equal(result.count_vectors[3], result.count_vectors[1])


def test_method1_empty_candidate_cannot_win_with_zero_sse() -> None:
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    folds = [
        Method1FoldPredictions(
            validation_idx=np.arange(4, dtype=np.int64),
            predictions={
                1: np.array([1.1, 2.1, 3.1, 4.1]),
                2: np.array([1.2, 2.2, 3.2, 4.2]),
                3: np.full(4, np.nan),
                "all": np.array([1.5, 2.5, 3.5, 4.5]),
            },
        )
    ]

    result = select_method1_candidate(
        n_obs=4,
        actual=actual,
        folds=folds,
        local_candidates=[1, 2, 3],
        evaluator=_sse,
        objective="min",
    )

    assert 3 in result.excluded
    assert 3 not in result.scores
    assert result.selected == 1


def test_method1_applies_early_stop_after_complete_pooled_scores() -> None:
    actual = np.zeros(8, dtype=np.float64)
    candidates = {
        1: np.full(8, 0.1),
        2: np.full(8, 0.2),
        3: np.full(8, 0.3),
        4: np.full(8, 0.4),
        5: np.zeros(8),
        "all": np.full(8, 0.05),
    }
    folds = [
        Method1FoldPredictions(
            validation_idx=np.arange(0, 4, dtype=np.int64),
            predictions={key: value[:4] for key, value in candidates.items()},
        ),
        Method1FoldPredictions(
            validation_idx=np.arange(4, 8, dtype=np.int64),
            predictions={key: value[4:] for key, value in candidates.items()},
        ),
    ]

    result = select_method1_candidate(
        n_obs=8,
        actual=actual,
        folds=folds,
        local_candidates=[1, 2, 3, 4, 5],
        evaluator=_sse,
        objective="min",
    )

    assert result.stopping_k == 4
    assert result.eligible == (1, 2, 3, 4, "all")
    assert 5 in result.excluded
    assert result.selected == "all"


def test_method1_all_is_evaluated_even_after_local_stop() -> None:
    actual = np.zeros(4, dtype=np.float64)
    folds = [
        Method1FoldPredictions(
            validation_idx=np.arange(4, dtype=np.int64),
            predictions={
                1: np.full(4, 0.2),
                2: np.full(4, 0.3),
                3: np.full(4, 0.4),
                4: np.full(4, 0.5),
                "all": np.zeros(4),
            },
        )
    ]

    result = select_method1_candidate(
        n_obs=4,
        actual=actual,
        folds=folds,
        local_candidates=[1, 2, 3, 4],
        evaluator=_sse,
        objective="min",
    )

    assert result.stopping_k == 4
    assert result.selected == "all"
    assert result.scores["all"] == pytest.approx(0.0)


def test_method1_first_candidate_wins_exact_tie() -> None:
    actual = np.zeros(3, dtype=np.float64)
    folds = [
        Method1FoldPredictions(
            validation_idx=np.arange(3, dtype=np.int64),
            predictions={
                1: np.ones(3),
                2: np.ones(3),
                "all": np.ones(3),
            },
        )
    ]

    result = select_method1_candidate(
        n_obs=3,
        actual=actual,
        folds=folds,
        local_candidates=[1, 2],
        evaluator=_sse,
        objective="min",
    )

    assert result.selected == 1


def test_method1_rejects_missing_reference_coverage() -> None:
    with pytest.raises(ValueError, match="candidate 1 has no OOF coverage"):
        select_method1_candidate(
            n_obs=2,
            actual=np.zeros(2),
            folds=[
                Method1FoldPredictions(
                    validation_idx=np.arange(2, dtype=np.int64),
                    predictions={1: np.full(2, np.nan), "all": np.zeros(2)},
                )
            ],
            local_candidates=[1],
            evaluator=_sse,
            objective="min",
        )
