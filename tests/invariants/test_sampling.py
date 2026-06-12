from __future__ import annotations

import numpy as np

from pynns.categorical import _balance_class_training, _down_sample_rows, _up_sample_rows


def test_down_and_up_sample_match_r_class_counts_and_grouping() -> None:
    x = np.column_stack(
        (np.arange(1, 11, dtype=np.float64), np.arange(10, 0, -1, dtype=np.float64))
    )
    y = np.array([1.0] * 8 + [2.0] * 2)
    classes = np.array([1.0, 2.0, 3.0])

    down_x, down_y = _down_sample_rows(
        x,
        y,
        classes=classes,
        rng=np.random.default_rng(1),
    )
    up_x, up_y = _up_sample_rows(
        x,
        y,
        classes=classes,
        rng=np.random.default_rng(1),
    )
    balanced_x, balanced_y = _balance_class_training(
        x,
        y,
        classes=classes,
        rng=np.random.default_rng(1),
    )

    assert down_x.shape == (4, 2)
    assert up_x.shape == (16, 2)
    assert balanced_x.shape == (20, 2)
    np.testing.assert_array_equal(down_y, np.array([1.0, 1.0, 2.0, 2.0]))
    np.testing.assert_array_equal(up_y[:8], np.ones(8))
    np.testing.assert_array_equal(up_y[8:], np.full(8, 2.0))
    np.testing.assert_array_equal(balanced_y[:4], down_y)
    np.testing.assert_array_equal(balanced_y[4:], up_y)
    assert 3.0 not in balanced_y


def test_balance_samples_already_balanced_data() -> None:
    x = np.arange(12, dtype=np.float64).reshape(6, 2)
    y = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    classes = np.array([1.0, 2.0])

    balanced_x, balanced_y = _balance_class_training(
        x,
        y,
        classes=classes,
        rng=np.random.default_rng(2),
    )

    assert balanced_x.shape == (12, 2)
    np.testing.assert_array_equal(balanced_y[:3], np.ones(3))
    np.testing.assert_array_equal(balanced_y[3:6], np.full(3, 2.0))
    np.testing.assert_array_equal(balanced_y[6:9], np.ones(3))
    np.testing.assert_array_equal(balanced_y[9:], np.full(3, 2.0))


def test_balance_tiny_minority_with_replacement_and_seed_determinism() -> None:
    x = np.arange(24, dtype=np.float64).reshape(12, 2)
    y = np.array([1.0] * 11 + [2.0])
    classes = np.array([1.0, 2.0])

    first_x, first_y = _balance_class_training(
        x,
        y,
        classes=classes,
        rng=np.random.default_rng(3),
    )
    second_x, second_y = _balance_class_training(
        x,
        y,
        classes=classes,
        rng=np.random.default_rng(3),
    )

    np.testing.assert_array_equal(first_x, second_x)
    np.testing.assert_array_equal(first_y, second_y)
    assert first_x.shape == (24, 2)
    assert np.count_nonzero(first_y == 1.0) == 12
    assert np.count_nonzero(first_y == 2.0) == 12
    assert np.unique(first_x[first_y == 2.0], axis=0).shape[0] == 1


def test_balance_respects_explicit_class_order() -> None:
    x = np.arange(18, dtype=np.float64).reshape(9, 2)
    y = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 1.0, 3.0, 1.0, 1.0])
    classes = np.array([3.0, 1.0, 2.0])

    balanced_x, balanced_y = _balance_class_training(
        x,
        y,
        classes=classes,
        rng=np.random.default_rng(4),
    )

    assert balanced_x.shape == (21, 2)
    np.testing.assert_array_equal(balanced_y[:2], np.full(2, 3.0))
    np.testing.assert_array_equal(balanced_y[2:4], np.full(2, 1.0))
    np.testing.assert_array_equal(balanced_y[4:6], np.full(2, 2.0))
    np.testing.assert_array_equal(balanced_y[6:11], np.full(5, 3.0))
    np.testing.assert_array_equal(balanced_y[11:16], np.full(5, 1.0))
    np.testing.assert_array_equal(balanced_y[16:], np.full(5, 2.0))
