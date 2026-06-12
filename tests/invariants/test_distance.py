from __future__ import annotations

import numpy as np

from pynns import nns_distance, nns_distance_bulk


def test_nns_distance_self_target_returns_nearest_y_hat() -> None:
    rpm = _rpm()

    assert nns_distance(rpm, rpm[0, :-1], k=1) == rpm[0, -1]


def test_nns_distance_bulk_shape_and_finiteness() -> None:
    rpm = _rpm()
    result = nns_distance_bulk(rpm, rpm[:3, :-1], k=2)

    assert result.shape == (3,)
    assert np.all(np.isfinite(result))


def test_nns_distance_bulk_k_all_is_finite() -> None:
    rpm = _rpm()
    result = nns_distance_bulk(rpm, rpm[:2, :-1], k="all")

    assert np.all(np.isfinite(result))


def test_nns_distance_class_returns_observed_code() -> None:
    rpm = _class_rpm()
    result = nns_distance(rpm, np.array([2.5, 0.9]), k=3, class_="class")

    assert result in set(rpm[:, -1])


def test_nns_distance_bulk_class_matches_bulk_numeric_shape() -> None:
    rpm = _class_rpm()
    result = nns_distance_bulk(rpm, rpm[:3, :-1], k=2, class_="class")

    assert result.shape == (3,)
    assert np.all(np.isfinite(result))


def _rpm() -> np.ndarray:
    row = np.arange(1, 8, dtype=np.float64)
    features = np.column_stack((row, row**2, np.sin(row)))
    y_hat = row / 10.0
    return np.column_stack((features, y_hat))


def _class_rpm() -> np.ndarray:
    row = np.arange(1, 8, dtype=np.float64)
    features = np.column_stack((row, np.sin(row)))
    y_hat = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 1.0])
    return np.column_stack((features, y_hat))
