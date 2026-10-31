"""The multivariate engine's fast paths must reproduce the straightforward
reference formulations bit for bit (they are performance rewrites only)."""

from __future__ import annotations

import numpy as np
import pytest

from nns._reg_engine import _mreg_distances


def _reference_distances(
    rpm_x: np.ndarray, xtest: np.ndarray, dist: str, mins: np.ndarray, maxs: np.ndarray
) -> np.ndarray:
    """The former (m, n, p) broadcast implementation."""
    if dist == "FACTOR":
        return np.mean(rpm_x[None, :, :] != xtest[:, None, :], axis=2, dtype=np.float64)
    ranges = maxs - mins
    active = np.isfinite(ranges) & (ranges > 0)
    if not np.any(active):
        return np.zeros((xtest.shape[0], rpm_x.shape[0]))
    z = (xtest[:, None, active] - rpm_x[None, :, active]) / ranges[active]
    if dist == "NNS":
        return np.sum(np.abs(z) + z * z, axis=2)
    if dist == "L1":
        return np.sum(np.abs(z), axis=2)
    return np.sqrt(np.sum(z * z, axis=2))


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and np.array_equal(a, b, equal_nan=True)


@pytest.mark.parametrize("dist", ["NNS", "L1", "L2", "FACTOR"])
@pytest.mark.parametrize("p", [1, 2, 5, 8, 9, 13, 17, 40])
def test_column_wise_distances_match_broadcast_reference(dist: str, p: int) -> None:
    rng = np.random.default_rng(100 + p)
    rpm = rng.normal(size=(120, p)) * rng.exponential(size=p) * 10
    if p > 3:
        rpm[:, 1] = 2.5  # constant (inactive) column
        rpm[:, 2] = np.round(rpm[:, 2])
    xtest = rng.normal(size=(30, p)) * 5
    xtest[:5] = rpm[:5]
    mins, maxs = rpm.min(axis=0), rpm.max(axis=0)
    for xt in (xtest, np.asfortranarray(xtest), xtest[:1]):
        assert _same(
            _mreg_distances(rpm, xt, dist, mins, maxs),
            _reference_distances(rpm, xt, dist, mins, maxs),
        )


def test_column_wise_distances_all_constant_columns() -> None:
    rpm = np.ones((4, 3))
    xtest = np.zeros((2, 3))
    out = _mreg_distances(rpm, xtest, "NNS", rpm.min(axis=0), rpm.max(axis=0))
    assert _same(out, np.zeros((2, 4)))
