"""The multivariate engine's fast paths must reproduce the straightforward
reference formulations bit for bit (they are performance rewrites only)."""

from __future__ import annotations

import numpy as np
import pytest

from nns import nns_reg
from nns._reg_engine import _mreg_build_rpm, _mreg_distances, _mreg_predict, _reduce_value


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


def _reference_build_rpm(
    x: np.ndarray, y: np.ndarray, ids: np.ndarray, noise_reduction: str, is_class: bool
) -> tuple[np.ndarray, np.ndarray]:
    """The former one-reducer-call-per-cell-per-column implementation."""
    order = np.argsort(ids, kind="stable")
    ids_sorted, xs, ys = ids[order], x[order], y[order]
    _, starts = np.unique(ids_sorted, return_index=True)
    starts = np.sort(starts)
    ends = np.append(starts[1:], ids_sorted.size)
    rows = [
        [_reduce_value(xs[s:e, j], noise_reduction, False) for j in range(x.shape[1])]
        for s, e in zip(starts, ends, strict=True)
    ]
    yhat = [
        _reduce_value(ys[s:e], noise_reduction, is_class)
        for s, e in zip(starts, ends, strict=True)
    ]
    return np.asarray(rows, dtype=np.float64), np.asarray(yhat, dtype=np.float64)


@pytest.mark.parametrize("noise_reduction", ["off", "mean", "median", "mode"])
@pytest.mark.parametrize("is_class", [False, True])
def test_rpm_cell_reduction_matches_per_cell_reducer(
    noise_reduction: str, is_class: bool
) -> None:
    rng = np.random.default_rng(7)
    for trial in range(40):
        n = int(rng.integers(2, 200))
        x = rng.normal(size=(n, 3)) * 10.0 ** int(rng.integers(-2, 3))
        if trial % 2:
            x = np.round(x)
        x[rng.random(size=x.shape) < 0.05] = 0.0
        x[rng.random(size=x.shape) < 0.05] = -0.0
        if trial % 5 == 0:
            x[rng.random(size=x.shape) < 0.03] = np.nan
        y = rng.integers(0, 3, size=n).astype(float) if is_class else rng.normal(size=n)
        ids = np.asarray([str(v) for v in rng.integers(0, max(2, n // 2), size=n)])
        got = _mreg_build_rpm(x, y, ids, noise_reduction, is_class)
        ref = _reference_build_rpm(x, y, ids, noise_reduction, is_class)
        for a, b in zip(got, ref, strict=True):
            assert _same(a, b)
            assert np.array_equal(np.signbit(a), np.signbit(b))


@pytest.mark.parametrize("noise_reduction", ["off", "mean", "median"])
@pytest.mark.parametrize("type_", [None, "CLASS"])
def test_order_max_identity_shortcut_matches_k1_distance_rule(
    noise_reduction: str, type_: str | None
) -> None:
    rng = np.random.default_rng(11)
    x = rng.integers(0, 4, size=(300, 3)).astype(float)  # many duplicate rows
    x[:100] = rng.normal(size=(100, 3))
    y = x.sum(axis=1) + rng.normal(size=300)
    if type_ is not None:
        y = (y > np.median(y)).astype(float)
    fit = nns_reg(x, y, order="max", noise_reduction=noise_reduction, type=type_)
    rpm = fit["RPM"]
    rpm_x = np.column_stack([rpm[k] for k in rpm if k != "y.hat"])
    expected = _mreg_predict(
        x, rpm_x, rpm["y.hat"], 1, "NNS", x.min(axis=0), x.max(axis=0), type_ is not None
    )
    assert expected is not None
    if type_ is not None:
        expected = np.unique(y)[np.argmin(np.abs(expected[:, None] - np.unique(y)), axis=1)]
    assert fit["n.best"] == 1
    assert _same(np.asarray(fit["Fitted.xy"]["y.hat"]), expected)
