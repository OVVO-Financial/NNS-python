"""The multivariate engine's fast paths must reproduce the straightforward
reference formulations bit for bit (they are performance rewrites only)."""

from __future__ import annotations

import numpy as np
import pytest

from nns import nns_reg
from nns._reg_engine import (
    _group_reduce_sorted,
    _mreg_build_rpm,
    _mreg_distances,
    _mreg_ensemble_weights,
    _mreg_predict,
    _mreg_predict_path,
    _mreg_weighted_mode,
    _mreg_weighted_mode_rows,
    _nearest_tie_values,
    _reduce_value,
    _stable_topk,
)
from nns.central_tendencies import nns_gravity


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
def test_order_max_identity_shortcut_matches_k1_distance_rule(
    noise_reduction: str
) -> None:
    rng = np.random.default_rng(11)
    x = rng.integers(0, 4, size=(300, 3)).astype(float)  # many duplicate rows
    x[:100] = rng.normal(size=(100, 3))
    y = x.sum(axis=1) + rng.normal(size=300)
    # Numeric REG follows NNS.M.reg; R14 CLASS uses a different nominal
    # probability model, so it must not be tested against its RPM y.hat rule.
    fit = nns_reg(x, y, order="max", noise_reduction=noise_reduction)
    rpm = fit["RPM"]
    rpm_x = np.column_stack([rpm[k] for k in rpm if k != "y.hat"])
    expected = _mreg_predict(
        x, rpm_x, rpm["y.hat"], 1, "NNS", x.min(axis=0), x.max(axis=0), False
    )
    assert expected is not None
    assert fit["n.best"] == 1
    assert _same(np.asarray(fit["Fitted.xy"]["y.hat"]), expected)


def test_order_max_nominal_class_uses_r14_probability_contract() -> None:
    rng = np.random.default_rng(11)
    x = rng.integers(0, 4, size=(300, 3)).astype(float)
    y = (x[:, 0] + 2 * x[:, 1] > 5).astype(float)
    fit = nns_reg(x, y, order="max", type="CLASS")
    assert fit["n.best"] == 1
    assert "y.hat" not in fit["RPM"]  # centres, not REG target averages
    fitted = fit["Fitted.xy"]["y.hat"]
    assert set(np.unique(fitted)).issubset(set(np.unique(y)))
    posterior = np.column_stack(list(fit["Fitted.probabilities"].values()))
    np.testing.assert_allclose(np.sum(posterior, axis=1), 1.0, atol=1e-12)
    assert fit["R2"] == pytest.approx(float(np.mean(fitted == y)))


def _tied_distances(rng: np.random.Generator, c: int, n: int) -> np.ndarray:
    d = np.round(rng.random((c, n)) * 6) / 6  # heavy exact ties
    d[: c // 2] = rng.random((c // 2, n))
    d[0, :] = 0.5  # one row entirely tied
    return d


@pytest.mark.parametrize("k", [1, 2, 5, 17, 39, 40, 41])
def test_stable_topk_matches_full_stable_argsort(k: int) -> None:
    rng = np.random.default_rng(k)
    d = _tied_distances(rng, 25, 40)
    expected = np.argsort(d, axis=1, kind="stable")[:, :k]
    assert np.array_equal(_stable_topk(d, k), expected)


@pytest.mark.parametrize("is_class", [False, True])
def test_nearest_tie_values_match_per_row_reducer(is_class: bool) -> None:
    rng = np.random.default_rng(3)
    d = _tied_distances(rng, 60, 30)
    d[5, 3] = np.nan
    yhat = rng.integers(0, 3, size=30).astype(float) if is_class else rng.normal(size=30)
    yhat[4] = -0.0
    d[7] = 1.0
    d[7, 4] = 0.0  # unique nearest row holding -0.0
    expected = np.empty(d.shape[0])
    for i in range(d.shape[0]):
        tied = yhat[d[i] == d[i].min()]
        expected[i] = (
            _mreg_weighted_mode(tied, np.ones(tied.size))
            if is_class
            else nns_gravity(tied[np.isfinite(tied)])
        )
    got = _nearest_tie_values(d, yhat, is_class)
    assert _same(got, expected)
    assert np.array_equal(np.signbit(got), np.signbit(expected))


def test_weighted_mode_rows_match_per_row_mode() -> None:
    rng = np.random.default_rng(5)
    values = rng.integers(0, 4, size=(200, 7)).astype(float)
    weights = rng.integers(0, 3, size=(200, 7)) / 3.0  # exact ties between classes
    weights[3] = 0.0
    weights[4, 2] = np.nan
    values[5, 1] = np.nan
    weights[6] = -1.0  # nothing kept
    expected = np.array([_mreg_weighted_mode(v, w) for v, w in zip(values, weights, strict=True)])
    assert _same(_mreg_weighted_mode_rows(values, weights), expected)


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"type": "CLASS"}, {"order": "max"}, {"n_best": 1}, {"confidence_interval": 0.9}],
)
def test_multivariate_point_only_matches_full_fit(kwargs: dict[str, object]) -> None:
    rng = np.random.default_rng(13)
    x = rng.normal(size=(250, 3))
    y = x[:, 0] - x[:, 1] ** 2 + rng.normal(scale=0.3, size=250)
    if kwargs.get("type") == "CLASS":
        y = (y > 0).astype(float)
    point = rng.normal(size=(20, 3))
    full = nns_reg(x, y, point_est=point, **kwargs)
    only = nns_reg(x, y, point_est=point, point_only=True, **kwargs)
    assert only["R2"] is None
    assert only["Fitted.xy"] is None
    assert only["n.best"] == full["n.best"]
    assert _same(np.asarray(only["Point.est"]), np.asarray(full["Point.est"]))
    for key in full["RPM"]:
        assert _same(np.asarray(only["RPM"][key]), np.asarray(full["RPM"][key]))
    if full["pred.int"] is not None:
        for key in full["pred.int"]:
            assert _same(np.asarray(only["pred.int"][key]), np.asarray(full["pred.int"][key]))


def _reference_ensemble_weights(dk: np.ndarray) -> np.ndarray:
    """A per-row scalar transcription of R's C++ ``predict_sorted_k_noalloc``
    weights (src/NNS_mreg_predict.cpp), the oracle for the vectorised, cached
    implementation."""
    import math

    from nns._rnmath import dexp, dlnorm_log, dnorm, dt

    def normalize(values: list[float]) -> list[float]:
        cleaned = [v if math.isfinite(v) and v >= 0.0 else 0.0 for v in values]
        total = 0.0
        for v in cleaned:
            total += v
        if total <= 0.0:
            return [1.0 / len(cleaned)] * len(cleaned)
        return [v / total for v in cleaned]

    def sample_sd(values: list[float]) -> float:
        mu = 0.0
        for v in values:
            mu += v
        mu /= len(values)
        acc = 0.0
        for v in values:
            acc += (v - mu) * (v - mu)
        return math.sqrt(acc / (len(values) - 1))

    m, k = dk.shape
    if k == 1:
        return np.ones((m, 1))
    ranks = [float(i + 1) for i in range(k)]
    uniform = [1.0 / k] * k
    exponential = normalize([dexp(r, float(k)) for r in ranks])
    rank_sd = sample_sd(ranks)
    lognormal = normalize([abs(dlnorm_log(r, 0.0, rank_sd)) for r in ranks])[::-1]
    power = normalize([1.0 / (r * r) for r in ranks])
    out = np.empty((m, k))
    for row in range(m):
        d = [float(v) for v in dk[row]]
        student = normalize([dt(v, float(k)) for v in d])
        inverse = normalize([1.0 / max(v, 1e-12) for v in d])
        sd = sample_sd(d)
        var = sd * sd
        normal = normalize([dnorm(v, 0.0, sd) for v in d]) if sd > 0 else [0.0] * k
        rbf = normalize([math.exp(-v / (2.0 * var)) for v in d]) if var > 0 else [0.0] * k
        total = [
            uniform[i] + student[i] + inverse[i] + exponential[i] + lognormal[i] + power[i]
            + normal[i] + rbf[i]
            for i in range(k)
        ]
        out[row] = normalize(total)
    return out


@pytest.mark.parametrize("k", [2, 3, 7, 40, 129])
def test_cached_rank_weights_match_reference(k: int) -> None:
    rng = np.random.default_rng(k)
    dk = np.sort(rng.exponential(size=(50, k)), axis=1)
    dk[0] = 0.0  # zero spread row
    for _ in range(2):  # second call is served from the cache
        assert _same(_mreg_ensemble_weights(dk), _reference_ensemble_weights(dk))


def test_predict_path_matches_single_k_predictions() -> None:
    rng = np.random.default_rng(17)
    rpm = rng.normal(size=(80, 3))
    yhat = rng.normal(size=80)
    xt = rng.normal(size=(15, 3))
    mins, maxs = rpm.min(axis=0), rpm.max(axis=0)
    path = _mreg_predict_path(xt, rpm, yhat, 12, "NNS", mins, maxs)
    for k in range(1, 13):
        single = _mreg_predict(xt, rpm, yhat, k, "NNS", mins, maxs, False)
        assert single is not None
        assert _same(path[:, k - 1], single)


def test_stack_rank_average_matches_average_ranks() -> None:
    from scipy.stats import rankdata

    from nns.stack import _rank_average

    rng = np.random.default_rng(19)
    for n in (0, 1, 2, 5, 50, 500):
        v = np.round(rng.normal(size=n), 1)
        if n > 2:
            v[0] = -0.0
            v[1] = 0.0
        assert _same(_rank_average(v), rankdata(v, method="average").astype(np.float64))


def _reference_group_reduce(
    keys: np.ndarray, values: np.ndarray, noise_reduction: str, is_class: bool
) -> tuple[np.ndarray, np.ndarray]:
    """The former loop: split on 15-significant-digit key strings (R prints
    -0 as "0", so signed zeros share a key)."""
    order = np.argsort(keys, kind="stable")
    ks, vs = keys[order], values[order]
    labels = [f"{v + 0.0:.15g}" for v in ks]
    out_keys, reduced, start = [], [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            out_keys.append(float(labels[start]))
            reduced.append(_reduce_value(vs[start:i], noise_reduction, is_class))
            start = i
    return np.asarray(out_keys, dtype=np.float64), np.asarray(reduced, dtype=np.float64)


@pytest.mark.parametrize("noise_reduction", ["off", "mean", "median", "mode"])
@pytest.mark.parametrize("is_class", [False, True])
def test_univariate_group_reduce_matches_reference(noise_reduction: str, is_class: bool) -> None:
    rng = np.random.default_rng(23)
    for _ in range(30):
        n = int(rng.integers(1, 150))
        keys = np.round(rng.normal(size=n), int(rng.integers(0, 3)))
        idx = rng.integers(0, n, size=max(1, n // 4))
        keys[idx] = keys[idx] * (1 + 2e-16)  # equal to 15 significant digits
        keys = np.append(keys, [0.1 + 0.2, 0.3])
        values = rng.integers(0, 3, size=keys.size).astype(float)
        if not is_class:
            values += rng.normal(size=keys.size)
        values[rng.random(keys.size) < 0.1] = -0.0
        got = _group_reduce_sorted(keys, values, noise_reduction, is_class)
        ref = _reference_group_reduce(keys, values, noise_reduction, is_class)
        for a, b in zip(got, ref, strict=True):
            assert _same(a, b)
            assert np.array_equal(np.signbit(a), np.signbit(b))


def _numpy_only(monkeypatch: pytest.MonkeyPatch) -> None:
    import nns._reg_engine as engine

    monkeypatch.setattr(engine, "native_fn", lambda name: None)


@pytest.mark.parametrize("dist", ["NNS", "L1", "L2", "FACTOR"])
@pytest.mark.parametrize("is_class", [False, True])
def test_native_mreg_predict_matches_numpy_path(
    dist: str, is_class: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    # NNS-core's port of R's NNS_mreg_predict_v2_cpp / _path_v2_cpp. On data
    # without exact distance ties it agrees with the NumPy path to rounding.
    from nns._native import native_fn

    if native_fn("mreg_predict") is None:
        pytest.skip("native extension not available")
    rng = np.random.default_rng(31)
    rpm = rng.normal(size=(250, 4))
    yhat = np.round(rng.normal(size=250)) if is_class else rng.normal(size=250)
    if dist == "FACTOR":
        rpm = np.round(rpm)
    noise = 0.0 if dist == "FACTOR" else 0.05 * rng.normal(size=(60, 4))
    xt = rpm[rng.integers(0, 250, 60)] + noise
    mins, maxs = rpm.min(axis=0), rpm.max(axis=0)
    native = [_mreg_predict(xt, rpm, yhat, k, dist, mins, maxs, is_class) for k in (1, 3, 40)]
    path = None if is_class else _mreg_predict_path(xt, rpm, yhat, 25, dist, mins, maxs)
    _numpy_only(monkeypatch)
    for k, got in zip((1, 3, 40), native, strict=True):
        expected = _mreg_predict(xt, rpm, yhat, k, dist, mins, maxs, is_class)
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-14)
    if path is not None:
        expected_path = _mreg_predict_path(xt, rpm, yhat, 25, dist, mins, maxs)
        np.testing.assert_allclose(path, expected_path, rtol=1e-12, atol=1e-14)


def test_native_mreg_predict_breaks_distance_ties_like_numpy_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nns import nns_reg
    from nns._native import native_fn

    if native_fn("mreg_predict") is None:
        pytest.skip("native extension not available")
    x = np.array([[0.0, 2.0], [6.0, 6.0], [0.0, 2.0], [0.0, 1.0]])
    y = np.array([0.0, 10.0, 20.0, 30.0])
    point = np.array([[6.0, 0.0]])

    native = nns_reg(x, y, point_est=point, order="max", n_best=1, dist="L1")["Point.est"]
    _numpy_only(monkeypatch)
    fallback = nns_reg(x, y, point_est=point, order="max", n_best=1, dist="L1")["Point.est"]

    np.testing.assert_array_equal(native, fallback)
