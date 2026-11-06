"""The NumPy fallbacks used without the compiled extension."""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from nns._native import native_fn
from nns.central_tendencies import _continuous_mode, _discrete_mode
from nns.dependence import _gravity_python

pytestmark = pytest.mark.invariant


def test_nns_disable_native_forces_the_fallbacks() -> None:
    code = (
        "from nns._native import native_fn, get_num_threads\n"
        "import nns\n"
        "assert native_fn('lpm') is None and get_num_threads() == 1\n"
        "print(nns.nns_gravity([1.0, 2.0, 2.0, 5.0]))\n"
    )
    env = dict(os.environ, NNS_DISABLE_NATIVE="1")
    done = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr
    assert float(done.stdout) > 0.0


def _vectors() -> list[np.ndarray]:
    rng = np.random.default_rng(12)
    out = []
    for trial in range(600):
        n = int(rng.integers(1, 250))
        kind = trial % 6
        if kind == 0:
            x = rng.normal(size=n)
        elif kind == 1:
            x = rng.integers(0, 5, n).astype(float)
        elif kind == 2:
            x = np.round(rng.standard_t(2, size=n), 2)
        elif kind == 3:
            x = np.r_[rng.normal(size=n), 0.0, -0.0, -0.0]
        elif kind == 4:
            x = np.r_[rng.normal(size=n), -1e5]
        else:
            x = np.r_[5.0 + 1e-7 * rng.normal(size=n), 6.0]
        out.append(x)
    return out


@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("multi", [False, True])
def test_mode_fallback_is_bit_identical_to_native(
    discrete: bool, multi: bool, request: pytest.FixtureRequest
) -> None:
    native = native_fn("mode")
    if native is None:
        pytest.skip("native extension not available")
    if discrete and not multi:
        reason = "the vendored NNS-core still averages tied discrete modes"
        request.applymarker(pytest.mark.xfail(strict=True, reason=reason))
    fallback = _discrete_mode if discrete else _continuous_mode
    for x in _vectors():
        expected = np.asarray(native(np.ascontiguousarray(x), discrete, multi))
        got = fallback(x, multi)
        assert got.shape == expected.shape
        assert np.array_equal(got, expected)
        assert np.array_equal(np.signbit(got), np.signbit(expected))


@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.xfail(strict=True, reason="the vendored NNS-core still averages tied discrete modes")
def test_discrete_mode_fallback_matches_native_on_tiny_samples(multi: bool) -> None:
    native = native_fn("mode")
    if native is None:
        pytest.skip("native extension not available")
    for values in ([1.0, 3.0], [1.0, 2.0], [2.0, 2.0, 7.0], [4.0], [1.0, 1.0, 3.0, 3.0]):
        x = np.asarray(values)
        expected = np.asarray(native(np.ascontiguousarray(x), True, multi))
        got = _discrete_mode(x, multi)
        assert got.shape == expected.shape
        assert np.array_equal(got, expected)


def test_gravity_fallback_keeps_signed_zeros_like_native() -> None:
    native = native_fn("gravity_exact")
    if native is None:
        pytest.skip("native extension not available")
    for x in ([-0.0], [-0.0, -0.0, 1.0], [-0.0] * 5, [-0.0, 0.0, 0.0, -0.0, 3.0]):
        values = np.asarray(x, dtype=np.float64)
        expected = float(native(values))
        got = _gravity_python(values)
        assert got == expected
        assert np.signbit(got) == np.signbit(expected)


def _dep_pairs() -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(31)
    x = rng.normal(size=100)
    binary = np.r_[np.zeros(51), np.ones(49)]
    ternary = np.repeat([-1.0, 0.0, 1.0], 100)
    pairs = [
        (x, -x),  # every cell fully discordant
        (binary, binary),  # 49 * (1 / 49) != 1
        (rng.permutation(ternary), rng.permutation(ternary)),  # ties at the mean
    ]
    for trial in range(60):
        n = int(rng.integers(10, 200))
        a = rng.integers(-2, 3, n).astype(float)
        b = a * (trial % 3 - 1) + rng.integers(-1, 2, n) * (trial % 2)
        pairs.append((a, b.astype(float)))
    return pairs


@pytest.mark.parametrize("asym", [False, True])
def test_dep_fallback_matches_native_on_ties(
    asym: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    native = native_fn("nns_dep_pair")
    if native is None:
        pytest.skip("native extension not available")
    import nns.dependence as dependence

    pairs = _dep_pairs()
    expected = [native(np.ascontiguousarray(a), np.ascontiguousarray(b), asym) for a, b in pairs]
    monkeypatch.setattr(dependence, "native_fn", lambda name: None)
    for (a, b), want in zip(pairs, expected, strict=True):
        got = dependence.nns_dep(a, b, asym=asym)
        # The fallback adds the cell terms in first-seen order, R in hash order.
        assert got["Correlation"] == pytest.approx(want["Correlation"], abs=1e-12)
        assert got["Dependence"] == pytest.approx(want["Dependence"], abs=1e-12)


def test_nd_degree0_fallback_matches_native_and_partitions_the_sample() -> None:
    from nns.dependence import co_lpm_nd, co_upm_nd, dpm_nd

    natives = [native_fn(name) for name in ("clpm_nd", "cupm_nd", "dpm_nd")]
    rng = np.random.default_rng(7)
    for trial in range(40):
        n, d = int(rng.integers(1, 80)), int(rng.integers(2, 5))
        data = rng.integers(-1, 2, (n, d)).astype(float)
        target = data[int(rng.integers(0, n))] if trial % 2 else np.zeros(d)
        got = [f(data, target, 0.0) for f in (co_lpm_nd, co_upm_nd, dpm_nd)]
        assert sum(got) == pytest.approx(1.0, abs=1e-12)
        if all(f is not None for f in natives):
            flat = np.ascontiguousarray(data.T).ravel()
            want = [f(flat, n, d, target, 0.0, False) for f in natives]
            assert got == want


def test_continuous_mode_selects_r14_discrete_value_not_a_mean() -> None:
    # R 14.0 treats evenly spaced discrete-valued samples as exact values;
    # an average of tied modes would yield 9.5, which is not observed.
    x = np.repeat(np.arange(20, dtype=np.float64), 400)
    modes = _continuous_mode(x, True)
    np.testing.assert_array_equal(modes, np.arange(20, dtype=np.float64))
    assert _continuous_mode(x, False)[0] == pytest.approx(9.0, abs=1e-12)
