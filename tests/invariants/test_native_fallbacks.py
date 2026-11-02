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
def test_mode_fallback_is_bit_identical_to_native(discrete: bool, multi: bool) -> None:
    native = native_fn("mode")
    if native is None:
        pytest.skip("native extension not available")
    fallback = _discrete_mode if discrete else _continuous_mode
    for x in _vectors():
        expected = np.asarray(native(np.ascontiguousarray(x), discrete, multi))
        got = fallback(x, multi)
        assert got.shape == expected.shape
        assert np.array_equal(got, expected)
        assert np.array_equal(np.signbit(got), np.signbit(expected))


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
