from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import nns
from nns._native import native_fn, native_threads

pytestmark = pytest.mark.invariant


def _require_native() -> None:
    if native_fn("_set_call_threads") is None:
        pytest.skip("native extension not available")


def test_set_num_threads_round_trips_and_restores_the_default() -> None:
    _require_native()
    default = nns.get_num_threads()
    try:
        nns.set_num_threads(3)
        assert nns.get_num_threads() == 3
        nns.set_num_threads(None)
        assert nns.get_num_threads() == default
        nns.set_num_threads(0)
        assert nns.get_num_threads() == default
    finally:
        nns.set_num_threads(None)
    with pytest.raises(ValueError, match="positive integer"):
        nns.set_num_threads(1.5)  # type: ignore[arg-type]


def test_native_threads_is_scoped_to_the_block_and_validates() -> None:
    _require_native()
    set_call = native_fn("_set_call_threads")
    assert set_call is not None
    with native_threads(2):
        assert set_call(0) == 2
        set_call(2)
        with native_threads(1):
            assert set_call(1) == 1
        assert set_call(2) == 2
    assert set_call(0) == 0
    for bad in (1.5, True, "2"):
        with pytest.raises(ValueError, match="ncores"), native_threads(bad):
            pass
    # 0 / negative values were silently ignored before ncores was honoured; they now
    # mean "default" with a DeprecationWarning instead of raising.
    for legacy in (0, -1):
        with pytest.warns(DeprecationWarning, match="ncores"), native_threads(legacy):
            pass


def test_thread_count_does_not_change_results() -> None:
    rng = np.random.default_rng(6)
    x = rng.normal(size=20_000)
    targets = np.linspace(-3.0, 3.0, 200)
    matrix = rng.normal(size=(5_000, 6))
    results = []
    for ncores in (1, 2, 4):
        with native_threads(ncores):
            results.append(
                (
                    np.asarray(nns.lpm(3.0, targets, x)),
                    np.asarray(nns.upm(0.5, targets, x)),
                    nns.pm_matrix(1.0, 1.0, "mean", matrix, True)["cov.matrix"],
                    nns.co_lpm_nd(matrix, np.mean(matrix, axis=0), 1.0),
                )
            )
    for other in results[1:]:
        for expected, actual in zip(results[0], other, strict=True):
            np.testing.assert_array_equal(actual, expected)


def test_ncores_is_honoured_without_a_warning() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(120, 2))
    y = x[:, 0] + x[:, 1] ** 2
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        serial = nns.nns_reg(x, y, ncores=1)["Fitted.xy"]["y.hat"]
        threaded = nns.nns_reg(x, y, ncores=4)["Fitted.xy"]["y.hat"]
    np.testing.assert_array_equal(serial, threaded)
    with pytest.warns(DeprecationWarning, match="ncores"):
        legacy = nns.nns_reg(x, y, ncores=0)["Fitted.xy"]["y.hat"]
    np.testing.assert_array_equal(serial, legacy)


def test_native_kernels_run_concurrently_from_python_threads() -> None:
    # The heavy kernels release the GIL; concurrent calls must stay correct.
    rng = np.random.default_rng(8)
    x = rng.normal(size=50_000)
    targets = np.linspace(-2.0, 2.0, 64)
    expected = np.asarray(nns.lpm(2.0, targets, x))

    def work() -> bool:
        with native_threads(1):
            return all(
                np.array_equal(np.asarray(nns.lpm(2.0, targets, x)), expected) for _ in range(5)
            )

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(work) for _ in range(4)]
    # result() re-raises a worker's exception in the test thread.
    assert all(future.result() for future in futures)


def test_warnings_point_at_the_callers_line() -> None:
    # Public entry points are wrapped by honours_ncores; warnings must still be
    # attributed to the user's call, not to the wrapper.
    t = np.arange(40, dtype=np.float64)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        nns.nns_var(np.column_stack((np.sin(t), np.cos(t))), h=2, tau=1, nowcast=True)
    nowcast = [w for w in caught if "nowcast" in str(w.message)]
    assert nowcast
    assert nowcast[0].filename == __file__


def test_fast_lm_fallback_matches_the_native_kernel_bit_for_bit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nns._helpers as helpers

    if helpers.native_fn("fast_lm") is None:
        pytest.skip("native extension not available")
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.390137, 2.847274, 3.173205])
    native = helpers._fast_lm(x, y)
    monkeypatch.setattr(helpers, "native_fn", lambda name: None)
    assert helpers._fast_lm(x, y) == native
