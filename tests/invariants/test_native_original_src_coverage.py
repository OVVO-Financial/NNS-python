from __future__ import annotations

import importlib
from collections.abc import Iterator
from types import ModuleType
from typing import Any, cast

import numpy as np
import pytest

from pynns import (
    co_lpm,
    co_upm,
    d_lpm,
    d_upm,
    lpm,
    lpm_ratio,
    pm_matrix,
    upm,
    upm_ratio,
)

core_module = importlib.import_module("pynns.core")
co_moments_module = importlib.import_module("pynns.co_moments")
pm_matrix_module = importlib.import_module("pynns.pm_matrix")


def _native() -> ModuleType:
    return cast(ModuleType, pytest.importorskip("pynns._nnscore"))


pytestmark = pytest.mark.invariant


@pytest.fixture()
def native() -> ModuleType:
    return cast(ModuleType, pytest.importorskip("pynns._nnscore"))


@pytest.fixture()
def disable_native(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(core_module, "nnscore", lambda: None)
    monkeypatch.setattr(co_moments_module, "nnscore", lambda: None)
    monkeypatch.setattr(pm_matrix_module, "nnscore", lambda: None)
    yield


def test_direct_native_partial_moment_smoke(native: ModuleType) -> None:
    x = np.array([-2.0, -1.0, 0.5, 3.0], dtype=np.float64)
    y = np.array([1.0, -0.5, 2.0, 4.0], dtype=np.float64)
    targets = np.array([-1.0, 0.0, 1.0], dtype=np.float64)

    assert native.lpm(2.0, 0.0, x) == pytest.approx(1.25)
    assert native.upm(2.0, 0.0, x) == pytest.approx(2.3125)
    if hasattr(native, "lpm_ratio_v"):
        np.testing.assert_allclose(
            native.lpm_ratio_v(2.0, targets, x),
            lpm_ratio(2.0, targets, x),
        )
    if hasattr(native, "upm_ratio_v"):
        np.testing.assert_allclose(
            native.upm_ratio_v(2.0, targets, x),
            upm_ratio(2.0, targets, x),
        )

    if hasattr(native, "co_lpm"):
        assert np.isfinite(native.co_lpm(1.0, 1.0, x, y, 0.0, 1.0))
    if hasattr(native, "co_upm"):
        assert np.isfinite(native.co_upm(1.0, 1.0, x, y, 0.0, 1.0))
    if hasattr(native, "d_lpm"):
        assert np.isfinite(native.d_lpm(1.0, 1.0, x, y, 0.0, 1.0))
    if hasattr(native, "d_upm"):
        assert np.isfinite(native.d_upm(1.0, 1.0, x, y, 0.0, 1.0))

    matrix = np.array(
        [[-2.0, 1.0], [-1.0, -0.5], [0.5, 2.0], [3.0, 4.0]], dtype=np.float64
    )
    target = np.mean(matrix, axis=0).astype(np.float64)
    if hasattr(native, "pm_matrix"):
        native_pm = native.pm_matrix(
            1.0,
            1.0,
            np.ascontiguousarray(target),
            np.ascontiguousarray(np.ravel(matrix, order="F")),
            matrix.shape[0],
            matrix.shape[1],
            True,
            False,
        )
        assert native_pm["dim"] == 2
        assert set(native_pm) >= {"cupm", "dupm", "dlpm", "clpm", "cov.matrix", "dim"}


def test_direct_native_fast_lm_smoke(native: ModuleType) -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    y = np.array([3.0, 5.0, 7.0, 9.0], dtype=np.float64)

    fit = native.fast_lm(x, y)
    np.testing.assert_allclose(fit["coef"], [1.0, 2.0], atol=1e-12)
    np.testing.assert_allclose(fit["residuals"], np.zeros_like(x), atol=1e-12)

    if hasattr(native, "fast_lm_mult"):
        design = np.column_stack([x, x**2]).astype(np.float64)
        mult = native.fast_lm_mult(
            np.ascontiguousarray(np.ravel(design, order="F")),
            y,
            design.shape[0],
            design.shape[1],
        )
        assert len(mult["coefficients"]) == 3
        np.testing.assert_allclose(mult["fitted_values"], y, atol=1e-10)


def test_direct_native_internal_helper_smoke(native: ModuleType) -> None:
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    matrix = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)

    if hasattr(native, "is_discrete"):
        assert native.is_discrete(x) is True
    if hasattr(native, "vec_sd"):
        assert native.vec_sd(x) == pytest.approx(np.std(x, ddof=1))
    if hasattr(native, "col_sd"):
        np.testing.assert_allclose(
            native.col_sd(np.ascontiguousarray(np.ravel(matrix, order="F")), 3, 2),
            np.std(matrix, axis=0, ddof=1),
        )
    if hasattr(native, "factor_2_dummy"):
        dummy = native.factor_2_dummy([1, 2, 3, 2], ["a", "b", "c"])
        assert dummy["nrow"] == 4
        assert dummy["ncol"] == 2
        assert list(dummy["names"]) == ["b", "c"]
    if hasattr(native, "factor_2_dummy_fr"):
        dummy_fr = native.factor_2_dummy_fr([1, 2, 3, 2], ["a", "b", "c"])
        assert dummy_fr["nrow"] == 4
        assert dummy_fr["ncol"] == 3
        assert list(dummy_fr["names"]) == ["a", "b", "c"]


class _FinitePartialMomentNative:
    @staticmethod
    def lpm(*args: Any, **kwargs: Any) -> float:
        return 0.0

    @staticmethod
    def upm(*args: Any, **kwargs: Any) -> float:
        return 0.0


def test_public_lpm_upm_non_finite_values_use_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(core_module, "nnscore", lambda: _FinitePartialMomentNative())
    x = np.array([1.0, np.nan, 3.0], dtype=np.float64)

    assert np.isnan(lpm(1.0, 0.0, x))
    assert np.isnan(upm(1.0, 0.0, x))


@pytest.mark.parametrize(
    ("native_call", "fallback_call"),
    [
        (
            lambda: lpm(2.0, np.array([-1.0, 0.0, 1.0]), _x()),
            lambda: lpm(2.0, np.array([-1.0, 0.0, 1.0]), _x()),
        ),
        (
            lambda: upm(2.0, np.array([-1.0, 0.0, 1.0]), _x()),
            lambda: upm(2.0, np.array([-1.0, 0.0, 1.0]), _x()),
        ),
        (
            lambda: lpm_ratio(2.0, np.array([-1.0, 0.0, 1.0]), _x()),
            lambda: lpm_ratio(2.0, np.array([-1.0, 0.0, 1.0]), _x()),
        ),
        (
            lambda: upm_ratio(2.0, np.array([-1.0, 0.0, 1.0]), _x()),
            lambda: upm_ratio(2.0, np.array([-1.0, 0.0, 1.0]), _x()),
        ),
        (
            lambda: co_lpm(1.0, _x(), _y(), np.array([0.0, 1.0]), np.array([1.0])),
            lambda: co_lpm(1.0, _x(), _y(), np.array([0.0, 1.0]), np.array([1.0])),
        ),
        (
            lambda: co_upm(1.0, _x(), _y(), np.array([0.0, 1.0]), np.array([1.0])),
            lambda: co_upm(1.0, _x(), _y(), np.array([0.0, 1.0]), np.array([1.0])),
        ),
        (
            lambda: d_lpm(1.0, 1.0, _x(), _y(), np.array([0.0, 1.0]), np.array([1.0])),
            lambda: d_lpm(1.0, 1.0, _x(), _y(), np.array([0.0, 1.0]), np.array([1.0])),
        ),
        (
            lambda: d_upm(1.0, 1.0, _x(), _y(), np.array([0.0, 1.0]), np.array([1.0])),
            lambda: d_upm(1.0, 1.0, _x(), _y(), np.array([0.0, 1.0]), np.array([1.0])),
        ),
    ],
)
def test_public_partial_moment_fallback_matches_native(
    native_call: Any,
    fallback_call: Any,
    disable_native: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del disable_native
    fallback = fallback_call()
    native_module = _native()
    monkeypatch.setattr(core_module, "nnscore", lambda: native_module)
    monkeypatch.setattr(co_moments_module, "nnscore", lambda: native_module)
    actual = native_call()
    np.testing.assert_allclose(actual, fallback)


def test_public_pm_matrix_fallback_matches_native(
    disable_native: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    del disable_native
    matrix = np.array([[-2.0, 1.0], [-1.0, -0.5], [0.5, 2.0], [3.0, 4.0]], dtype=np.float64)
    fallback = pm_matrix(1.0, 1.0, "mean", matrix, True, norm=False)

    native_module = _native()
    monkeypatch.setattr(pm_matrix_module, "nnscore", lambda: native_module)
    actual = pm_matrix(1.0, 1.0, "mean", matrix, True, norm=False)

    assert actual.keys() == fallback.keys()
    for key in actual:
        np.testing.assert_allclose(actual[key], fallback[key])


def test_public_api_fallback_works_without_native(disable_native: None) -> None:
    del disable_native
    x = _x()
    y = _y()
    assert np.isfinite(lpm(2.0, 0.0, x))
    assert np.isfinite(upm(2.0, 0.0, x))
    assert np.isfinite(lpm_ratio(2.0, 0.0, x))
    assert np.isfinite(upm_ratio(2.0, 0.0, x))
    assert np.isfinite(co_lpm(1.0, x, y, 0.0, 1.0))
    assert np.isfinite(co_upm(1.0, x, y, 0.0, 1.0))
    assert np.isfinite(d_lpm(1.0, 1.0, x, y, 0.0, 1.0))
    assert np.isfinite(d_upm(1.0, 1.0, x, y, 0.0, 1.0))
    assert pm_matrix(1.0, 1.0, "mean", np.column_stack([x, y]), True)


def _x() -> np.ndarray[Any, np.dtype[np.float64]]:
    return np.array([-2.0, -1.0, 0.5, 3.0], dtype=np.float64)


def _y() -> np.ndarray[Any, np.dtype[np.float64]]:
    return np.array([1.0, -0.5, 2.0, 4.0], dtype=np.float64)
