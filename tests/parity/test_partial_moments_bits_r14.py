"""R 14.0 parity, to the last bit, for the partial-moment family.

* ``NNS.moments`` is R 14's: ``mean()`` deviations, ``mean(z^3)``/``mean(z^4)``
  with R's ``^`` (libm ``pow``) and the variance raised by ``^1.5``.
* ``Co.LPM``/``Co.UPM``/``D.LPM``/``D.UPM``, the ``_nD`` co-moments and
  ``PM.matrix`` follow R's C++: integer degrees by repeated multiplication,
  fractional ones by libm ``pow``, double sums accumulated row by row and scaled
  by ``1 / n``. ``PM.matrix`` fills the upper triangle and mirrors it (DUPM and
  DLPM crossed), and its default target is Rcpp's ``colMeans``.

Each case runs with the native kernels and with the NumPy fallback.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest
from _r import nns_frame_call, r_column

import nns

pytestmark = pytest.mark.parity

co_moments_module = importlib.import_module("nns.co_moments")
pm_matrix_module = importlib.import_module("nns.pm_matrix")


def _num(values: Any) -> dict[str, Any]:
    return r_column([float(v) for v in np.asarray(values, dtype=np.float64).ravel()], "numeric")


@pytest.mark.parametrize(("population", "flag"), [(True, "TRUE"), (False, "FALSE")])
def test_nns_moments_match_r_bitwise(population: bool, flag: str) -> None:
    rng = np.random.default_rng(401)
    x = rng.exponential(size=75) ** 2
    expected = nns_frame_call(
        "pm14.moments",
        f"NNS.moments(x, population = {flag})",
        {"x": _num(x)},
        exact=True,
    )
    actual = nns.nns_moments(x, population=population)
    for key in ("mean", "variance", "skewness", "kurtosis"):
        assert actual[key] == expected[key], key


@pytest.mark.parametrize("native", [True, False])
@pytest.mark.parametrize(("degree_x", "degree_y"), [(3, 1), (0.5, 1.5), (2, 0)])
def test_bivariate_co_moments_match_r_bitwise(
    monkeypatch: pytest.MonkeyPatch, native: bool, degree_x: float, degree_y: float
) -> None:
    rng = np.random.default_rng(402)
    x = rng.normal(size=90)
    y = x * x + rng.normal(size=90) * 0.3
    tx, ty = x[:4], y[:4]
    expected = nns_frame_call(
        "pm14.co_moments",
        f"list(clpm = Co.LPM({degree_x}, x, y, tx, ty, degree_y = {degree_y}), "
        f"cupm = Co.UPM({degree_x}, x, y, tx, ty, degree_y = {degree_y}), "
        f"dlpm = D.LPM({degree_x}, {degree_y}, x, y, tx, ty), "
        f"dupm = D.UPM({degree_x}, {degree_y}, x, y, tx, ty))",
        {"x": _num(x), "y": _num(y), "tx": _num(tx), "ty": _num(ty)},
        exact=True,
    )
    if not native:
        monkeypatch.setattr(co_moments_module, "native_fn", lambda name: None)
    actual = {
        "clpm": nns.co_lpm(degree_x, x, y, tx, ty, degree_y=degree_y),
        "cupm": nns.co_upm(degree_x, x, y, tx, ty, degree_y=degree_y),
        "dlpm": nns.d_lpm(degree_x, degree_y, x, y, tx, ty),
        "dupm": nns.d_upm(degree_x, degree_y, x, y, tx, ty),
    }
    for key, value in expected.items():
        np.testing.assert_array_equal(np.atleast_1d(actual[key]), np.atleast_1d(value), key)


@pytest.mark.parametrize("degree", [2, 0.5, 3])
def test_nd_co_moments_match_r_bitwise(degree: float) -> None:
    rng = np.random.default_rng(403)
    data = rng.normal(size=(70, 3))
    target = data[5]
    expected = nns_frame_call(
        "pm14.nd",
        f"X <- matrix(v, ncol = 3); list(clpm = Co.LPM_nD(X, t, degree = {degree}), "
        f"cupm = Co.UPM_nD(X, t, degree = {degree}), dpm = DPM_nD(X, t, degree = {degree}))",
        {"v": _num(data.ravel(order="F")), "t": _num(target)},
        exact=True,
    )
    assert nns.co_lpm_nd(data, target, degree=degree) == expected["clpm"]
    assert nns.co_upm_nd(data, target, degree=degree) == expected["cupm"]
    assert nns.dpm_nd(data, target, degree=degree) == expected["dpm"]


@pytest.mark.parametrize("native", [True, False])
@pytest.mark.parametrize(
    ("lpm_degree", "upm_degree", "pop_adj", "norm"),
    [(0.5, 3, True, False), (2, 1, False, True), (0, 0, True, True), (3, 3, True, False)],
)
def test_pm_matrix_matches_r_bitwise(
    monkeypatch: pytest.MonkeyPatch,
    native: bool,
    lpm_degree: float,
    upm_degree: float,
    pop_adj: bool,
    norm: bool,
) -> None:
    rng = np.random.default_rng(404)
    data = rng.normal(size=(140, 4)) + np.array([0.0, 2.0, -1.0, 5.0])
    expected = nns_frame_call(
        "pm14.pm_matrix",
        f"lapply(PM.matrix({lpm_degree}, {upm_degree}, 'mean', matrix(v, ncol = 4), "
        f"{'TRUE' if pop_adj else 'FALSE'}, norm = {'TRUE' if norm else 'FALSE'}), unname)",
        {"v": _num(data.ravel(order="F"))},
        exact=True,
    )
    if not native:
        monkeypatch.setattr(pm_matrix_module, "native_fn", lambda name: None)
    # A Fortran-ordered copy must give the same bits (the default target is a
    # row-by-row sum whatever the memory layout).
    for layout in (data, np.asfortranarray(data)):
        actual = nns.pm_matrix(lpm_degree, upm_degree, "mean", layout, pop_adj, norm=norm)
        assert set(expected) == {"cupm", "dupm", "dlpm", "clpm", "cov.matrix"}
        for key, rows in expected.items():
            np.testing.assert_array_equal(actual[key], np.asarray(rows, dtype=np.float64), key)
