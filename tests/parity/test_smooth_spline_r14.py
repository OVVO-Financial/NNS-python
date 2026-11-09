"""R 14.0 parity, to the last bit, for NNS.reg(smooth = TRUE) and what builds on it.

``NNS.reg`` fits ``stats::smooth.spline(rp$x, rp$y, spar = (dep + 0.5) / 2)``;
the port reproduces R 4.5.0's smoothing spline line by line
(``nns._rsmspline``), so the knots, coefficients and predictions, ``dy.dx``
and the smoothed fits are R's own bits.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest
from _r import nns_frame_call, r_column

import nns
from nns._rsmspline import smooth_spline

pytestmark = pytest.mark.parity

reg_engine = importlib.import_module("nns._reg_engine")


def _num(values: Any) -> dict[str, Any]:
    return r_column([float(v) for v in np.asarray(values, dtype=np.float64).ravel()], "numeric")


@pytest.mark.parametrize(("n", "duplicates"), [(9, False), (60, True), (240, False)])
@pytest.mark.parametrize("spar", [0.3, 0.55, 0.75])
def test_smooth_spline_matches_r_bitwise(n: int, duplicates: bool, spar: float) -> None:
    rng = np.random.default_rng(701 + n)
    x = rng.uniform(-2.0, 4.0, n)
    if duplicates:
        x = np.round(x, 1)
    y = np.sin(x) + rng.normal(size=n) * 0.3
    lo, hi = float(x.min()), float(x.max())
    xo = np.array([lo - 0.5, lo, 0.1, 1.7, hi, hi + 0.75])
    expected = nns_frame_call(
        "smspl14.fit",
        f"f <- smooth.spline(x, y, spar = {spar!r}); "
        "list(knot = f$fit$knot, coef = f$fit$coef, p = predict(f, xo)$y)",
        {"x": _num(x), "y": _num(y), "xo": _num(xo)},
        exact=True,
    )
    fit = smooth_spline(x, y, spar)
    np.testing.assert_array_equal(fit.knot[1:], np.asarray(expected["knot"]))
    np.testing.assert_array_equal(fit.coef[1:], np.asarray(expected["coef"]))
    np.testing.assert_array_equal(fit.predict(xo), np.asarray(expected["p"]))


def test_smooth_spline_needs_four_unique_x_like_r() -> None:
    with pytest.raises(ValueError, match="need at least four unique 'x' values"):
        smooth_spline(np.array([1.0, 2.0, 3.0, 3.0, 2.0]), np.arange(5.0), 0.5)


@pytest.mark.parametrize("native", [True, False])
def test_nns_reg_smooth_matches_r_bitwise(monkeypatch: pytest.MonkeyPatch, native: bool) -> None:
    rng = np.random.default_rng(702)
    x = rng.normal(size=150)
    y = x**3 - x + rng.normal(size=150) * 0.4
    point = np.array([-3.5, -0.2, 0.9, 3.2])
    expected = nns_frame_call(
        "smspl14.reg",
        "r <- NNS.reg(x, y, smooth = TRUE, point.est = pe, plot = FALSE); "
        "list(fit = r$Fitted.xy$y.hat, point = r$Point.est)",
        {"x": _num(x), "y": _num(y), "pe": _num(point)},
        exact=True,
    )
    if not native:
        monkeypatch.setattr(reg_engine, "native_fn", lambda name: None)
    actual = nns.nns_reg(x, y, smooth=True, point_est=point)
    np.testing.assert_array_equal(actual["Fitted.xy"]["y.hat"], np.asarray(expected["fit"]))
    np.testing.assert_array_equal(actual["Point.est"], np.asarray(expected["point"]))


def test_dy_dx_matches_r_bitwise() -> None:
    rng = np.random.default_rng(703)
    x = rng.uniform(-3.0, 3.0, 120)
    y = np.sin(x) + rng.normal(size=120) * 0.1
    expected = nns_frame_call(
        "smspl14.dydx",
        "r <- dy.dx(x, y, eval.point = c(-1.5, 0.2, 2)); "
        "list(first = r$first.derivative, second = r$second.derivative)",
        {"x": _num(x), "y": _num(y)},
        exact=True,
    )
    actual = nns.dy_dx(x, y, eval_point=np.array([-1.5, 0.2, 2.0]))
    np.testing.assert_array_equal(actual["first.derivative"], np.asarray(expected["first"]))
    np.testing.assert_array_equal(actual["second.derivative"], np.asarray(expected["second"]))


def test_numeric_multivariate_fit_has_no_class_levels_like_r() -> None:
    rng = np.random.default_rng(704)
    x = rng.normal(size=(40, 2))
    y = x[:, 0] + rng.normal(size=40) * 0.1
    expected = nns_frame_call(
        "smspl14.mreg_names",
        "list(reg = names(NNS.reg(matrix(v, ncol = 2), y, plot = FALSE)), "
        "mreg = names(NNS:::NNS.M.reg(matrix(v, ncol = 2), y)))",
        {"v": _num(x.ravel(order="F")), "y": _num(y)},
        exact=True,
    )
    assert "class.levels" not in expected["reg"]
    assert "class.levels" in expected["mreg"]
    assert "class.levels" not in nns.nns_reg(x, y)
    assert nns.nns_m_reg(x, y)["class.levels"] is None
