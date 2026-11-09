"""R 14.0 parity for numerical derivatives, partial-moment VaR and NNS.VAR contracts.

The derivative routines and the degree-1 VaR inversion feed exact comparisons
(root brackets, stencil half-widths, tied rows), so these checks follow R's own
arithmetic: R's ``mean()``/``sum()``/``cumsum()``/``colMeans()`` accumulate in
long double, R's ``x^2`` is ``x * x``, ``solve()`` is reference LAPACK ``dgesv``
and ``round()`` is nmath ``fround``. Test functions passed to ``NNS.diff`` are
written so that R and Python evaluate them with the same arithmetic (``v * v``
rather than ``v ** 2``, which calls the C ``pow``).
"""

from __future__ import annotations

import cmath
import math
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from _r import nns_frame_call, r_column
from _tolerances import COMPOUND

import nns
from nns._rblas import r_solve

pytestmark = pytest.mark.parity


def _num(values: Any) -> dict[str, Any]:
    return r_column([float(v) for v in np.asarray(values, dtype=np.float64).ravel()], "numeric")


def _complex_aware(real_fn: Callable[[float], float], complex_fn: Callable[[complex], complex]):
    return lambda v: complex_fn(v) if isinstance(v, complex) else real_fn(v)


_FUNCTIONS = {
    "poly": ("function(v) 3*v^2 - 2*v + 1", lambda v: 3 * (v * v) - 2 * v + 1),
    "cube": ("function(v) v^3", lambda v: v**3),
    "exp": (
        "function(v) exp(v / 2)",
        _complex_aware(lambda v: math.exp(v / 2), lambda v: cmath.exp(v / 2)),
    ),
    "sin": ("function(v) sin(v)", _complex_aware(math.sin, cmath.sin)),
    "log": ("function(v) log(v)", _complex_aware(math.log, cmath.log)),
}


@pytest.mark.parametrize("name", sorted(_FUNCTIONS))
@pytest.mark.parametrize(
    ("point", "h", "tol", "digits"),
    [(1.5, None, 1e-10, 12), (3.0, 0.5, 1e-8, 8), (0.5, None, 1e-6, 6)],
)
def test_nns_diff_matches_r_exactly(
    name: str, point: float, h: float | None, tol: float, digits: int
) -> None:
    r_fn, py_fn = _FUNCTIONS[name]
    h_arg = "" if h is None else f", h = {h!r}"
    expected = nns_frame_call(
        "diff14.full",
        f"res <- NNS.diff({r_fn}, {point!r}{h_arg}, tol = {tol!r}, digits = {digits}); "
        "unname(as.numeric(res[, 1]))",
        {},
        exact=True,
    )
    actual = nns.nns_diff(py_fn, point, h=h, tol=tol, digits=digits)
    values = np.array(list(actual.values()), dtype=np.float64)
    np.testing.assert_array_equal(values, np.asarray(expected, dtype=np.float64))


def test_dy_dx_any_character_eval_point_is_the_overall_gradient() -> None:
    rng = np.random.default_rng(51)
    x = np.sort(rng.uniform(-3, 3, 120))
    y = np.sin(2 * x)
    expected = nns_frame_call(
        "dydx14.character", "c(dy.dx(x, y, eval.point = 'obs'), dy.dx(x, y, 'overall'))",
        {"x": _num(x), "y": _num(y)}, exact=True,
    )
    actual = [nns.dy_dx(x, y, eval_point="obs"), nns.dy_dx(x, y, eval_point="overall")]
    np.testing.assert_array_equal(actual, np.asarray(expected, dtype=np.float64))


def test_dy_dx_two_valued_response_is_a_regression_as_in_r() -> None:
    rng = np.random.default_rng(52)
    x = np.sort(rng.uniform(0, 4, 90))
    y = (x + rng.normal(size=90) * 0.5 > 2).astype(float)
    expected = nns_frame_call(
        "dydx14.binary",
        "r <- dy.dx(x, y, eval.point = c(1, 2, 3)); "
        "list(o = dy.dx(x, y, 'overall'), f = r$first.derivative, s = r$second.derivative)",
        {"x": _num(x), "y": _num(y)},
    )
    assert nns.dy_dx(x, y, eval_point="overall") == pytest.approx(
        float(np.asarray(expected["o"])), abs=COMPOUND
    )
    actual = nns.dy_dx(x, y, eval_point=np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(actual["first.derivative"], expected["f"], rtol=0, atol=COMPOUND)
    np.testing.assert_allclose(actual["second.derivative"], expected["s"], rtol=0, atol=COMPOUND)


@pytest.mark.parametrize("wrt", [1, 2])
def test_dy_d_on_tied_coordinates_matches_r(wrt: int) -> None:
    """Stencil half-widths come from LPM.VaR(., 1, x); a few-ulp difference in
    that inversion moved the nodes and changed the derivative at tied rows."""
    rng = np.random.default_rng(53)
    x = np.round(rng.normal(size=(200, 2)), 1)
    y = x[:, 0] + x[:, 1] ** 2
    expected = nns_frame_call(
        "dyd14.ties",
        f"res <- dy.d_(matrix(v, ncol = 2), y, wrt = {wrt}, eval.points = 'obs', "
        "messages = FALSE); list(f = res['First', 1][[1]], s = res['Second', 1][[1]])",
        {"v": _num(x.ravel(order="F")), "y": _num(y)},
    )
    actual = nns.dy_d(x, y, wrt=wrt, eval_points="obs", messages=False)
    np.testing.assert_allclose(actual["First"], expected["f"], rtol=0, atol=COMPOUND)
    np.testing.assert_allclose(actual["Second"], expected["s"], rtol=0, atol=COMPOUND)


@pytest.mark.parametrize("degree", [1, 2, 3, 4, 0.5, 1.5])
def test_partial_moment_var_matches_r_bitwise(degree: float) -> None:
    rng = np.random.default_rng(54)
    probs = [0.0, 0.01, 0.025, 0.1, 0.33, 0.5, 0.77, 0.9, 0.975, 1.0]
    for shift, ties in ((0.0, False), (0.0, True), (1e6, False)):
        x = rng.normal(size=60) * 7 + shift
        if ties:
            x = np.round(x)
        expected = nns_frame_call(
            "var14.bits",
            f"c(LPM.VaR(p, {degree!r}, x), UPM.VaR(p, {degree!r}, x))",
            {"x": _num(x), "p": _num(probs)},
            exact=True,
        )
        actual = np.concatenate(
            [nns.lpm_var(np.array(probs), degree, x), nns.upm_var(np.array(probs), degree, x)]
        )
        np.testing.assert_array_equal(actual, np.asarray(expected, dtype=np.float64))


def test_solve_matches_r_bitwise() -> None:
    rng = np.random.default_rng(55)
    for n in (2, 3, 4, 5):
        for vandermonde in (False, True):
            if vandermonde:
                z = np.sort(rng.uniform(-1, 1, n))
                a = np.vander(z, n, increasing=True).T
            else:
                a = rng.normal(size=(n, n))
            b = rng.normal(size=n)
            expected = nns_frame_call(
                "solve14.bits",
                f"as.numeric(solve(matrix(a, {n}), b))",
                {"a": _num(a.ravel(order="F")), "b": _num(b)},
                exact=True,
            )
            np.testing.assert_array_equal(r_solve(a, b), np.asarray(expected, dtype=np.float64))


def _panel(seed: int, n: int = 60) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(1, n + 1)
    a = np.sin(2 * np.pi * t / 12) * 2 + rng.normal(size=n) * 0.3 + 10
    b = 0.6 * np.r_[a[0], a[:-1]] + rng.normal(size=n) * 0.3 + 3
    c = np.cumsum(rng.normal(size=n)) + 20
    x = np.column_stack([a, b, c])
    x[-2:, 2] = np.nan
    return x


@pytest.mark.parametrize("validation", ["pipeline", "observed"])
def test_nns_var_validation_modes_and_metadata_follow_r(validation: str) -> None:
    import pandas as pd

    x = _panel(56)
    flat = x.ravel(order="F")
    data = {"v": _num(np.nan_to_num(flat)), "na": _num(np.flatnonzero(np.isnan(flat)) + 1)}
    expected = nns_frame_call(
        "var14.validation",
        "M <- matrix(v, ncol = 3); M[na] <- NA; X <- as.data.frame(M); "
        "colnames(X) <- c('A', 'B', 'C'); "
        f"r <- NNS.VAR(X, h = 3, tau = 1, status = FALSE, ncores = 1, "
        f"validation = '{validation}'); list(ens = as.numeric(as.matrix(r$ensemble)), "
        "mode = r$validation$mode, origin = r$validation$origin, "
        "horizon = r$validation$horizon, "
        "targets = if (is.null(r$validation$observed.targets)) NULL "
        "else unname(r$validation$observed.targets))",
        data,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actual = nns.nns_var(
            pd.DataFrame(x, columns=list("ABC")), 3, tau=1, ncores=1, validation=validation
        )
    np.testing.assert_allclose(
        np.asarray(actual["ensemble"]).ravel(order="F"), expected["ens"], rtol=0, atol=COMPOUND
    )
    meta = actual["validation"]
    assert meta["mode"] == expected["mode"]
    assert meta["origin"] == float(np.asarray(expected["origin"]))
    assert meta["horizon"] == float(np.asarray(expected["horizon"]))
    if expected["targets"] is None:
        assert meta["observed.targets"] is None
    else:
        assert list(meta["observed.targets"].values()) == list(np.asarray(expected["targets"]))
