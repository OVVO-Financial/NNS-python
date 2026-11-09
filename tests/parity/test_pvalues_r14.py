"""R 14.0 parity for the permutation p-values of NNS.seas, NNS.dep and NNS.caus.

R draws every permutation with ``sample()``/``sample.int()``, which the port
reproduces from R's Mersenne-Twister stream: with ``random_seed = s`` (``seed``
for ``nns_causation``, as R's ``NNS.caus(seed = )``) the nulls, p-values and
intervals are R's after ``set.seed(s)``, bit for bit.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from _r import nns_frame_call, r_column

import nns
from nns.causation import causal_matrix

pytestmark = pytest.mark.parity


def _num(values: Any) -> dict[str, Any]:
    return r_column([float(v) for v in np.asarray(values, dtype=np.float64).ravel()], "numeric")


def _seasonal(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    return 10.0 + 2.0 * np.sin(2.0 * np.pi * t / 12.0) + rng.normal(size=n) * 0.6


@pytest.mark.parametrize(("modulo", "r_modulo"), [(None, "NULL"), (6, "6")])
def test_nns_seas_p_values_match_r_under_set_seed(modulo: int | None, r_modulo: str) -> None:
    x = _seasonal(301, 72)
    expected = nns_frame_call(
        "pval14.seas",
        f"set.seed(5); r <- NNS.seas(x, modulo = {r_modulo}, mod.only = FALSE, plot = FALSE, "
        "p.value = TRUE, nperm = 25); list(p = r$all.periods$Period, "
        "pv = r$all.periods$p.value, periods = r$periods, best = r$best.period)",
        {"x": _num(x)},
        exact=True,
    )
    actual = nns.nns_seas(x, modulo=modulo, mod_only=False, p_value=True, nperm=25, random_seed=5)
    table = actual["all.periods"]
    np.testing.assert_array_equal(table["Period"], np.asarray(expected["p"], dtype=np.int64))
    np.testing.assert_array_equal(table["p.value"], np.asarray(expected["pv"], dtype=np.float64))
    np.testing.assert_array_equal(actual["periods"], np.atleast_1d(expected["periods"]))
    assert actual["best.period"] == int(np.asarray(expected["best"]))


@pytest.mark.parametrize("asym", [False, True])
def test_nns_dep_p_values_match_r_under_set_seed(asym: bool) -> None:
    rng = np.random.default_rng(302)
    x = rng.normal(size=60)
    y = x * x + rng.normal(size=60) * 0.5
    flag = "TRUE" if asym else "FALSE"
    expected = nns_frame_call(
        "pval14.dep",
        f"set.seed(21); lapply(NNS.dep(x, y, asym = {flag}, p.value = TRUE), unname)",
        {"x": _num(x), "y": _num(y)},
        exact=True,
    )
    actual = nns.nns_dep(x, y, asym=asym, p_value=True, random_seed=21)
    assert set(actual) == set(expected)
    for key, value in expected.items():
        np.testing.assert_array_equal(np.atleast_1d(actual[key]), np.atleast_1d(value), key)


def test_nns_dep_p_value_needs_both_variables() -> None:
    with pytest.raises(ValueError, match=r"p\.value = TRUE requires both x and y"):
        nns.nns_dep(np.ones((5, 2)), p_value=True)


@pytest.mark.parametrize(
    ("tau", "r_tau", "permute"),
    [(0, "0", "auto"), (0, "0", "both"), (2, "2", "auto"), (2, "2", "x"), ("ts", '"ts"', "shift")],
)
def test_nns_caus_p_values_match_r(tau: int | str, r_tau: str, permute: str) -> None:
    # n = 150 so that the period of 12 is within sqrt(n), as tau = "ts" requires.
    x = _seasonal(303, 150)
    rng = np.random.default_rng(304)
    y = np.r_[x[3:], x[:3]] + rng.normal(size=150) * 0.3
    expected = nns_frame_call(
        "pval14.caus",
        f"r <- NNS.caus(x, y, tau = {r_tau}, p.value = TRUE, nperm = 15, permute = '{permute}', "
        "seed = 99, conf.int = 0.9); lapply(c(as.list(r$causation), r$p.value), "
        "function(v) if (is.numeric(v)) unname(v) else v)",
        {"x": _num(x), "y": _num(y)},
        exact=True,
    )
    actual = nns.nns_causation(
        x, y, tau=tau, p_value=True, nperm=15, permute=permute, seed=99, conf_int=0.9
    )
    flat = {**actual["causation"], **actual["p.value"]}
    assert set(flat) == set(expected)
    for key, value in expected.items():
        if isinstance(value, str):
            assert flat[key] == value
        else:
            np.testing.assert_array_equal(np.atleast_1d(flat[key]), np.atleast_1d(value), key)


@pytest.mark.parametrize(("tau", "permute"), [(0, "auto"), (1, "auto"), (0, "y")])
def test_nns_caus_matrix_p_values_match_r(tau: int, permute: str) -> None:
    rng = np.random.default_rng(305)
    base = rng.normal(size=40)
    m = np.column_stack([base, 0.5 * base + rng.normal(size=40), rng.normal(size=40)])
    expected = nns_frame_call(
        "pval14.caus_matrix",
        f"lapply(NNS.caus(matrix(v, ncol = 3), tau = {tau}, p.value = TRUE, nperm = 10, "
        f"permute = '{permute}', seed = 7), unname)",
        {"v": _num(m.ravel(order="F"))},
        exact=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actual = causal_matrix(m, tau=tau, p_value=True, nperm=10, permute=permute, seed=7)
        assert nns.nns_causation(m, tau=tau, p_value=True, nperm=10, permute=permute, seed=7)[
            "p.value_matrix"
        ].tolist() == actual["p.value_matrix"].tolist()
    assert set(actual) == set(expected)
    for key, rows in expected.items():
        np.testing.assert_array_equal(actual[key], np.asarray(rows, dtype=np.float64), key)


def test_permute_accepts_r_partial_matching() -> None:
    x = _seasonal(306, 30)
    y = np.roll(x, 2)
    full = nns.nns_causation(x, y, tau=1, p_value=True, nperm=5, permute="shift", seed=1)
    partial = nns.nns_causation(x, y, tau=1, p_value=True, nperm=5, permute="sh", seed=1)
    assert full["p.value"]["permute"] == partial["p.value"]["permute"] == "shift"
    with pytest.raises(ValueError, match="'arg' should be one of"):
        nns.nns_causation(x, y, p_value=True, permute="bogus")
