"""R 14.0 parity for the maximum-entropy bootstrap and what is built on it.

``NNS.meboot`` draws only through ``runif``, which the port reproduces from R's
Mersenne-Twister stream, so with ``random_seed = s`` the replicates are R's
after ``set.seed(s)``, bit for bit; so are ``NNS.MC``'s, ``NNS.SS``'s bootstrap
bounds and ``NNS.ARMA``'s prediction intervals, which resample the forecast path
with ``NNS.MC``. The deterministic steps follow R's arithmetic (``optimize``,
``qnorm``, trimmed ``mean``, long-double sums), and ``force.clt`` spreads the
replicate means by ``sd / sqrt(n)``, R 14's correction of meboot's
``sqrt(reps)``.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from _r import nns_frame_call, r_column

import nns
from nns.mc import _format_r_number

pytestmark = pytest.mark.parity


def _num(values: Any) -> dict[str, Any]:
    return r_column([float(v) for v in np.asarray(values, dtype=np.float64).ravel()], "numeric")


def _series(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    return 10.0 + 0.2 * t + 2.0 * np.sin(2.0 * np.pi * t / 7.0) + rng.normal(size=n)


def _rows(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(len(value), -1)


@pytest.mark.parametrize(
    ("r_args", "py_args"),
    [
        ("reps = 6, rho = 0.5", {"reps": 6, "rho": 0.5}),
        ("reps = 4, rho = c(-1, 0.3)", {"reps": 4, "rho": [-1.0, 0.3]}),
        ("reps = 3, rho = 0.4, type = 'pearson'", {"reps": 3, "rho": 0.4, "type": "pearson"}),
        ("reps = 3, rho = 0.6, type = 'nnsdep'", {"reps": 3, "rho": 0.6, "type": "nnsdep"}),
        ("reps = 5, target_drift = 0.05", {"reps": 5, "target_drift": 0.05}),
        ("reps = 5, rho = 0.2, drift = FALSE", {"reps": 5, "rho": 0.2, "drift": False}),
        (
            "reps = 4, rho = 0.5, xmin = 0, xmax = 30",
            {"reps": 4, "rho": 0.5, "xmin": 0.0, "xmax": 30.0},
        ),
        (
            "reps = 4, rho = 0.5, scl.adjustment = TRUE, sym = TRUE, trim = 0.25",
            {"reps": 4, "rho": 0.5, "scl_adjustment": True, "sym": True, "trim": 0.25},
        ),
    ],
)
def test_nns_meboot_replicates_are_rs_under_set_seed(r_args: str, py_args: dict[str, Any]) -> None:
    x = _series(201, 40)
    expected = nns_frame_call(
        "boot14.meboot",
        f"set.seed(42); r <- NNS.meboot(x, {r_args}); lapply(seq_len(ncol(r)), function(j) "
        "list(rep = unname(r['replicates', j][[1]]), ens = r['ensemble', j][[1]], "
        "dvtrim = r['dvtrim', j][[1]], desintxb = r['desintxb', j][[1]]))",
        {"x": _num(x)},
        exact=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actual = nns.nns_meboot(x, random_seed=42, **py_args)
    results = [actual] if isinstance(actual, dict) else actual
    assert len(results) == len(expected)
    for got, want in zip(results, expected, strict=True):
        np.testing.assert_array_equal(np.asarray(got["replicates"]), _rows(want["rep"]))
        np.testing.assert_array_equal(got["ensemble"], np.asarray(want["ens"], dtype=np.float64))
        assert got["dvtrim"] == float(np.asarray(want["dvtrim"]))
        np.testing.assert_array_equal(got["desintxb"], np.asarray(want["desintxb"]))


def test_force_clt_spreads_replicate_means_like_means_of_n_observations() -> None:
    x = _series(202, 30)
    result = nns.nns_meboot(x, reps=200, rho=0.0, random_seed=7)
    assert isinstance(result, dict)
    replicate_means = np.mean(np.asarray(result["replicates"]), axis=0)
    # sd(x) / sqrt(n) with n = 30, whatever the number of replicates.
    assert np.std(replicate_means, ddof=1) == pytest.approx(
        np.std(x, ddof=1) / np.sqrt(30.0), rel=1e-3
    )


@pytest.mark.parametrize(
    ("lower", "upper", "by", "exp", "extra_r", "extra_py"),
    [
        (-1.0, 1.0, 0.5, 1.0, "", {}),
        (-0.5, 0.8, 0.1, 0.5, "", {}),
        (0.2, 0.9, 0.35, 3.0, ", target_drift = 0.02", {"target_drift": 0.02}),
    ],
)
def test_nns_mc_matches_r_under_set_seed(
    lower: float, upper: float, by: float, exp: float, extra_r: str, extra_py: dict[str, Any]
) -> None:
    x = _series(203, 24)
    expected = nns_frame_call(
        "boot14.mc",
        f"set.seed(9); r <- NNS.MC(x, reps = 3, lower_rho = {lower!r}, upper_rho = {upper!r}, "
        f"by = {by!r}, exp = {exp!r}{extra_r}); list(ensemble = r$ensemble, "
        "names = names(r$replicates), reps = lapply(unname(r$replicates), unname))",
        {"x": _num(x)},
        exact=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actual = nns.nns_mc(
            x, reps=3, lower_rho=lower, upper_rho=upper, by=by, exp=exp, random_seed=9, **extra_py
        )
    np.testing.assert_array_equal(actual["ensemble"], np.asarray(expected["ensemble"]))
    replicates = actual["replicates"]
    assert isinstance(replicates, dict)
    assert list(replicates) == list(expected["names"])
    for got, want in zip(replicates.values(), expected["reps"], strict=True):
        np.testing.assert_array_equal(np.asarray(got), _rows(want))


def test_rho_labels_follow_r_as_character() -> None:
    values = [1e-04, 5e-04, 0.0001234, 0.1, 1 / 3, 1e5, 123456.0, -0.99, 0.3, 1e-06, 0.001,
              2.5e-05, 0.12345678901234567, -0.6000000000000001, 0.0, -1.0]
    expected = nns_frame_call("boot14.labels", "as.character(v)", {"v": _num(values)})
    assert [_format_r_number(v) for v in values] == list(expected)


def test_nns_mc_validates_its_arguments_as_r_does() -> None:
    x = _series(204, 20)
    with pytest.raises(ValueError, match="exp must be a single finite number > 0"):
        nns.nns_mc(x, exp=0.0)
    with pytest.raises(ValueError, match="lower_rho and upper_rho must satisfy"):
        nns.nns_mc(x, lower_rho=-1.5)
    with pytest.raises(ValueError, match="by must be a single finite number > 0"):
        nns.nns_mc(x, by=-0.1)


@pytest.mark.parametrize(("kind", "rho"), [("shift", 1.0), ("ties", 0.0), ("constant", -0.5)])
def test_nns_ss_bootstrap_bounds_match_r_under_set_seed(kind: str, rho: float) -> None:
    rng = np.random.default_rng(205)
    if kind == "shift":
        x, y = rng.normal(size=25) + 0.3, rng.normal(size=30)
    elif kind == "ties":
        x, y = np.round(rng.normal(size=25) * 2), np.round(rng.normal(size=30) * 2)
    else:
        x, y = np.full(25, 1.5), rng.normal(size=30)
    expected = nns_frame_call(
        "boot14.ss",
        "set.seed(11); NNS.SS(x, y, confidence.interval = TRUE, reps = 20, ci = 0.9, "
        f"rho = {rho!r})",
        {"x": _num(x), "y": _num(y)},
        exact=True,
    )
    actual = nns.nns_ss(x, y, confidence_interval=True, reps=20, ci=0.9, rho=rho, random_seed=11)
    for key in ("p_gt", "p_tie", "p_star", "lower", "upper"):
        assert float(np.asarray(actual[key])) == float(np.asarray(expected[key])), key
    np.testing.assert_array_equal(actual["boot_vals"], np.asarray(expected["boot_vals"]))


@pytest.mark.parametrize(
    ("h", "seasonal", "method"),
    [(4, "TRUE", "nonlin"), (5, "12", "lin"), (3, "c(12, 6)", "both"), (6, "FALSE", "means")],
)
def test_nns_arma_prediction_intervals_match_r_under_set_seed(
    h: int, seasonal: str, method: str
) -> None:
    rng = np.random.default_rng(206)
    t = np.arange(72, dtype=np.float64)
    x = np.round(10.0 + 3.0 * np.sin(2.0 * np.pi * t / 12.0) + rng.normal(size=72), 1)
    expected = nns_frame_call(
        "boot14.arma",
        f"set.seed(13); r <- NNS.ARMA(x, h = {h}, seasonal.factor = {seasonal}, "
        f"method = '{method}', plot = FALSE, seasonal.plot = FALSE, pred.int = 0.9); "
        "lapply(as.list(r), as.numeric)",
        {"x": _num(x)},
        exact=True,
    )
    seasonal_py: Any = {"TRUE": True, "FALSE": False, "12": 12, "c(12, 6)": [12, 6]}[seasonal]
    actual = nns.nns_arma(
        x, h=h, seasonal_factor=seasonal_py, method=method, pred_int=0.9, random_seed=13
    )
    assert isinstance(actual, dict)
    assert list(actual) == list(expected)
    for key, values in expected.items():
        np.testing.assert_array_equal(actual[key], np.asarray(values, dtype=np.float64), key)


@pytest.mark.parametrize("kind", ["seasonal", "ties"])
def test_nns_seas_scores_match_r_bitwise(kind: str) -> None:
    rng = np.random.default_rng(207)
    t = np.arange(120, dtype=np.float64)
    x = (20.0 + 0.3 * t) * (1.0 + 0.2 * np.sin(2.0 * np.pi * t / 12.0)) + rng.normal(size=120)
    if kind == "ties":
        x = np.round(x)
    expected = nns_frame_call(
        "boot14.seas",
        "a <- NNS.seas(x, plot = FALSE)$all.periods; "
        "list(p = a$Period, cv = a$Coefficient.of.Variation, "
        "vcv = a$Variable.Coefficient.of.Variation)",
        {"x": _num(x)},
        exact=True,
    )
    actual = nns.nns_seas(x)["all.periods"]
    assert isinstance(actual, dict)
    np.testing.assert_array_equal(actual["Period"], np.asarray(expected["p"], dtype=np.int64))
    np.testing.assert_array_equal(actual["Coefficient.of.Variation"], np.asarray(expected["cv"]))
    np.testing.assert_array_equal(
        actual["Variable.Coefficient.of.Variation"], np.asarray(expected["vcv"])
    )
