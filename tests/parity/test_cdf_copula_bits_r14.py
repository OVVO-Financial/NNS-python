"""R 14.0 parity, to the last bit, for NNS.CDF, NNS.copula, NNS.rescale and the
many-target partial moments.

* Beyond 32 targets R's ``LPM``/``UPM`` (and so ``LPM.ratio`` and ``NNS.CDF``)
  switch to a sorted-anchor prefix backend for integer degrees; the NumPy
  fallback reproduces its recurrence.
* ``NNS.CDF`` takes ``-log`` with libm, the joint survival with R's long-double
  ``rowSums``, and the joint CDF with R's batch kernel.
* ``NNS.copula`` sums with R's long-double ``sum``/``prod``/``mean`` and returns
  ``mean(...)^(1/2)``, which is ``pow``, not ``sqrt``.
* ``NNS.rescale(method = "riskneutral")`` takes the mean in one long-double pass.

Each case runs with the native kernels and with the NumPy fallbacks.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest
from _r import nns_frame_call, r_column

import nns

pytestmark = pytest.mark.parity


def _num(values: Any) -> dict[str, Any]:
    return r_column([float(v) for v in np.asarray(values, dtype=np.float64).ravel()], "numeric")


def _disable_native(monkeypatch: pytest.MonkeyPatch) -> None:
    for module in ("nns.core", "nns.pm_matrix", "nns.cdf"):
        monkeypatch.setattr(importlib.import_module(module), "native_fn", lambda name: None)


@pytest.mark.parametrize("native", [True, False])
@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_many_target_partial_moments_match_r_bitwise(
    monkeypatch: pytest.MonkeyPatch, native: bool, degree: int
) -> None:
    rng = np.random.default_rng(501)
    x = np.round(rng.normal(size=160) * 3.0, 1)
    targets = np.concatenate([np.sort(x), [x.min() - 1.0, x.max() + 1.0]])
    expected = nns_frame_call(
        "cdf14.prefix",
        f"list(lpm = LPM({degree}, t, x), upm = UPM({degree}, t, x), "
        f"ratio = LPM.ratio({degree}, t, x))",
        {"x": _num(x), "t": _num(targets)},
        exact=True,
    )
    if not native:
        _disable_native(monkeypatch)
    np.testing.assert_array_equal(nns.lpm(degree, targets, x), expected["lpm"])
    np.testing.assert_array_equal(nns.upm(degree, targets, x), expected["upm"])
    np.testing.assert_array_equal(nns.lpm_ratio(degree, targets, x), expected["ratio"])


@pytest.mark.parametrize("native", [True, False])
@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("kind", ["CDF", "survival", "cumulative hazard"])
def test_univariate_cdf_matches_r_bitwise(
    monkeypatch: pytest.MonkeyPatch, native: bool, degree: int, kind: str
) -> None:
    rng = np.random.default_rng(502)
    x = rng.exponential(size=150)
    target = float(np.median(x))
    expected = nns_frame_call(
        "cdf14.univariate",
        f"r <- NNS.CDF(x, degree = {degree}, target = tg, type = '{kind}', plot = FALSE); "
        "list(f = unname(as.list(r$Function)), tv = r$target.value)",
        {"x": _num(x), "tg": _num([target])},
        exact=True,
    )
    if not native:
        _disable_native(monkeypatch)
    actual = nns.nns_cdf(x, degree=degree, target=target, type=kind)
    columns = list(actual["Function"].values())
    for column, want in zip(columns, expected["f"], strict=True):
        np.testing.assert_array_equal(column, np.asarray(want, dtype=np.float64))
    np.testing.assert_array_equal(
        np.atleast_1d(actual["target.value"]), np.atleast_1d(expected["tv"])
    )


@pytest.mark.parametrize("native", [True, False])
@pytest.mark.parametrize(
    ("columns", "degree", "kind"),
    [(3, 0, "survival"), (3, 0, "cumulative hazard"), (2, 1, "CDF"), (3, 2, "survival")],
)
def test_multivariate_cdf_matches_r_bitwise(
    monkeypatch: pytest.MonkeyPatch, native: bool, columns: int, degree: int, kind: str
) -> None:
    rng = np.random.default_rng(503)
    data = rng.normal(size=(60, columns))
    target = np.median(data, axis=0)
    expected = nns_frame_call(
        "cdf14.multivariate",
        f"r <- NNS.CDF(matrix(v, ncol = {columns}), degree = {degree}, target = tg, "
        f"type = '{kind}', plot = FALSE); list(cdf = r$Function$CDF, tv = r$target.value)",
        {"v": _num(data.ravel(order="F")), "tg": _num(target)},
        exact=True,
    )
    if not native:
        _disable_native(monkeypatch)
    actual = nns.nns_cdf(data, degree=degree, target=target, type=kind)
    np.testing.assert_array_equal(actual["Function"]["CDF"], np.asarray(expected["cdf"]))
    np.testing.assert_array_equal(
        np.atleast_1d(actual["target.value"]), np.atleast_1d(expected["tv"])
    )


@pytest.mark.parametrize("native", [True, False])
@pytest.mark.parametrize("continuous", [True, False])
@pytest.mark.parametrize("columns", [2, 3, 4])
def test_nns_copula_matches_r_bitwise(
    monkeypatch: pytest.MonkeyPatch, native: bool, continuous: bool, columns: int
) -> None:
    rng = np.random.default_rng(504)
    base = rng.normal(size=300)
    data = np.column_stack(
        [base * (j + 1) * 0.4 + rng.normal(size=300) for j in range(columns)]
    )
    flag = "TRUE" if continuous else "FALSE"
    expected = nns_frame_call(
        "cdf14.copula",
        f"NNS.copula(matrix(v, ncol = {columns}), continuous = {flag})",
        {"v": _num(data.ravel(order="F"))},
        exact=True,
    )
    if not native:
        _disable_native(monkeypatch)
    assert nns.nns_copula(data, continuous=continuous) == expected


@pytest.mark.parametrize("kind", ["Terminal", "Discounted"])
def test_risk_neutral_rescale_matches_r_bitwise(kind: str) -> None:
    rng = np.random.default_rng(505)
    x = np.concatenate([rng.normal(-40.0, 5.0, 30), rng.normal(10.0, 2.0, 50), [1e4]])
    expected = nns_frame_call(
        "cdf14.rescale",
        f"NNS.rescale(x, 100, 0.05, method = 'riskneutral', T = 2, type = '{kind}')",
        {"x": _num(x)},
        exact=True,
    )
    actual = nns.nns_rescale(x, 100, 0.05, method="riskneutral", time_to_maturity=2, type=kind)
    np.testing.assert_array_equal(actual, np.asarray(expected, dtype=np.float64))


def test_minmax_rescale_warns_on_identical_values_like_r() -> None:
    with pytest.warns(UserWarning, match="All x identical: returning midpoint values"):
        out = nns.nns_rescale(np.full(4, 2.0), 0.0, 1.0)
    np.testing.assert_array_equal(out, np.full(4, 0.5))
