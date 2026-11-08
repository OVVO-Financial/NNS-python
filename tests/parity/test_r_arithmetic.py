"""Bit-for-bit agreement with the R arithmetic NNS results depend on.

NNS feeds several intermediate quantities into exact comparisons: the
dimension-reduction index is cut by ``findInterval`` at regression points built
from the same index, class probabilities are compared for ties, VAR rounds its
correlation cutoffs to two digits and Stack rounds a random holdout share to
three. A last-bit difference in any of them can therefore change a fitted
partition, a gradient or a selected model, so these checks compare exact
doubles (``nns_frame_call(..., exact=True)`` stores 17 significant digits).

The R values come from the pinned oracle. Matrix products follow R's bundled
reference BLAS, which is what R uses unless it is relinked to another BLAS.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from _r import nns_frame_call, r_column

import nns
from nns._dep_r14 import dep_pair_r14
from nns._rblas import r_crossprod_vec, r_matvec
from nns._reg_engine import _dimred_coefficients
from nns._rmean import (
    r_col_means,
    r_cor,
    r_cor_matrix,
    r_mean,
    r_quantile,
    r_rank_average,
    r_round,
    r_round_array,
    r_row_means,
    r_seq_by,
    r_sum,
)
from nns._rnmath import dhyper, pnorm

pytestmark = pytest.mark.parity


def _num(values: Any) -> dict[str, Any]:
    return r_column([float(v) for v in np.asarray(values, dtype=np.float64).ravel()], "numeric")


def _exact(label: str, expr: str, **data: Any) -> Any:
    return nns_frame_call(label, expr, {k: _num(v) for k, v in data.items()}, exact=True)


def _bits(actual: Any, expected: Any) -> None:
    actual_arr = np.asarray(actual, dtype=np.float64)
    expected_arr = np.asarray(expected, dtype=np.float64).reshape(actual_arr.shape)
    np.testing.assert_array_equal(actual_arr, expected_arr)


def _hyper_grid() -> np.ndarray:
    rng = np.random.default_rng(20261108)
    rows = []
    for m in [*range(2, 41), 57, 100, 233, 400]:
        for _ in range(6):
            a, b = int(rng.integers(0, m + 1)), int(rng.integers(0, m + 1))
            lo, hi = max(0, a + b - m), min(a, b)
            for k in sorted({lo, (lo + hi) // 2, hi}):
                rows.append((k, a, m - a, b))
    return np.asarray(sorted(set(rows)), dtype=np.float64)


def test_dhyper_matches_r_bitwise() -> None:
    grid = _hyper_grid()
    expected = _exact(
        "rarith.dhyper",
        "dhyper(k, r, b, n)",
        k=grid[:, 0],
        r=grid[:, 1],
        b=grid[:, 2],
        n=grid[:, 3],
    )
    _bits([dhyper(*row) for row in grid], expected)


def test_pnorm_matches_r_bitwise() -> None:
    rng = np.random.default_rng(7)
    z = np.concatenate(
        [
            rng.normal(size=400) * 3,
            rng.uniform(-40.0, 10.0, size=200),
            [0.0, 0.67448975, -0.67448975, 5.656854249492381, -5.656854249492381],
            [8.2924, -38.4674, 1e-17, -1e-17, 37.0, -37.9],
        ]
    )
    expected = _exact("rarith.pnorm", "pnorm(z)", z=z)
    _bits([pnorm(float(v)) for v in z], expected)


@pytest.mark.parametrize("digits", [1, 2, 3, 6, 12, -1])
def test_round_matches_r_bitwise(digits: int) -> None:
    rng = np.random.default_rng(digits + 100)
    x = np.concatenate(
        [
            rng.uniform(-2.0, 2.0, 600),
            np.round(rng.uniform(-2.0, 2.0, 300), 3) + 0.005,
            rng.uniform(0.20, 1.0 / 3.0, 300),
            np.arange(-150, 150) / 1000 + 0.0005,
        ]
    )
    expected = _exact(f"rarith.round{digits}", f"round(x, {digits})", x=x)
    _bits(r_round_array(x, digits), expected)
    _bits([r_round(float(v), digits) for v in x], expected)


@pytest.mark.parametrize(
    ("start", "stop", "by"),
    [(0.01, 0.99, 0.01), (0.0, 1.0, 0.01), (0.05, 0.95, 0.05), (-1.0, 2.5, 0.1)],
)
def test_seq_by_matches_r_bitwise(start: float, stop: float, by: float) -> None:
    expected = _exact("rarith.seq", f"seq({start!r}, {stop!r}, by = {by!r})")
    _bits(r_seq_by(start, stop, by), expected)


@pytest.mark.parametrize("type_", [7, 8])
def test_quantile_matches_r_bitwise(type_: int) -> None:
    rng = np.random.default_rng(type_)
    probs = [0.0, 0.01, 0.025, 0.05, 0.1, 0.25, 0.333, 0.5, 0.77, 0.95, 0.975, 1.0]
    for n in (1, 2, 3, 7, 50, 401):
        x = rng.normal(size=n) * 10 ** rng.uniform(-3, 3)
        expected = _exact(
            f"rarith.quantile{type_}",
            f"quantile(x, p, type = {type_}, names = FALSE)",
            x=x,
            p=probs,
        )
        _bits(r_quantile(x, probs, type_), expected)


def test_summaries_match_r_bitwise() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(257, 5)) * 1e3 + 1e6
    expected = _exact(
        "rarith.summaries",
        "X <- matrix(v, ncol = 5); c(mean(X[, 1]), sum(X[, 2]), colMeans(X), rowMeans(X))",
        v=x.ravel(order="F"),
    )
    actual = np.concatenate(
        [[r_mean(x[:, 0]), r_sum(x[:, 1])], r_col_means(x), r_row_means(x)]
    )
    _bits(actual, expected)


def test_correlations_match_r_bitwise() -> None:
    rng = np.random.default_rng(12)
    x = rng.normal(size=(83, 4))
    x[:, 1] = x[:, 0] ** 2 + rng.normal(size=83) * 0.1
    x[:, 3] = np.round(x[:, 3])
    expected = _exact(
        "rarith.cor",
        "X <- matrix(v, ncol = 4); c(cor(X[, 1], X[, 2]), "
        "cor(X[, 3], X[, 4], method = 'spearman'), as.numeric(cor(X)), "
        "as.numeric(cor(X, method = 'spearman')))",
        v=x.ravel(order="F"),
    )
    ranks = np.column_stack([r_rank_average(x[:, j]) for j in range(4)])
    actual = np.concatenate(
        [
            [r_cor(x[:, 0], x[:, 1])],
            [r_cor(r_rank_average(x[:, 2]), r_rank_average(x[:, 3]))],
            r_cor_matrix(x).ravel(order="F"),
            r_cor_matrix(ranks).ravel(order="F"),
        ]
    )
    _bits(actual, expected)


def test_reference_blas_products_match_r_bitwise() -> None:
    rng = np.random.default_rng(13)
    design = rng.uniform(size=(61, 4))
    coef = np.array([0.31, -1.7, 0.0, 2.2])
    weights = rng.uniform(size=9)
    shares = rng.uniform(size=(9, 3))
    expected = _exact(
        "rarith.blas",
        "D <- matrix(d, ncol = 4); S <- matrix(s, ncol = 3); "
        "c(as.numeric(D %*% cf), as.numeric(crossprod(w, S)))",
        d=design.ravel(order="F"),
        cf=coef,
        w=weights,
        s=shares.ravel(order="F"),
    )
    _bits(np.concatenate([r_matvec(design, coef), r_crossprod_vec(weights, shares)]), expected)


def _dependence_pairs() -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(14)
    pairs = []
    for i, n in enumerate([8, 25, 60, 60, 150, 401, 900]):
        x = rng.normal(size=n)
        if i % 3 == 0:
            y = x**2 + rng.normal(size=n) * 0.2
        elif i % 3 == 1:
            x = np.round(x)
            y = np.round(rng.normal(size=n))
        else:
            y = (x > 0).astype(float) + rng.normal(size=n) * 0.3
        pairs.append((x, y))
    return pairs


@pytest.mark.parametrize("asym", [False, True])
def test_nns_dep_matches_r_bitwise(asym: bool) -> None:
    flag = "TRUE" if asym else "FALSE"
    for x, y in _dependence_pairs():
        expected = _exact("rarith.dep", f"unname(unlist(NNS.dep(x, y, asym = {flag})))", x=x, y=y)
        public = nns.nns_dep(x, y, asym=asym)
        _bits([public["Correlation"], public["Dependence"]], expected)
        # The independent pure-Python fallback is held to the same standard.
        _bits(dep_pair_r14(x, y, asym), expected)


@pytest.mark.parametrize("method", ["cor", "NNS.dep", "NNS.caus", "all"])
def test_dimension_reduction_weights_match_r_bitwise(method: str) -> None:
    rng = np.random.default_rng(15)
    for n, ties in ((40, False), (150, True)):
        x = rng.normal(size=(n, 3))
        if ties:
            x = np.round(x)
        y = x[:, 0] + x[:, 1] ** 2 + rng.normal(size=n) * 0.3
        expected = _exact(
            "rarith.dimred",
            f"NNS:::.nns_reg_dimred_coefficients(matrix(v, ncol = 3), y, '{method}', NULL, 0)",
            v=x.ravel(order="F"),
            y=y,
        )
        _bits(_dimred_coefficients(x, y, method, None, 0), expected)


def test_dimension_reduction_fit_ids_match_r_on_boundary_rows() -> None:
    """An index value equal to a regression point keeps R's cell and gradient.

    Before the reference-order index and R's dependence weights, Python's index
    differed from R's in the last bit, so rows sitting on regression points fell
    into the neighbouring cell (different ``NNS.ID`` and ``gradient``).
    """
    rng = np.random.default_rng(12)
    x = rng.normal(size=(150, 3))
    x[:, 1] = (x[:, 1] > 0).astype(float)
    y = x[:, 0] + 2 * x[:, 1] + rng.normal(size=150) * 0.2
    expected = _exact(
        "rarith.dimred.fit",
        "f <- NNS.reg(matrix(v, ncol = 3), y, dim.red.method = 'NNS.dep', threshold = 0.1, "
        "plot = FALSE)$Fitted.xy; c(f$x, f$NNS.ID, f$gradient, f$y.hat)",
        v=x.ravel(order="F"),
        y=y,
    )
    fit = nns.nns_reg(x, y, dim_red_method="NNS.dep", threshold=0.1)["Fitted.xy"]
    actual = np.concatenate([fit["x"], fit["NNS.ID"], fit["gradient"], fit["y.hat"]])
    _bits(actual, expected)


@pytest.mark.parametrize("linear", [False, True])
def test_nns_norm_matches_r_bitwise(linear: bool) -> None:
    rng = np.random.default_rng(16)
    x = np.abs(rng.normal(size=(30, 3))) * np.array([1.0, 25.0, 400.0]) + 0.5
    flag = "TRUE" if linear else "FALSE"
    expected = _exact(
        "rarith.norm",
        f"as.numeric(NNS.norm(matrix(v, ncol = 3), linear = {flag}))",
        v=x.ravel(order="F"),
    )
    _bits(nns.nns_norm(x, linear=linear).ravel(order="F"), expected)


def test_nns_norm_constant_column_is_na_as_in_r() -> None:
    """A constant column has zero spread for R's cor(), so NNS.norm is all NA.

    NumPy's one-pass mean of a repeated 0.1 is one ulp off, which made the
    former np.corrcoef path report a tiny spread and finite output.
    """
    x = np.column_stack([np.full(8, 0.1), np.linspace(1.0, 2.0, 8)])
    expected = _exact(
        "rarith.norm.constant",
        "as.numeric(suppressWarnings(NNS.norm(matrix(v, ncol = 2))))",
        v=x.ravel(order="F"),
    )
    # An all-NA numeric vector decodes as the strings "NA".
    assert all(value in ("NA", None) or np.isnan(value) for value in expected)
    assert np.all(np.isnan(nns.nns_norm(x)))
