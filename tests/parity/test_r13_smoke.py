from __future__ import annotations

import numpy as np
import pytest
from _r import nns, nns_stack_numeric
from _tolerances import COMPOUND, EXACT

from nns import lpm, nns_arma, nns_copula, nns_reg, nns_stack, pm_matrix, upm


@pytest.mark.parity
def test_r_nns_13_partial_moment_smoke_values() -> None:
    x = np.array([-2.0, -1.0, 0.5, 3.0])

    np.testing.assert_allclose(lpm(2, 0, x), 1.25, atol=EXACT)
    np.testing.assert_allclose(upm(2, 0, x), 2.3125, atol=EXACT)


@pytest.mark.parity
def test_r_nns_13_regression_points_smoke_value() -> None:
    result = nns_reg(
        np.array([1.0, 2.0]),
        np.array([148.0, 135.0]),
        return_values=False,
        plot=False,
        multivariate_call=True,
    )

    # R 13.3 + patches (the x centroid of a cell is no longer
    # rounded for integer-valued x, so the centre point stays at x = 1.5).
    np.testing.assert_allclose(result["x"], np.array([1.0, 1.5, 2.0]), atol=EXACT)
    np.testing.assert_allclose(result["y"], np.array([148.0, 141.5, 135.0]), atol=EXACT)


@pytest.mark.parity
def test_r_nns_13_copula_smoke_values() -> None:
    a = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    b = np.array([1, 2, 1, 4, 3], dtype=np.float64)
    cc = np.array([2, 1, 3, 5, 4], dtype=np.float64)
    a2 = np.column_stack((a, b))
    a3 = np.column_stack((a, b, cc))

    np.testing.assert_allclose(nns_copula(a2, continuous=True), 1.0, atol=EXACT)
    np.testing.assert_allclose(nns_copula(a2, continuous=False), 1.0, atol=EXACT)
    # R 13.3 + patches ("NNS.copula: dimension-generic perfect-dependence
    # early return"): every pair of a3 is fully concordant about the column means
    # (pair sum 3 = choose(3, 2)), so the copula is 1; R 13.3 compared the sum with
    # 1 and gave 0.9660918 / 0.9309493.
    np.testing.assert_allclose(nns_copula(a3, continuous=True), 1.0, atol=EXACT)
    np.testing.assert_allclose(nns_copula(a3, continuous=False), 1.0, atol=EXACT)


@pytest.mark.parity
def test_r_nns_13_pm_matrix_names_smoke() -> None:
    variable = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 1.0], [4.0, 4.0], [5.0, 3.0]])

    result = pm_matrix(1, 1, None, variable, True, names=["a", "b"])

    assert result["names"] == ["a", "b"]
    assert set(result) == {"cupm", "dupm", "dlpm", "clpm", "cov.matrix", "names"}
    for key in ("cupm", "dupm", "dlpm", "clpm", "cov.matrix"):
        assert result[key].shape == (2, 2)


@pytest.mark.parity
def test_r_nns_14_arma_airline_smoke_values() -> None:
    series = np.array(
        [
            112,
            118,
            132,
            129,
            121,
            135,
            148,
            148,
            136,
            119,
            104,
            118,
            115,
            126,
            141,
            135,
            125,
            149,
            170,
            170,
            158,
            133,
            114,
            140,
        ],
        dtype=np.float64,
    )

    seasonal = nns_arma(series, h=6, seasonal_factor=12, method="lin")
    nonseasonal = nns_arma(series, h=4, seasonal_factor=False, method="nonlin")

    np.testing.assert_allclose(seasonal, np.array([118, 134, 150, 141, 129, 163]), atol=EXACT)
    reference = nns(
        "NNS.ARMA",
        series.tolist(),
        4,
        None,
        False,
        None,
        1,
        None,
        True,
        False,
        "nonlin",
        False,
        False,
        False,
        False,
        None,
    )
    np.testing.assert_allclose(nonseasonal, reference, atol=COMPOUND)


@pytest.mark.parity
@pytest.mark.stochastic
def test_r_nns_14_seeded_stack_smoke_sample() -> None:
    x0 = np.linspace(0.0, 1.0, 12)
    x = np.column_stack((x0, np.sin(x0)))
    y = 1.0 + 2.0 * x[:, 0] - x[:, 1]

    result = nns_stack(
        x,
        y,
        x[:3],
        cv_size=0.25,
        folds=2,
        method=[1, 2],
        stack=True,
        random_seed=123,
    )

    expected = nns_stack_numeric(
        x.tolist(),
        y.tolist(),
        x[:3].tolist(),
        cv_size=0.25,
        folds=2,
        method=[1, 2],
        order=None,
        stack=True,
        dim_red_method="cor",
        seed=123,
    )
    assert isinstance(expected, dict)
    for key in ("stack", "reg", "dim.red", "NNS.reg.n.best", "NNS.dim.red.threshold"):
        np.testing.assert_allclose(result[key], expected[key], atol=COMPOUND)
