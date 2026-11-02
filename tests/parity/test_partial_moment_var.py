from __future__ import annotations

import numpy as np
import pytest
from _r import nns
from _tolerances import STOCHASTIC

from nns import lpm_ratio, lpm_var, upm_ratio, upm_var

PERCENTILES = [-0.2, 0.0, 0.01, 0.05, 0.25, 0.5, 0.9, 1.0, 1.4]


@pytest.mark.parity
@pytest.mark.parametrize("degree", [0.0, 1.0, 2.0, 3.0, 1.5, 0.5])
@pytest.mark.parametrize(("py_fn", "r_fn"), [(lpm_var, "LPM.VaR"), (upm_var, "UPM.VaR")])
def test_partial_moment_var_vector_percentile_matches_r(
    py_fn: object, r_fn: str, degree: float
) -> None:
    # R's LPM.VaR / UPM.VaR are vectorised over percentile (clamped to [0, 1]).
    rng = np.random.default_rng(42)
    values = rng.standard_t(4, size=400)

    expected = np.asarray(nns(r_fn, PERCENTILES, degree, values.tolist()), dtype=np.float64)
    actual = py_fn(np.asarray(PERCENTILES), degree, values)  # type: ignore[operator]

    assert isinstance(actual, np.ndarray)
    assert actual.shape == (len(PERCENTILES),)
    np.testing.assert_allclose(actual, expected, atol=STOCHASTIC)


@pytest.mark.parametrize("degree", [0.5, 1.5, 2.5])
def test_fractional_degree_var_inverts_the_ratio(degree: float) -> None:
    # Patched R: the fractional-degree VaR solves ratio(t) = p with uniroot at
    # tol = sqrt(eps) * range(x); 13.3's optimize() missed by up to 4e-4 in ratio.
    rng = np.random.default_rng(3)
    values = 0.02 * rng.standard_t(3, size=2000)
    levels = np.array([0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    lower = lpm_var(levels, degree, values)
    upper = upm_var(levels, degree, values)
    lower_ratio = np.array([lpm_ratio(degree, t, values) for t in lower]).reshape(-1)
    upper_ratio = np.array([upm_ratio(degree, t, values) for t in upper]).reshape(-1)
    np.testing.assert_allclose(lower_ratio, levels, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(upper_ratio, levels, rtol=0.0, atol=1e-6)
    # p = 0 / 1 are the ends of the support, not optimize()'s interior stopping point.
    assert lpm_var([0.0, 1.0], degree, values).tolist() == [values.min(), values.max()]
    assert upm_var([0.0, 1.0], degree, values).tolist() == [values.max(), values.min()]


@pytest.mark.parametrize("degree", [0.0, 1.0, 2.0, 4.0, 0.5])
def test_partial_moment_var_vector_equals_scalar_calls(degree: float) -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(size=300)
    for fn in (lpm_var, upm_var):
        vector = fn(PERCENTILES, degree, values)
        scalars = [fn(p, degree, values) for p in PERCENTILES]
        assert all(isinstance(v, float) for v in scalars)
        np.testing.assert_array_equal(vector, np.asarray(scalars))
    grid = np.array([[0.1, 0.2], [0.3, 0.4]])
    assert lpm_var(grid, degree, values).shape == (2, 2)
    assert lpm_var(np.array([]), degree, values).shape == (0,)
