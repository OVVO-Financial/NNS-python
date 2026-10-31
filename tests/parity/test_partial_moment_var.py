from __future__ import annotations

import numpy as np
import pytest
from _r import nns
from _tolerances import STOCHASTIC

from nns import lpm_var, upm_var

PERCENTILES = [-0.2, 0.0, 0.01, 0.05, 0.25, 0.5, 0.9, 1.0, 1.4]


@pytest.mark.parity
@pytest.mark.parametrize("degree", [0.0, 1.0, 2.0, 3.0, 1.5])
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
