from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pynns import pm_matrix


@given(
    arrays(
        dtype=np.float64,
        shape=st.tuples(
            st.integers(min_value=3, max_value=30),
            st.integers(min_value=2, max_value=6),
        ),
        elements=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
)
def test_pm_matrix_reconstruction_and_psd_properties(variable: np.ndarray) -> None:
    result = pm_matrix(2, 2, "mean", variable, pop_adj=True)

    np.testing.assert_allclose(
        result["clpm"] + result["cupm"] - result["dlpm"] - result["dupm"],
        result["cov.matrix"],
        atol=0.0,
    )
    assert np.linalg.eigvalsh(result["clpm"]).min() > -1e-10
    assert np.linalg.eigvalsh(result["cupm"]).min() > -1e-10
