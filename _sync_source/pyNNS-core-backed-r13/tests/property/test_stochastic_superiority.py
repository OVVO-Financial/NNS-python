from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pynns import nns_ss


@given(
    arrays(
        np.float64,
        st.integers(min_value=2, max_value=80),
        elements=st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False),
    ),
    arrays(
        np.float64,
        st.integers(min_value=2, max_value=80),
        elements=st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False),
    ),
)
def test_nns_ss_probability_invariants_hold(x: np.ndarray, y: np.ndarray) -> None:
    xy = nns_ss(x, y)
    yx = nns_ss(y, x)

    assert 0.0 <= xy["p_gt"] <= 1.0
    assert 0.0 <= xy["p_tie"] <= 1.0
    assert 0.0 <= xy["p_star"] <= 1.0
    assert xy["p_star"] == np.float64(xy["p_gt"] + 0.5 * xy["p_tie"])
    np.testing.assert_allclose(np.float64(xy["p_star"]) + np.float64(yx["p_star"]), 1.0)
