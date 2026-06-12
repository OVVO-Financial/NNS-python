from __future__ import annotations

import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pynns import nns_copula

finite_arrays = arrays(
    dtype=np.float64,
    shape=st.integers(min_value=8, max_value=100),
    elements=st.floats(
        min_value=-1e6,
        max_value=1e6,
        allow_nan=False,
        allow_infinity=False,
        width=64,
    ),
)


@given(finite_arrays, finite_arrays)
def test_nns_copula_bounds_hold_for_random_pairs(x: np.ndarray, y: np.ndarray) -> None:
    size = min(x.size, y.size)
    x = x[:size]
    y = y[:size]
    assume(np.ptp(x) > 0.0)
    assume(np.ptp(y) > 0.0)

    result = nns_copula(x, y)

    assert result >= 0.0
    assert result <= 1.0
