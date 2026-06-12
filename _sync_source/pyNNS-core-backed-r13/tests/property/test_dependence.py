from __future__ import annotations

import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pynns import nns_dep

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


@given(finite_arrays, finite_arrays, st.booleans())
def test_nns_dep_bounds_hold_for_random_pairs(
    x: np.ndarray,
    y: np.ndarray,
    asym: bool,
) -> None:
    size = min(x.size, y.size)
    x = x[:size]
    y = y[:size]
    assume(np.unique(x).size > 1)
    assume(np.unique(y).size > 1)

    result = nns_dep(x, y, asym=asym)

    assert result["Dependence"] >= -1e-12
    assert result["Dependence"] <= 1.0 + 1e-12
