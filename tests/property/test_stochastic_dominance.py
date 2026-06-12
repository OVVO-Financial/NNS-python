from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pynns import fsd, ssd, tsd

finite_arrays = arrays(
    dtype=np.float64,
    shape=st.integers(min_value=2, max_value=100),
    elements=st.floats(
        min_value=-1e6,
        max_value=1e6,
        allow_nan=False,
        allow_infinity=False,
        width=64,
    ),
)


@given(finite_arrays, finite_arrays)
def test_sd_antisymmetry_holds_for_random_pairs(x: np.ndarray, y: np.ndarray) -> None:
    size = min(x.size, y.size)
    x = x[:size]
    y = y[:size]

    assert fsd(x, y) == -fsd(y, x)
    assert ssd(x, y) == -ssd(y, x)
    assert tsd(x, y) == -tsd(y, x)
