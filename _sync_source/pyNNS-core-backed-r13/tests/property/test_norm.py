from __future__ import annotations

import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pynns import nns_norm

finite_matrices = arrays(
    dtype=np.float64,
    shape=st.tuples(st.integers(min_value=8, max_value=100), st.integers(min_value=2, max_value=8)),
    elements=st.floats(
        min_value=0.1,
        max_value=1e6,
        allow_nan=False,
        allow_infinity=False,
        width=64,
    ),
)


@given(finite_matrices, st.booleans())
def test_nns_norm_shape_and_finiteness_hold_for_random_matrices(
    x: np.ndarray,
    linear: bool,
) -> None:
    assume(np.all(np.std(x, axis=0) > 0.0))

    result = nns_norm(x, linear=linear)

    assert result.shape == x.shape
    assert np.all(np.isfinite(result))
