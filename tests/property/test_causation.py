from __future__ import annotations

import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from nns import nns_causation

finite_arrays = arrays(
    dtype=np.float64,
    shape=st.integers(min_value=8, max_value=100),
    elements=st.floats(
        min_value=-1e6,
        max_value=1e6,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,  # see tests/property/test_regression.py
        width=64,
    ),
)


@given(finite_arrays, finite_arrays)
def test_nns_causation_bounds_hold_for_random_pairs(
    x: np.ndarray,
    y: np.ndarray,
) -> None:
    size = min(x.size, y.size)
    x = x[:size]
    y = y[:size]
    # NNS.caus normalises by the column means (NNS.norm), which overflows for
    # ranges like 1e-249; R stops with "variable must be finite" there too. That
    # scale dependence is a design question for R, so keep
    # the property to ranges where the normalisation is representable.
    assume(np.ptp(x) > 1e-100)
    assume(np.ptp(y) > 1e-100)

    result = nns_causation(x, y)
    directional = list(result.values())[:2]
    net = next(value for key, value in result.items() if key.startswith("C("))

    assert all(0.0 <= value <= 1.0 for value in directional)
    assert abs(net) <= 100.0
