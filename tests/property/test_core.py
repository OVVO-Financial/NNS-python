from __future__ import annotations

import numpy as np
from _tolerances import COMPOUND
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pynns import lpm, upm

finite_arrays = arrays(
    dtype=np.float64,
    shape=st.integers(min_value=2, max_value=200),
    elements=st.floats(
        min_value=-1e6,
        max_value=1e6,
        allow_nan=False,
        allow_infinity=False,
        width=64,
    ),
)


@given(finite_arrays, st.sampled_from([1.0, 2.0]), st.data())
def test_partial_moments_are_non_negative_and_finite(
    x: np.ndarray,
    degree: float,
    data: st.DataObject,
) -> None:
    target = data.draw(st.floats(min_value=float(x.min()), max_value=float(x.max()), width=64))

    lower = lpm(degree, target, x)
    upper = upm(degree, target, x)

    assert np.isfinite(lower)
    assert np.isfinite(upper)
    assert lower >= 0
    assert upper >= 0


@given(finite_arrays)
def test_mean_equivalence_holds_for_finite_arrays(x: np.ndarray) -> None:
    assert np.isclose(x.mean(), upm(1, 0, x) - lpm(1, 0, x), atol=COMPOUND)
