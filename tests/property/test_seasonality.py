from __future__ import annotations

from typing import Any, cast

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pynns import nns_seas

finite_arrays = arrays(
    dtype=np.float64,
    shape=st.integers(min_value=5, max_value=200),
    elements=st.floats(
        min_value=-1e4,
        max_value=1e4,
        allow_nan=False,
        allow_infinity=False,
        width=64,
    ),
)


@given(finite_arrays)
def test_nns_seas_random_arrays_return_valid_shape(values: np.ndarray) -> None:
    result = nns_seas(values)

    _assert_valid_result(result, values.size)


@given(st.integers(min_value=5, max_value=200), st.floats(min_value=-100.0, max_value=100.0))
def test_nns_seas_constant_arrays_return_valid_shape(size: int, value: float) -> None:
    result = nns_seas(np.full(size, value, dtype=np.float64))

    _assert_valid_result(result, size)


@given(st.integers(min_value=6, max_value=100))
def test_nns_seas_zero_mean_arrays_return_valid_shape(size: int) -> None:
    base = np.tile(np.array([-1.0, 1.0]), size // 2 + 1)[:size]

    result = nns_seas(base)

    _assert_valid_result(result, size)


def _assert_valid_result(result: dict[str, object], size: int) -> None:
    periods = cast(np.ndarray, result["periods"])
    table = cast(dict[str, Any], result["all.periods"])
    assert result["best.period"] == int(periods[0])
    assert table["Period"].shape == periods.shape
    assert np.all(periods >= 0)
    assert np.all(periods < size / 2.0)
    cv = table["Coefficient.of.Variation"]
    assert np.all(np.isfinite(cv) | np.isinf(cv))
