from __future__ import annotations

import numpy as np
from _tolerances import COMPOUND
from hypothesis import given
from hypothesis import strategies as st

from pynns import co_lpm, co_upm, d_lpm, d_upm

finite_values = st.lists(
    st.floats(
        min_value=-1e6,
        max_value=1e6,
        allow_nan=False,
        allow_infinity=False,
        width=64,
    ),
    min_size=2,
    max_size=200,
)


@given(finite_values, finite_values)
def test_co_moments_are_non_negative_and_finite(
    x_values: list[float],
    y_values: list[float],
) -> None:
    size = min(len(x_values), len(y_values))
    x = np.asarray(x_values[:size], dtype=np.float64)
    y = np.asarray(y_values[:size], dtype=np.float64)
    target_x = float(x.mean())
    target_y = float(y.mean())

    results = [
        co_lpm(1, x, y, target_x, target_y),
        co_upm(1, x, y, target_x, target_y),
        d_lpm(1, 1, x, y, target_x, target_y),
        d_upm(1, 1, x, y, target_x, target_y),
    ]

    for result in results:
        assert np.isfinite(result)
        assert result >= 0


@given(finite_values, finite_values)
def test_covariance_decomposition_holds(x_values: list[float], y_values: list[float]) -> None:
    size = min(len(x_values), len(y_values))
    x = np.asarray(x_values[:size], dtype=np.float64)
    y = np.asarray(y_values[:size], dtype=np.float64)
    target_x = float(x.mean())
    target_y = float(y.mean())

    decomposition = (
        co_lpm(1, x, y, target_x, target_y)
        + co_upm(1, x, y, target_x, target_y)
        - d_lpm(1, 1, x, y, target_x, target_y)
        - d_upm(1, 1, x, y, target_x, target_y)
    )

    assert np.isclose(np.cov(x, y, ddof=0)[0, 1], decomposition, atol=COMPOUND)
