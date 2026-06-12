from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pynns import nns_anova

MIN_MEANINGFUL_RANGE = np.finfo(np.float64).tiny


@given(
    arrays(
        np.float64,
        20,
        elements=st.floats(
            min_value=-100.0,
            max_value=100.0,
            allow_nan=False,
            allow_infinity=False,
            width=64,
        ),
    ),
    arrays(
        np.float64,
        20,
        elements=st.floats(
            min_value=-100.0,
            max_value=100.0,
            allow_nan=False,
            allow_infinity=False,
            width=64,
        ),
    ),
)
def test_nns_anova_binary_certainty_bounds(x: np.ndarray, y: np.ndarray) -> None:
    assume(np.ptp(x) > MIN_MEANINGFUL_RANGE)
    assume(np.ptp(y) > MIN_MEANINGFUL_RANGE)
    result = nns_anova(x, y, confidence_interval=None)

    assert isinstance(result, dict)
    assert 0.0 <= result["Certainty"] <= 1.0
    assert 0.0 <= result["Control_CDF"] <= 1.0
    assert 0.0 <= result["Treatment_CDF"] <= 1.0


@given(
    arrays(
        np.float64,
        (20, 3),
        elements=st.floats(
            min_value=-100.0,
            max_value=100.0,
            allow_nan=False,
            allow_infinity=False,
            width=64,
        ),
    )
)
def test_nns_anova_pairwise_bounds(x: np.ndarray) -> None:
    assume(all(np.ptp(x[:, col]) > MIN_MEANINGFUL_RANGE for col in range(x.shape[1])))
    result = nns_anova(x, confidence_interval=None, pairwise=True)

    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 3)
    assert np.all((0.0 <= result) & (result <= 1.0))


@given(
    arrays(
        np.float64,
        12,
        elements=st.floats(
            min_value=-10.0,
            max_value=10.0,
            allow_nan=False,
            allow_infinity=False,
            width=64,
        ),
    ),
    arrays(
        np.float64,
        12,
        elements=st.floats(
            min_value=-10.0,
            max_value=10.0,
            allow_nan=False,
            allow_infinity=False,
            width=64,
        ),
    ),
)
@pytest.mark.stochastic
def test_nns_anova_robust_bounds(x: np.ndarray, y: np.ndarray) -> None:
    assume(np.ptp(x) > MIN_MEANINGFUL_RANGE)
    assume(np.ptp(y) > MIN_MEANINGFUL_RANGE)
    result = nns_anova(x, y, robust=True, confidence_interval=None, random_seed=123)

    assert isinstance(result, dict)
    assert 0.0 <= result["Robust Certainty Estimate"] <= 1.0
    assert 0.0 <= result["Lower Bound Robust Certainty"] <= 1.0
    assert 0.0 <= result["Upper Bound Robust Certainty"] <= 1.0
