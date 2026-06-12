from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pynns import ecdf_pm, kurt_pm, mean_pm, skew_pm, var_pm


@given(
    arrays(
        dtype=np.float64,
        shape=st.integers(min_value=3, max_value=50),
        elements=st.floats(
            min_value=-100.0,
            max_value=100.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
)
def test_classical_pm_matches_numpy_and_scipy(x: np.ndarray) -> None:
    from scipy import stats  # type: ignore[import-untyped]

    assume(np.var(x) > 1e-24)

    assert mean_pm(x) == pytest.approx(np.mean(x), abs=1e-12)
    assert var_pm(x) == pytest.approx(np.var(x), abs=2e-12)
    assert var_pm(x, ddof=1) == pytest.approx(np.var(x, ddof=1), abs=2e-12)
    assert skew_pm(x) == pytest.approx(stats.skew(x, bias=True), abs=1e-10)
    assert kurt_pm(x) == pytest.approx(stats.kurtosis(x, fisher=True, bias=True), abs=1e-10)
    assert kurt_pm(x, excess=False) == pytest.approx(
        stats.kurtosis(x, fisher=False, bias=True),
        abs=1e-10,
    )

    sorted_x = np.sort(x)
    np.testing.assert_allclose(
        ecdf_pm(x),
        np.searchsorted(sorted_x, sorted_x, side="right") / x.size,
    )
    np.testing.assert_allclose(
        ecdf_pm(x, sorted_x),
        np.searchsorted(sorted_x, sorted_x, side="right") / x.size,
    )
