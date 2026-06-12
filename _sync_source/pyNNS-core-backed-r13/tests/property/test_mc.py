from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pynns import nns_mc

pytestmark = pytest.mark.stochastic

finite_arrays = arrays(
    dtype=np.float64,
    shape=st.integers(min_value=10, max_value=100),
    elements=st.floats(
        min_value=-50.0,
        max_value=50.0,
        allow_nan=False,
        allow_infinity=False,
        width=64,
    ),
)


def _is_valid_rho_target_input(x: np.ndarray) -> bool:
    if not np.all(np.isfinite(x)) or np.ptp(x) <= 1e-8:
        return False
    time = np.arange(1, x.size + 1, dtype=np.float64)
    fitted = np.polyval(np.polyfit(time, x, 1), time)
    residuals = x - fitted
    if np.ptp(residuals) <= 1e-8 or np.std(residuals) <= 1e-8:
        return False
    rounded = np.round(residuals, decimals=12)
    if np.unique(rounded).size < 4:
        return False
    ranks = np.argsort(np.argsort(rounded, kind="stable"), kind="stable").astype(np.float64)
    return bool(np.std(ranks) > 1e-8)


@given(
    finite_arrays,
    st.sampled_from([1, 5]),
    st.sampled_from([0.5, 1.0]),
    st.sampled_from([1.0, 2.0]),
    st.integers(min_value=0, max_value=10000),
)
def test_nns_mc_random_inputs_have_valid_shape(
    x: np.ndarray,
    reps: int,
    by: float,
    exp: float,
    seed: int,
) -> None:
    assume(_is_valid_rho_target_input(x))

    result = nns_mc(x, reps=reps, lower_rho=-1.0, upper_rho=1.0, by=by, exp=exp, random_seed=seed)

    assert result["ensemble"].shape == (x.size,)
    assert np.all(np.isfinite(result["ensemble"]))
    assert len(result["replicates"]) >= 1
    for matrix in result["replicates"].values():
        assert matrix.shape == (x.size, reps)
        assert np.all(np.isfinite(matrix))


@given(finite_arrays, st.integers(min_value=0, max_value=10000))
def test_nns_mc_same_seed_is_deterministic(x: np.ndarray, seed: int) -> None:
    assume(_is_valid_rho_target_input(x))

    first = nns_mc(x, reps=2, lower_rho=-1.0, upper_rho=1.0, by=1.0, random_seed=seed)
    second = nns_mc(x, reps=2, lower_rho=-1.0, upper_rho=1.0, by=1.0, random_seed=seed)

    np.testing.assert_array_equal(first["ensemble"], second["ensemble"])
