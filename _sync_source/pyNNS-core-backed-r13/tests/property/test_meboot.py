from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pynns import nns_meboot

pytestmark = pytest.mark.stochastic

finite_arrays = arrays(
    dtype=np.float64,
    shape=st.integers(min_value=5, max_value=100),
    elements=st.floats(
        min_value=-50.0,
        max_value=50.0,
        allow_nan=False,
        allow_infinity=False,
        width=64,
    ),
)


def _valid_rho_target_input(x: np.ndarray) -> bool:
    time = np.arange(1, x.size + 1, dtype=np.float64)
    fitted = np.polyval(np.polyfit(time, x, 1), time)
    residuals = x - fitted
    return bool(
        np.ptp(x) > 1e-8
        and np.std(residuals) > 1e-8
        and np.unique(np.round(residuals, decimals=12)).size >= 3
    )


@given(
    finite_arrays,
    st.sampled_from([1, 5, 10]),
    st.sampled_from([-1.0, 0.0, 0.5, 1.0]),
    st.integers(min_value=0, max_value=10000),
)
def test_nns_meboot_random_inputs_have_valid_shape(
    x: np.ndarray,
    reps: int,
    rho: float,
    seed: int,
) -> None:
    assume(_valid_rho_target_input(x))

    result = nns_meboot(x, reps=reps, rho=rho, random_seed=seed)

    assert result["replicates"].shape == (x.size, reps)
    assert result["ensemble"].shape == (x.size,)
    assert np.all(np.isfinite(result["replicates"]))
    assert np.all(np.isfinite(result["ensemble"]))


@given(finite_arrays, st.integers(min_value=0, max_value=10000))
def test_nns_meboot_same_seed_is_deterministic(x: np.ndarray, seed: int) -> None:
    assume(_valid_rho_target_input(x))

    first = nns_meboot(x, reps=3, rho=0.0, random_seed=seed)
    second = nns_meboot(x, reps=3, rho=0.0, random_seed=seed)

    np.testing.assert_array_equal(first["replicates"], second["replicates"])
