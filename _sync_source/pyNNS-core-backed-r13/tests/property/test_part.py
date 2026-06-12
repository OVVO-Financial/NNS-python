from __future__ import annotations

from typing import cast

import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pynns import nns_part
from pynns.part import NoiseReduction

finite_arrays = arrays(
    dtype=np.float64,
    shape=st.integers(min_value=4, max_value=100),
    elements=st.floats(
        min_value=-1e6,
        max_value=1e6,
        allow_nan=False,
        allow_infinity=False,
        width=64,
    ),
)


@given(
    finite_arrays,
    finite_arrays,
    st.sampled_from([None, "XONLY", "Y"]),
    st.sampled_from(["off", "mean", "median", "mode", "mode_class"]),
    st.integers(min_value=0, max_value=5),
    st.integers(min_value=0, max_value=16),
    st.booleans(),
)
def test_nns_part_shape_invariants_hold_for_random_pairs(
    x: np.ndarray,
    y: np.ndarray,
    part_type: str | None,
    noise: str,
    order: int,
    obs_req: int,
    min_obs_stop: bool,
) -> None:
    size = min(x.size, y.size)
    x = x[:size]
    y = y[:size]
    assume(np.ptp(x) > 0.0)
    assume(np.ptp(y) > 0.0)

    result = nns_part(
        x,
        y,
        type=part_type,
        order=order,
        obs_req=obs_req,
        min_obs_stop=min_obs_stop,
        noise_reduction=cast(NoiseReduction, noise),
    )
    dt = result["dt"]
    rp = result["regression.points"]
    assert isinstance(dt, dict)
    assert isinstance(rp, dict)

    quadrants = dt["quadrant"].astype(str)
    prior = dt["prior.quadrant"].astype(str)

    assert dt["x"].shape == (size,)
    assert dt["y"].shape == (size,)
    assert quadrants.shape == (size,)
    assert prior.shape == (size,)
    assert rp["quadrant"].size == np.unique(prior).size
    assert 0 <= result["order"] <= int(np.floor(np.log2(size)))
