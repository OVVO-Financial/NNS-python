from __future__ import annotations

import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pynns import nns_sd_cluster


@given(
    data=arrays(
        np.float64,
        st.tuples(st.integers(min_value=10, max_value=60), st.integers(min_value=2, max_value=8)),
        elements=st.floats(-20.0, 20.0, allow_nan=False, allow_infinity=False),
    ),
    degree=st.integers(min_value=1, max_value=3),
    min_cluster=st.integers(min_value=1, max_value=4),
)
def test_nns_sd_cluster_structural_invariants_hold(
    data: np.ndarray,
    degree: int,
    min_cluster: int,
) -> None:
    assume(np.all(np.ptp(data, axis=0) > 1e-12))
    cols = data.shape[1]

    result = nns_sd_cluster(data, degree=degree, min_cluster=min_cluster)
    clusters = result["Clusters"]

    assert list(clusters) == [f"Cluster_{index}" for index in range(1, len(clusters) + 1)]
    members = [name for cluster in clusters.values() for name in cluster]
    if min_cluster >= cols:
        assert members == []
    else:
        expected_names = [f"X_{index + 1}" for index in range(cols)]
        assert sorted(members, key=lambda item: int(item.split("_")[1])) == expected_names
        assert len(members) == len(set(members))
