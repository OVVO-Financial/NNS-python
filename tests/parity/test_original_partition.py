from __future__ import annotations

import numpy as np
import pytest

from pynns import nns_part

from ._original import expected, r_string_vector_assignment, r_vector


@pytest.mark.parity
def test_original_partition_map_matches_r_fixture_order_rows_and_orientation() -> None:
    x = r_vector("test_Partition_Map.R", "x")
    y = r_vector("test_Partition_Map.R", "y")
    exp = expected("test_Partition_Map.R")

    actual = nns_part(x, y, min_obs_stop=True)

    assert actual["order"] == exp["order"]
    np.testing.assert_allclose(actual["dt"]["x"], x, atol=1e-12)
    np.testing.assert_allclose(actual["dt"]["y"], y, atol=1e-12)
    np.testing.assert_array_equal(
        actual["dt"]["quadrant"],
        r_string_vector_assignment("test_Partition_Map.R", "T_DT$quadrant"),
    )
    np.testing.assert_array_equal(
        actual["dt"]["prior.quadrant"],
        r_string_vector_assignment("test_Partition_Map.R", "T_DT$prior.quadrant"),
    )
    np.testing.assert_array_equal(
        actual["regression.points"]["quadrant"], exp["regression_points"]["quadrant"]
    )
    np.testing.assert_allclose(
        actual["regression.points"]["x"], exp["regression_points"]["x"], atol=1e-5
    )
    np.testing.assert_allclose(
        actual["regression.points"]["y"], exp["regression_points"]["y"], atol=1e-5
    )
