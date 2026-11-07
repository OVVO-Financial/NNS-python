from __future__ import annotations

import numpy as np
import pytest
from _r import nns

from nns import nns_part

from ._original import r_vector


@pytest.mark.parity
def test_original_partition_map_matches_current_r14_order_rows_and_orientation() -> None:
    # Retain the historical input example, but obtain the reference from the
    # pinned R14 package. Its observation-centered gravity changes this map.
    x = r_vector("test_Partition_Map.R", "x")
    y = r_vector("test_Partition_Map.R", "y")
    expected = nns("NNS.part", x.tolist(), y.tolist(), False, None, None, 8, True, "off")
    actual = nns_part(x, y, min_obs_stop=True)
    assert isinstance(expected, dict)
    assert actual["order"] == expected["order"]
    for table in ("dt", "regression.points"):
        assert set(actual[table]) == set(expected[table])
        for column, values in actual[table].items():
            if "quadrant" in column:
                np.testing.assert_array_equal(values, expected[table][column])
            else:
                np.testing.assert_allclose(values, expected[table][column], atol=1e-12)
