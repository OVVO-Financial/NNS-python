from __future__ import annotations

import numpy as np

from nns._indices import ordered_complement


def test_ordered_complement_preserves_source_index_order() -> None:
    all_index = np.array([4, 2, 0, 3, 1], dtype=np.int64)
    excluded = np.array([0, 3], dtype=np.int64)

    result = ordered_complement(all_index, excluded)

    np.testing.assert_array_equal(result, np.array([4, 2, 1], dtype=np.int64))
