"""Index helpers shared by resampling-heavy estimators."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def ordered_complement(
    all_index: NDArray[np.int64],
    excluded: NDArray[np.int64],
) -> NDArray[np.int64]:
    """Return ``all_index`` values not present in ``excluded`` without sorting.

    The stack/boost split builders already create unique validation indices
    against a dense ``0..n-1`` index vector.  A boolean complement avoids
    ``np.setdiff1d``'s sort/unique work while preserving the input row order.
    """
    if all_index.size == 0 or excluded.size == 0:
        return all_index.copy()
    mask_size = int(max(np.max(all_index), np.max(excluded))) + 1
    excluded_mask = np.zeros(mask_size, dtype=bool)
    excluded_mask[excluded] = True
    return all_index[~excluded_mask[all_index]]
