"""R-compatible means (long-double accumulation, as R and NNS's C++ do).

``np.mean`` sums in double with pairwise blocks, so a group of tied values
can come out one ulp off (``np.mean(np.full(30, 19.58))`` is
19.580000000000002); R returns the tied value. These helpers accumulate
sequentially in ``np.longdouble`` (80-bit extended on x86-64, the same type R
and NNS's C++ use there) so the results match R bit for bit.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def r_mean(values: NDArray[np.float64]) -> float:
    """R's ``mean()`` of a double vector (summary.c): a long-double sum / n,
    then, when finite, the second pass ``s += sum(x - s) / n``."""
    v = np.asarray(values, dtype=np.float64).ravel()
    n = v.size
    if n == 0:
        return float("nan")
    ld = v.astype(np.longdouble)
    s = (np.cumsum(ld)[-1] + 0.0) / n  # R accumulates from 0.0L
    if np.isfinite(float(s)):
        s = s + np.cumsum(ld - s)[-1] / n
    return float(s)


def r_mean_no_na(values: NDArray[np.float64]) -> float:
    """NNS.part's C++ ``mean_no_na``: long-double sum of the non-NaN values
    divided by their count (single pass)."""
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[~np.isnan(v)]
    if v.size == 0:
        return float("nan")
    # R starts from 0.0L; "+ 0.0" reproduces that for all-(-0.0) input.
    return float((np.cumsum(v.astype(np.longdouble))[-1] + 0.0) / v.size)


def r_group_means_no_na(
    values: NDArray[np.float64], groups: NDArray[np.intp], n_groups: int
) -> NDArray[np.float64]:
    """``r_mean_no_na`` of ``values[groups == g]`` for every g, each group
    accumulated in its original element order (``np.add.at`` is sequential)."""
    v = np.asarray(values, dtype=np.float64).ravel()
    keep = ~np.isnan(v)
    sums = np.zeros(n_groups, dtype=np.longdouble)
    np.add.at(sums, groups[keep], v[keep].astype(np.longdouble))
    counts = np.bincount(groups[keep], minlength=n_groups)
    out = np.full(n_groups, np.nan, dtype=np.float64)
    has = counts > 0
    out[has] = (sums[has] / counts[has]).astype(np.float64)
    return out
