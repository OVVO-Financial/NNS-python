"""Partitions on data whose gravity histogram exceeds 1e6 bins.

The native gravity_exact kernel used to clip its dense histogram at 1e6 bins,
which moved the mode term onto the outlier; R does not cap. These cases have a
single outlier that pushes the bin count of the first gravity call past 1e6.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from _r import nns
from _tolerances import EXACT

from nns import nns_dep, nns_part
from nns.dependence import (
    _gravity,
    _gravity_python,
    _quartiles_like_r_code,
    _sparse_bin_counts,
)


def _outlier_pair(n: int = 2000, outlier: float = -1e5) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(10)
    x = rng.normal(size=n)
    y = x + 0.3 * rng.normal(size=n)
    x[0] = outlier
    y[0] = outlier + 0.05
    return x, y


@pytest.mark.parity
def test_gravity_with_outlier_matches_r() -> None:
    x, _ = _outlier_pair()

    expected = float(np.asarray(nns("NNS.gravity", x.tolist(), False)))

    assert _gravity(x) == pytest.approx(expected, abs=1e-12)
    assert _gravity_python(x) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parity
@pytest.mark.parametrize("part_type", [None, "XONLY"])
def test_nns_part_with_outlier_matches_r(part_type: str | None) -> None:
    x, y = _outlier_pair()

    expected = cast(
        dict, nns("NNS.part", x.tolist(), y.tolist(), False, part_type, 2, 8, True, "off")
    )
    actual = nns_part(x, y, type=part_type, order=2)

    rp = expected["regression.points"]
    assert actual["order"] == int(np.asarray(expected["order"]).item())
    np.testing.assert_array_equal(actual["dt"]["quadrant"], np.asarray(expected["dt"]["quadrant"]))
    np.testing.assert_array_equal(
        actual["regression.points"]["quadrant"], np.asarray(rp["quadrant"])
    )
    np.testing.assert_allclose(actual["regression.points"]["x"], np.asarray(rp["x"]), atol=EXACT)
    np.testing.assert_allclose(actual["regression.points"]["y"], np.asarray(rp["y"]), atol=EXACT)


@pytest.mark.parity
@pytest.mark.parametrize("asym", [False, True])
def test_nns_dep_with_outlier_matches_r(asym: bool) -> None:
    # With n = 1e4 the capped kernel gave Dependence 0.5995 (asym 0.5766).
    x, y = _outlier_pair(n=10_000)

    expected = cast(dict, nns("NNS.dep", x.tolist(), y.tolist(), asym, False, False))
    actual = nns_dep(x, y, asym=asym)

    np.testing.assert_allclose(
        actual["Correlation"], float(np.asarray(expected["Correlation"])), atol=EXACT
    )
    np.testing.assert_allclose(
        actual["Dependence"], float(np.asarray(expected["Dependence"])), atol=EXACT
    )


def test_python_gravity_uses_r14_type7_and_observation_centers() -> None:
    # R 14.0: type-7 quartiles, with modal centers drawn from observations.
    # The older vendored native gravity_exact is deliberately NOT a reference
    # until NNS-core has been synchronized with the R 14.0 implementation.
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert _gravity_python(x) == pytest.approx(2.5, abs=1e-12)
    assert _gravity(x) == pytest.approx(2.5, abs=1e-12)

    # Modal neighborhood contains the three zero observations; the mean and
    # type-7 upper quartile still incorporate the distant observation.
    y = np.array([0.0, 0.0, 0.0, 100.0])
    assert _gravity_python(y) == pytest.approx(9.375, abs=1e-12)
    assert _gravity(y) == pytest.approx(9.375, abs=1e-12)


def test_sparse_gravity_needs_no_dense_histogram() -> None:
    # About 7e8 bins in R's dense layout (below its int overflow; about 3 GB of
    # counts in R); the sorted-run histogram only stores the occupied bins.
    x, _ = _outlier_pair(n=10_000, outlier=-1e7)
    values = np.sort(x)
    q1, _, q3 = _quartiles_like_r_code(values)
    width = (q3 - q1) * values.size**-0.5
    bin_count, used_width, occupied, counts = _sparse_bin_counts(values, width, float(values[0]))
    assert bin_count > 100_000_000
    assert used_width == width
    assert occupied.size == counts.size <= values.size
    assert int(counts.sum()) == values.size
    assert np.all(np.diff(occupied) > 0)
    assert np.isfinite(_gravity(x))
