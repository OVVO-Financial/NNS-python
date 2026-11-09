"""R 14.0 parity, to the last bit, for NNS.distance and NNS.distance.bulk.

Both follow R's C++ kernels: ``1 / range`` column scaling, distances summed
column by column, R's nmath densities and sequential sums. Tied distances fall
where libstdc++'s ``std::partial_sort``/``std::sort`` leave them, which the port
replays (``nns._stdsort``); the fixtures below were produced by compiled
libstdc++.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from _r import nns_frame_call, r_column

import nns
from nns._stdsort import partial_sort, std_sort

pytestmark = pytest.mark.parity


def _num(values: Any) -> dict[str, Any]:
    return r_column([float(v) for v in np.asarray(values, dtype=np.float64).ravel()], "numeric")


def _tied_rpm(seed: int, n: int, classes: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    features = rng.integers(0, 4, size=(n, 2)).astype(np.float64)
    y = rng.integers(1, 4, size=n).astype(np.float64) if classes else np.round(
        rng.normal(size=n) * 3.0, 2
    )
    return np.column_stack([features, y])


_RPM = "rpm <- data.frame(X1 = f1, X2 = f2, y.hat = yh)"


def _rpm_data(rpm: np.ndarray) -> dict[str, Any]:
    return {"f1": _num(rpm[:, 0]), "f2": _num(rpm[:, 1]), "yh": _num(rpm[:, 2])}


@pytest.mark.parametrize("classes", [False, True])
@pytest.mark.parametrize(("k", "r_k"), [(3, "3"), (7, "7"), ("all", "'all'"), (0, "0")])
def test_nns_distance_with_tied_distances_matches_r_bitwise(
    classes: bool, k: int | str, r_k: str
) -> None:
    rpm = _tied_rpm(601, 40, classes)
    dest = np.array([1.0, 2.0])
    r_class = "'class'" if classes else "NULL"
    expected = nns_frame_call(
        "dist14.scalar",
        f"{_RPM}; NNS.distance(rpm, dest, k = {r_k}, class = {r_class})",
        {**_rpm_data(rpm), "dest": _num(dest)},
        exact=True,
    )
    actual = nns.nns_distance(rpm, dest, k=k, class_="class" if classes else None)
    assert actual == expected


@pytest.mark.parametrize("classes", [False, True])
@pytest.mark.parametrize(("k", "r_k"), [(2, "2"), (5, "5"), ("all", "'all'")])
def test_nns_distance_bulk_with_tied_distances_matches_r_bitwise(
    classes: bool, k: int | str, r_k: str
) -> None:
    rpm = _tied_rpm(602, 50, classes)
    tests = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 1.0], [1.5, 1.5]])
    r_class = "'class'" if classes else "NULL"
    expected = nns_frame_call(
        "dist14.bulk",
        f"{_RPM}; NNS:::NNS.distance.bulk(rpm, matrix(xv, ncol = 2), k = {r_k}, class = {r_class})",
        {**_rpm_data(rpm), "xv": _num(tests.ravel(order="F"))},
        exact=True,
    )
    actual = nns.nns_distance_bulk(rpm, tests, k=k, class_="class" if classes else None)
    np.testing.assert_array_equal(actual, np.asarray(expected, dtype=np.float64))


def test_continuous_distances_match_r_bitwise() -> None:
    rng = np.random.default_rng(603)
    rpm = np.column_stack([rng.normal(size=(80, 2)), rng.normal(size=80)])
    dest = rng.normal(size=2)
    expected = nns_frame_call(
        "dist14.continuous",
        f"{_RPM}; c(NNS.distance(rpm, dest, k = 6), "
        "NNS:::NNS.distance.bulk(rpm, matrix(dest, ncol = 2), k = 6))",
        {**_rpm_data(rpm), "dest": _num(dest)},
        exact=True,
    )
    single = nns.nns_distance(rpm, dest, k=6)
    bulk = nns.nns_distance_bulk(rpm, dest.reshape(1, -1), k=6)
    np.testing.assert_array_equal([single, bulk[0]], np.asarray(expected, dtype=np.float64))


# Orders printed by compiled libstdc++ (g++ -O2) for these keys.
_ALL_TIED = [0] * 52
_ALL_TIED_SORT = [
    39, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 26, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    49, 50, 51, 13, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 14, 15, 16, 17, 18, 19, 20, 21, 22,
    23, 24, 25,
]
_BINARY = [
    1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1,
]
_BINARY_SORT = [
    35, 31, 29, 28, 27, 26, 25, 24, 23, 22, 21, 32, 19, 33, 17, 36, 13, 11, 10, 7, 4, 2, 43, 34,
    37, 38, 39, 40, 41, 42, 0, 30, 20, 18, 16, 15, 14, 12, 9, 8, 6, 5, 3, 1,
]
_BINARY_PARTIAL_10 = [17, 4, 13, 7, 10, 19, 21, 22, 11, 2]


def test_sort_replay_matches_libstdcxx_on_ties() -> None:
    order = list(range(52))
    std_sort(order, [float(v) for v in _ALL_TIED])
    assert order == _ALL_TIED_SORT
    order = list(range(52))
    partial_sort(order, [float(v) for v in _ALL_TIED], 4)
    assert order[:4] == [1, 3, 0, 2]
    order = list(range(44))
    std_sort(order, [float(v) for v in _BINARY])
    assert order == _BINARY_SORT
    order = list(range(44))
    partial_sort(order, [float(v) for v in _BINARY], 10)
    assert order[:10] == _BINARY_PARTIAL_10
