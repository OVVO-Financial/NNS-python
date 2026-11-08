"""R 14.0 parity for the stochastic-dominance contracts beyond the pairwise verdict.

* ``NNS.SD.efficient.set`` lists its columns by ``LPM(degree, max, column)``,
  which R's C++ forms by repeated multiplication and a row-order sum; columns
  whose keys tie in exact arithmetic (permutations, duplicates) keep their tie
  and R's index tie break.
* R's ``.uni`` tests, efficient set and SD clustering accept infinite values
  (only ``NNS.FSD``/``NNS.SSD``/``NNS.TSD`` reject them, through the plotted
  partial-moment curves), comparing them as R's C++ does.
* Inputs are validated in R's order and with R's messages, and
  ``NNS.SD.cluster`` compares ``min_cluster`` as a real number.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from _r import nns_frame_call, r_column

import nns

pytestmark = pytest.mark.parity


def _num(values: Any) -> dict[str, Any]:
    return r_column([float(v) for v in np.asarray(values, dtype=np.float64).ravel()], "numeric")


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    return [str(v) for v in (value if isinstance(value, list) else [value])]


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_efficient_set_order_keeps_r_ties_for_permuted_and_duplicated_columns(
    degree: int,
) -> None:
    rng = np.random.default_rng(91)
    base = rng.normal(size=(60, 4))
    columns = [base[:, i % 4] for i in range(8)]
    columns[5] = rng.permutation(columns[5])
    columns[7] = rng.permutation(columns[7])
    m = np.column_stack(columns)
    expected = nns_frame_call(
        "sd14.efficient_order",
        f"NNS.SD.efficient.set(matrix(v, ncol = 8), {degree}, status = FALSE)",
        {"v": _num(m.ravel(order="F"))},
    )
    actual = [f"X_{i + 1}" for i in nns.sd_efficient_set(m, degree)]
    assert actual == _strings(expected)


def _with_infinities(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.normal(size=25) + 0.3, rng.normal(size=31)


@pytest.mark.parametrize(("x_inf", "y_inf"), [("Inf", ""), ("", "-Inf"), ("Inf", "-Inf")])
def test_uni_tests_compare_infinite_values_as_r_does(x_inf: str, y_inf: str) -> None:
    x, y = _with_infinities(92)
    if x_inf:
        x[4] = float(x_inf)
    if y_inf:
        y[7] = float(y_inf)
    setup = "x <- xv; y <- yv; " + (f"x[5] <- {x_inf}; " if x_inf else "") + (
        f"y[8] <- {y_inf}; " if y_inf else ""
    )
    expected = nns_frame_call(
        "sd14.uni_infinite",
        setup + "c(NNS.FSD.uni(x, y), NNS.FSD.uni(y, x), "
        "NNS.FSD.uni(x, y, type = 'continuous'), NNS.FSD.uni(y, x, type = 'continuous'), "
        "NNS.SSD.uni(x, y), NNS.SSD.uni(y, x), NNS.TSD.uni(x, y), NNS.TSD.uni(y, x))",
        {
            "xv": _num(np.where(np.isfinite(x), x, 0.0)),
            "yv": _num(np.where(np.isfinite(y), y, 0.0)),
        },
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        actual = [
            nns.fsd_uni(x, y),
            nns.fsd_uni(y, x),
            nns.fsd_uni(x, y, type="continuous"),
            nns.fsd_uni(y, x, type="continuous"),
            nns.ssd_uni(x, y),
            nns.ssd_uni(y, x),
            nns.tsd_uni(x, y),
            nns.tsd_uni(y, x),
        ]
    assert actual == [int(v) for v in np.asarray(expected)]


@pytest.mark.parametrize(("degree", "type_"), [(1, "discrete"), (1, "continuous"), (2, "discrete")])
def test_efficient_set_and_clusters_accept_infinite_values_like_r(degree: int, type_: str) -> None:
    rng = np.random.default_rng(93)
    m = rng.normal(size=(30, 6)) + rng.uniform(-0.5, 0.5, 6)
    m[3, 1] = np.inf
    m[9, 4] = -np.inf
    expected = nns_frame_call(
        "sd14.matrix_infinite",
        "M <- matrix(v, ncol = 6); M[4, 2] <- Inf; M[10, 5] <- -Inf; "
        f"list(eff = NNS.SD.efficient.set(M, {degree}, type = '{type_}', status = FALSE), "
        f"clu = NNS.SD.cluster(M, degree = {degree}, type = '{type_}')$Clusters)",
        {"v": _num(np.where(np.isfinite(m), m, 0.0).ravel(order="F"))},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        efficient = nns.sd_efficient_set(m, degree, type=type_)
        clusters = nns.nns_sd_cluster(m, degree=degree, type=type_)["Clusters"]
    assert [f"X_{i + 1}" for i in efficient] == _strings(expected["eff"])
    assert clusters == {name: _strings(members) for name, members in expected["clu"].items()}


def test_top_level_tests_reject_infinite_values_as_r_does() -> None:
    # NNS.FSD/SSD/TSD first evaluate LPM(.ratio)(degree, sort(c(x, y)), x), which
    # checks the variable x and then the target grid.
    x = np.array([1.0, 2.0, 3.0])
    for function in (nns.fsd, nns.ssd, nns.tsd):
        with pytest.raises(ValueError, match="`variable` must be finite"):
            function(np.array([1.0, np.inf]), x)
        with pytest.raises(ValueError, match="`target` must be finite"):
            function(x, np.array([-np.inf, 1.0]))


def test_validation_follows_r_order_and_messages() -> None:
    missing = "You have some missing values, please address"
    empty = "Each sample needs at least one observation"
    with pytest.raises(ValueError, match=missing):
        nns.fsd(np.array([1.0, np.nan]), np.array([np.inf]))
    with pytest.raises(ValueError, match=missing):
        nns.ssd_uni(np.array([1.0, 2.0]), np.array([0.0, np.nan]))
    with pytest.raises(ValueError, match=empty):
        nns.fsd(np.array([]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match=empty):
        nns.tsd_uni(np.array([1.0, 2.0]), np.array([]))
    # R's C++ efficient set: no columns, then empty columns, then the degree,
    # then missing values.
    assert nns.sd_efficient_set(np.zeros((3, 0)), 7) == []
    with pytest.raises(ValueError, match=empty):
        nns.sd_efficient_set(np.zeros((0, 2)), 9)
    with pytest.raises(ValueError, match="degree must be 1, 2, or 3"):
        nns.sd_efficient_set(np.array([[1.0, 4.0], [2.0, np.nan]]), 5)
    with pytest.raises(ValueError, match=missing):
        nns.sd_efficient_set(np.array([[1.0, 4.0], [2.0, np.nan]]), 2)
    # as.matrix(vector) is one column.
    assert nns.sd_efficient_set(np.array([3.0, 1.0, 2.0]), 2) == [0]


def test_sd_cluster_validates_lazily_and_compares_min_cluster_as_a_real() -> None:
    # No more columns than min_cluster: R never searches, so nothing is checked.
    data = np.array([[1.0, np.nan], [2.0, 4.0]])
    assert nns.nns_sd_cluster(data, degree=9, min_cluster=2)["Clusters"] == {
        "Cluster_1": ["X_1", "X_2"]
    }
    rng = np.random.default_rng(94)
    m = rng.normal(size=(40, 5)) + np.linspace(0.0, 0.6, 5)
    flat = _num(m.ravel(order="F"))
    for min_cluster in (1.5, -1):
        expected = nns_frame_call(
            "sd14.cluster_min",
            "NNS.SD.cluster(matrix(v, ncol = 5), degree = 2, "
            f"min_cluster = {min_cluster})$Clusters",
            {"v": flat},
        )
        actual = nns.nns_sd_cluster(m, degree=2, min_cluster=min_cluster)["Clusters"]
        assert actual == {name: _strings(members) for name, members in expected.items()}
