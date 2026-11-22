"""R 14 parity for the stochastic-dominance contracts beyond the pairwise verdict.

* ``NNS.SD.efficient.set`` lists its columns by ``LPM(degree, max, column)``,
  which R's C++ forms by repeated multiplication and a row-order sum; columns
  whose keys tie in exact arithmetic (permutations, duplicates) keep their tie
  and R's index tie break.
* Discrete first-order comparisons accept infinite values.
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
        setup + "c(NNS.FSD.uni(x, y), NNS.FSD.uni(y, x))",
        {
            "xv": _num(np.where(np.isfinite(x), x, 0.0)),
            "yv": _num(np.where(np.isfinite(y), y, 0.0)),
        },
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        actual = [nns.fsd_uni(x, y), nns.fsd_uni(y, x)]
    assert actual == [int(v) for v in np.asarray(expected)]


@pytest.mark.parametrize(("degree", "type_"), [(1, "continuous"), (2, "discrete"), (3, "discrete")])
def test_sd_rejects_infinities(degree, type_):
    uni = (nns.fsd_uni, nns.ssd_uni, nns.tsd_uni)[degree - 1]
    r = ("NNS.FSD.uni", "NNS.SSD.uni", "NNS.TSD.uni")[degree - 1]
    extra = ", 'continuous'" if degree == 1 else ""
    expected = nns_frame_call(
        "sd14.uni_infinite", f"tryCatch({r}(c(0,Inf),1:2{extra}), error=conditionMessage)", {}
    )
    y = [0, 1]
    for inf in (-np.inf, np.inf):
        x = [0, inf]
        for args in ((x, y), (y, x)):
            with pytest.raises(ValueError) as caught:
                uni(*args, **({"type": type_} if degree == 1 else {}))
            assert str(caught.value) == expected


@pytest.mark.parametrize(("degree", "type_"), [(1, "continuous"), (2, "discrete"), (3, "discrete")])
@pytest.mark.parametrize(("fn", "r"), [
    (nns.sd_efficient_set, "NNS.SD.efficient.set"), (nns.nns_sd_cluster, "NNS.SD.cluster"),
])
def test_matrix_search_rejects_infinities(fn, r, degree, type_):
    expected = nns_frame_call(
        "sd14.matrix_infinite",
        f"tryCatch({r}(cbind(c(0,Inf),1:2), {degree}, '{type_}'), error=conditionMessage)", {}
    )
    for inf in (-np.inf, np.inf):
        with pytest.raises(ValueError) as caught:
            fn(np.column_stack(([0, inf], [1, 2])), degree=degree, type=type_)
        assert str(caught.value) == expected


def test_discrete_efficient_set_and_clusters_accept_infinite_values_like_r() -> None:
    degree, type_ = 1, "discrete"
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
    # R 14.2 NNS.SSD/TSD always evaluate LPM(degree, sort(c(x, y)), x), which
    # checks the variable x and then the target grid. NNS.FSD evaluates its
    # LPM.ratio curves only for the plot (R's default, plot = TRUE); without one
    # the verdict is NNS.FSD.uni both ways, whose discrete test takes infinite
    # values and whose continuous test rejects them.
    x = np.array([1.0, 2.0, 3.0])
    for function in (nns.ssd, nns.tsd):
        for plot in (False, True):
            with pytest.raises(ValueError, match="`variable` must be finite"):
                function(np.array([1.0, np.inf]), x, plot=plot)
            with pytest.raises(ValueError, match="`target` must be finite"):
                function(x, np.array([-np.inf, 1.0]), plot=plot)
    for type_ in ("discrete", "continuous"):
        with pytest.raises(ValueError, match="`variable` must be finite"):
            nns.fsd(np.array([1.0, np.inf]), x, type_, plot=True)
        with pytest.raises(ValueError, match="`target` must be finite"):
            nns.fsd(x, np.array([-np.inf, 1.0]), type_, plot=True)
    # NNS.FSD(c(1, Inf), 1:3, plot = FALSE) is "NO FSD EXISTS" and
    # NNS.FSD(1:3, c(-Inf, 1), plot = FALSE) is "X FSD Y".
    assert nns.fsd(np.array([1.0, np.inf]), x) == 0
    assert nns.fsd(x, np.array([-np.inf, 1.0])) == 1
    with pytest.raises(ValueError, match="You have some infinite values, please address"):
        nns.fsd(np.array([1.0, np.inf]), x, "continuous")


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
