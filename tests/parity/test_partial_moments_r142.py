from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from _r import nns_frame_call

import nns
from nns._native import native_fn
from nns.cdf import _co_lpm_nd_batch

pytestmark = pytest.mark.parity


def oracle(expr):
    return nns_frame_call("pm142", expr, {}, exact=True)


def error(expr):
    return oracle(f"tryCatch({expr}, error = conditionMessage)")


@pytest.mark.parametrize("name", ["lpm", "upm"])
def test_scalar_target_coercion(name):
    fn = getattr(nns, name)
    r = name.upper()
    x = [1, 2, 3, 10]
    for target, rt in [(None, "NULL"), ("2", "'2'")]:
        for excess in (False, True):
            assert fn(1, target, x, excess) == oracle(
                f"{r}(1, {rt}, c(1,2,3,10), {str(excess).upper()})"
            )
    for target, rt in [("bad", "'bad'"), (pd.Categorical([2]), "factor(2)")]:
        msg = error(f"{r}(1, {rt}, c(1,2,3,10))")
        for excess in (False, True):
            with pytest.raises(ValueError) as caught:
                fn(1, target, x, excess)
            assert str(caught.value) == msg


@pytest.mark.parametrize(
    ("name", "r"),
    [
        ("lpm_ratio", "LPM.ratio"),
        ("upm_ratio", "UPM.ratio"),
        ("co_lpm", "Co.LPM"),
        ("co_upm", "Co.UPM"),
        ("d_lpm", "D.LPM"),
        ("d_upm", "D.UPM"),
        ("pm_matrix", "PM.matrix"),
    ],
)
def test_typed_targets(name, r):
    fn = getattr(nns, name)
    x = [0, 2, 4]
    matrix = np.column_stack((x, x))

    def args(t, rt):
        if name == "pm_matrix":
            return (1, 1, t, matrix, False), f"1, 1, {rt}, cbind(x,x), FALSE"
        if name.endswith("ratio"):
            return (1, t, x), f"1, {rt}, x"
        degrees = (1, 1) if name.startswith("d_") else (1,)
        return (*degrees, x, x, None, t), f"{'1, ' * len(degrees)}x, x, NULL, {rt}"

    for t, rt in [
        ([True, False], "c(TRUE,FALSE)"),
        (np.array([1, 3]), "c(1L,3L)"),
        ([Decimal("1"), Decimal("3")], "c(1L,3L)"),
        (np.array([Fraction(1), Fraction(3)], dtype=object), "c(1L,3L)"),
        (np.array(["1970-01-02", "1970-01-04"], dtype="datetime64[ns]"), "c(1L,3L)"),
    ]:
        actual, rexpr = args(t, rt)
        result = fn(*actual)
        suffix = "$clpm" if name == "pm_matrix" else ""
        expected = oracle(f"x <- c(0,2,4); {r}({rexpr}){suffix}")
        np.testing.assert_array_equal(result["clpm"] if name == "pm_matrix" else result, expected)

    for t, rt in [("2", "'2'"), (pd.Categorical([2]), "factor(2)"), ([1, np.nan], "c(1,NA)")]:
        actual, rexpr = args(t, rt)
        expected = error(f"{{x <- c(0,2,4); {r}({rexpr})}}")
        with pytest.raises(ValueError) as caught:
            fn(*actual)
        assert str(caught.value) == expected
    if name == "pm_matrix":
        with pytest.raises(ValueError) as caught:
            fn(1, 1, [True], matrix, False)
        assert str(caught.value) == error("PM.matrix(1,1,TRUE,cbind(1:3,1:3),FALSE)")
        for alias in ("mean", "means"):
            expected = oracle(f"x <- c(0,2,4); PM.matrix(1,1,'{alias}',cbind(x,x),FALSE)$clpm")
            np.testing.assert_array_equal(fn(1, 1, alias, matrix, False)["clpm"], expected)


@pytest.mark.parametrize(
    ("name", "r", "arguments"),
    [
        ("lpm", "LPM", (1, 2, "f")),
        ("upm", "UPM", (1, 2, "f")),
        ("lpm_ratio", "LPM.ratio", (1, 2, "f")),
        ("upm_ratio", "UPM.ratio", (1, 2, "f")),
        ("nns_moments", "NNS.moments", ("f",)),
        ("nns_cdf", "NNS.CDF", ("f",)),
        ("lpm_var", "LPM.VaR", (0.5, 1, "f")),
        ("upm_var", "UPM.VaR", (0.5, 1, "f")),
        ("lpm_var", "LPM.VaR", (0.5, 1, "tbl")),
        ("upm_var", "UPM.VaR", (0.5, 1, "tbl")),
        *(
            (name, r, args)
            for name, r, deg in [
                ("co_lpm", "Co.LPM", (1,)),
                ("co_upm", "Co.UPM", (1,)),
                ("d_lpm", "D.LPM", (1, 1)),
                ("d_upm", "D.UPM", (1, 1)),
            ]
            for args in [(*deg, "f", "x", 2, 2), (*deg, "x", "f", 2, 2)]
        ),
        *(
            (name, r, args)
            for name, r in [
                ("fsd", "NNS.FSD"),
                ("ssd", "NNS.SSD"),
                ("tsd", "NNS.TSD"),
                ("fsd_uni", "NNS.FSD.uni"),
                ("ssd_uni", "NNS.SSD.uni"),
                ("tsd_uni", "NNS.TSD.uni"),
            ]
            for args in [("x", "f"), ("f", "x")]
        ),
    ],
)
def test_factor_data(name, r, arguments):
    x = [10, 20, 30]
    f = pd.Categorical(x)
    values = {"x": x, "f": f, "tbl": pd.DataFrame({"a": x, "b": f})}
    setup = "x <- c(10,20,30); f <- factor(x)"
    if "tbl" in arguments:
        setup += "; tbl <- structure(data.frame(a=x,b=f), class=c('data.table','data.frame'))"
    expected = error(f"{{{setup}; {r}({','.join(map(str, arguments))})}}")
    with pytest.raises(ValueError) as caught:
        getattr(nns, name)(*(values.get(a, a) for a in arguments))
    assert str(caught.value) == expected


@pytest.mark.parametrize("degree", [0.5, 1, 2])
def test_zero_total_quadrants(degree):
    x = np.column_stack(([-1, 0, 1, 2], [5] * 4))
    t = np.array([0.0, 5.0])
    expected = oracle(
        f"X <- cbind(c(-1,0,1,2),5); t <- c(0,5); d <- {degree}; "
        "c(Co.LPM_nD(X,t,d), Co.UPM_nD(X,t,d), DPM_nD(X,t,d))"
    )
    for name, key, value in zip(
        ("co_lpm_nd", "co_upm_nd", "dpm_nd"),
        ("clpm_nd", "cupm_nd", "dpm_nd"),
        expected,
        strict=True,
    ):
        assert getattr(nns, name)(x, t, degree) == value
        assert getattr(nns, name)(x, t, degree, False) == 0
        native = native_fn(key)
        if native:
            assert native(x.ravel(order="F").astype(float), 4, 2, t, degree, True) == value


@pytest.mark.parametrize("kind", ["CDF", "survival"])
def test_zero_total_cdf_and_survival(kind):
    x = np.array([[-1, -1], [0, 0], [1, 1], [2, 1]]) * 1e-200
    expected = oracle(
        "X <- cbind(c(-1,0,1,2),c(-1,0,1,1)) * 1e-200; "
        f"a <- NNS.CDF(X, 1, c(0,0), '{kind}', FALSE); c(a$Function$CDF, a$target.value)"
    )
    upper = kind == "survival"
    moment = nns.co_upm_nd if upper else nns.co_lpm_nd
    t = np.zeros(2)
    assert moment(x, t, 1) == expected[-1]
    native = native_fn("cupm_nd" if upper else "clpm_nd")
    if native:
        assert native(x.ravel(order="F"), 4, 2, t, 1, True) == expected[-1]
    actual = nns.nns_cdf(x, 1, [0, 0], type=kind)
    np.testing.assert_array_equal(
        np.r_[actual["Function"]["CDF"], actual["target.value"]], expected
    )


@pytest.mark.parametrize("degrees", [(1, 1), (0.5, 2), (1, 0)])
def test_zero_total_matrix_cells(degrees):
    x = np.column_stack(([1, 2, 5, 0, 2], [2] * 5, [4, 0, 1, 0, 2]))
    expected = oracle(
        f"PM.matrix({degrees[0]},{degrees[1]},c(2,2,1),"
        "cbind(c(1,2,5,0,2),2,c(4,0,1,0,2)),TRUE,TRUE)"
    )
    for key, value in nns.pm_matrix(*degrees, [2, 2, 1], x, True, True).items():
        np.testing.assert_array_equal(value, expected[key])


@pytest.mark.parametrize("alias", [pd.Series(["mean"], index=["cutoff"]), pd.Categorical(["mean"])])
def test_matrix_rejects_named_and_factor_aliases(alias):
    with pytest.raises(ValueError, match="must be numeric"):
        nns.pm_matrix(1, 1, alias, np.eye(2), False)


def test_matrix_target_shape_and_broadcast():
    x = np.arange(12).reshape(3, 4)
    for target, expected in [
        (Decimal("2"), [2] * 4),
        (np.datetime64("1970-01-03"), [2] * 4),
        (np.array([[1, 3], [2, 4]]), [1, 2, 3, 4]),
    ]:
        actual = nns.pm_matrix(1, 1, target, x, False)
        for key, value in nns.pm_matrix(1, 1, expected, x, False).items():
            np.testing.assert_array_equal(actual[key], value)
    with pytest.raises(ValueError, match="cols != target vector length"):
        nns.pm_matrix(1, 1, np.array([[1]]), x, False)


@pytest.mark.parametrize(("fn", "r"), [(nns.lpm_var, "LPM.VaR"), (nns.upm_var, "UPM.VaR")])
@pytest.mark.parametrize(
    ("labels", "r_labels"),
    [([], "integer(0)"), ([30], "30"), ([None], "NA"), ([10, 30], "c(10,30)")],
)
def test_var_factor_column(fn, r, labels, r_labels):
    factor = pd.Categorical(labels, categories=[10, 30])
    frame = pd.DataFrame({"a": [10] * len(labels), "b": factor})
    expected = error(
        f"{r}(0.5,1,data.frame(a=rep(10,{len(labels)}),b=factor({r_labels},levels=c(10,30))))"
    )
    with pytest.raises(ValueError) as caught:
        fn(0.5, 1, frame)
    assert str(caught.value) == expected


@pytest.mark.parametrize("excess", [False, True])
def test_nonfinite_target_before_degree(excess):
    with pytest.raises(ValueError, match=r"target.*must be finite"):
        nns.lpm(-1, np.inf, [1, 3], excess)


def test_validation_order():
    factor = pd.Categorical([1, 3])
    bad = np.array([[1, 1], [3, np.inf]])
    for fn, args, error in [
        (nns.lpm_ratio, (-1, 2, factor), "degree"),
        (nns.lpm_ratio, (0, "2", [1, np.inf]), "target.*must be numeric"),
        (nns.lpm_ratio, (0, np.datetime64("NaT"), [1, 3]), "target.*must be finite"),
        (nns.co_lpm, (-1, factor, [1, 3], 2, 2), "degree"),
        (nns.co_lpm, (0, [1], [1, 3], "2", 2), "target.*must be numeric"),
        (nns.co_lpm, (0, [1, np.inf], [1, 3], [], 2), "target.*empty"),
        (nns.pm_matrix, (0, 0, [2], bad, False), "cols != target vector length"),
        (nns.pm_matrix, (0, 0, "wrong", bad, False), "target.*must be numeric"),
        (nns.nns_cdf, (factor, 0, None, "bad"), "factor"),
        (nns.nns_cdf, ([1, 3], 0, 7, "bad"), "out of bounds"),
        (nns.nns_cdf, (np.eye(2), 0, [7, 7], "hazard"), "out of bounds"),
        (nns.co_lpm_nd, (bad, [2]), "data must be finite"),
    ]:
        with pytest.raises(ValueError, match=error):
            fn(*args)
    with pytest.warns(UserWarning, match="type needs"), pytest.raises(ValueError):
        nns.fsd_uni([], [1, 3], type="bad")


def test_batch_degree_zero_tie():
    x = np.array([[1.0]])
    np.testing.assert_array_equal(_co_lpm_nd_batch(x, x, 0, strict=True), [0])
    np.testing.assert_array_equal(_co_lpm_nd_batch(x, x, 0), [1])


def test_native_batch_raw():
    batch = native_fn("clpm_nd_batch")
    if batch is None:
        pytest.skip("raw batch requires the native backend")
    args = (np.array([0.0, 4.0]), 2, 1, np.array([1.0]), 1, 1.0)
    np.testing.assert_array_equal(batch(*args, False), [0.5])
    np.testing.assert_array_equal(batch(*args, True), [0.25])
