"""Column names, factor levels and class levels on the regression/stack/boost paths.

R's NNS.reg / NNS.M.reg / NNS.stack / NNS.boost work on data frames: columns keep
their names (prediction columns are matched by name) and their type (character
and factor columns are factors, coded in their declared level order). These
tests run the same pandas inputs through the port and through R.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from nns import nns_boost, nns_m_reg, nns_reg, nns_stack
from tests._r import nns_frame_call, r_column
from tests._tolerances import COMPOUND

pytestmark = pytest.mark.parity


def _spec(frame: pd.DataFrame) -> dict[str, Any]:
    cols = []
    for name in frame.columns:
        col = frame[name]
        if isinstance(col.dtype, pd.CategoricalDtype):
            cols.append(r_column(col.astype(object).tolist(), "factor", name,
                                 col.cat.categories.tolist()))
        elif not pd.api.types.is_numeric_dtype(col.dtype):
            cols.append(r_column(col.tolist(), "character", name))
        else:
            cols.append(r_column(col.astype(float).tolist(), "numeric", name))
    return {"frame": cols}


def _vec(series: pd.Series) -> dict[str, Any]:
    if isinstance(series.dtype, pd.CategoricalDtype):
        return r_column(series.astype(object).tolist(), "factor",
                        levels=series.cat.categories.tolist())
    if not pd.api.types.is_numeric_dtype(series.dtype):
        return r_column(series.tolist(), "character")
    return r_column(series.astype(float).tolist(), "numeric")


def _num(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _cars() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = 40
    wt = np.round(rng.uniform(1.5, 5.5, n), 3)
    hp = np.round(rng.uniform(50, 330, n))
    disp = np.round(wt * 60 + rng.normal(0, 20, n), 1)
    cyl = np.where(hp < 120, "four", np.where(hp < 200, "six", "eight"))
    mpg = np.round(40 - 4 * wt - 0.03 * hp + rng.normal(0, 1.5, n), 2)
    return pd.DataFrame({"wt": wt, "hp": hp, "disp": disp, "cyl": cyl, "mpg": mpg})


def _flowers(seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = np.repeat(["setosa", "versicolor", "virginica"], 20)
    centre = {"setosa": 1.0, "versicolor": 2.0, "virginica": 3.0}
    base = np.asarray([centre[v] for v in labels])
    frame = pd.DataFrame(
        {
            "a": np.round(base + rng.normal(0, 0.45, labels.size), 3),
            "b": np.round(2 * base + rng.normal(0, 0.9, labels.size), 3),
            "c": np.round(rng.normal(0, 1, labels.size), 3),
            "species": labels,
        }
    )
    # Shuffle so the first-appearance order (virginica first) differs from the
    # sorted / declared order.
    order = rng.permutation(labels.size).tolist()
    first = next(i for i in order if labels[i] == "virginica")
    order = [first] + [i for i in order if i != first]
    return frame.iloc[order].reset_index(drop=True)


# --------------------------------------------------------------------------
# REG-03: point.est columns are matched by name
# --------------------------------------------------------------------------

def test_nns_reg_point_est_columns_matched_by_name() -> None:
    cars = _cars()
    x = cars[["wt", "hp"]]
    permuted = x.iloc[:4][["hp", "wt"]]
    expected = nns_frame_call(
        "labels.reg.point_names",
        "NNS.reg(x, y, point.est = pe, ncores = 1)$Point.est",
        {"x": _spec(x), "y": _vec(cars.mpg), "pe": _spec(permuted)},
    )
    actual = nns_reg(x, cars.mpg, point_est=permuted)
    np.testing.assert_allclose(actual["Point.est"], _num(expected), atol=COMPOUND)
    np.testing.assert_allclose(
        actual["Point.est"], nns_reg(x, cars.mpg, point_est=x.iloc[:4])["Point.est"]
    )
    # A named row (df.iloc[i]) is a named vector: also matched by name.
    row = nns_reg(x, cars.mpg, point_est=permuted.iloc[0])["Point.est"]
    np.testing.assert_allclose(row, actual["Point.est"][:1])


def test_point_est_partial_name_overlap_is_an_error() -> None:
    cars = _cars()
    x = cars[["wt", "hp"]]
    bad = cars[["wt", "disp"]].iloc[:2]
    with pytest.raises(ValueError, match=r"Missing: hp; extra: disp"):
        nns_reg(x, cars.mpg, point_est=bad)
    # Disjoint names (and unnamed arrays) are positional, as in R.
    other = x.iloc[:2].set_axis(["p", "q"], axis=1)
    np.testing.assert_allclose(
        nns_reg(x, cars.mpg, point_est=other)["Point.est"],
        nns_reg(x, cars.mpg, point_est=x.iloc[:2].to_numpy())["Point.est"],
    )


# --------------------------------------------------------------------------
# REG-12 / REG-15: declared response and predictor levels
# --------------------------------------------------------------------------

def test_nns_reg_categorical_response_uses_declared_levels() -> None:
    flowers = _flowers()
    levels = ["virginica", "versicolor", "setosa"]
    y = pd.Series(pd.Categorical(flowers.species, categories=levels))
    x = flowers[["a", "b"]]
    expected = nns_frame_call(
        "labels.reg.factor_y",
        "r <- NNS.reg(x, y, point.est = x[1:6, ], ncores = 1); "
        "list(pe = r$Point.est, levels = r$class.levels, r2 = r$R2)",
        {"x": _spec(x), "y": _vec(y)},
    )
    assert isinstance(expected, dict)
    actual = nns_reg(x, y, point_est=x.iloc[:6])
    assert actual["class.levels"] == levels == expected["levels"]
    np.testing.assert_allclose(actual["Point.est"], _num(expected["pe"]), atol=COMPOUND)
    np.testing.assert_allclose(actual["R2"], _num(expected["r2"]), atol=COMPOUND)
    # class_levels= declares the same order for plain strings.
    by_levels = nns_reg(x, flowers.species, point_est=x.iloc[:6], class_levels=levels)
    np.testing.assert_allclose(by_levels["Point.est"], actual["Point.est"])
    assert by_levels["class.levels"] == levels


@pytest.mark.parametrize("dummy", [True, False])
def test_nns_reg_categorical_predictor_uses_declared_levels(dummy: bool) -> None:
    cars = _cars()
    cyl = pd.Categorical(cars.cyl, categories=["four", "six", "eight"])
    x = pd.DataFrame({"wt": cars.wt, "cyl": cyl})
    expected = nns_frame_call(
        "labels.reg.factor_x",
        "r <- NNS.reg(x, y, point.est = x[c(1, 5, 9), ], factor.2.dummy = d, ncores = 1); "
        "list(pe = r$Point.est, r2 = r$R2, names = names(r$RPM))",
        {"x": _spec(x), "y": _vec(cars.mpg), "d": r_column([dummy], "logical")},
    )
    assert isinstance(expected, dict)
    actual = nns_reg(x, cars.mpg, point_est=x.iloc[[0, 4, 8]], factor_2_dummy=dummy)
    np.testing.assert_allclose(actual["Point.est"], _num(expected["pe"]), atol=COMPOUND)
    np.testing.assert_allclose(actual["R2"], _num(expected["r2"]), atol=COMPOUND)
    assert list(actual["RPM"]) == list(expected["names"])
    # factor_levels= gives the same coding for a plain string column.
    plain = pd.DataFrame({"wt": cars.wt, "cyl": cars.cyl})
    declared = nns_reg(
        plain, cars.mpg, point_est=plain.iloc[[0, 4, 8]], factor_2_dummy=dummy,
        factor_levels={"cyl": ["four", "six", "eight"]},
    )
    np.testing.assert_allclose(declared["Point.est"], actual["Point.est"])


def test_numeric_looking_strings_are_factor_levels() -> None:
    cars = _cars()
    gear = pd.Series(np.where(cars.hp < 150, "3", np.where(cars.hp < 250, "4", "10")))
    x = pd.DataFrame({"gear": gear, "wt": cars.wt})
    expected = nns_frame_call(
        "labels.reg.numeric_strings",
        "r <- NNS.reg(x, y, ncores = 1); list(r2 = r$R2, names = names(r$RPM))",
        {"x": _spec(x), "y": _vec(cars.mpg)},
    )
    assert isinstance(expected, dict)
    actual = nns_reg(x, cars.mpg)
    # R orders character levels by first appearance; the port sorts them, which
    # only permutes the dummy columns (same fit).
    assert sorted(actual["RPM"]) == sorted(expected["names"])
    np.testing.assert_allclose(actual["R2"], _num(expected["r2"]), atol=COMPOUND)


def test_integer_codes_with_class_levels_are_labelled_codes() -> None:
    # y already coded 1..K plus the labels of those codes (an R factor's codes
    # and levels): the codes are kept and class_levels names them.
    flowers = _flowers()
    levels = ["setosa", "versicolor", "virginica"]
    codes = np.asarray([levels.index(v) + 1.0 for v in flowers.species])
    x = flowers[["a", "b"]]
    by_code = nns_reg(x, codes, point_est=x.iloc[:6], class_levels=levels)
    by_label = nns_reg(x, flowers.species, point_est=x.iloc[:6], class_levels=levels)
    assert by_code["class.levels"] == levels
    np.testing.assert_array_equal(by_code["Point.est"], by_label["Point.est"])
    with pytest.raises(ValueError, match="absent from class_levels"):
        nns_reg(x, np.r_[codes[:-1], 4.0], class_levels=levels)


def test_object_numeric_response_is_regression() -> None:
    cars = _cars()
    as_object = nns_reg(cars.wt.to_numpy(), cars.mpg.to_numpy().astype(object))
    numeric = nns_reg(cars.wt.to_numpy(), cars.mpg.to_numpy())
    assert as_object["class.levels"] is None
    np.testing.assert_allclose(as_object["R2"], numeric["R2"])
    np.testing.assert_allclose(as_object["Fitted.xy"]["y.hat"], numeric["Fitted.xy"]["y.hat"])


def test_missing_values_in_categorical_columns_raise() -> None:
    cars = _cars()
    g = pd.Series(cars.cyl.to_numpy(dtype=object))
    g.iloc[3] = None
    with pytest.raises(ValueError, match="Predictor 'g' contains missing values"):
        nns_reg(pd.DataFrame({"g": g, "wt": cars.wt}), cars.mpg)
    good = pd.DataFrame({"g": cars.cyl, "wt": cars.wt})
    with pytest.raises(ValueError, match=r"\[point.est\] predictor 'g' contains missing"):
        nns_reg(good, cars.mpg, point_est=pd.DataFrame({"g": [None], "wt": [3.0]}))
    with pytest.raises(ValueError, match=r"\[y\] contains missing values"):
        nns_reg(cars.wt, g)
    with pytest.raises(ValueError, match="contains missing values"):
        nns_stack(pd.DataFrame({"g": g, "wt": cars.wt}), cars.mpg, method=1)


# --------------------------------------------------------------------------
# REG-15 / REG-16: nns_m_reg
# --------------------------------------------------------------------------

def test_nns_m_reg_raw_factor_frame_matches_r() -> None:
    cars = _cars()
    x = cars[["cyl", "wt"]]
    expected = nns_frame_call(
        "labels.mreg.raw_factor",
        "r <- NNS:::NNS.M.reg(x, y, point.est = x[1:3, ], ncores = 1); "
        "list(pe = r$Point.est, r2 = r$R2, names = names(r$RPM), nbest = r$n.best)",
        {"x": _spec(x), "y": _vec(cars.mpg)},
    )
    assert isinstance(expected, dict)
    actual = nns_m_reg(x, cars.mpg, point_est=x.iloc[:3])
    # Sorted vs first-appearance level order only permutes the dummy columns.
    assert sorted(actual["RPM"]) == sorted(expected["names"])
    np.testing.assert_allclose(actual["Point.est"], _num(expected["pe"]), atol=COMPOUND)
    np.testing.assert_allclose(actual["R2"], _num(expected["r2"]), atol=COMPOUND)
    assert actual["n.best"] == int(_num(expected["nbest"]))
    # Column names are kept for numeric frames too.
    assert list(nns_m_reg(cars[["wt", "hp"]], cars.mpg)["RPM"]) == ["wt", "hp", "y.hat"]


def test_nns_m_reg_class_levels_stay_classification() -> None:
    flowers = _flowers()
    levels = ["virginica", "versicolor", "setosa"]
    x = flowers[["a", "b"]]
    y = pd.Series(pd.Categorical(flowers.species, categories=levels))
    expected = nns_frame_call(
        "labels.mreg.factor_y",
        "r <- NNS:::NNS.M.reg(x, y, point.est = x[1:6, ], ncores = 1); "
        "list(pe = r$Point.est, r2 = r$R2, levels = r$class.levels)",
        {"x": _spec(x), "y": _vec(y)},
    )
    assert isinstance(expected, dict)
    actual = nns_m_reg(x, flowers.species, point_est=x.iloc[:6], class_levels=levels)
    assert actual["class.levels"] == levels == expected["levels"]
    np.testing.assert_allclose(actual["Point.est"], _num(expected["pe"]), atol=COMPOUND)
    np.testing.assert_allclose(actual["R2"], _num(expected["r2"]), atol=COMPOUND)
    assert set(np.unique(actual["Fitted.xy"]["y.hat"])) <= {1.0, 2.0, 3.0}


# --------------------------------------------------------------------------
# STK-08 / STK-04 / STK-07: nns_stack
# --------------------------------------------------------------------------

def test_nns_stack_test_columns_matched_by_name_and_named_outputs() -> None:
    cars = _cars()
    x = cars[["wt", "hp", "disp"]]
    test = x.iloc[:5][["disp", "hp", "wt"]]
    expected = nns_frame_call(
        "labels.stack.names",
        "r <- NNS.stack(x, y, IVs.test = te, folds = 2, status = FALSE, ncores = 1); "
        "list(stack = r$stack, vars = r$NNS.dim.red.variables, "
        "coef = r$NNS.dim.red.coefficients)",
        {"x": _spec(x), "y": _vec(cars.mpg), "te": _spec(test)},
    )
    assert isinstance(expected, dict)
    actual = nns_stack(x, cars.mpg, test, folds=2)
    np.testing.assert_allclose(actual["stack"], _num(expected["stack"]), atol=COMPOUND)
    unpermuted = nns_stack(x, cars.mpg, x.iloc[:5], folds=2)
    np.testing.assert_allclose(actual["stack"], unpermuted["stack"])
    assert actual["NNS.dim.red.variables"] == list(np.atleast_1d(expected["vars"]))
    coef = expected["coef"]
    assert isinstance(coef, dict)
    assert list(actual["NNS.dim.red.coefficients"]) == list(coef["names"])
    np.testing.assert_allclose(
        list(actual["NNS.dim.red.coefficients"].values()), _num(coef["values"]),
        atol=COMPOUND,
    )


def test_nns_stack_categorical_response_uses_declared_levels() -> None:
    flowers = _flowers()
    levels = ["setosa", "versicolor", "virginica"]
    x = flowers[["a", "b", "c"]]
    y = pd.Series(pd.Categorical(flowers.species, categories=levels))
    expected = nns_frame_call(
        "labels.stack.factor_y",
        "r <- NNS.stack(x[1:50, ], y[1:50], IVs.test = x[51:60, ], folds = 3, "
        "status = FALSE, ncores = 1); "
        "list(stack = r$stack, reg = r$reg, levels = r$class.levels, w = r$weights)",
        {"x": _spec(x), "y": _vec(y)},
    )
    assert isinstance(expected, dict)
    assert flowers.species.iloc[0] == "virginica"
    actual = nns_stack(x.iloc[:50], y.iloc[:50], x.iloc[50:60], folds=3)
    assert actual["class.levels"] == levels == expected["levels"]
    np.testing.assert_allclose(actual["reg"], _num(expected["reg"]), atol=COMPOUND)
    np.testing.assert_allclose(actual["stack"], _num(expected["stack"]), atol=COMPOUND)
    # Plain strings are coded in sorted order (R's factor order), and
    # class_levels= declares an order explicitly.
    plain = nns_stack(x.iloc[:50], flowers.species.iloc[:50], x.iloc[50:60], folds=3)
    np.testing.assert_allclose(plain["stack"], actual["stack"])
    reverse = nns_stack(
        x.iloc[:50], flowers.species.iloc[:50], x.iloc[50:60], folds=3,
        class_levels=levels[::-1],
    )
    assert reverse["class.levels"] == levels[::-1]
    # Integer codes with class_levels naming them.
    codes = np.asarray([levels.index(v) + 1.0 for v in flowers.species.iloc[:50]])
    by_code = nns_stack(x.iloc[:50], codes, x.iloc[50:60], folds=3, class_levels=levels)
    assert by_code["class.levels"] == levels
    np.testing.assert_allclose(by_code["stack"], actual["stack"])


def test_nns_stack_factor_predictor_levels_and_names() -> None:
    cars = _cars()
    flag = np.where(cars.disp > cars.disp.median(), "1", "0")
    x = pd.DataFrame({"wt": cars.wt, "flag": flag, "hp": cars.hp})
    declared = x.assign(flag=pd.Categorical(flag, categories=["1", "0"]))
    expected = nns_frame_call(
        "labels.stack.factor_x",
        "r <- NNS.stack(x, y, IVs.test = x[1:5, ], folds = 2, status = FALSE, ncores = 1); "
        "list(stack = r$stack, coef = r$NNS.dim.red.coefficients)",
        {"x": _spec(declared), "y": _vec(cars.mpg)},
    )
    assert isinstance(expected, dict)
    actual = nns_stack(declared, cars.mpg, declared.iloc[:5], folds=2)
    coef = expected["coef"]
    assert isinstance(coef, dict)
    # The "0"/"1" strings are a factor (one dummy per level), not a number.
    assert list(actual["NNS.dim.red.coefficients"]) == list(coef["names"])
    assert list(coef["names"]) == ["wt", "flag__X1", "flag__X0", "hp"]
    np.testing.assert_allclose(actual["stack"], _num(expected["stack"]), atol=COMPOUND)
    by_levels = nns_stack(
        x, cars.mpg, x.iloc[:5], folds=2, factor_levels={"flag": ["1", "0"]}
    )
    np.testing.assert_allclose(by_levels["stack"], actual["stack"])


def test_nns_stack_univariate_all_n_best_is_na_like_r() -> None:
    rs = np.random.RandomState(5)
    n = 200
    a = rs.normal(size=n)
    rs.normal(size=n)
    rs.normal(size=n)
    y = rs.normal(size=n)
    train = pd.DataFrame({"a": a[:160]})
    test = pd.DataFrame({"a": a[160:]})
    expected = nns_frame_call(
        "labels.stack.nbest_all",
        "r <- NNS.stack(x, y, IVs.test = te, status = FALSE, ncores = 1); "
        "list(nbest = r$NNS.reg.n.best, reg = r$reg)",
        {"x": _spec(train), "y": r_column(y[:160].tolist(), "numeric"), "te": _spec(test)},
    )
    assert isinstance(expected, dict)
    actual = nns_stack(train, y[:160], test)
    assert np.isnan(_num(expected["nbest"]))
    assert np.isnan(actual["NNS.reg.n.best"])
    np.testing.assert_allclose(actual["reg"], _num(expected["reg"]), atol=COMPOUND)


# --------------------------------------------------------------------------
# STK-08: nns_boost feature names
# --------------------------------------------------------------------------

def test_nns_boost_feature_names_are_column_names() -> None:
    cars = _cars()
    x = cars[["wt", "hp", "disp"]]
    expected = nns_frame_call(
        "labels.boost.names",
        "r <- NNS.boost(x[1:30, ], y[1:30], IVs.test = x[31:40, ], learner.trials = 7, "
        "epochs = 10, CV.size = 0.25, status = FALSE, feature.importance = FALSE); "
        "list(results = r$results, freq = r$feature.frequency)",
        {"x": _spec(x), "y": _vec(cars.mpg)},
    )
    assert isinstance(expected, dict)
    actual = nns_boost(x.iloc[:30], cars.mpg.iloc[:30], x.iloc[30:], learner_trials=7,
                       epochs=10, cv_size=0.25)
    freq = expected["freq"]
    assert isinstance(freq, dict)
    assert list(actual["feature.frequency"]) == list(freq["names"])
    np.testing.assert_allclose(
        list(actual["feature.frequency"].values()), _num(freq["values"])
    )
    np.testing.assert_allclose(actual["results"], _num(expected["results"]), atol=COMPOUND)
