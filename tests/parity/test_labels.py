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

from nns import nns_reg
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
