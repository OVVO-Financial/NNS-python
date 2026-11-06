from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from _r import nns_stack_factor_predictor, nns_stack_mixed_factor_predictor, nns_stack_numeric
from _tolerances import COMPOUND

from nns import nns_stack


@pytest.mark.parity
@pytest.mark.parametrize("method", [[1], [2], [1, 2]])
@pytest.mark.parametrize("stack", [True, False])
def test_nns_stack_numeric_matches_r(method: list[int], stack: bool) -> None:
    x = np.linspace(-2.0, 2.0, 30)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = x + np.sin(x) + 0.25 * np.cos(x)
    point = variable[:5]

    expected = nns_stack_numeric(
        variable.tolist(),
        y.tolist(),
        point.tolist(),
        cv_size=0.25,
        folds=2,
        method=method,
        order=None,
        stack=stack,
        dim_red_method="cor",
    )
    actual = nns_stack(
        variable,
        y,
        point,
        cv_size=0.25,
        folds=2,
        method=method,
        stack=stack,
        dim_red_method="cor",
        dist=None,
    )

    _assert_stack_matches(actual, expected, exact_probability_threshold=False)


@pytest.mark.parity
def test_nns_stack_equal_dim_red_matches_r() -> None:
    x = np.linspace(-1.5, 1.5, 36)
    variable = np.column_stack((x, x**2, np.sin(x)))
    y = 0.5 * x + x**2 - 0.25 * np.sin(x)
    point = variable[::9]

    expected = nns_stack_numeric(
        variable.tolist(),
        y.tolist(),
        point.tolist(),
        cv_size=0.25,
        folds=2,
        method=[2],
        order=2,
        stack=False,
        dim_red_method="equal",
    )
    actual = nns_stack(
        variable,
        y,
        point,
        cv_size=0.25,
        folds=2,
        method=2,
        order=2,
        stack=False,
        dim_red_method="equal",
        dist=None,
    )

    _assert_stack_matches(actual, expected, exact_probability_threshold=False)


@pytest.mark.parity
def test_nns_stack_factor_predictor_method1_matches_r() -> None:
    x = np.asarray(["b", "a", "b", "c", "a", "c", "b", "a"])
    y = np.asarray([2.0, 1.0, 3.0, 4.0, 1.5, 3.5, 2.5, 1.25])
    point = np.asarray(["a", "c", "b"])
    levels = ["a", "b", "c"]

    expected = nns_stack_factor_predictor(
        x.tolist(),
        y.tolist(),
        point.tolist(),
        levels=levels,
        cv_size=0.25,
        folds=1,
        method=[1],
        order=None,
        stack=True,
        dim_red_method="cor",
    )
    actual = nns_stack(
        x,
        y,
        point,
        factor_levels=levels,
        cv_size=0.25,
        folds=1,
        method=1,
        stack=True,
        dim_red_method="cor",
        dist=None,
    )

    _assert_stack_matches(actual, expected, exact_probability_threshold=False)


@pytest.mark.parity
def test_nns_stack_factor_predictor_method2_factor_only_matches_r_fallback() -> None:
    x = np.asarray(["b", "a", "b", "c", "a", "c", "b", "a"])
    y = np.asarray([2.0, 1.0, 3.0, 4.0, 1.5, 3.5, 2.5, 1.25])
    point = np.asarray(["a", "c", "b"])
    levels = ["a", "b", "c"]

    expected = nns_stack_factor_predictor(
        x.tolist(),
        y.tolist(),
        point.tolist(),
        levels=levels,
        cv_size=0.25,
        folds=1,
        method=[2],
        order=None,
        stack=True,
        dim_red_method="cor",
    )
    actual = nns_stack(
        x,
        y,
        point,
        factor_levels=levels,
        cv_size=0.25,
        folds=1,
        method=2,
        stack=True,
        dim_red_method="cor",
        dist=None,
    )

    _assert_stack_matches(actual, expected, exact_probability_threshold=False)


@pytest.mark.parity
def test_nns_stack_factor_predictor_method12_factor_only_matches_r_fallback() -> None:
    x = np.asarray(["b", "a", "b", "c", "a", "c", "b", "a"])
    y = np.asarray([2.0, 1.0, 3.0, 4.0, 1.5, 3.5, 2.5, 1.25])
    point = np.asarray(["a", "c", "b"])
    levels = ["a", "b", "c"]

    expected = nns_stack_factor_predictor(
        x.tolist(),
        y.tolist(),
        point.tolist(),
        levels=levels,
        cv_size=0.25,
        folds=1,
        method=[1, 2],
        order=None,
        stack=True,
        dim_red_method="cor",
    )
    actual = nns_stack(
        x,
        y,
        point,
        factor_levels=levels,
        cv_size=0.25,
        folds=1,
        method=(1, 2),
        stack=True,
        dim_red_method="cor",
        dist=None,
    )

    _assert_stack_matches(actual, expected, exact_probability_threshold=False)


@pytest.mark.parity
def test_nns_stack_mixed_factor_predictor_method2_matches_r() -> None:
    x = np.asarray(["b", "a", "b", "c", "a", "c", "b", "a"], dtype=object)
    z = np.arange(1, x.size + 1, dtype=np.float64) / 10.0
    variable = np.column_stack((x, z.astype(object)))
    y = np.asarray([2.0, 1.0, 3.0, 4.0, 1.5, 3.5, 2.5, 1.25])
    point_factor = np.asarray(["a", "c", "b"], dtype=object)
    point_z = np.asarray([0.15, 0.55, 0.75], dtype=object)
    point = np.column_stack((point_factor, point_z))
    levels = ["a", "b", "c"]

    expected = nns_stack_mixed_factor_predictor(
        x.tolist(),
        z.tolist(),
        y.tolist(),
        point_factor.tolist(),
        [0.15, 0.55, 0.75],
        levels=levels,
        cv_size=0.25,
        folds=1,
        method=[2],
        order=None,
        stack=True,
        dim_red_method="cor",
    )
    actual = nns_stack(
        variable,
        y,
        point,
        factor_levels=(levels, None),
        cv_size=0.25,
        folds=1,
        method=2,
        stack=True,
        dim_red_method="cor",
        dist=None,
    )

    _assert_stack_matches(
        actual, expected, exact_probability_threshold=False, predictor_names=["x", "z"]
    )


@pytest.mark.parity
def test_nns_stack_mixed_factor_predictor_method12_matches_r() -> None:
    x = np.asarray(["b", "a", "b", "c", "a", "c", "b", "a"], dtype=object)
    z = np.arange(1, x.size + 1, dtype=np.float64) / 10.0
    variable = np.column_stack((x, z.astype(object)))
    y = np.asarray([2.0, 1.0, 3.0, 4.0, 1.5, 3.5, 2.5, 1.25])
    point_factor = np.asarray(["a", "c", "b"], dtype=object)
    point_z = np.asarray([0.15, 0.55, 0.75], dtype=object)
    point = np.column_stack((point_factor, point_z))
    levels = ["a", "b", "c"]

    expected = nns_stack_mixed_factor_predictor(
        x.tolist(),
        z.tolist(),
        y.tolist(),
        point_factor.tolist(),
        [0.15, 0.55, 0.75],
        levels=levels,
        cv_size=0.25,
        folds=1,
        method=[1, 2],
        order=None,
        stack=True,
        dim_red_method="cor",
    )
    actual = nns_stack(
        variable,
        y,
        point,
        factor_levels=(levels, None),
        cv_size=0.25,
        folds=1,
        method=(1, 2),
        stack=True,
        dim_red_method="cor",
        dist=None,
    )

    _assert_stack_matches(
        actual, expected, exact_probability_threshold=False, predictor_names=["x", "z"]
    )


@pytest.mark.parity
@pytest.mark.parametrize(
    ("method", "ts_test"),
    [([1], 5), ([1], 10), ([2], 5), ([2], 10), ([1, 2], 10)],
)
def test_nns_stack_ts_test_matches_r(method: list[int], ts_test: int) -> None:
    x = np.linspace(-2.0, 2.0, 40)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = x + np.sin(x) + 0.25 * np.cos(x)
    point = variable[:5]

    expected = nns_stack_numeric(
        variable.tolist(),
        y.tolist(),
        point.tolist(),
        cv_size=0.25,
        folds=1,
        method=method,
        order=None,
        stack=True,
        dim_red_method="cor",
        ts_test=ts_test,
    )
    actual = nns_stack(
        variable,
        y,
        point,
        cv_size=0.25,
        folds=1,
        method=method,
        stack=True,
        dim_red_method="cor",
        ts_test=ts_test,
        dist=None,
    )

    _assert_stack_matches(actual, expected, exact_probability_threshold=False)


@pytest.mark.parity
def test_nns_stack_var_like_ts_test_matches_r() -> None:
    h = 5
    x = np.linspace(-2.0, 2.0, 40)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = x + np.sin(x) + 0.25 * np.cos(x)
    point = variable[-h:]
    ts_test = max(2 * h, int(0.2 * y.size))

    expected = nns_stack_numeric(
        variable.tolist(),
        y.tolist(),
        point.tolist(),
        cv_size=0.25,
        folds=1,
        method=[1, 2],
        order=None,
        stack=True,
        dim_red_method="cor",
        ts_test=ts_test,
    )
    actual = nns_stack(
        variable,
        y,
        point,
        cv_size=0.25,
        folds=1,
        method=(1, 2),
        stack=True,
        dim_red_method="cor",
        ts_test=ts_test,
        dist=None,
    )

    _assert_stack_matches(actual, expected, exact_probability_threshold=False)


@pytest.mark.parity
@pytest.mark.parametrize("dim_red_method", ["cor", "nns.dep", "nns.caus", "all"])
def test_nns_stack_dim_red_selection_matches_r(dim_red_method: str) -> None:
    # y is driven by the first two predictors; the third is noise. The test
    # rows sit beyond the training range, as NNS.VAR forecast rows do.
    rng = np.random.default_rng(11)
    n, h = 60, 4
    t = np.arange(n + h, dtype=np.float64)
    drift = np.cumsum(rng.standard_normal(n + h))
    variable = np.column_stack((drift, np.sin(t / 3.0), rng.standard_normal(n + h)))
    y = 0.8 * drift + 0.5 * np.sin(t / 3.0) + 0.1 * rng.standard_normal(n + h)
    ts_test = max(2 * h, int(np.ceil(0.2 * n)))

    expected = nns_stack_numeric(
        variable[:n].tolist(),
        y[:n].tolist(),
        variable[n:].tolist(),
        cv_size=0.25,
        folds=1,
        method=[1, 2],
        order=None,
        stack=True,
        dim_red_method=dim_red_method,
        ts_test=ts_test,
    )
    actual = nns_stack(
        variable[:n],
        y[:n],
        variable[n:],
        folds=1,
        method=(1, 2),
        stack=True,
        dim_red_method=dim_red_method,
        ts_test=ts_test,
        dist=None,
    )

    assert isinstance(expected, dict)
    assert _SELECTION_KEYS <= set(expected)
    _assert_stack_matches(actual, expected)


@pytest.mark.parity
@pytest.mark.parametrize("folds", [3, 5])
def test_nns_stack_boston_cor_partitions_match_r(folds: int) -> None:
    # The cor coefficients and the X* projections feed NNS partitions, where a
    # last-ulp difference can move a regression point. The port reproduces R's
    # long-double cor and its summation orders so the folds partition alike.
    data = np.genfromtxt(
        Path(__file__).resolve().parents[1] / "data" / "boston_housing.csv",
        delimiter=",",
        skip_header=1,
    )
    x, y = data[:356, :-1], data[:356, -1]
    x_test = data[356:380, :-1]

    expected = nns_stack_numeric(
        x.tolist(),
        y.tolist(),
        x_test.tolist(),
        cv_size=0.25,
        folds=folds,
        method=[1, 2],
        order=None,
        stack=True,
        dim_red_method="cor",
        seed=123,
    )
    actual = nns_stack(
        x,
        y,
        x_test,
        cv_size=0.25,
        folds=folds,
        method=(1, 2),
        stack=True,
        dim_red_method="cor",
        dist=None,
        seed=123,
    )

    _assert_stack_matches(actual, expected)


@pytest.mark.parity
@pytest.mark.parametrize("method", [[1], [2], [1, 2]])
def test_nns_stack_pred_int_matches_r(method: list[int]) -> None:
    x = np.linspace(-2.0, 2.0, 40)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = x + np.sin(x) + 0.25 * np.cos(x)
    point = variable[:5]

    expected = nns_stack_numeric(
        variable.tolist(),
        y.tolist(),
        point.tolist(),
        cv_size=0.25,
        folds=1,
        method=method,
        order=None,
        stack=True,
        dim_red_method="cor",
        pred_int=0.95,
    )
    actual = nns_stack(
        variable,
        y,
        point,
        cv_size=0.25,
        folds=1,
        method=method,
        stack=True,
        dim_red_method="cor",
        pred_int=0.95,
        dist=None,
    )

    _assert_stack_matches(actual, expected, exact_probability_threshold=False)


@pytest.mark.parity
@pytest.mark.parametrize("method", [[1], [2], [1, 2]])
def test_nns_stack_binary_class_matches_r(method: list[int]) -> None:
    x = np.linspace(-2.0, 2.0, 36)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = np.where(x + np.sin(x) > 0.0, 2.0, 1.0)
    point = variable[::9]

    expected = nns_stack_numeric(
        variable.tolist(),
        y.tolist(),
        point.tolist(),
        cv_size=0.25,
        folds=1,
        method=method,
        order=None,
        stack=True,
        dim_red_method="cor",
        type="class",
    )
    actual = nns_stack(
        variable,
        y,
        point,
        cv_size=0.25,
        folds=1,
        method=method,
        stack=True,
        dim_red_method="cor",
        type="class",
        dist=None,
    )

    _assert_stack_matches(actual, expected, exact_probability_threshold=False)


@pytest.mark.parity
@pytest.mark.parametrize("method", [[1], [2], [1, 2]])
def test_nns_stack_binary_class_pred_int_matches_r(method: list[int]) -> None:
    x = np.linspace(-2.0, 2.0, 36)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = np.where(x + np.sin(x) > 0.0, 2.0, 1.0)
    point = variable[::9]

    expected = nns_stack_numeric(
        variable.tolist(),
        y.tolist(),
        point.tolist(),
        cv_size=0.25,
        folds=1,
        method=method,
        order=None,
        stack=True,
        dim_red_method="cor",
        type="class",
        pred_int=0.95,
    )
    actual = nns_stack(
        variable,
        y,
        point,
        cv_size=0.25,
        folds=1,
        method=method,
        stack=True,
        dim_red_method="cor",
        type="class",
        pred_int=0.95,
        dist=None,
    )

    _assert_stack_matches(actual, expected, exact_probability_threshold=False)
    assert isinstance(actual["pred.int"], dict)
    assert all(values.shape == actual["stack"].shape for values in actual["pred.int"].values())
    if method == [1, 2]:
        assert set(actual["pred.int"]) == set(actual["reg.pred.int"])
        for values in actual["pred.int"].values():
            np.testing.assert_allclose(values, np.round(values))


@pytest.mark.parity
@pytest.mark.parametrize("method", [[1], [2], [1, 2]])
def test_nns_stack_multiclass_matches_r(method: list[int]) -> None:
    x = np.linspace(-2.0, 2.0, 36)
    variable = np.column_stack((x, x**2, np.sin(x)))
    y = np.where(x < -0.5, 1.0, np.where(x > 0.75, 3.0, 2.0))
    point = variable[[0, 7, 18, 31]]

    expected = nns_stack_numeric(
        variable.tolist(),
        y.tolist(),
        point.tolist(),
        cv_size=0.25,
        folds=1,
        method=method,
        order=1,
        stack=True,
        dim_red_method="cor",
        type="class",
    )
    actual = nns_stack(
        variable,
        y,
        point,
        cv_size=0.25,
        folds=1,
        method=method,
        order=1,
        stack=True,
        dim_red_method="cor",
        type="class",
        dist=None,
    )

    _assert_stack_matches(actual, expected, exact_probability_threshold=False)


@pytest.mark.parity
def test_nns_stack_factor_like_class_pred_int_matches_r() -> None:
    x = np.linspace(-2.0, 2.0, 30)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    labels = np.where(x < -0.5, "A", np.where(x > 0.75, "C", "B"))
    point = variable[::10]

    expected = nns_stack_numeric(
        variable.tolist(),
        labels.tolist(),
        point.tolist(),
        cv_size=0.25,
        folds=1,
        method=[1, 2],
        order=1,
        stack=True,
        dim_red_method="cor",
        type="class",
        class_levels=["A", "B", "C"],
        pred_int=0.95,
    )
    actual = nns_stack(
        variable,
        labels,
        point,
        cv_size=0.25,
        folds=1,
        method=(1, 2),
        order=1,
        stack=True,
        dim_red_method="cor",
        type="class",
        class_levels=["A", "B", "C"],
        pred_int=0.95,
        dist=None,
    )

    _assert_stack_matches(actual, expected, exact_probability_threshold=False)


@pytest.mark.parity
def test_nns_stack_factor_like_class_matches_r() -> None:
    x = np.linspace(-2.0, 2.0, 30)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    labels = np.where(x < -0.5, "A", np.where(x > 0.75, "C", "B"))
    point = variable[::10]

    expected = nns_stack_numeric(
        variable.tolist(),
        labels.tolist(),
        point.tolist(),
        cv_size=0.25,
        folds=1,
        method=[1, 2],
        order=1,
        stack=True,
        dim_red_method="cor",
        type="class",
        class_levels=["A", "B", "C"],
    )
    actual = nns_stack(
        variable,
        labels,
        point,
        cv_size=0.25,
        folds=1,
        method=(1, 2),
        order=1,
        stack=True,
        dim_red_method="cor",
        type="class",
        class_levels=["A", "B", "C"],
        dist=None,
    )

    _assert_stack_matches(actual, expected, exact_probability_threshold=False)


def test_nns_stack_raw_character_class_encodes_like_r() -> None:
    # R's NNS.stack codes character class labels 1..K (base 1) natively; the
    # port matches and exposes the mapping via class.levels.
    x = np.linspace(-2.0, 2.0, 20)
    variable = np.column_stack((x, np.sin(x)))
    labels = np.where(x > 0.0, "B", "A")

    result = nns_stack(variable, labels, variable[:3], type="class", cv_size=0.25, folds=1)

    assert list(result["class.levels"]) == ["A", "B"]
    codes = np.asarray(result["stack"], dtype=np.float64)
    assert np.all(np.isin(codes[np.isfinite(codes)], [1.0, 2.0]))


@pytest.mark.parity
@pytest.mark.stochastic
@pytest.mark.parametrize("method", [[1], [2], [1, 2]])
def test_nns_stack_balance_binary_class_matches_r_structure(method: list[int]) -> None:
    x = np.linspace(-2.0, 2.0, 48)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = np.where(x < 1.0, 1.0, 2.0)
    point = variable[[2, 12, 28, 42]]

    expected = nns_stack_numeric(
        variable.tolist(),
        y.tolist(),
        point.tolist(),
        cv_size=0.25,
        folds=1,
        method=method,
        order=None,
        stack=True,
        dim_red_method="cor",
        type="class",
        balance=True,
        seed=42,
    )
    actual = nns_stack(
        variable,
        y,
        point,
        cv_size=0.25,
        folds=1,
        method=method,
        stack=True,
        dim_red_method="cor",
        type="class",
        balance=True,
        random_seed=42,
        dist=None,
    )

    _assert_stack_class_structure(actual, expected, point_rows=point.shape[0], classes=np.unique(y))


@pytest.mark.parity
@pytest.mark.parametrize("columns", [1, 2])
def test_nns_stack_all_candidate_is_scored_with_the_deployed_estimator(columns: int) -> None:
    # The Method 1 "ALL" candidate was scored out of fold as the
    # constant training mean, but the deployed model is the univariate NNS.reg fit
    # or NNS.reg(n.best = "all"). On pure noise ALL used to win with the SSE of a
    # constant (the same OBJfn.reg for one and two columns).
    rng = np.random.default_rng(5)
    variable = np.round(rng.standard_normal((200, 2)), 8)[:, :columns]
    y = np.round(rng.standard_normal(200), 8)

    expected = nns_stack_numeric(
        variable[:160].tolist(),
        y[:160].tolist(),
        variable[160:].tolist(),
        cv_size=0.25,
        folds=5,
        method=[1],
        order=None,
        stack=False,
        dim_red_method="cor",
        seed=5,
    )
    actual = nns_stack(
        variable[:160],
        y[:160],
        variable[160:],
        cv_size=0.25,
        folds=5,
        method=(1,),
        stack=False,
        dim_red_method="cor",
        random_seed=5,
        dist=None,
    )

    assert isinstance(expected, dict)
    np.testing.assert_allclose(
        actual["OBJfn.reg"], float(np.asarray(expected["OBJfn.reg"])), rtol=1e-10
    )
    np.testing.assert_allclose(actual["reg"], np.asarray(expected["reg"], dtype=np.float64))


@pytest.mark.parity
@pytest.mark.stochastic
def test_nns_stack_balance_keeps_a_single_member_class_like_r() -> None:
    # sample(g, k) on a one-member class g sampled from 1:g and
    # dropped the rare class; R now indexes g[sample.int(length(g), k)], which is
    # what NNS Python's R-sample emulation already did. Seeds now reach R's
    # NNS.stack(seed =) (it re-seeded with its default 123).
    x = np.linspace(-2.0, 2.0, 60)
    variable = np.column_stack((x, np.sin(3.0 * x)))
    y = np.where(x < -0.05, 1.0, 2.0)
    y[[20, 45]] = 3.0
    point = variable[[3, 20, 33, 45, 58]]

    expected = nns_stack_numeric(
        variable.tolist(),
        y.tolist(),
        point.tolist(),
        cv_size=0.25,
        folds=1,
        method=[1],
        order=None,
        stack=False,
        dim_red_method="cor",
        type="class",
        balance=True,
        seed=1,
    )
    actual = nns_stack(
        variable,
        y,
        point,
        cv_size=0.25,
        folds=1,
        method=(1,),
        stack=False,
        dim_red_method="cor",
        type="class",
        balance=True,
        random_seed=1,
        dist=None,
    )

    _assert_stack_class_structure(actual, expected, point_rows=point.shape[0], classes=np.unique(y))
    assert isinstance(expected, dict)
    np.testing.assert_allclose(actual["reg"], np.asarray(expected["reg"], dtype=np.float64))


@pytest.mark.parity
@pytest.mark.stochastic
def test_nns_stack_balance_multiclass_and_factor_structure() -> None:
    x = np.linspace(-2.0, 2.0, 45)
    variable = np.column_stack((x, x**2, np.sin(x)))
    labels = np.where(x < -0.75, "A", np.where(x > 1.0, "C", "B"))
    point = variable[[0, 11, 30, 44]]

    expected = nns_stack_numeric(
        variable.tolist(),
        labels.tolist(),
        point.tolist(),
        cv_size=0.25,
        folds=1,
        method=[1, 2],
        order=1,
        stack=True,
        dim_red_method="cor",
        type="class",
        class_levels=["A", "B", "C"],
        balance=True,
        seed=7,
    )
    actual = nns_stack(
        variable,
        labels,
        point,
        cv_size=0.25,
        folds=1,
        method=(1, 2),
        order=1,
        stack=True,
        dim_red_method="cor",
        type="class",
        class_levels=["A", "B", "C"],
        balance=True,
        random_seed=7,
        dist=None,
    )

    _assert_stack_class_structure(
        actual,
        expected,
        point_rows=point.shape[0],
        classes=np.array([1.0, 2.0, 3.0]),
    )


@pytest.mark.parity
@pytest.mark.stochastic
def test_nns_stack_balance_class_pred_int_matches_r_structure() -> None:
    x = np.linspace(-2.0, 2.0, 48)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = np.where(x < 1.0, 1.0, 2.0)
    point = variable[[2, 12, 28, 42]]

    expected = nns_stack_numeric(
        variable.tolist(),
        y.tolist(),
        point.tolist(),
        cv_size=0.25,
        folds=1,
        method=[1],
        order=None,
        stack=True,
        dim_red_method="cor",
        type="class",
        balance=True,
        seed=42,
        pred_int=0.95,
    )
    actual = nns_stack(
        variable,
        y,
        point,
        cv_size=0.25,
        folds=1,
        method=(1,),
        stack=True,
        dim_red_method="cor",
        type="class",
        balance=True,
        random_seed=42,
        pred_int=0.95,
        dist=None,
    )

    _assert_stack_class_structure(
        actual,
        expected,
        point_rows=point.shape[0],
        classes=np.unique(y),
        expect_pred_int=True,
    )


@pytest.mark.parity
@pytest.mark.stochastic
def test_nns_stack_balance_type_none_forces_class_path() -> None:
    x = np.linspace(-2.0, 2.0, 40)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = np.where(x < 1.25, 1.0, 2.0)
    point = variable[:5]

    expected = nns_stack_numeric(
        variable.tolist(),
        y.tolist(),
        point.tolist(),
        cv_size=0.25,
        folds=1,
        method=[1],
        order=None,
        stack=True,
        dim_red_method="cor",
        type=None,
        balance=True,
        seed=9,
    )
    actual = nns_stack(
        variable,
        y,
        point,
        cv_size=0.25,
        folds=1,
        method=1,
        balance=True,
        random_seed=9,
        dist=None,
    )

    _assert_stack_class_structure(
        actual,
        expected,
        point_rows=point.shape[0],
        classes=np.array([1.0, 2.0]),
    )


def test_nns_stack_balance_raw_character_class_encodes_like_r() -> None:
    # balance=TRUE still codes raw character labels natively, matching R.
    x = np.linspace(-2.0, 2.0, 20)
    variable = np.column_stack((x, np.sin(x)))
    labels = np.where(x > 0.0, "B", "A")

    result = nns_stack(
        variable,
        labels,
        variable[:3],
        type="class",
        cv_size=0.25,
        folds=1,
        balance=True,
        random_seed=1,
    )

    assert list(result["class.levels"]) == ["A", "B"]
    codes = np.asarray(result["stack"], dtype=np.float64)
    assert np.all(np.isin(codes[np.isfinite(codes)], [1.0, 2.0]))


# Out-of-fold objective scores (and the probability threshold derived from the
# OOF ensemble) depend on the validation-split draw. Under random holdout / k-fold
# splits the port's numpy Generator diverges from R's Mersenne sample(), so those
# scalars match only distributionally even though the final (whole-training) fits
# and predictions match R exactly. See the RNG note in nns/stack.py.
_RNG_DEPENDENT_KEYS = {"OBJfn.reg", "OBJfn.dim.red", "probability.threshold"}

# Dimension-reduction selection keys added in NNS 13.2 (NNS.VAR relevant
# variables). Cache entries recorded before they existed lack them; when present
# they are compared, and dedicated tests below always cover them.
_SELECTION_KEYS = {"NNS.dim.red.variables", "NNS.dim.red.coefficients"}


def _assert_selection_matches(
    key: str, actual: Any, expected: Any, predictor_names: list[str] | None = None
) -> None:
    if actual is None:
        assert expected is None or expected == {} or np.asarray(expected).size == 0
        return
    if key == "NNS.dim.red.coefficients":
        # Regression has a single named numeric coefficient vector, whereas
        # R 14.0 classification returns a class-keyed list of *equations*.
        # Compare every equation's variables and numerical coefficients;
        # silently flattening class indices would lose their identity.
        assert isinstance(actual, dict)
        if isinstance(expected, dict) and all(
            isinstance(v, dict) for v in expected.values()
        ):
            assert set(actual) == set(expected)
            for class_name, reference in expected.items():
                equation = actual[class_name]
                assert isinstance(equation, dict)
                assert set(equation) == set(reference)
                assert list(equation["Variable"]) == list(reference["Variable"])
                np.testing.assert_allclose(
                    np.asarray(equation["Coefficient"], dtype=np.float64),
                    _numeric(reference["Coefficient"]),
                    atol=COMPOUND,
                )
        else:
            actual_coef = np.asarray(list(actual.values()), dtype=np.float64)
            np.testing.assert_allclose(actual_coef, _numeric(expected), atol=COMPOUND)
        return
    # Predictor names, as R reports them: V1..Vp for an unnamed matrix. Where
    # the R reference used a named data.frame, map the port's positional
    # V<i> names to those R names.
    expected_names = [str(v) for v in np.atleast_1d(np.asarray(expected, dtype=object))]
    assert all(isinstance(v, str) for v in actual)
    if predictor_names is None:
        assert list(actual) == expected_names
    else:
        assert [predictor_names[int(v[1:]) - 1] for v in actual] == expected_names


def _assert_stack_matches(
    actual: dict[str, Any],
    expected: Any,
    *,
    exact_probability_threshold: bool = True,
    exact_objective: bool = True,
    predictor_names: list[str] | None = None,
) -> None:
    assert isinstance(expected, dict)
    assert set(actual) - _SELECTION_KEYS == set(expected) - _SELECTION_KEYS
    for key in actual:
        if key in _SELECTION_KEYS:
            if key in expected:
                _assert_selection_matches(key, actual[key], expected[key], predictor_names)
            continue
        if key in _RNG_DEPENDENT_KEYS and not exact_objective:
            # Random-split OOF scalar: assert finiteness agrees with R, not value.
            actual_finite = actual[key] is not None and np.isfinite(float(actual[key]))
            expected_finite = bool(np.all(np.isfinite(_numeric(expected[key]))))
            assert actual_finite == expected_finite
            continue
        if key == "probability.threshold" and not exact_probability_threshold:
            assert np.isfinite(float(actual[key]))
            assert 0.0 <= float(actual[key]) <= 1.0
            assert np.isfinite(float(_numeric(expected[key])))
            continue
        if actual[key] is None:
            # R uses a scalar NA_real_ placeholder for an absent method's
            # predictions/interval; the port surfaces that as None.
            expected_value = expected[key]
            if expected_value is None or expected_value == {}:
                continue
            expected_arr = _numeric(expected_value)
            assert expected_arr.size == 0 or np.all(np.isnan(expected_arr))
        elif key == "weights":
            # R returns the named vector c(reg=, dim.red=); jsonlite serializes
            # it as a plain array, while the port keeps the named dict.
            actual_weights = np.array(
                [actual[key]["reg"], actual[key]["dim.red"]], dtype=np.float64
            )
            np.testing.assert_allclose(actual_weights, _numeric(expected[key]), atol=COMPOUND)
        elif key == "class.levels":
            assert [str(v) for v in actual[key]] == [str(v) for v in expected[key]]
        elif isinstance(actual[key], dict):
            assert isinstance(expected[key], dict)
            assert set(actual[key]) == set(expected[key])
            for column, values in actual[key].items():
                np.testing.assert_allclose(
                    np.asarray(values, dtype=np.float64),
                    _numeric(expected[key][column]),
                    atol=COMPOUND,
                )
        else:
            np.testing.assert_allclose(
                np.asarray(actual[key], dtype=np.float64),
                _numeric(expected[key]),
                atol=COMPOUND,
            )


def _numeric(value: object) -> np.ndarray:
    if isinstance(value, str):
        if value == "NA":
            return np.asarray(np.nan, dtype=np.float64)
        if value == "Inf":
            return np.asarray(np.inf, dtype=np.float64)
        if value == "-Inf":
            return np.asarray(-np.inf, dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def _assert_stack_class_structure(
    actual: dict[str, Any],
    expected: Any,
    *,
    point_rows: int,
    classes: np.ndarray,
    expect_pred_int: bool = False,
) -> None:
    assert isinstance(expected, dict)
    assert set(actual) - _SELECTION_KEYS == set(expected) - _SELECTION_KEYS
    for key in ("reg", "dim.red", "stack"):
        if actual[key] is None:
            # Absent method: R returns a scalar NA_real_, the port returns None.
            assert np.all(np.isnan(_numeric(expected[key])))
            continue
        actual_values = np.asarray(actual[key], dtype=np.float64)
        expected_values = _numeric(expected[key])
        if expected_values.shape == ():
            assert actual_values.shape == (point_rows,)
            assert np.all(np.isnan(actual_values))
            assert np.isnan(float(expected_values))
            continue
        assert actual_values.shape == expected_values.shape
        if actual_values.ndim > 0:
            assert actual_values.shape == (point_rows,)
            finite_actual = actual_values[np.isfinite(actual_values)]
            assert np.all(np.isin(finite_actual, classes))
            finite_expected = expected_values[np.isfinite(expected_values)]
            assert np.all(np.isin(finite_expected, classes))
    assert np.isfinite(float(actual["probability.threshold"]))
    assert np.isfinite(float(_numeric(expected["probability.threshold"])))
    for key in ("reg.pred.int", "dim.red.pred.int", "pred.int"):
        actual_pred_int = actual[key]
        expected_pred_int = expected[key]
        if not expect_pred_int or actual_pred_int is None:
            assert actual_pred_int is None
            assert expected_pred_int is None
            continue
        assert isinstance(actual_pred_int, dict)
        assert isinstance(expected_pred_int, dict)
        assert set(actual_pred_int) == set(expected_pred_int)
        for values in actual_pred_int.values():
            assert values.shape == (point_rows,)
            assert np.all(np.isfinite(values))


@pytest.mark.parity
@pytest.mark.parametrize("balance", [False, True])
def test_nns_stack_univariate_class_threshold_uses_the_deployed_fit(balance: bool) -> None:
    # Patched R scores only the deployed univariate nns_reg fit for one predictor
    # (the k-NN candidates it used to rank were never deployed), so the class
    # threshold is tuned on that fit's out-of-fold predictions.
    rng = np.random.default_rng(12)
    x = np.round(np.concatenate([rng.normal(1.5, 0.3, 30), rng.normal(4.3, 0.5, 30),
                                 rng.normal(5.6, 0.6, 30)]), 6)
    y = np.repeat([1.0, 2.0, 3.0], 30)
    test = np.array([2.1, 2.4, 4.9, 5.1])

    expected = nns_stack_numeric(
        x.reshape(-1, 1).tolist(),
        y.tolist(),
        test.reshape(-1, 1).tolist(),
        cv_size=0.25,
        folds=5,
        method=[1],
        order=None,
        stack=False,
        dim_red_method="cor",
        type="CLASS",
        balance=balance,
        seed=1,
    )
    actual = nns_stack(
        x.reshape(-1, 1),
        y,
        test.reshape(-1, 1),
        type="CLASS",
        cv_size=0.25,
        folds=5,
        method=(1,),
        stack=False,
        balance=balance,
        random_seed=1,
    )

    assert isinstance(expected, dict)
    assert np.isnan(actual["NNS.reg.n.best"])
    np.testing.assert_allclose(
        actual["probability.threshold"], float(np.asarray(expected["probability.threshold"]))
    )
    np.testing.assert_allclose(actual["OBJfn.reg"], float(np.asarray(expected["OBJfn.reg"])))
    np.testing.assert_array_equal(actual["reg"], np.asarray(expected["reg"], dtype=np.float64))
