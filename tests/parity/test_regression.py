from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from _r import nns, nns_reg_factor_dimred, nns_reg_factor_predictor
from _tolerances import COMPOUND

from nns import nns_reg
from nns.part import NoiseReduction
from nns.regression import Order

SIZES = [50, 200, 1000]
RELATIONSHIPS = ["linear", "quadratic", "sin", "random"]
MODE_RELATIONSHIPS = ["linear", "quadratic", "sin", "random"]
CASES: list[tuple[int | str | None, str, np.ndarray | None]] = [
    (None, "off", None),
    (1, "mean", None),
    (2, "median", np.array([-3.0, -1.0, 0.25, 3.0])),
    ("max", "off", np.array([-3.0, 0.0, 3.0])),
]
MODE_ORDERS: list[int | None] = [None, 1, 2, 3, 5]
MEAN_POINT_EST_CASES = [
    np.array([-3.0]),
    np.array([3.0]),
    np.array([-3.0, -1.0, 0.0, 2.5]),
]
DIM_RED_METHODS: list[str | list[float]] = [
    "cor",
    "NNS.dep",
    "NNS.caus",
    "all",
    "equal",
    [1.0, 0.5, 0.25],
]
CI_REGRESSION_CASES: list[tuple[int | None, str]] = [
    (None, "off"),
    (1, "off"),
    (2, "off"),
    (1, "mean"),
    (None, "median"),
    (1, "median"),
    (2, "median"),
]
CLASS_REGRESSION_CASES: list[tuple[str, np.ndarray, np.ndarray]] = [
    ("binary", np.array([1, 1, 1, 2, 2, 2], dtype=np.float64), np.array([1.5, 4.5])),
    (
        "multiclass",
        np.array([1, 1, 2, 2, 3, 3, 2, 1, 3], dtype=np.float64),
        np.array([1.5, 5.5, 7.5]),
    ),
    ("zero_one", np.array([0, 0, 0, 1, 1, 1], dtype=np.float64), np.array([1.5, 4.5])),
]


@pytest.mark.parity
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("relationship", RELATIONSHIPS)
@pytest.mark.parametrize(("order", "noise", "point_est"), CASES)
def test_nns_reg_univariate_matches_r(
    rng: np.random.Generator,
    size: int,
    relationship: str,
    order: int | str | None,
    noise: str,
    point_est: np.ndarray | None,
) -> None:
    x, y = _relationship(relationship, size, rng)

    expected = _r_nns_reg(x, y, order=order, noise=noise, point_est=point_est)
    actual = nns_reg(
        x,
        y,
        order=cast(Order, order),
        noise_reduction=cast(NoiseReduction, noise),
        point_est=point_est,
    )

    _assert_reg_matches(actual, expected)


@pytest.mark.parity
def test_nns_reg_small_smooth_fallback_matches_r() -> None:
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 1.0])
    point = np.array([1.5, 2.5])

    expected = _r_nns_reg_smooth(x, y, point_est=point, confidence_interval=0.95)
    actual = nns_reg(x, y, point_est=point, smooth=True, confidence_interval=0.95)

    _assert_reg_matches(actual, expected)


@pytest.mark.parity
def test_nns_reg_order_max_smooth_fallback_matches_r() -> None:
    x = np.linspace(-2.0, 2.0, 20)
    y = np.sin(x)
    point = np.array([-1.5, 0.0, 1.5])

    expected = _r_nns_reg_smooth(
        x,
        y,
        order="max",
        point_est=point,
        confidence_interval=0.95,
    )
    actual = nns_reg(x, y, order="max", point_est=point, smooth=True, confidence_interval=0.95)

    _assert_reg_matches(actual, expected)


@pytest.mark.parity
def test_nns_reg_spline_eligible_smooth_matches_r() -> None:
    x = np.linspace(-2.0, 2.0, 40)
    y = np.sin(x) + 0.2 * x**2
    point = np.array([-1.5, 0.0, 1.5])

    expected = _r_nns_reg_smooth(
        x,
        y,
        order=2,
        point_est=point,
        confidence_interval=0.95,
    )
    actual = nns_reg(x, y, order=2, point_est=point, smooth=True, confidence_interval=0.95)

    _assert_reg_matches(actual, expected, atol=5e-5)


@pytest.mark.parity
def test_nns_reg_dimred_smooth_matches_r() -> None:
    x1 = np.linspace(-2.0, 2.0, 36)
    x = np.column_stack((x1, np.sin(x1), np.cos(x1)))
    y = x[:, 0] + x[:, 1] - 0.25 * x[:, 2]
    point = x[::12]

    expected = _r_nns_reg_dimred(
        x,
        y,
        order=2,
        dim_red_method="equal",
        threshold=0.0,
        point_est=point,
        point_only=False,
        confidence_interval=0.95,
        smooth=True,
    )
    actual = nns_reg(
        x,
        y,
        order=2,
        dim_red_method="equal",
        point_est=point,
        confidence_interval=0.95,
        smooth=True,
    )

    _assert_reg_matches(actual, expected, check_dimred=True, atol=5e-5)


@pytest.mark.parity
def test_nns_reg_dimred_smooth_out_of_range_points_match_r() -> None:
    x1 = np.linspace(-1.5, 1.5, 18)
    x2 = np.cos(np.linspace(0.0, 2.0, 18))
    x = np.column_stack((x1, x2))
    y = x[:, 0] ** 2 + 0.5 * x[:, 1] + np.sin(x[:, 0] * x[:, 1])
    h_step = 0.35294117647058826
    lower = x.copy()
    upper = x.copy()
    lower[:, 0] -= h_step
    upper[:, 0] += h_step
    point = np.vstack((lower, x, upper))

    expected = _r_nns_reg_dimred(
        x,
        y,
        order=None,
        dim_red_method="equal",
        threshold=0.0,
        point_est=point,
        point_only=True,
        smooth=True,
    )
    actual = nns_reg(
        x,
        y,
        dim_red_method="equal",
        point_est=point,
        point_only=True,
        smooth=True,
    )

    _assert_reg_matches(actual, expected, check_dimred=True, atol=5e-3)


@pytest.mark.parity
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("relationship", MODE_RELATIONSHIPS)
@pytest.mark.parametrize("order", MODE_ORDERS)
def test_nns_reg_mode_noise_reduction_matches_r(
    rng: np.random.Generator,
    size: int,
    relationship: str,
    order: int | None,
) -> None:
    x, y = _relationship(relationship, size, rng)

    expected = _r_nns_reg(x, y, order=order, noise="mode", point_est=None)
    actual = nns_reg(x, y, order=order, noise_reduction="mode")

    _assert_reg_matches(actual, expected)


@pytest.mark.parity
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("relationship", MODE_RELATIONSHIPS)
@pytest.mark.parametrize("order", MODE_ORDERS)
def test_nns_reg_mode_class_noise_reduction_matches_r(
    rng: np.random.Generator,
    size: int,
    relationship: str,
    order: int | None,
) -> None:
    x, y = _relationship(relationship, size, rng)

    expected = _r_nns_reg(x, y, order=order, noise="mode_class", point_est=None)
    actual = nns_reg(x, y, order=order, noise_reduction="mode_class")

    _assert_reg_matches(
        actual,
        expected,
        skip_standard_errors=order is None,
    )


@pytest.mark.parity
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("point_est", MEAN_POINT_EST_CASES)
def test_nns_reg_mean_out_of_range_point_est_matches_r(
    size: int,
    point_est: np.ndarray,
) -> None:
    x = np.linspace(-2.0, 2.0, size)
    y = np.sin(x)

    expected = _r_nns_reg(x, y, order=1, noise="mean", point_est=point_est)
    actual = nns_reg(x, y, order=1, noise_reduction="mean", point_est=point_est)

    _assert_reg_matches(actual, expected)


@pytest.mark.parity
@pytest.mark.parametrize("method", DIM_RED_METHODS)
def test_nns_reg_dim_red_matches_r(method: str | list[float]) -> None:
    x1 = np.linspace(-2.0, 2.0, 50)
    x = np.column_stack((x1, np.sin(x1), np.cos(x1)))
    y = x[:, 0] + x[:, 1] + 0.25 * x[:, 2]
    point_est = np.array([[0.0, 0.0, 1.0], [3.0, 0.0, 1.0]])

    expected = _r_nns_reg_dimred(
        x,
        y,
        order=2,
        dim_red_method=method,
        threshold=0.0,
        point_est=point_est,
        point_only=False,
    )
    actual = nns_reg(
        x,
        y,
        order=2,
        dim_red_method=method,
        point_est=point_est,
    )

    tolerance = 5e-2 if method == "NNS.caus" else COMPOUND
    _assert_reg_matches(actual, expected, check_dimred=True, atol=tolerance)


@pytest.mark.parity
@pytest.mark.parametrize("method", ["NNS.caus", "all"])
def test_nns_reg_dim_red_tau_ts_matches_r_fixed_uni_caus_lag(method: str) -> None:
    x1 = np.linspace(-2.0, 2.0, 40)
    x = np.column_stack((x1, np.sin(x1), np.cos(x1)))
    y = x[:, 0] + x[:, 1] + 0.25 * x[:, 2]
    point_est = x[:3]

    expected = _r_nns_reg_dimred(
        x,
        y,
        order=None,
        dim_red_method=method,
        tau="ts",
        threshold=0.0,
        point_est=point_est,
        point_only=False,
    )
    actual = nns_reg(
        x,
        y,
        dim_red_method=method,
        tau="ts",
        point_est=point_est,
    )

    assert isinstance(expected, dict)
    assert isinstance(expected["equation"], dict)
    assert isinstance(actual["equation"], dict)
    np.testing.assert_array_equal(
        actual["equation"]["Variable"].astype(str),
        _strings(expected["equation"]["Variable"]),
    )
    np.testing.assert_allclose(
        actual["equation"]["Coefficient"],
        _array(expected["equation"]["Coefficient"]),
        atol=5e-2,
    )
    assert isinstance(expected["x.star"], dict)
    assert isinstance(actual["x.star"], dict)
    np.testing.assert_allclose(actual["x.star"]["x"], _array(expected["x.star"]["x"]), atol=5e-2)
    np.testing.assert_allclose(actual["Point.est"], _array(expected["Point.est"]), atol=5e-2)


@pytest.mark.parity
def test_nns_reg_dim_red_point_only_matches_r() -> None:
    x1 = np.linspace(-2.0, 2.0, 50)
    x = np.column_stack((x1, np.sin(x1), np.cos(x1)))
    y = x[:, 0] + x[:, 1] + 0.25 * x[:, 2]
    point_est = np.array([[0.0, 0.0, 1.0], [3.0, 0.0, 1.0]])

    expected = _r_nns_reg_dimred(
        x,
        y,
        order=None,
        dim_red_method="equal",
        threshold=0.0,
        point_est=point_est,
        point_only=True,
    )
    actual = nns_reg(
        x,
        y,
        dim_red_method="equal",
        point_est=point_est,
        point_only=True,
    )

    _assert_reg_matches(actual, expected, check_dimred=True)


@pytest.mark.parity
def test_nns_reg_dim_red_multivariate_call_matches_r() -> None:
    x1 = np.linspace(-2.0, 2.0, 30)
    x = np.column_stack((x1, np.sin(x1), np.cos(x1)))
    y = x[:, 0] + x[:, 1] + 0.25 * x[:, 2]

    expected = _r_nns_reg_dimred(
        x,
        y,
        order=None,
        dim_red_method="equal",
        threshold=0.0,
        point_est=None,
        point_only=False,
        multivariate_call=True,
    )
    actual = nns_reg(x, y, dim_red_method="equal", multivariate_call=True)

    assert isinstance(expected, dict)
    assert set(actual) == set(expected) == {"x", "y"}
    np.testing.assert_allclose(actual["x"], _array(expected["x"]), atol=COMPOUND)
    np.testing.assert_allclose(actual["y"], _array(expected["y"]), atol=COMPOUND)


@pytest.mark.parity
def test_nns_reg_univariate_point_only_matches_r() -> None:
    x = np.linspace(-2.0, 2.0, 20)
    y = np.sin(x)
    point_est = np.array([-1.0, 0.0, 1.0])

    expected = _r_nns_reg(x, y, order=None, noise="off", point_est=point_est, point_only=True)
    actual = nns_reg(x, y, point_est=point_est, point_only=True)

    _assert_reg_matches(actual, expected)


@pytest.mark.parity
def test_nns_reg_univariate_matrix_point_est_matches_r_flattening() -> None:
    x = np.linspace(-2.0, 2.0, 20)
    y = np.sin(x)
    point_est = np.array([[-1.0, 1.0], [0.0, 2.0]])

    expected = _r_nns_reg(
        x,
        y,
        order=None,
        noise="off",
        point_est=np.array([-1.0, 0.0, 1.0, 2.0]),
    )
    actual = nns_reg(x, y, point_est=point_est)

    _assert_reg_matches(actual, expected)


@pytest.mark.parity
def test_nns_reg_dim_red_degenerate_equal_projection_matches_r() -> None:
    x = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, -1.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )
    y = 0.5 * x[:, 0] - 0.25 * x[:, 1]

    expected = _r_nns_reg_dimred(
        x,
        y,
        order=None,
        dim_red_method="equal",
        threshold=0.0,
        point_est=None,
        point_only=False,
    )
    actual = nns_reg(x, y, dim_red_method="equal")

    _assert_reg_matches(actual, expected, check_dimred=True)


@pytest.mark.parity
def test_nns_reg_factor_predictor_matches_r_full_rank_dummy_path() -> None:
    x = np.array(["b", "a", "b", "c"])
    y = np.array([2.0, 1.0, 3.0, 4.0])
    point_est = np.array(["a", "c"])
    levels = ["a", "b", "c"]

    expected = nns_reg_factor_predictor(
        x.tolist(),
        y.tolist(),
        point_est.tolist(),
        levels=levels,
        order=None,
    )
    actual = nns_reg(
        x,
        y,
        factor_2_dummy=True,
        factor_levels=levels,
        point_est=point_est,
    )

    assert isinstance(expected, dict)
    assert set(actual) == set(expected)
    np.testing.assert_allclose(actual["R2"], _array(expected["R2"]), atol=COMPOUND)
    np.testing.assert_allclose(actual["Point.est"], _array(expected["Point.est"]), atol=COMPOUND)
    for key in ("rhs.partitions", "RPM"):
        assert isinstance(actual[key], dict)
        assert isinstance(expected[key], dict)
        actual_items = list(actual[key].items())
        expected_table = expected[key]
        assert isinstance(expected_table, dict)
        expected_items = list(expected_table.items())
        assert len(actual_items) == len(expected_items)
        for (_, values), (_, expected_values) in zip(
            actual_items,
            expected_items,
            strict=True,
        ):
            np.testing.assert_allclose(values, _array(expected_values), atol=COMPOUND)

    assert isinstance(actual["Fitted.xy"], dict)
    assert isinstance(expected["Fitted.xy"], dict)
    np.testing.assert_array_equal(
        actual["Fitted.xy"]["NNS.ID"].astype(str),
        _strings(expected["Fitted.xy"]["NNS.ID"]),
    )
    actual_predictors = [
        values
        for column, values in actual["Fitted.xy"].items()
        if column not in {"y", "y.hat", "NNS.ID", "residuals"}
    ]
    expected_predictors = [
        values
        for column, values in expected["Fitted.xy"].items()
        if column not in {"y", "y.hat", "NNS.ID", "residuals"}
    ]
    assert len(actual_predictors) == len(expected_predictors)
    for values, expected_values in zip(actual_predictors, expected_predictors, strict=True):
        np.testing.assert_allclose(values, _array(expected_values), atol=COMPOUND)
    for column in ("y", "y.hat", "residuals"):
        np.testing.assert_allclose(
            actual["Fitted.xy"][column],
            _array(expected["Fitted.xy"][column]),
            atol=COMPOUND,
        )


@pytest.mark.parity
@pytest.mark.parametrize("method", ["cor", "equal", "NNS.dep"])
def test_nns_reg_factor_predictor_dim_red_matches_r(method: str) -> None:
    levels = ["a", "b", "c"]
    factor = np.array(["b", "a", "b", "c", "a", "c"], dtype=object)
    z = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=object)
    x = np.column_stack((factor, z))
    y = np.array([2.0, 1.0, 3.0, 4.0, 1.5, 4.5])
    point_est = np.array([["a", 1.5], ["c", 3.5]], dtype=object)

    expected = nns_reg_factor_dimred(
        factor.tolist(),
        [float(value) for value in z],
        y.tolist(),
        ["a", "c"],
        [1.5, 3.5],
        levels=levels,
        dim_red_method=method,
    )
    actual = nns_reg(
        x,
        y,
        factor_2_dummy=True,
        factor_levels=[levels, None],
        dim_red_method=method,
        point_est=point_est,
    )

    assert isinstance(expected, dict)
    assert isinstance(expected["equation"], dict)
    assert isinstance(actual["equation"], dict)
    np.testing.assert_allclose(
        actual["equation"]["Coefficient"],
        _array(expected["equation"]["Coefficient"]),
        atol=5e-2,
    )
    assert isinstance(expected["x.star"], dict)
    assert isinstance(actual["x.star"], dict)
    np.testing.assert_allclose(actual["x.star"]["x"], _array(expected["x.star"]["x"]), atol=5e-2)
    np.testing.assert_allclose(actual["Point.est"], _array(expected["Point.est"]), atol=5e-2)
    np.testing.assert_allclose(actual["R2"], _array(expected["R2"]), atol=5e-2)


@pytest.mark.parity
@pytest.mark.parametrize("relationship", ["linear", "quadratic", "sin"])
@pytest.mark.parametrize("confidence_interval", [0.8, 0.95])
@pytest.mark.parametrize(
    "point_est",
    [
        None,
        np.array([-1.0, 0.0, 1.0]),
        np.array([-3.0, -1.0, 0.0, 2.5]),
    ],
)
@pytest.mark.parametrize(("order", "noise"), CI_REGRESSION_CASES)
def test_nns_reg_confidence_interval_matches_r(
    rng: np.random.Generator,
    relationship: str,
    confidence_interval: float,
    point_est: np.ndarray | None,
    order: int | None,
    noise: str,
) -> None:
    x, y = _relationship(relationship, 50, rng)

    expected = _r_nns_reg(
        x,
        y,
        order=order,
        noise=noise,
        point_est=point_est,
        confidence_interval=confidence_interval,
    )
    actual = nns_reg(
        x,
        y,
        order=order,
        noise_reduction=cast(NoiseReduction, noise),
        point_est=point_est,
        confidence_interval=confidence_interval,
    )

    _assert_reg_matches(actual, expected)


@pytest.mark.parity
def test_nns_reg_below_range_point_est_pred_int_row_drop_matches_r() -> None:
    x = np.linspace(-2.0, 2.0, 50)
    y = np.sin(x)
    point_est = np.array([-3.0, -1.0, 0.0, 2.5])

    expected = _r_nns_reg(
        x,
        y,
        order=1,
        noise="off",
        point_est=point_est,
        confidence_interval=0.95,
    )
    actual = nns_reg(
        x,
        y,
        order=1,
        point_est=point_est,
        confidence_interval=0.95,
    )

    _assert_reg_matches(actual, expected)
    assert actual["pred.int"] is not None
    assert actual["pred.int"]["pred.int.neg"].shape == (3,)


@pytest.mark.parity
@pytest.mark.parametrize("order", [None, 1, 2])
@pytest.mark.parametrize(("name", "classes", "point_est"), CLASS_REGRESSION_CASES)
def test_nns_reg_classification_matches_r(
    order: int | None,
    name: str,
    classes: np.ndarray,
    point_est: np.ndarray,
) -> None:
    del name
    x = np.linspace(0.0, float(classes.size - 1), classes.size)

    expected = _r_nns_reg(
        x,
        classes,
        order=order,
        noise="off",
        point_est=point_est,
        type="class",
    )
    actual = nns_reg(x, classes, order=order, type="class", point_est=point_est)

    _assert_reg_matches(actual, expected)


@pytest.mark.parity
@pytest.mark.parametrize("confidence_interval", [0.8, 0.95])
@pytest.mark.parametrize("order", [None, 1, 2])
@pytest.mark.parametrize(("name", "classes", "point_est"), CLASS_REGRESSION_CASES)
def test_nns_reg_class_confidence_interval_matches_r(
    confidence_interval: float,
    order: int | None,
    name: str,
    classes: np.ndarray,
    point_est: np.ndarray,
) -> None:
    del name
    x = np.linspace(0.0, float(classes.size - 1), classes.size)

    expected = _r_nns_reg(
        x,
        classes,
        order=order,
        noise="off",
        point_est=point_est,
        confidence_interval=confidence_interval,
        type="class",
    )
    actual = nns_reg(
        x,
        classes,
        order=order,
        type="class",
        point_est=point_est,
        confidence_interval=confidence_interval,
    )

    _assert_reg_matches(actual, expected)


@pytest.mark.parity
def test_nns_reg_logical_auto_classification_matches_r() -> None:
    x = np.linspace(0.0, 5.0, 6)
    y = np.array([False, False, False, True, True, True])

    expected = _r_nns_reg(
        x,
        y.astype(np.float64),
        order=None,
        noise="off",
        point_est=np.array([1.5, 4.5]),
    )
    actual = nns_reg(x, y, point_est=np.array([1.5, 4.5]))

    _assert_reg_matches(actual, expected)


@pytest.mark.parity
def test_nns_reg_logical_auto_class_confidence_interval_matches_r() -> None:
    x = np.linspace(0.0, 5.0, 6)
    y = np.array([False, False, False, True, True, True])

    expected = _r_nns_reg(
        x,
        y.astype(np.float64),
        order=None,
        noise="off",
        point_est=np.array([1.5, 4.5]),
        confidence_interval=0.95,
    )
    actual = nns_reg(x, y, point_est=np.array([1.5, 4.5]), confidence_interval=0.95)

    _assert_reg_matches(actual, expected)


@pytest.mark.parity
def test_nns_reg_factor_levels_return_numeric_codes() -> None:
    x = np.linspace(0.0, 8.0, 9)
    labels = np.array(["B", "B", "A", "A", "C", "C", "A", "B", "C"])
    levels = ["A", "B", "C"]
    encoded = np.array([2, 2, 1, 1, 3, 3, 1, 2, 3], dtype=np.float64)

    expected = _r_nns_reg(
        x,
        encoded,
        order=1,
        noise="off",
        point_est=np.array([1.5, 5.5]),
        type="class",
    )
    actual = nns_reg(
        x,
        labels,
        order=1,
        type="class",
        point_est=np.array([1.5, 5.5]),
        class_levels=levels,
    )

    _assert_reg_matches(actual, expected)


@pytest.mark.parity
def test_nns_reg_factor_levels_class_confidence_interval_matches_r() -> None:
    x = np.linspace(0.0, 8.0, 9)
    labels = np.array(["B", "B", "A", "A", "C", "C", "A", "B", "C"])
    levels = ["A", "B", "C"]
    encoded = np.array([2, 2, 1, 1, 3, 3, 1, 2, 3], dtype=np.float64)

    expected = _r_nns_reg(
        x,
        encoded,
        order=1,
        noise="off",
        point_est=np.array([1.5, 5.5]),
        confidence_interval=0.95,
        type="class",
    )
    actual = nns_reg(
        x,
        labels,
        order=1,
        type="class",
        point_est=np.array([1.5, 5.5]),
        confidence_interval=0.95,
        class_levels=levels,
    )

    _assert_reg_matches(actual, expected)


@pytest.mark.parity
def test_nns_reg_class_confidence_interval_below_range_row_drop_matches_r() -> None:
    x = np.linspace(0.0, 11.0, 12)
    classes = np.array([1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2], dtype=np.float64)
    point_est = np.array([-1.0, 2.5, 6.5, 11.5])

    expected = _r_nns_reg(
        x,
        classes,
        order=1,
        noise="off",
        point_est=point_est,
        confidence_interval=0.95,
        type="class",
    )
    actual = nns_reg(
        x,
        classes,
        order=1,
        type="class",
        point_est=point_est,
        confidence_interval=0.95,
    )

    _assert_reg_matches(actual, expected)
    assert actual["Point.est"].shape == (4,)
    assert actual["pred.int"] is not None
    assert actual["pred.int"]["pred.int.neg"].shape == (3,)


@pytest.mark.parity
def test_nns_reg_raw_character_class_labels_raise() -> None:
    x = np.linspace(0.0, 5.0, 6)
    y = np.array(["A", "A", "A", "B", "B", "B"])

    with pytest.raises(ValueError, match="class_levels"):
        nns_reg(x, y, type="class")


def _r_nns_reg(
    x: np.ndarray,
    y: np.ndarray,
    *,
    order: int | str | None,
    noise: str,
    point_est: np.ndarray | None,
    confidence_interval: float | None = None,
    type: str | None = None,
    point_only: bool = False,
) -> Any:
    point_arg: list[float] | None = None if point_est is None else point_est.tolist()
    return nns(
        "NNS.reg",
        x.tolist(),
        y.tolist(),
        False,
        order,
        None,
        None,
        type,
        point_arg,
        "top",
        True,
        False,
        False,
        False,
        confidence_interval,
        0,
        None,
        False,
        noise,
        "L2",
        None,
        point_only,
        False,
    )


def _r_nns_reg_smooth(
    x: np.ndarray,
    y: np.ndarray,
    *,
    order: int | str | None = None,
    point_est: np.ndarray | None,
    confidence_interval: float | None = None,
) -> Any:
    point_arg: list[float] | None = None if point_est is None else point_est.tolist()
    return nns(
        "NNS.reg",
        x.tolist(),
        y.tolist(),
        False,
        order,
        None,
        None,
        None,
        point_arg,
        "top",
        True,
        False,
        False,
        False,
        confidence_interval,
        0,
        None,
        True,
        "off",
        "L2",
        None,
        False,
        False,
    )


def _r_nns_reg_dimred(
    x: np.ndarray,
    y: np.ndarray,
    *,
    order: int | str | None,
    dim_red_method: str | list[float],
    threshold: float,
    point_est: np.ndarray | None,
    point_only: bool,
    confidence_interval: float | None = None,
    tau: object | None = None,
    multivariate_call: bool = False,
    smooth: bool = False,
) -> Any:
    return nns(
        "NNS.reg",
        x.tolist(),
        y.tolist(),
        False,
        order,
        dim_red_method,
        tau,
        None,
        None if point_est is None else point_est.tolist(),
        "top",
        True,
        False,
        False,
        False,
        confidence_interval,
        threshold,
        None,
        smooth,
        "off",
        "L2",
        1,
        point_only,
        multivariate_call,
    )


def _assert_reg_matches(
    actual: dict[str, Any],
    expected: Any,
    *,
    skip_standard_errors: bool = False,
    check_dimred: bool = False,
    atol: float = COMPOUND,
) -> None:
    assert isinstance(expected, dict)
    assert set(actual) == set(expected)
    np.testing.assert_allclose(actual["R2"], _array(expected["R2"]), atol=atol)
    np.testing.assert_allclose(actual["SE"], _array(expected["SE"]), atol=atol)
    np.testing.assert_allclose(actual["Point.est"], _array(expected["Point.est"]), atol=atol)
    if actual["pred.int"] is None:
        assert _array(expected["pred.int"]).size == 0
    else:
        assert isinstance(actual["pred.int"], dict)
        assert isinstance(expected["pred.int"], dict)
        assert set(actual["pred.int"]) == set(expected["pred.int"])
        for column, values in actual["pred.int"].items():
            np.testing.assert_allclose(values, _array(expected["pred.int"][column]), atol=atol)
    if check_dimred:
        assert isinstance(expected["equation"], dict)
        assert isinstance(actual["equation"], dict)
        np.testing.assert_array_equal(
            actual["equation"]["Variable"].astype(str),
            _strings(expected["equation"]["Variable"]),
        )
        np.testing.assert_allclose(
            actual["equation"]["Coefficient"],
            _array(expected["equation"]["Coefficient"]),
            atol=atol,
        )
        assert isinstance(expected["x.star"], dict)
        assert isinstance(actual["x.star"], dict)
        np.testing.assert_allclose(
            actual["x.star"]["x"],
            _array(expected["x.star"]["x"]),
            atol=atol,
        )

    for key in ("derivative", "regression.points", "Fitted.xy"):
        assert isinstance(expected[key], dict)
        assert isinstance(actual[key], dict)
        assert set(actual[key]) == set(expected[key])
        for column in actual[key]:
            if skip_standard_errors and key == "Fitted.xy" and column == "standard.errors":
                continue
            if column == "NNS.ID":
                np.testing.assert_array_equal(
                    actual[key][column].astype(str),
                    _strings(expected[key][column]),
                )
            else:
                np.testing.assert_allclose(
                    actual[key][column],
                    _array(expected[key][column]),
                    atol=atol,
                )


def _array(value: object) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _strings(value: object) -> np.ndarray:
    if isinstance(value, list):
        return np.asarray(value, dtype=str)
    return np.asarray(value, dtype=str)


def _relationship(
    relationship: str,
    size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-2.0, 2.0, size)
    if relationship == "linear":
        return x, 1.5 + 0.7 * x + 0.02 * np.sin(np.arange(size))
    if relationship == "quadratic":
        return x, x * x
    if relationship == "cubic":
        return x, x**3
    if relationship == "sin":
        return x, np.sin(x)
    return x, rng.normal(size=size)
