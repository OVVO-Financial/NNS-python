from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from _r import dy_d_scalar, dy_d_scalar_mixed, dy_dx_numeric, dy_dx_overall, nns_diff_custom
from _tolerances import EXACT

from nns import dy_d, dy_dx, nns_diff

DIFF_PARITY = 1e-5
DY_D_PARITY = 1e-3


@pytest.mark.parity
@pytest.mark.parametrize(
    ("name", "func", "point"),
    [
        ("square", lambda x: x * x, 2.0),
        ("sin", np.sin, 1.0),
        ("exp", np.exp, 0.5),
        ("constant", lambda x: 5.0, 2.0),
        ("identity", lambda x: x, 3.0),
    ],
)
def test_nns_diff_derivative_matches_r(
    name: str,
    func: Any,
    point: float,
) -> None:
    expected = _r_nns_diff(name, point)
    actual = nns_diff(func, point)

    np.testing.assert_allclose(actual["DERIVATIVE"], expected["DERIVATIVE"], atol=DIFF_PARITY)
    np.testing.assert_allclose(
        actual["Value of f(x) at point"],
        expected["Value of f(x) at point"],
        atol=EXACT,
    )


@pytest.mark.parity
def test_dy_dx_overall_matches_r() -> None:
    x = np.linspace(-2.0, 2.0, 24)
    y = x + np.sin(x)

    expected = float(np.asarray(dy_dx_overall(x.tolist(), y.tolist()), dtype=np.float64))
    actual = dy_dx(x, y, eval_point="overall")

    assert actual == pytest.approx(expected, abs=EXACT)


@pytest.mark.parity
@pytest.mark.parametrize("eval_point", [[0.0], [-1.0, 0.0, 1.0]])
def test_dy_dx_numeric_eval_points_match_r(eval_point: list[float]) -> None:
    x = np.linspace(-2.0, 2.0, 24)
    y = x + np.sin(x)

    expected = _dict_array(dy_dx_numeric(x.tolist(), y.tolist(), eval_point))
    actual = dy_dx(x, y, eval_point=np.asarray(eval_point, dtype=np.float64))
    assert isinstance(actual, dict)

    assert list(actual) == list(expected)
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key], atol=5e-3, equal_nan=True)


@pytest.mark.parity
@pytest.mark.parametrize(
    ("wrt",),
    [
        (1,),
        (2,),
    ],
)
def test_dy_d_mean_wrt_match_r(wrt: int) -> None:
    x = np.column_stack(
        (np.array([-2, -1, 0, 1, 2], dtype=float), np.array([1, 3, 5, 7, 9], dtype=float))
    )
    y = 2 * x[:, 0] + 3 * x[:, 1]

    expected = _dict_array(dy_d_scalar(x.tolist(), y.tolist(), wrt, "mean"))
    actual = dy_d(x, y, wrt=wrt, eval_points="mean")

    assert actual.keys() == expected.keys()
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key], atol=DY_D_PARITY, equal_nan=True)


@pytest.mark.parity
def test_dy_d_nonlinear_wrt1_mean_matches_r() -> None:
    x = np.column_stack(
        (np.array([-2, -1, 0, 1, 2], dtype=float), np.array([1, 3, 5, 7, 9], dtype=float))
    )
    y = x[:, 0] ** 2 + np.sin(x[:, 1])

    expected = _dict_array(dy_d_scalar(x.tolist(), y.tolist(), 1, "mean"))
    actual = dy_d(x, y, wrt=1, eval_points="mean")

    assert actual.keys() == expected.keys()
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key], atol=DY_D_PARITY, equal_nan=True)


@pytest.mark.parity
@pytest.mark.parametrize("eval_points", ["mean", "median"])
def test_dy_d_scalar_wrt1_point_eval_modes_match_r(eval_points: str) -> None:
    x = np.column_stack(
        (np.array([-2, -1, 0, 1, 2], dtype=float), np.array([1, 3, 5, 7, 9], dtype=float))
    )
    y = 2 * x[:, 0] + 3 * x[:, 1]

    expected = _dict_array(dy_d_scalar(x.tolist(), y.tolist(), 1, eval_points))
    actual = dy_d(x, y, wrt=1, eval_points=eval_points)

    assert actual.keys() == expected.keys()
    for key in actual:
        actual_values = np.asarray(actual[key], dtype=np.float64).reshape(-1)
        expected_values = np.asarray(expected[key], dtype=np.float64).reshape(-1)
        assert actual_values.shape == expected_values.shape
        np.testing.assert_allclose(actual_values, expected_values, atol=DY_D_PARITY, equal_nan=True)


@pytest.mark.parity
def test_dy_d_scalar_wrt1_last_matches_r() -> None:
    x = np.column_stack(
        (np.linspace(-2.0, 2.0, 60), np.cos(np.linspace(0.0, 5.0, 60)))
    )
    y = 2 * x[:, 0] + 3 * x[:, 1]

    expected = _dict_array(dy_d_scalar(x.tolist(), y.tolist(), 1, "last"))
    actual = dy_d(x, y, wrt=1, eval_points="last")

    assert actual.keys() == expected.keys()
    for key in actual:
        actual_values = np.asarray(actual[key], dtype=np.float64).reshape(-1)
        expected_values = np.asarray(expected[key], dtype=np.float64).reshape(-1)
        assert actual_values.shape == expected_values.shape
        np.testing.assert_allclose(actual_values, expected_values, atol=DY_D_PARITY, equal_nan=True)


@pytest.mark.parity
@pytest.mark.parametrize("eval_points", ["obs", "apd"])
def test_dy_d_scalar_wrt1_distribution_eval_modes_match_r(eval_points: str) -> None:
    x1 = np.linspace(-1.5, 1.5, 18)
    x2 = np.cos(np.linspace(0.0, 2.0, 18))
    x = np.column_stack((x1, x2))
    y = x[:, 0] ** 2 + 0.5 * x[:, 1] + np.sin(x[:, 0] * x[:, 1])

    expected = _dict_array(dy_d_scalar(x.tolist(), y.tolist(), 1, eval_points))
    actual = dy_d(x, y, wrt=1, eval_points=eval_points)

    assert actual.keys() == expected.keys()
    for key in actual:
        actual_values = np.asarray(actual[key], dtype=np.float64).reshape(-1)
        expected_values = np.asarray(expected[key], dtype=np.float64).reshape(-1)
        assert actual_values.shape == expected_values.shape
        diagnostics = _relative_diagnostics(actual[key], expected[key])
        assert diagnostics["max_abs_diff"] <= 5e-3 or diagnostics["p95_rel_pct_masked"] <= 1.0
        np.testing.assert_allclose(
            actual_values,
            expected_values,
            atol=5e-3,
            rtol=1e-2,
            equal_nan=True,
        )


@pytest.mark.parametrize(
    ("wrt", "expected_first", "expected_second"),
    [
        (
            [1, 2],
            [3.990358, 1.995179],
            [-0.004758276, -0.001189569],
        ),
        (
            [1, 3],
            [0.997524, 0.498762],
            [-0.000848783, -0.000212196],
        ),
    ],
)
@pytest.mark.parity
def test_dy_d_vectorized_wrt_mean_matches_r(
    wrt: list[int],
    expected_first: list[float],
    expected_second: list[float],
) -> None:
    x = np.array([[-2, -1, 0, 1, 2], [1, 3, 5, 7, 9]]).T
    y = 2 * x[:, 0] + 3 * x[:, 1]
    if wrt == [1, 3]:
        x = np.column_stack((x, np.array([2, 4, 6, 8, 10], dtype=float)))
        y = x[:, 0] + 2 * x[:, 1] - x[:, 2]
    expected = {
        "First": np.array([expected_first], dtype=float),
        "Second": np.array([expected_second], dtype=float),
    }
    actual = dy_d(x, y, wrt=wrt, eval_points="mean")

    assert actual.keys() == expected.keys()
    for key in actual:
        assert actual[key].shape == (1, len(wrt))
        np.testing.assert_allclose(
            actual[key],
            expected[key],
            atol=DY_D_PARITY,
            equal_nan=True,
        )


@pytest.mark.parity
def test_dy_d_vectorized_wrt_nonlinear_mean_matches_r() -> None:
    x = np.column_stack(
        (np.array([-2, -1, 0, 1, 2], dtype=float), np.array([1, 3, 5, 7, 9], dtype=float))
    )
    y = x[:, 0] ** 2 + np.sin(x[:, 1])
    expected = {
        "First": np.array([[-0.06712002, -0.03356001]], dtype=float),
        "Second": np.array([[0.2593582, 0.06483955]], dtype=float),
    }
    actual = dy_d(x, y, wrt=[1, 2], eval_points="mean")

    assert actual.keys() == expected.keys()
    for key in actual:
        assert actual[key].shape == (1, 2)
        np.testing.assert_allclose(
            actual[key],
            expected[key],
            atol=DY_D_PARITY,
            equal_nan=True,
        )


@pytest.mark.parity
@pytest.mark.parametrize("eval_points", ["median", "last", "obs", "apd"])
def test_dy_d_vectorized_wrt_non_mean_modes_match_r(eval_points: str) -> None:
    x1 = np.linspace(-1.5, 1.5, 18)
    x2 = np.cos(np.linspace(0.0, 2.0, 18))
    x = np.column_stack((x1, x2))
    y = x[:, 0] ** 2 + 0.5 * x[:, 1] + np.sin(x[:, 0] * x[:, 1])

    expected = _stacked_scalar_dy_d(x, y, [1, 2], eval_points)
    actual = dy_d(x, y, wrt=np.array([1, 2]), eval_points=eval_points)

    atol = 3e-2 if eval_points == "apd" else 5e-3
    _assert_dy_d_dict_close(actual, expected, atol=atol, rtol=1e-2)


@pytest.mark.parity
@pytest.mark.parametrize("eval_points", ["mean"])
def test_dy_d_vectorized_wrt_mixed_modes_match_r(eval_points: str) -> None:
    x1 = np.linspace(-1.5, 1.5, 18)
    x2 = np.cos(np.linspace(0.0, 2.0, 18))
    x = np.column_stack((x1, x2))
    y = x[:, 0] ** 2 + 0.5 * x[:, 1] + np.sin(x[:, 0] * x[:, 1])

    expected = _stacked_scalar_dy_d_mixed(x, y, [1, 2], eval_points)
    actual = dy_d(x, y, wrt=np.array([1, 2]), eval_points=eval_points, mixed=True)

    _assert_dy_d_dict_close(actual, expected, atol=5e-3, rtol=1e-2)


@pytest.mark.parity
def test_dy_d_vectorized_wrt_numeric_eval_mixed_matches_r() -> None:
    x = np.column_stack((np.linspace(-1.0, 1.0, 12), np.cos(np.linspace(0.0, 2.0, 12))))
    y = x[:, 0] ** 2 + x[:, 1]
    eval_points = np.array([0.1, 0.4], dtype=np.float64)

    expected = _stacked_scalar_dy_d_mixed(x, y, [1, 2], eval_points)
    actual = dy_d(x, y, wrt=np.array([1, 2]), eval_points=eval_points, mixed=True)

    _assert_dy_d_dict_close(actual, expected, atol=5e-3, rtol=1e-2)


def _r_nns_diff(name: str, point: float) -> dict[str, float]:
    result = nns_diff_custom(name, point)
    assert isinstance(result, dict)
    return {
        key: float(np.asarray(value).reshape(-1)[0])
        for key, value in result.items()
        if isinstance(value, np.ndarray)
    }


def _dict_array(value: object) -> dict[str, np.ndarray]:
    if not isinstance(value, dict):
        raise AssertionError(f"Expected dictionary, got {type(value)!r}")
    return {key: np.asarray(item, dtype=np.float64) for key, item in value.items()}


def _stacked_scalar_dy_d(
    x: np.ndarray,
    y: np.ndarray,
    wrt_values: list[int],
    eval_points: str,
) -> dict[str, np.ndarray]:
    outputs = [
        _dict_array(dy_d_scalar(x.tolist(), y.tolist(), wrt, eval_points))
        for wrt in wrt_values
    ]
    return _stack_dy_d_outputs(outputs)


def _stacked_scalar_dy_d_mixed(
    x: np.ndarray,
    y: np.ndarray,
    wrt_values: list[int],
    eval_points: object,
) -> dict[str, np.ndarray]:
    point_arg = eval_points.tolist() if isinstance(eval_points, np.ndarray) else eval_points
    outputs = [
        _dict_array(dy_d_scalar_mixed(x.tolist(), y.tolist(), wrt, point_arg))
        for wrt in wrt_values
    ]
    return _stack_dy_d_outputs(outputs)


def _stack_dy_d_outputs(outputs: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        key: np.column_stack(
            [np.asarray(output[key], dtype=np.float64).reshape(-1) for output in outputs]
        )
        for key in ("First", "Second", "Mixed")
        if all(key in output for output in outputs)
    }


def _assert_dy_d_dict_close(
    actual: dict[str, np.ndarray],
    expected: dict[str, np.ndarray],
    *,
    atol: float,
    rtol: float,
) -> None:
    assert actual.keys() == expected.keys()
    for key in actual:
        assert actual[key].shape == expected[key].shape
        np.testing.assert_allclose(actual[key], expected[key], atol=atol, rtol=rtol, equal_nan=True)


def _relative_diagnostics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float | int]:
    actual_values = np.asarray(actual, dtype=np.float64)
    expected_values = np.asarray(expected, dtype=np.float64)
    diff = np.abs(actual_values - expected_values)
    finite = np.isfinite(diff)
    material = finite & (np.abs(expected_values) > 1e-8)
    if np.any(material):
        rel = 100.0 * diff[material] / np.abs(expected_values[material])
        max_rel = float(np.max(rel))
        p95_rel = float(np.percentile(rel, 95))
    else:
        max_rel = 0.0
        p95_rel = 0.0
    return {
        "max_abs_diff": float(np.max(diff[finite])) if np.any(finite) else 0.0,
        "max_rel_pct_masked": max_rel,
        "p95_rel_pct_masked": p95_rel,
        "near_zero_reference": int(np.count_nonzero(finite & ~material)),
    }
