from __future__ import annotations

import cmath
import math
from typing import Any

import numpy as np
import pytest
from _r import dy_d_scalar, dy_dx_numeric, dy_dx_overall, nns_diff_custom
from _tolerances import COMPOUND, EXACT

from nns import dy_d, dy_dx, nns_diff

# dy.d_ is evaluated through independent fitted NNS models in R and Python.
# Small fit differences are magnified by finite-difference denominators.
DY_D_PARITY = 5e-2


@pytest.mark.parity
@pytest.mark.parametrize(
    ("name", "func", "point"),
    [
        ("square", lambda x: x * x, 2.0),
        ("sin", np.sin, 1.0),
        ("exp", np.exp, 0.5),
        ("constant", lambda x: 5.0, 2.0),
        ("identity", lambda x: x, 3.0),
        ("square", lambda x: x * x, 3.0),
        ("cube", lambda x: x**3, 2.0),
        ("log", np.log, 5.0),
        # math.tanh is the C library tanh R uses; np.tanh can differ by an ulp,
        # which changes the bisection path.
        ("tanh", lambda x: cmath.tanh(x) if isinstance(x, complex) else math.tanh(x), 0.5),
        ("abs", abs, -1.0),
        ("quartic", lambda x: x**4 - 3 * x, -2.0),
    ],
)
def test_nns_diff_matches_r(name: str, func: Any, point: float) -> None:
    # The root finder mirrors R's uniroot (extendInt = "yes", tol = tol / 10 and the
    # bracket-on-B convergence test of the patched R), so every reported field,
    # including iterations and the convergence flags, matches R's 12-digit output.
    expected = _r_nns_diff(name, point)
    actual = nns_diff(func, point)
    for key, value in actual.items():
        if key not in expected:  # R NA
            assert np.isnan(value), key
            continue
        np.testing.assert_allclose(value, expected[key], rtol=EXACT, atol=0.0, err_msg=key)


@pytest.mark.parametrize(
    ("func", "point", "truth"),
    [
        (np.sin, 1.0, math.cos(1.0)),
        (lambda x: x**3, 2.0, 12.0),
        (np.log, 5.0, 0.2),
        (lambda x: x**4 - 3 * x, -2.0, -35.0),
    ],
)
def test_nns_diff_converged_derivative_is_accurate(func: Any, point: float, truth: float) -> None:
    # Patched R: "converged" used to be reported with errors of 4e-7 to 6e-4.
    result = nns_diff(func, point)
    assert result["converged"] == 1.0
    assert abs(result["DERIVATIVE"] - truth) < 1e-7


@pytest.mark.parity
def test_nns_diff_steps_past_a_zero_inferred_h_like_r() -> None:
    # uniroot evaluates the inferred-h equation at h = 0, where R's 0/0 is NaN.
    def quintic(x: Any) -> Any:
        return -0.25 + x * (-1.25 + x * (-2 + x * (-1.25 - 1.25 * x)))

    expected = _r_nns_diff_h("quintic", -0.25, 0.2)
    actual = nns_diff(quintic, -0.25, h=0.2)
    assert actual["converged"] == 1.0
    for key, value in actual.items():
        if key not in expected:
            assert np.isnan(value), key
            continue
        np.testing.assert_allclose(value, expected[key], rtol=EXACT, atol=0.0, err_msg=key)


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
        np.testing.assert_allclose(actual[key], expected[key], atol=COMPOUND, equal_nan=True)


@pytest.mark.parity
@pytest.mark.parametrize("size", [30, 60, 250])
def test_dy_dx_numeric_bandwidths_match_r(size: int) -> None:
    # R steps by gravity(diff(sort(unique(x)))) * h_s (the bandwidth value, not its
    # index; the patched R sorts x first); for n < 49 R keeps only the bandwidths
    # that occur once in h_s.
    # x on a 1/1024 grid makes x * x * x exact, so the cache key matches on every numpy build.
    rng = np.random.default_rng(size)
    x = rng.integers(-2048, 2049, size) / 1024.0
    y = x * x * x - x
    eval_point = [-1.5, 0.0, 0.5, 1.5]
    expected = _dict_array(dy_dx_numeric(x.tolist(), y.tolist(), eval_point))
    actual = dy_dx(x, y, eval_point=np.asarray(eval_point, dtype=np.float64))
    assert isinstance(actual, dict)
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key], atol=COMPOUND, equal_nan=True)


@pytest.mark.parity
def test_dy_dx_numeric_does_not_depend_on_row_order() -> None:
    # R 13.3 took diff() of x in row order: 500 unsorted grid points in [-2, 2]
    # gave dy/dx of x^3 of 3.7 to 3.9 everywhere (truth 6.75 0 0.75 6.75).
    rng = np.random.default_rng(1)
    x = rng.integers(-2048, 2049, 500) / 1024.0
    y = x * x * x
    eval_point = [-1.5, 0.0, 0.5, 1.5]
    expected = _dict_array(dy_dx_numeric(x.tolist(), y.tolist(), eval_point))
    actual = dy_dx(x, y, eval_point=np.asarray(eval_point, dtype=np.float64))
    order = np.argsort(x)
    sorted_actual = dy_dx(x[order], y[order], eval_point=np.asarray(eval_point))
    assert isinstance(actual, dict) and isinstance(sorted_actual, dict)
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key], atol=COMPOUND, equal_nan=True)
        np.testing.assert_allclose(actual[key], sorted_actual[key], atol=1e-12)
    np.testing.assert_allclose(actual["first.derivative"], [6.75, 0.0, 0.75, 6.75], atol=0.2)


@pytest.mark.parity
def test_dy_dx_numeric_boundary_eval_points_match_r() -> None:
    # At min(x) and max(x) one run is 0: R uses the one-sided slope and the
    # second derivative is NaN.
    rng = np.random.default_rng(7)
    x = np.sort(rng.uniform(0.0, 2.0 * np.pi, 120))
    y = np.cos(x)
    eval_point = [float(x[0]), 3.0, float(x[-1])]
    expected = _dict_array(dy_dx_numeric(x.tolist(), y.tolist(), eval_point))
    actual = dy_dx(x, y, eval_point=np.asarray(eval_point, dtype=np.float64))
    assert isinstance(actual, dict)
    assert np.isnan(actual["second.derivative"][[0, 2]]).all()
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key], atol=COMPOUND, equal_nan=True)


@pytest.mark.parity
@pytest.mark.parametrize("wrt", [1, 2])
def test_dy_d_mean_wrt_matches_r(wrt: int) -> None:
    x = np.column_stack(
        (np.array([-2, -1, 0, 1, 2], dtype=float), np.array([1, 3, 5, 7, 9], dtype=float))
    )
    y = 2 * x[:, 0] + 3 * x[:, 1]
    expected = _dict_array(dy_d_scalar(x.tolist(), y.tolist(), wrt, "mean"))
    actual = dy_d(x, y, wrt=wrt, eval_points="mean")
    _assert_dy_d_dict_close(actual, expected)


@pytest.mark.parity
@pytest.mark.parametrize("wrt", [1, 2])
@pytest.mark.parametrize("eval_points", ["mean", "median"])
def test_dy_d_irrelevant_regressor_gets_no_slope_like_r(wrt: int, eval_points: str) -> None:
    # Patched R: f(x +/- h e_wrt) comes from the multivariate NNS.reg fit on
    # the actual regressors. The old NNS.stack fit on rowMeans(x) gave every
    # regressor about the index slope / p (y = 2*x1: dy/dx2 about 1).
    rng = np.random.default_rng(2)
    x = np.round(rng.standard_normal((200, 2)), 8)
    y = 2.0 * x[:, 0]
    expected = _dict_array(dy_d_scalar(x.tolist(), y.tolist(), wrt, eval_points))
    actual = dy_d(x, y, wrt=wrt, eval_points=eval_points, messages=False)
    _assert_dy_d_dict_close(actual, expected)
    if wrt == 2:
        assert abs(float(np.ravel(actual["First"])[0])) < 0.25
    else:
        assert float(np.ravel(actual["First"])[0]) > 1.0


@pytest.mark.parity
def test_dy_d_nonlinear_wrt1_mean_matches_r() -> None:
    x = np.column_stack(
        (np.array([-2, -1, 0, 1, 2], dtype=float), np.array([1, 3, 5, 7, 9], dtype=float))
    )
    y = x[:, 0] ** 2 + np.sin(x[:, 1])
    expected = _dict_array(dy_d_scalar(x.tolist(), y.tolist(), 1, "mean"))
    actual = dy_d(x, y, wrt=1, eval_points="mean")
    _assert_dy_d_dict_close(actual, expected)


@pytest.mark.parity
@pytest.mark.parametrize("eval_points", ["mean", "median"])
def test_dy_d_scalar_mean_median_match_r(eval_points: str) -> None:
    x = np.column_stack(
        (np.array([-2, -1, 0, 1, 2], dtype=float), np.array([1, 3, 5, 7, 9], dtype=float))
    )
    y = 2 * x[:, 0] + 3 * x[:, 1]
    expected = _dict_array(dy_d_scalar(x.tolist(), y.tolist(), 1, eval_points))
    actual = dy_d(x, y, wrt=1, eval_points=eval_points)
    _assert_dy_d_dict_close(actual, expected)


@pytest.mark.parity
def test_dy_d_scalar_last_matches_r() -> None:
    x = np.column_stack((np.linspace(-2.0, 2.0, 60), np.cos(np.linspace(0.0, 5.0, 60))))
    y = 2 * x[:, 0] + 3 * x[:, 1]
    expected = _dict_array(dy_d_scalar(x.tolist(), y.tolist(), 1, "last"))
    actual = dy_d(x, y, wrt=1, eval_points="last")
    _assert_dy_d_dict_close(actual, expected)


@pytest.mark.parity
@pytest.mark.parametrize("eval_points", ["obs", "apd"])
def test_dy_d_scalar_distribution_modes_match_r(eval_points: str) -> None:
    x1 = np.linspace(-1.5, 1.5, 18)
    x2 = np.cos(np.linspace(0.0, 2.0, 18))
    x = np.column_stack((x1, x2))
    y = x[:, 0] ** 2 + 0.5 * x[:, 1] + np.sin(x[:, 0] * x[:, 1])
    expected = _dict_array(dy_d_scalar(x.tolist(), y.tolist(), 1, eval_points))
    actual = dy_d(x, y, wrt=1, eval_points=eval_points)
    _assert_dy_d_dict_close(actual, expected, rtol=1e-2)


@pytest.mark.parity
@pytest.mark.parametrize("wrt", [[1, 2], [1, 3]])
def test_dy_d_vectorized_wrt_mean_matches_scalar_python_calls(wrt: list[int]) -> None:
    x = np.array([[-2, -1, 0, 1, 2], [1, 3, 5, 7, 9]], dtype=float).T
    y = 2 * x[:, 0] + 3 * x[:, 1]
    if wrt == [1, 3]:
        x = np.column_stack((x, np.array([2, 4, 6, 8, 10], dtype=float)))
        y = x[:, 0] + 2 * x[:, 1] - x[:, 2]
    expected = _stacked_scalar_python_dy_d(x, y, wrt, "mean")
    actual = dy_d(x, y, wrt=wrt, eval_points="mean")
    _assert_dy_d_dict_close(actual, expected, atol=1e-12)


@pytest.mark.parity
def test_dy_d_vectorized_wrt_nonlinear_mean_matches_scalar_python_calls() -> None:
    x = np.column_stack(
        (np.array([-2, -1, 0, 1, 2], dtype=float), np.array([1, 3, 5, 7, 9], dtype=float))
    )
    y = x[:, 0] ** 2 + np.sin(x[:, 1])
    expected = _stacked_scalar_python_dy_d(x, y, [1, 2], "mean")
    actual = dy_d(x, y, wrt=[1, 2], eval_points="mean")
    _assert_dy_d_dict_close(actual, expected, atol=1e-12)


@pytest.mark.parity
@pytest.mark.parametrize("eval_points", ["median", "last", "obs", "apd"])
def test_dy_d_vectorized_wrt_non_mean_modes_match_scalar_python_calls(eval_points: str) -> None:
    x1 = np.linspace(-1.5, 1.5, 18)
    x2 = np.cos(np.linspace(0.0, 2.0, 18))
    x = np.column_stack((x1, x2))
    y = x[:, 0] ** 2 + 0.5 * x[:, 1] + np.sin(x[:, 0] * x[:, 1])
    expected = _stacked_scalar_python_dy_d(x, y, [1, 2], eval_points)
    actual = dy_d(x, y, wrt=np.array([1, 2]), eval_points=eval_points)
    _assert_dy_d_dict_close(actual, expected, atol=1e-12)


@pytest.mark.parity
def test_dy_d_vectorized_wrt_mixed_mean_matches_scalar_python_calls() -> None:
    x1 = np.linspace(-1.5, 1.5, 18)
    x2 = np.cos(np.linspace(0.0, 2.0, 18))
    x = np.column_stack((x1, x2))
    y = x[:, 0] ** 2 + 0.5 * x[:, 1] + np.sin(x[:, 0] * x[:, 1])
    expected = _stacked_scalar_python_dy_d(x, y, [1, 2], "mean", mixed=True)
    actual = dy_d(x, y, wrt=np.array([1, 2]), eval_points="mean", mixed=True)
    _assert_dy_d_dict_close(actual, expected, atol=1e-12)


@pytest.mark.parity
def test_dy_d_vectorized_wrt_numeric_eval_mixed_matches_scalar_python_calls() -> None:
    x = np.column_stack((np.linspace(-1.0, 1.0, 12), np.cos(np.linspace(0.0, 2.0, 12))))
    y = x[:, 0] ** 2 + x[:, 1]
    eval_points = np.array([0.1, 0.4], dtype=np.float64)
    expected = _stacked_scalar_python_dy_d(x, y, [1, 2], eval_points, mixed=True)
    actual = dy_d(x, y, wrt=np.array([1, 2]), eval_points=eval_points, mixed=True)
    _assert_dy_d_dict_close(actual, expected, atol=1e-12)


def _r_nns_diff(name: str, point: float) -> dict[str, float]:
    return _r_nns_diff_h(name, point, None)


def _r_nns_diff_h(name: str, point: float, h: float | None) -> dict[str, float]:
    result = nns_diff_custom(name, point, h)
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


def _stacked_scalar_python_dy_d(
    x: np.ndarray,
    y: np.ndarray,
    wrt_values: list[int],
    eval_points: object,
    *,
    mixed: bool = False,
) -> dict[str, np.ndarray]:
    outputs = [
        dy_d(x, y, wrt=wrt, eval_points=eval_points, mixed=mixed, messages=False)
        for wrt in wrt_values
    ]
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
    atol: float = DY_D_PARITY,
    rtol: float = 1e-7,
) -> None:
    assert actual.keys() == expected.keys()
    for key in actual:
        actual_values = np.asarray(actual[key], dtype=np.float64).reshape(-1)
        expected_values = np.asarray(expected[key], dtype=np.float64).reshape(-1)
        assert actual_values.shape == expected_values.shape
        np.testing.assert_allclose(
            actual_values,
            expected_values,
            atol=atol,
            rtol=rtol,
            equal_nan=True,
        )
