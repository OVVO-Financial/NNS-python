from __future__ import annotations

import numpy as np
import pytest

from pynns import dy_d, dy_dx, nns_diff


def test_nns_diff_constant_derivative_is_zero() -> None:
    result = nns_diff(lambda x: 12.0, 3.0)

    assert result["DERIVATIVE"] == pytest.approx(0.0)


def test_nns_diff_identity_derivative_is_one() -> None:
    result = nns_diff(lambda x: x, -2.0)

    assert result["DERIVATIVE"] == pytest.approx(1.0)


def test_nns_diff_smooth_function_derivative_has_bounded_error() -> None:
    point = 1.25
    result = nns_diff(np.sin, point)

    assert result["DERIVATIVE"] == pytest.approx(np.cos(point), abs=1e-6)


def test_dy_dx_numeric_eval_point_returns_derivative_table() -> None:
    x = np.linspace(-2.0, 2.0, 24)
    y = x + np.sin(x)

    result = dy_dx(x, y, eval_point=np.array([-1.0, 0.0, 1.0]))

    assert isinstance(result, dict)
    assert list(result) == ["eval.point", "first.derivative", "second.derivative"]
    assert all(value.shape == (3,) for value in result.values())
    assert np.all(np.isfinite(result["first.derivative"]))


def test_dy_d_vectorized_wrt_obs_is_implemented() -> None:
    x = np.random.RandomState(0).randn(40, 3)
    y = x[:, 0] + 2.0 * x[:, 1] - x[:, 2]

    result = dy_d(x, y, wrt=np.array([1, 2]), eval_points="obs")

    assert result.keys() == {"First", "Second"}
    assert result["First"].shape == (40, 2)
    assert result["Second"].shape == (40, 2)


def test_dy_d_vectorized_wrt_mixed_three_column_input_falls_back_to_first_second() -> None:
    x = np.random.RandomState(1).randn(40, 3)
    y = x[:, 0] + x[:, 1] + x[:, 2]

    result = dy_d(x, y, wrt=np.array([1, 2]), eval_points="mean", mixed=True)

    assert result.keys() == {"First", "Second"}
    assert result["First"].shape == (1, 2)
    assert result["Second"].shape == (1, 2)


def test_dy_d_vectorized_wrt_mixed_two_column_input_returns_mixed() -> None:
    x = np.random.RandomState(1).randn(40, 2)
    y = x[:, 0] + x[:, 1]

    result = dy_d(x, y, wrt=np.array([1, 2]), eval_points="mean", mixed=True)

    assert result.keys() == {"First", "Second", "Mixed"}
    assert result["First"].shape == (1, 2)
    assert result["Second"].shape == (1, 2)
    assert result["Mixed"].shape == (1, 2)


def test_dy_d_vectorized_wrt_obs_mixed_uses_pointwise_python_shape() -> None:
    x = np.random.RandomState(3).randn(24, 2)
    y = x[:, 0] ** 2 + x[:, 1]

    result = dy_d(x, y, wrt=np.array([1, 2]), eval_points="obs", mixed=True)

    assert result.keys() == {"First", "Second", "Mixed"}
    assert result["First"].shape == (24, 2)
    assert result["Second"].shape == (24, 2)
    assert result["Mixed"].shape == (24, 2)
    assert np.all(np.isfinite(result["Mixed"]))


def test_dy_d_vectorized_wrt_apd_mixed_remains_invalid() -> None:
    x = np.random.RandomState(1).randn(40, 2)
    y = x[:, 0] + x[:, 1]

    with pytest.raises(ValueError, match="Mixed Derivatives are only for 2 IV"):
        dy_d(x, y, wrt=np.array([1, 2]), eval_points="apd", mixed=True)


def test_dy_d_vectorized_wrt_mean_is_implemented() -> None:
    x = np.random.RandomState(2).randn(40, 2)
    y = x[:, 0] * 2.0 - x[:, 1]
    result = dy_d(x, y, wrt=np.array([1, 2]), eval_points="mean")

    assert isinstance(result, dict)
    assert result.keys() == {"First", "Second"}
    assert result["First"].shape == (1, 2)
    assert result["Second"].shape == (1, 2)


def test_dy_d_point_modes_preserve_linear_slope_direction() -> None:
    rng = np.random.default_rng(0)
    x = np.column_stack((rng.uniform(-2.0, 2.0, 120), rng.uniform(-1.0, 1.0, 120)))
    positive = 2.0 * x[:, 0] + 0.5 * x[:, 1]
    negative = -2.0 * x[:, 0] + 0.5 * x[:, 1]

    for eval_points in ("mean", "median", "last"):
        assert dy_d(x, positive, wrt=1, eval_points=eval_points)["First"][0] > 0.0
        assert dy_d(x, negative, wrt=1, eval_points=eval_points)["First"][0] < 0.0
