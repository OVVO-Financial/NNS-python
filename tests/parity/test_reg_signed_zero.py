"""-0.0 and 0.0 are one group key in R.

R 14.2 groups by ``match(x, sort(unique(x)))``; ``unique()`` treats -0 and 0 as
equal and keeps the first one, so the group's key carries the sign of the zero
that appears first.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from _r import nns

from nns import nns_reg
from nns._reg_engine import _group_reduce_sorted


def _r_reg(x: np.ndarray, y: np.ndarray, order: Any, noise: str, point: np.ndarray) -> Any:
    # Positional NNS.reg arguments, as in tests/parity/test_regression.py.
    return nns(
        "NNS.reg",
        x.tolist(),
        y.tolist(),
        False,
        order,
        None,
        None,
        None,
        point.tolist(),
        "top",
        True,
        False,
        False,
        False,
        None,
        0,
        None,
        False,
        noise,
        None,
        None,
        False,
        False,
    )


def _signed_zero_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(29)
    x = np.round(rng.normal(size=120), 1)  # np.round(-0.04, 1) is -0.0
    x[:6] = [-0.0, 0.0, -0.0, 0.0, -0.0, 0.0]
    y = np.sin(x) + np.round(rng.normal(size=120), 2)
    return x, y


@pytest.mark.parity
@pytest.mark.parametrize("noise", ["off", "median"])
def test_order_max_groups_signed_zeros_like_r(noise: str) -> None:
    x, y = _signed_zero_data()
    point = np.array([-0.05, 0.0, 0.05])
    expected = _r_reg(x, y, "max", noise, point)
    actual = nns_reg(x, y, order="max", noise_reduction=noise, point_est=point)
    rp, expected_rp = actual["regression.points"], expected["regression.points"]
    assert np.asarray(rp["x"]).size == np.unique(x).size
    for key in ("x", "y"):
        np.testing.assert_allclose(np.asarray(rp[key]), np.asarray(expected_rp[key]), atol=1e-14)
    np.testing.assert_allclose(
        np.asarray(actual["Fitted.xy"]["y.hat"]),
        np.asarray(expected["Fitted.xy"]["y.hat"]),
        atol=1e-14,
    )
    np.testing.assert_allclose(
        np.asarray(actual["Point.est"]), np.asarray(expected["Point.est"]), atol=1e-14
    )


@pytest.mark.parametrize(
    ("keys", "negative"), [([-0.0, 0.0, -0.0, 1.0], True), ([0.0, -0.0, -0.0, 1.0], False)]
)
def test_group_reduce_merges_signed_zero_keys(keys: list[float], negative: bool) -> None:
    # R 14.2: x <- c(-0, 0, -0, 1); 1 / sort(unique(x)) is c(-Inf, 1).
    values = np.array([1.0, 2.0, 3.0, 4.0])
    out_keys, reduced = _group_reduce_sorted(np.array(keys), values, "median", False)
    np.testing.assert_array_equal(out_keys, [0.0, 1.0])
    assert bool(np.signbit(out_keys[0])) is negative
    np.testing.assert_array_equal(reduced, [2.0, 4.0])
