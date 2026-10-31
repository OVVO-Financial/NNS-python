from __future__ import annotations

import numpy as np
import pytest

from nns import nns_part


def test_nns_part_return_shape_and_quadrant_lengths() -> None:
    x = np.linspace(-2.0, 2.0, 100)
    y = np.sin(x)

    result = nns_part(x, y, order=3, obs_req=3, min_obs_stop=False)
    dt = result["dt"]
    rp = result["regression.points"]
    assert isinstance(dt, dict)
    assert isinstance(rp, dict)

    quadrants = dt["quadrant"].astype(str)
    prior = dt["prior.quadrant"].astype(str)

    assert dt["x"].shape == x.shape
    assert dt["y"].shape == y.shape
    assert all(value.startswith("q") for value in quadrants)
    assert all(value.startswith(("q", "pq")) for value in prior)
    assert all(
        len(prev) == 2 if prev == "pq" else len(prev) == len(current) - 1
        for current, prev in zip(quadrants, prior, strict=True)
    )
    assert rp["quadrant"].size == np.unique(prior).size


def test_nns_part_order_is_bounded() -> None:
    x = np.linspace(0.0, 1.0, 64)
    y = x[::-1]

    result = nns_part(x, y, order=20, obs_req=0, min_obs_stop=False)

    assert 0 <= result["order"] <= int(np.floor(np.log2(x.size)))


def test_nns_part_order_max_returns_maximum_partition() -> None:
    # The repaired NNS.part supports order="max" (the kNN-equivalent maximum
    # partition) instead of rejecting it.
    x = np.linspace(0.0, 1.0, 10)

    result = nns_part(x, x, order="max")

    assert set(result) == {"order", "dt", "regression.points"}
    assert np.asarray(result["regression.points"]["x"]).size == x.size


def test_nns_part_accepts_numpy_and_integral_float_orders() -> None:
    # R accepts any integral numeric order/obs.req (e.g. 3 or 3.0).
    rng = np.random.default_rng(3)
    x = rng.normal(size=200)
    y = x + rng.normal(size=200)
    expected = nns_part(x, y, order=3, obs_req=8)
    for order, obs_req in [(np.int64(3), np.int32(8)), (3.0, 8.0), (np.float64(3.0), 8)]:
        actual = nns_part(x, y, order=order, obs_req=obs_req)
        assert actual["order"] == expected["order"]
        np.testing.assert_array_equal(actual["dt"]["quadrant"], expected["dt"]["quadrant"])
        np.testing.assert_array_equal(
            actual["regression.points"]["y"], expected["regression.points"]["y"]
        )


@pytest.mark.parametrize("order", [0, -1, 2.5, float("inf"), float("nan")])
def test_nns_part_rejects_non_positive_or_fractional_order(order: float) -> None:
    # R: "[order] must be NULL, 'max', or a positive integer." (order = 0 was
    # silently treated as 1.)
    x = np.arange(20.0)
    with pytest.raises(ValueError, match="positive integer"):
        nns_part(x, x, order=order)


@pytest.mark.parametrize("obs_req", [-1, 2.5, float("nan")])
def test_nns_part_rejects_invalid_obs_req(obs_req: float) -> None:
    x = np.arange(20.0)
    with pytest.raises(ValueError, match="obs_req"):
        nns_part(x, x, obs_req=obs_req)


@pytest.mark.parametrize("bad", [True, "3", [3]])
def test_nns_part_rejects_non_numeric_order(bad: object) -> None:
    x = np.arange(20.0)
    with pytest.raises((TypeError, ValueError)):
        nns_part(x, x, order=bad)  # type: ignore[arg-type]
