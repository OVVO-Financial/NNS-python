from __future__ import annotations

import numpy as np
import pytest

from pynns import nns_part


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


def test_nns_part_rejects_order_max_instead_of_matching_installed_r_useless_na_path() -> None:
    x = np.linspace(0.0, 1.0, 10)

    with pytest.raises(TypeError):
        nns_part(x, x, order="max")
