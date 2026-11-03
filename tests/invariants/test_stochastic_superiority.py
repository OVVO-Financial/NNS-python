from __future__ import annotations

import numpy as np
import pytest

from nns import nns_ss


@pytest.mark.stochastic
def test_nns_ss_confidence_interval_shapes_and_seed_determinism() -> None:
    x = np.linspace(-1.0, 2.0, 20) + 0.2 * np.sin(np.arange(20, dtype=np.float64))
    y = np.linspace(-1.5, 1.5, 20) + 0.3 * np.cos(np.arange(20, dtype=np.float64))

    first = nns_ss(x, y, confidence_interval=True, reps=6, ci=0.8, rho=0.0, random_seed=123)
    second = nns_ss(x, y, confidence_interval=True, reps=6, ci=0.8, rho=0.0, random_seed=123)
    third = nns_ss(x, y, confidence_interval=True, reps=6, ci=0.8, rho=0.0, random_seed=124)

    assert set(first) == {"p_gt", "p_tie", "p_star", "lower", "upper", "ci", "reps", "boot_vals"}
    assert first["ci"] == 0.8
    assert first["reps"] == 6
    assert first["boot_vals"].shape == (6,)
    assert np.isfinite(first["lower"])
    assert np.isfinite(first["upper"])
    np.testing.assert_array_equal(first["boot_vals"], second["boot_vals"])
    assert not np.array_equal(first["boot_vals"], third["boot_vals"])


@pytest.mark.stochastic
def test_nns_ss_constant_groups_bootstrap_to_themselves() -> None:
    r = nns_ss(
        np.array([1.0, 1.0, 1.0, 1.0]),
        np.array([2.0, 2.0, 2.0, 2.0]),
        confidence_interval=True,
        reps=5,
        rho=0.0,
        random_seed=1,
    )
    assert (r["p_star"], r["lower"], r["upper"]) == (0.0, 0.0, 0.0)


def test_nns_ss_accepts_numpy_bool_flags() -> None:
    x = np.linspace(-1.0, 2.0, 20)
    y = np.linspace(-1.5, 1.5, 20)

    assert nns_ss(x, y, confidence_interval=np.False_) == nns_ss(x, y)
    result = nns_ss(x, y, confidence_interval=np.True_, reps=4, rho=0.0, random_seed=3)
    expected = nns_ss(x, y, confidence_interval=True, reps=4, rho=0.0, random_seed=3)
    np.testing.assert_array_equal(result["boot_vals"], expected["boot_vals"])
    with pytest.raises(ValueError, match="TRUE/FALSE"):
        nns_ss(x, y, confidence_interval=1)  # type: ignore[arg-type]


def test_nns_ss_gives_an_interval_when_one_group_is_constant() -> None:
    # R NNS.SS: NNS.meboot cannot bootstrap a constant group, so this errored.
    x = np.ones(30)
    y = np.tile([0.0, 1.0], 15)
    r = nns_ss(x, y, confidence_interval=True, reps=99, random_seed=1)
    assert r["p_star"] == 0.75
    assert np.all(np.isfinite(r["boot_vals"]))
    assert r["lower"] <= 0.75 <= r["upper"]
    r = nns_ss(y, x, confidence_interval=True, reps=99, random_seed=1)
    assert r["p_star"] == 0.25
    assert r["lower"] <= 0.25 <= r["upper"]


def test_nns_ss_interval_for_two_constant_groups_is_the_point_estimate() -> None:
    r = nns_ss(np.full(10, 2.0), np.ones(15), confidence_interval=True, reps=9)
    assert (r["lower"], r["upper"]) == (1.0, 1.0)
    r = nns_ss(np.ones(10), np.ones(15), confidence_interval=True, reps=9)
    assert (r["lower"], r["upper"]) == (0.5, 0.5)
