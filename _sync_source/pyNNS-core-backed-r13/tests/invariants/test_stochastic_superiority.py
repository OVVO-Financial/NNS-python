from __future__ import annotations

import numpy as np
import pytest

from pynns import nns_ss


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
def test_nns_ss_degenerate_ci_raises_like_meboot_path() -> None:
    with pytest.raises(ValueError):
        nns_ss(
            np.array([1.0, 1.0, 1.0, 1.0]),
            np.array([2.0, 2.0, 2.0, 2.0]),
            confidence_interval=True,
            reps=5,
            rho=0.0,
            random_seed=1,
        )
