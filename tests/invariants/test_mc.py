from __future__ import annotations

import numpy as np
import pytest

from nns import nns_mc

pytestmark = pytest.mark.stochastic


def test_nns_mc_shapes_and_finite_outputs() -> None:
    x = np.linspace(-2.0, 4.0, 30) + 0.1 * np.sin(np.arange(30, dtype=np.float64))

    result = nns_mc(x, reps=4, lower_rho=-1.0, upper_rho=1.0, by=1.0, random_seed=21)

    assert result["ensemble"].shape == (30,)
    assert len(result["replicates"]) == 3
    assert np.all(np.isfinite(result["ensemble"]))
    for matrix in result["replicates"].values():
        assert matrix.shape == (30, 4)
        assert np.all(np.isfinite(matrix))


def test_nns_mc_random_seed_is_reproducible() -> None:
    x = np.linspace(-2.0, 4.0, 30) + 0.1 * np.sin(np.arange(30, dtype=np.float64))

    first = nns_mc(x, reps=3, lower_rho=-1.0, upper_rho=1.0, by=1.0, random_seed=22)
    second = nns_mc(x, reps=3, lower_rho=-1.0, upper_rho=1.0, by=1.0, random_seed=22)
    third = nns_mc(x, reps=3, lower_rho=-1.0, upper_rho=1.0, by=1.0, random_seed=23)

    np.testing.assert_array_equal(first["ensemble"], second["ensemble"])
    assert not np.array_equal(first["ensemble"], third["ensemble"])
    for key in first["replicates"]:
        np.testing.assert_array_equal(first["replicates"][key], second["replicates"][key])


def test_nns_mc_xmin_xmax_clipping_is_respected() -> None:
    x = np.linspace(-2.0, 4.0, 30) + 0.1 * np.sin(np.arange(30, dtype=np.float64))

    result = nns_mc(
        x,
        reps=3,
        lower_rho=-1.0,
        upper_rho=1.0,
        by=1.0,
        xmin=-1.0,
        xmax=2.0,
        random_seed=24,
    )

    for matrix in result["replicates"].values():
        assert np.min(matrix) >= -1.0
        assert np.max(matrix) <= 2.0


def test_nns_mc_lower_greater_than_upper_errors() -> None:
    x = np.linspace(-2.0, 4.0, 30) + 0.1 * np.sin(np.arange(30, dtype=np.float64))

    with pytest.raises(ValueError, match="rho grid"):
        nns_mc(x, lower_rho=1.0, upper_rho=-1.0, by=0.5)


def test_nns_mc_zero_by_errors() -> None:
    x = np.linspace(-2.0, 4.0, 30) + 0.1 * np.sin(np.arange(30, dtype=np.float64))

    with pytest.raises(ValueError, match="by"):
        nns_mc(x, by=0.0)


def test_nns_mc_recycles_vector_target_drift_along_rho_grid() -> None:
    t = np.arange(1, 41, dtype=np.float64)
    x = 50.0 + t + 3.0 * np.sin(t)

    with pytest.warns(RuntimeWarning, match="not a multiple"):
        result = nns_mc(x, reps=4, by=1.0, target_drift=[0.5, 4.0], random_seed=3)

    assert list(result["replicates"]) == ["rho = 1", "rho = 0", "rho = -1"]
    assert result["ensemble"].shape == (40,)


def test_nns_mc_target_drift_takes_precedence_over_scale_like_r() -> None:
    x = np.linspace(1.0, 5.0, 30) + 0.2 * np.sin(np.arange(30, dtype=np.float64))

    both = nns_mc(x, reps=3, by=1.0, target_drift=0.3, target_drift_scale=5.0, random_seed=8)
    drift_only = nns_mc(x, reps=3, by=1.0, target_drift=0.3, random_seed=8)

    np.testing.assert_array_equal(both["ensemble"], drift_only["ensemble"])


def test_nns_mc_rejects_target_drift_longer_than_rho_grid() -> None:
    x = np.linspace(1.0, 5.0, 30)

    with pytest.raises(ValueError, match="rho grid"):
        nns_mc(x, reps=3, by=1.0, target_drift=[0.1, 0.2, 0.3, 0.4])
