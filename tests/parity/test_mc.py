from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from _r import RValue, nns_mc_grid, nns_mc_stat_summary

from pynns import nns_mc
from pynns.mc import _format_r_number, _generate_mc_rhos


@pytest.mark.parity
@pytest.mark.parametrize(
    ("lower", "upper", "by", "exp"),
    [
        (-1.0, 1.0, 0.5, 1.0),
        (-1.0, 1.0, 0.25, 2.0),
        (-0.5, 0.8, 0.1, 1.5),
        (0.0, 1.0, 0.2, 1.0),
        (-1.0, 0.0, 0.2, 1.0),
    ],
)
def test_nns_mc_rho_grid_matches_r(lower: float, upper: float, by: float, exp: float) -> None:
    expected = cast(
        dict[str, RValue],
        nns_mc_grid(lower_rho=lower, upper_rho=upper, by=by, exp=exp),
    )

    actual = _generate_mc_rhos(lower, upper, by, exp)
    actual_names = [f"rho = {_format_r_number(value)}" for value in actual]

    np.testing.assert_allclose(actual, _array(expected["values"]), atol=1e-12)
    assert actual_names == expected["names"]


@pytest.mark.parity
def test_nns_mc_return_names_match_r() -> None:
    x = np.linspace(-2.0, 2.0, 12) + 0.1 * np.sin(np.arange(12, dtype=np.float64))
    expected = cast(dict[str, RValue], nns_mc_grid(lower_rho=-1.0, upper_rho=1.0, by=1.0, exp=1.0))

    result = nns_mc(x, reps=2, lower_rho=-1.0, upper_rho=1.0, by=1.0, random_seed=10)

    assert set(result) == {"ensemble", "replicates"}
    assert list(result["replicates"].keys()) == expected["names"]
    assert len(result["replicates"]) == _array(expected["values"]).size


def test_nns_mc_sampling_vignette_smoke() -> None:
    x = np.linspace(1.0, 4.0, 20) + 0.1 * np.sin(np.arange(20, dtype=np.float64))

    result = nns_mc(x, reps=1, lower_rho=-1.0, upper_rho=1.0, by=0.5, random_seed=12)

    assert list(result["replicates"]) == [
        "rho = 1",
        "rho = 0.5",
        "rho = 0",
        "rho = -0.5",
        "rho = -1",
    ]
    assert result["ensemble"].shape == (20,)
    assert all(matrix.shape == (20, 1) for matrix in result["replicates"].values())


def test_nns_mc_sampling_vignette_target_drift_smoke() -> None:
    x = np.linspace(1.0, 4.0, 20) + 0.1 * np.sin(np.arange(20, dtype=np.float64))

    result = nns_mc(
        x,
        reps=1,
        lower_rho=-1.0,
        upper_rho=1.0,
        by=0.5,
        target_drift=0.05,
        random_seed=13,
    )

    assert result["ensemble"].shape == (20,)
    assert np.all(np.isfinite(result["ensemble"]))


@pytest.mark.parity
@pytest.mark.stochastic
def test_nns_mc_statistical_summary_is_close_to_r() -> None:
    x = (np.linspace(-2.0, 3.0, 25) + 0.2 * np.sin(np.arange(25, dtype=np.float64))).tolist()

    expected = np.asarray(
        nns_mc_stat_summary(x, reps=20, lower_rho=-1.0, upper_rho=1.0, by=1.0, seed=123)
    )
    result = nns_mc(
        np.asarray(x),
        reps=20,
        lower_rho=-1.0,
        upper_rho=1.0,
        by=1.0,
        random_seed=123,
    )
    block_sds = [
        np.median(np.std(matrix, axis=0, ddof=1)) for matrix in result["replicates"].values()
    ]
    actual = np.array(
        [
            np.mean(result["ensemble"]),
            np.std(result["ensemble"], ddof=1),
            np.median(block_sds),
        ]
    )

    np.testing.assert_allclose(actual, expected, rtol=0.4, atol=0.4)


def _array(value: RValue) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise AssertionError(f"Expected R array, got {type(value)!r}")
    return value
