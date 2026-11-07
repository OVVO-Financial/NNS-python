from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from _r import RValue, nns_meboot_diagnostics, nns_meboot_stat_summary
from _tolerances import COMPOUND

from nns import nns_meboot


def _diagnostic_series() -> list[np.ndarray]:
    idx = np.arange(20, dtype=np.float64)
    return [
        np.linspace(-5.0, 5.0, 20) + 0.1 * np.sin(idx),
        np.array([1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0, 29.0]),
        np.array([-3.0, -2.5, -1.7, -0.2, 0.1, 1.4, 2.2, 4.9]),
    ]


@pytest.mark.parity
def test_nns_meboot_rho_none_matches_installed_r_empty_behavior() -> None:
    assert nns_meboot(np.arange(1, 8, dtype=np.float64), rho=None) == {}


@pytest.mark.parity
def test_nns_meboot_length_one_returns_x_only() -> None:
    result = nns_meboot(np.array([5.0]), rho=0.0)

    assert set(result) == {"x"}
    np.testing.assert_array_equal(result["x"], np.array([5.0]))


@pytest.mark.parity
@pytest.mark.parametrize("x", _diagnostic_series())
def test_nns_meboot_deterministic_diagnostics_match_r(x: np.ndarray) -> None:
    expected = nns_meboot_diagnostics(x.tolist(), rho=0.0, reps=2, seed=1)
    actual = nns_meboot(
        x,
        reps=2,
        rho=0.0,
        random_seed=1,
        force_clt=False,
        expand_sd=False,
    )
    expected_dict = cast(dict[str, RValue], expected)

    for key in ("x", "xx", "z", "dv", "desintxb", "ordxx"):
        np.testing.assert_allclose(actual[key], _array(expected_dict[key]), atol=COMPOUND)
    for key in ("dvtrim", "xmin", "xmax"):
        assert actual[key] == pytest.approx(_scalar(expected_dict[key]), abs=COMPOUND)
    assert actual["kappa"] == expected_dict["kappa"]


@pytest.mark.parity
def test_nns_meboot_symmetric_diagnostics_match_r() -> None:
    x = np.linspace(-5.0, 5.0, 20) + 0.1 * np.sin(np.arange(20, dtype=np.float64))

    expected = nns_meboot_diagnostics(x.tolist(), rho=0.0, reps=2, sym=True, seed=2)
    actual = nns_meboot(
        x,
        reps=2,
        rho=0.0,
        sym=True,
        random_seed=2,
        force_clt=False,
        expand_sd=False,
    )
    expected_dict = cast(dict[str, RValue], expected)

    for key in ("xx", "z", "desintxb"):
        np.testing.assert_allclose(actual[key], _array(expected_dict[key]), atol=COMPOUND)


@pytest.mark.parity
def test_nns_meboot_errors_match_r_categories() -> None:
    with pytest.raises(ValueError, match="missing values"):
        nns_meboot(np.array([1.0, np.nan, 3.0]), rho=0.0)

    with pytest.raises(ValueError):
        nns_meboot(np.array([1.0, np.inf, 3.0]), rho=0.0)


@pytest.mark.parity
@pytest.mark.stochastic
def test_nns_meboot_statistical_summary_is_close_to_r() -> None:
    x = (np.linspace(-3.0, 4.0, 30) + 0.2 * np.sin(np.arange(30, dtype=np.float64))).tolist()

    expected = np.asarray(nns_meboot_stat_summary(x, rho=0.0, reps=100, seed=123))
    actual_result = nns_meboot(
        np.asarray(x),
        reps=100,
        rho=0.0,
        random_seed=123,
    )
    replicates = actual_result["replicates"]
    actual = np.array(
        [
            np.mean(actual_result["ensemble"]),
            np.std(actual_result["ensemble"], ddof=1),
            np.median(np.mean(replicates, axis=0)),
            np.median(np.std(replicates, axis=0, ddof=1)),
        ]
    )

    np.testing.assert_allclose(actual, expected, rtol=0.35, atol=0.35)


def _array(value: RValue) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise AssertionError(f"Expected R array, got {type(value)!r}")
    return value


def _scalar(value: RValue) -> float:
    if not isinstance(value, np.ndarray):
        raise AssertionError(f"Expected R scalar array, got {type(value)!r}")
    return float(value)


@pytest.mark.parity
def test_nns_meboot_scale_adjustment_uses_residual_scale_like_r() -> None:
    # Patched R measures kappa against the residuals' mean and sd (it used the
    # levels', which shrank every replicate toward mean(x)).
    x = 2.0 * np.arange(60, dtype=np.float64) + 5.0 * np.sin(np.arange(60, dtype=np.float64))

    expected = nns_meboot_diagnostics(x.tolist(), rho=0.0, reps=2, scl_adjustment=True, seed=1)
    actual = nns_meboot(x, reps=2, rho=0.0, scl_adjustment=True, random_seed=1)

    kappa = _scalar(cast(dict[str, RValue], expected)["kappa"])
    assert actual["kappa"] == pytest.approx(kappa, abs=COMPOUND)
    assert abs(kappa) < 0.1


def test_nns_meboot_scale_adjustment_scales_the_residuals_not_the_levels() -> None:
    t = np.arange(60, dtype=np.float64)
    x = 2.0 * t + 5.0 * np.sin(t)
    kwargs = {"reps": 2, "rho": 0.0, "expand_sd": False, "force_clt": False, "random_seed": 1}

    adjusted = nns_meboot(x, scl_adjustment=True, **kwargs)
    plain = nns_meboot(x, scl_adjustment=False, **kwargs)

    time = t + 1.0
    slope, intercept = np.polyfit(time, x, 1)
    fitted = (intercept + slope * time)[:, np.newaxis]
    residual_mean = float(np.mean(x - fitted[:, 0]))
    kappa = adjusted["kappa"]
    assert kappa is not None and abs(kappa) > 1e-3
    np.testing.assert_allclose(
        adjusted["replicates"] - plain["replicates"],
        kappa * (plain["replicates"] - fitted - residual_mean),
        atol=1e-5,
    )


@pytest.mark.parity
@pytest.mark.stochastic
@pytest.mark.parametrize(("drift", "target_drift"), [(False, None), (True, 0.05)])
def test_nns_meboot_changed_drift_keeps_residual_scale_like_r(
    drift: bool, target_drift: float | None
) -> None:
    # Patched R skips the level-space sd expansion once the original drift is
    # replaced, and centres the new baseline on mean(x).
    t = np.arange(30, dtype=np.float64)
    x = (0.3 * t + np.sin(t)).tolist()

    expected = np.asarray(
        nns_meboot_stat_summary(
            x, rho=0.0, reps=100, seed=123, drift=drift, target_drift=target_drift
        )
    )
    result = nns_meboot(
        np.asarray(x), reps=100, rho=0.0, drift=drift, target_drift=target_drift, random_seed=123
    )
    replicates = result["replicates"]
    actual = np.array(
        [
            np.mean(result["ensemble"]),
            np.std(result["ensemble"], ddof=1),
            np.median(np.mean(replicates, axis=0)),
            np.median(np.std(replicates, axis=0, ddof=1)),
        ]
    )

    np.testing.assert_allclose(actual, expected, rtol=0.35, atol=0.35)
    assert actual[3] < 0.5 * float(np.std(x, ddof=1))
    # force_clt spreads the replicate means as the new baseline plus the original
    # residuals, not as x (whose sd includes the replaced drift).
    time = np.arange(1.0, 31.0)
    slope, intercept = np.polyfit(time, x, 1)
    residuals = np.asarray(x) - (intercept + slope * time)
    new_slope = 0.0 if target_drift is None else target_drift
    reference = float(np.mean(x)) + new_slope * (time - 15.5) + residuals
    np.testing.assert_allclose(
        np.std(np.mean(replicates, axis=0), ddof=1),
        np.std(reference, ddof=1) / np.sqrt(100.0),
        rtol=1e-4,
    )
