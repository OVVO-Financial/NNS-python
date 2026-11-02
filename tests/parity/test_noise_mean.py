"""noise_reduction = "mean" uses R's long-double means.

np.mean of 30 tied 19.58 values is 19.580000000000002; R returns 19.58. On
Boston's ``indus`` that ulp moved a partition boundary above the tied
observations and changed 30 multivariate ``NNS.ID`` values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from _r import nns

from nns import nns_part, nns_reg
from nns._rmean import r_group_means_no_na, r_mean, r_mean_no_na

BOSTON_CSV = Path(__file__).resolve().parents[1] / "data" / "boston_housing.csv"


def _boston() -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(BOSTON_CSV, delimiter=",", skip_header=1)
    return data[:, :-1], data[:, -1]


def _r_reg_mean(x: np.ndarray, y: np.ndarray, point_est: np.ndarray) -> Any:
    # Positional NNS.reg arguments, as in tests/parity/test_regression.py.
    return nns(
        "NNS.reg", x.tolist(), y.tolist(), False, None, None, None, None,
        point_est.tolist(), "top", True, False, False, False, None, 0, None, False,
        "mean", None, None, False, False,
    )


def _floats(values: Any) -> np.ndarray:
    return np.asarray([np.nan if v is None else v for v in np.ravel(values)], dtype=float)


@pytest.mark.parity
def test_boston_multivariate_mean_ids_and_partitions_match_r() -> None:
    x, y = _boston()
    expected = _r_reg_mean(x, y, x[:20])
    actual = nns_reg(x, y, noise_reduction="mean", point_est=x[:20])

    ids = np.asarray(actual["Fitted.xy"]["NNS.ID"]).astype(str)
    np.testing.assert_array_equal(ids, np.asarray(expected["Fitted.xy"]["NNS.ID"], dtype=str))
    for name, column in actual["rhs.partitions"].items():
        np.testing.assert_allclose(
            np.asarray(column, dtype=float),
            _floats(expected["rhs.partitions"][name]),
            rtol=1e-13,
            atol=0.0,
        )
    for name, column in actual["RPM"].items():
        np.testing.assert_allclose(
            np.asarray(column, dtype=float), _floats(expected["RPM"][name]), rtol=1e-13
        )
    np.testing.assert_allclose(
        np.asarray(actual["Fitted.xy"]["y.hat"], dtype=float),
        _floats(expected["Fitted.xy"]["y.hat"]),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(actual["Point.est"], dtype=float),
        _floats(expected["Point.est"]),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parity
@pytest.mark.parametrize("part_type", [None, "XONLY"])
def test_nns_part_mean_on_tied_values_matches_r(part_type: str | None) -> None:
    x, _ = _boston()
    indus, nox = x[:, 2], x[:, 4]
    expected = nns("NNS.part", indus.tolist(), nox.tolist(), False, part_type, 4, 0, True, "mean")
    actual = nns_part(
        indus, nox, type=part_type, order=4, obs_req=0, min_obs_stop=True,
        noise_reduction="mean",
    )
    rp, expected_rp = actual["regression.points"], expected["regression.points"]
    np.testing.assert_array_equal(
        np.asarray(rp["quadrant"]).astype(str), np.asarray(expected_rp["quadrant"], dtype=str)
    )
    np.testing.assert_array_equal(
        np.asarray(actual["dt"]["quadrant"]).astype(str),
        np.asarray(expected["dt"]["quadrant"], dtype=str),
    )
    for key in ("x", "y"):
        np.testing.assert_allclose(
            np.asarray(rp[key], dtype=float), _floats(expected_rp[key]), rtol=1e-14, atol=0.0
        )


def test_r_means_are_exact_on_tied_values() -> None:
    tied = np.full(30, 19.58)
    assert float(np.mean(tied)) != 19.58  # the drift this guards against
    assert r_mean(tied) == 19.58
    assert r_mean_no_na(tied) == 19.58
    groups = np.repeat(np.arange(3), 10)
    values = np.repeat([19.58, 0.1, 6.2], 10)
    np.testing.assert_array_equal(r_group_means_no_na(values, groups, 3), [19.58, 0.1, 6.2])


def test_r_means_follow_r_rules() -> None:
    # mean_no_na drops NaN only; r_mean is applied to finite values by callers.
    assert r_mean_no_na(np.array([1.0, np.nan, 2.0])) == 1.5
    assert np.isnan(r_mean_no_na(np.array([np.nan])))
    assert r_mean_no_na(np.array([1.0, np.inf])) == np.inf
    assert np.isnan(r_mean(np.array([], dtype=float)))
    assert r_mean(np.array([-0.0, -0.0])) == 0.0
    assert not np.signbit(r_mean(np.array([-0.0])))
    # R: mean(c(1e308, 1e308)) is 1e308 (long double avoids the overflow).
    assert r_mean(np.array([1e308, 1e308])) == 1e308
