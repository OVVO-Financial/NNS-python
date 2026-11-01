from __future__ import annotations

import numpy as np
import pytest
from _tolerances import EXACT

from nns import co_lpm, co_upm, d_lpm, d_upm, lpm, lpm_ratio, pm_matrix, upm


def test_co_lpm_self_equals_lpm_with_doubled_degree() -> None:
    x = np.array([-2.0, -1.0, 0.5, 3.0])
    target = 0.25

    assert np.isclose(co_lpm(2, x, x, target, target), lpm(4, target, x), atol=EXACT)


def test_co_upm_self_equals_upm_with_doubled_degree() -> None:
    x = np.array([-2.0, -1.0, 0.5, 3.0])
    target = 0.25

    assert np.isclose(co_upm(2, x, x, target, target), upm(4, target, x), atol=EXACT)


def test_divergent_moment_transpose_relation() -> None:
    x = np.array([-2.0, -1.0, 0.5, 3.0])
    y = np.array([1.0, -0.5, 2.0, -3.0])
    target_x = x.mean()
    target_y = y.mean()

    assert np.isclose(
        d_lpm(1, 1, x, y, target_x, target_y),
        d_upm(1, 1, y, x, target_y, target_x),
        atol=EXACT,
    )


def test_covariance_decomposition_with_per_variable_means() -> None:
    x = np.array([-2.0, -1.0, 0.5, 3.0])
    y = np.array([1.0, -0.5, 2.0, -3.0])
    target_x = x.mean()
    target_y = y.mean()

    decomposition = (
        co_lpm(1, x, y, target_x, target_y)
        + co_upm(1, x, y, target_x, target_y)
        - d_lpm(1, 1, x, y, target_x, target_y)
        - d_upm(1, 1, x, y, target_x, target_y)
    )

    assert np.isclose(np.cov(x, y, ddof=0)[0, 1], decomposition, atol=EXACT)


def test_co_moments_are_non_negative() -> None:
    x = np.array([-2.0, -1.0, 0.5, 3.0])
    y = np.array([1.0, -0.5, 2.0, -3.0])

    assert co_lpm(2, x, y, 0.0, 0.0) >= 0
    assert co_upm(2, x, y, 0.0, 0.0) >= 0
    assert d_lpm(2, 2, x, y, 0.0, 0.0) >= 0
    assert d_upm(2, 2, x, y, 0.0, 0.0) >= 0


def test_co_lpm_is_symmetric() -> None:
    x = np.array([-2.0, -1.0, 0.5, 3.0])
    y = np.array([1.0, -0.5, 2.0, -3.0])
    target_x = x.mean()
    target_y = y.mean()

    assert np.isclose(
        co_lpm(2, x, y, target_x, target_y),
        co_lpm(2, y, x, target_y, target_x),
        atol=EXACT,
    )


def test_co_upm_is_symmetric() -> None:
    x = np.array([-2.0, -1.0, 0.5, 3.0])
    y = np.array([1.0, -0.5, 2.0, -3.0])
    target_x = x.mean()
    target_y = y.mean()

    assert np.isclose(
        co_upm(2, x, y, target_x, target_y),
        co_upm(2, y, x, target_y, target_x),
        atol=EXACT,
    )


def test_co_moments_and_pm_matrix_reject_non_finite_input() -> None:
    # Patched R: Co.LPM/Co.UPM/D.LPM/D.UPM/PM.matrix stop on NA/Inf instead of
    # counting NA observations in n (Co.LPM(0, x, x, 6, 6) was 0.8 with one NA).
    x = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
    finite = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    for function in (co_lpm, co_upm):
        with pytest.raises(ValueError, match="x must contain only finite values"):
            function(0, x, finite, 6.0, 6.0)
        with pytest.raises(ValueError, match="y must contain only finite values"):
            function(0, finite, x, 6.0, 6.0)
        with pytest.raises(ValueError, match="target_x must contain only finite values"):
            function(0, finite, finite, np.nan, 6.0)
    for function in (d_lpm, d_upm):
        with pytest.raises(ValueError, match="x must contain only finite values"):
            function(1, 1, x, finite, 0.0, 0.0)
        with pytest.raises(ValueError, match="target_y must contain only finite values"):
            function(1, 1, finite, finite, 0.0, np.inf)
    with pytest.raises(ValueError, match="variable must contain only finite values"):
        pm_matrix(1, 1, None, np.column_stack([x, finite]), pop_adj=False)
    with pytest.raises(ValueError, match="target must contain only finite values"):
        pm_matrix(1, 1, np.nan, np.column_stack([finite, finite]), pop_adj=False)


@pytest.mark.parametrize("degree", [np.nan, -1.0])
def test_partial_moments_reject_nan_and_negative_degrees(degree: float) -> None:
    # Patched R stops on a negative or NaN degree; a NaN used to pass `degree < 0`.
    x = np.array([1.0, 2.0, 3.0])
    variable = np.column_stack([x, x[::-1]])
    calls = [
        lambda: lpm(degree, 2.0, x),
        lambda: upm(degree, 2.0, x),
        lambda: lpm_ratio(degree, 2.0, x),
        lambda: co_lpm(degree, x, x, 2.0, 2.0),
        lambda: co_upm(1.0, x, x, 2.0, 2.0, degree_y=degree),
        lambda: pm_matrix(degree, 1, None, variable, pop_adj=False),
        lambda: pm_matrix(1, degree, None, variable, pop_adj=False),
    ]
    for call in calls:
        with pytest.raises(ValueError, match="degree must be non-negative"):
            call()
