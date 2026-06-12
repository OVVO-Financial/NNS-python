from __future__ import annotations

import numpy as np
from _tolerances import EXACT

from pynns import co_lpm, co_upm, d_lpm, d_upm, lpm, upm


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
