from __future__ import annotations

import importlib
import tracemalloc
import warnings

import numpy as np
import pytest
from _tolerances import EXACT

from nns import lpm, lpm_ratio, nns_cdf, upm, upm_ratio

core_module = importlib.import_module("nns.core")


def test_mean_decomposes_into_upm_minus_lpm() -> None:
    x = np.array([-2.0, -1.0, 0.5, 3.0])

    assert np.isclose(x.mean(), upm(1, 0, x) - lpm(1, 0, x), atol=EXACT)


def test_population_variance_decomposes_into_second_partial_moments() -> None:
    x = np.array([-2.0, -1.0, 0.5, 3.0])
    target = x.mean()

    assert np.isclose(np.var(x, ddof=0), upm(2, target, x) + lpm(2, target, x), atol=EXACT)


def test_lpm_zero_at_sorted_points_is_empirical_cdf() -> None:
    x = np.array([-3.0, -1.0, 0.5, 2.0, 4.0])
    sorted_x = np.sort(x)
    expected = np.arange(1, x.size + 1) / x.size

    np.testing.assert_allclose(lpm(0, sorted_x, x), expected, atol=EXACT)


def test_degree_zero_partition_at_target_equality() -> None:
    x = np.array([1.0, 5.0, 10.0])

    assert lpm(0, 5.0, x) == pytest.approx(2 / 3, rel=EXACT)
    assert upm(0, 5.0, x) == pytest.approx(1 / 3, rel=EXACT)
    assert lpm(0, 5.0, x) + upm(0, 5.0, x) == pytest.approx(1.0, rel=EXACT)


def test_lpm_is_non_negative() -> None:
    x = np.array([-2.0, 0.0, 3.0])

    assert lpm(2, 1.0, x) >= 0


def test_upm_is_non_negative() -> None:
    x = np.array([-2.0, 0.0, 3.0])

    assert upm(2, 1.0, x) >= 0


def test_partial_moment_sum_positive_unless_constant_equal_to_target() -> None:
    x = np.array([-2.0, 0.0, 3.0])

    assert lpm(2, 1.0, x) + upm(2, 1.0, x) > 0
    assert lpm(2, 2.0, np.array([2.0, 2.0])) + upm(2, 2.0, np.array([2.0, 2.0])) == 0


def test_lpm_upm_symmetry() -> None:
    x = np.array([-2.0, -1.0, 0.5, 3.0])
    target = 0.75

    assert np.isclose(lpm(2, target, x), upm(2, -target, -x), atol=EXACT)


def test_lpm_ratio_bounds() -> None:
    x = np.array([-2.0, -1.0, 0.5, 3.0])
    result = lpm_ratio(2, 0.0, x)

    assert 0 <= result <= 1


def test_default_target_is_r_mean_for_a_constant_decimal_sample() -> None:
    from nns import co_lpm, co_upm

    # np.mean gives 0.6999999999999998 here; R's mean gives 0.7, so R reports 1 / 0.
    x = np.full(3, 0.7)

    assert lpm_ratio(0.0, None, x) == 1.0
    assert upm_ratio(0.0, None, x) == 0.0
    assert co_lpm(0.0, x, x, None, None) == 1.0
    assert co_upm(0.0, x, x, None, None) == 0.0


def test_lpm_ratio_and_upm_ratio_sum_to_one_when_defined() -> None:
    x = np.array([-2.0, -1.0, 0.5, 3.0])

    assert lpm_ratio(2, 0.0, x) + upm_ratio(2, 0.0, x) == pytest.approx(1.0, rel=EXACT)


def test_partial_moment_ratios_degenerate_denominator_use_degree0_ratio() -> None:
    # R fix 2cf9cbc: the 0/0 ratio (every observation equals the target) is the
    # degree-0 ratio, i.e. the ECDF value: LPM.ratio 1, UPM.ratio 0.
    x = np.array([2.0, 2.0])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        lower = lpm_ratio(2, 2.0, x)
        upper = upm_ratio(2, 2.0, x)
        lower_vector = lpm_ratio(1, np.array([2.0, 2.0]), x)

    assert lower == 1.0
    assert upper == 0.0
    np.testing.assert_array_equal(lower_vector, [1.0, 1.0])
    assert [warning for warning in caught if issubclass(warning.category, RuntimeWarning)] == []


@pytest.mark.parametrize("degree", [0.0, 0.5, 2.0])
def test_numpy_fallback_moments_are_blocked_over_targets_bit_identically(
    monkeypatch: pytest.MonkeyPatch, degree: float
) -> None:
    monkeypatch.setattr(core_module, "native_fn", lambda name: None)
    monkeypatch.setattr(core_module, "_TARGET_BLOCK", 64)
    rng = np.random.default_rng(7)
    x = rng.normal(size=301)
    targets = np.concatenate([x, rng.normal(size=50)])
    deviations = targets[:, np.newaxis] - x
    if degree == 0.0:
        dense_lower = np.mean(deviations >= 0.0, axis=1)
        dense_upper = np.mean(deviations < 0.0, axis=1)
    else:
        dense_lower = np.mean(np.maximum(0.0, deviations) ** degree, axis=1)
        dense_upper = np.mean(np.maximum(0.0, -deviations) ** degree, axis=1)

    np.testing.assert_array_equal(lpm(degree, targets, x), dense_lower)
    np.testing.assert_array_equal(upm(degree, targets, x), dense_upper)
    if degree > 0.0:
        dense_ratio = dense_lower / (dense_lower + dense_upper)
        np.testing.assert_array_equal(lpm_ratio(degree, targets, x), dense_ratio)


def test_numpy_fallback_cdf_does_not_allocate_an_n_by_n_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(core_module, "native_fn", lambda name: None)
    monkeypatch.setattr(core_module, "_TARGET_BLOCK", 64)
    n = 2000
    tracemalloc.start()
    try:
        nns_cdf(np.linspace(-1.0, 1.0, n), degree=2)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert peak < n * n * 8


@pytest.mark.parametrize("function", [lpm_ratio, upm_ratio])
def test_partial_moment_ratios_reject_non_finite_targets(function: object) -> None:
    # R fix 41c3297: LPM.ratio/UPM.ratio stop on a non-finite target or variable.
    x = np.array([1.0, 2.0, 3.0])
    for target in (np.nan, np.array([1.0, np.inf])):
        with pytest.raises(ValueError, match="target must contain only finite values"):
            function(1, target, x)  # type: ignore[operator]
    with pytest.raises(ValueError, match="x must contain only finite values"):
        function(0, 2.0, np.array([1.0, np.nan]))  # type: ignore[operator]


def test_lpm_upm_reject_non_finite_targets_like_r() -> None:
    x = np.array([1.0, 2.0, 3.0])
    for function in (lpm, upm):
        for degree in (0.0, 1.0):
            for target in (np.nan, np.inf, np.array([1.0, np.nan])):
                with pytest.raises(ValueError, match="`target` must be finite"):
                    function(degree, target, x)
