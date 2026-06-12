from __future__ import annotations

import numpy as np
import pytest
from _r import factor_dummy_custom
from _tolerances import EXACT

from pynns import encode_factor_codes, factor_2_dummy, factor_2_dummy_fr


@pytest.mark.parity
@pytest.mark.parametrize("full_rank", [False, True])
def test_factor_dummy_helpers_match_r_explicit_levels(full_rank: bool) -> None:
    values = ["B", "A", "B", "C", "A"]
    levels = ["A", "B", "C"]

    expected = factor_dummy_custom(values, levels, full_rank=full_rank)
    assert isinstance(expected, dict)
    actual = (
        factor_2_dummy_fr(values, levels=levels)
        if full_rank
        else factor_2_dummy(
            values,
            levels=levels,
        )
    )

    assert list(actual) == list(expected)
    for key, expected_values in expected.items():
        assert isinstance(expected_values, np.ndarray)
        np.testing.assert_allclose(actual[key], expected_values, atol=EXACT)


@pytest.mark.parity
def test_factor_2_dummy_drops_base_level_like_r() -> None:
    result = factor_2_dummy(["B", "A", "B"], levels=["A", "B", "C"])

    assert list(result) == ["B", "C"]
    np.testing.assert_array_equal(result["B"], np.array([1.0, 0.0, 1.0]))
    np.testing.assert_array_equal(result["C"], np.array([0.0, 0.0, 0.0]))


@pytest.mark.parity
def test_factor_2_dummy_fr_keeps_all_levels_like_r() -> None:
    result = factor_2_dummy_fr(["B", "A", "B"], levels=["A", "B", "C"])

    assert list(result) == ["A", "B", "C"]
    np.testing.assert_array_equal(result["A"], np.array([0.0, 1.0, 0.0]))
    np.testing.assert_array_equal(result["B"], np.array([1.0, 0.0, 1.0]))
    np.testing.assert_array_equal(result["C"], np.array([0.0, 0.0, 0.0]))


@pytest.mark.parity
def test_factor_helpers_numeric_and_logical_fallbacks() -> None:
    np.testing.assert_array_equal(factor_2_dummy([1, 2, 1])["x"], np.array([1.0, 2.0, 1.0]))
    np.testing.assert_array_equal(
        factor_2_dummy_fr([True, False, True])["x"],
        np.array([1.0, 0.0, 1.0]),
    )


@pytest.mark.parity
def test_encode_factor_codes_preserves_explicit_level_order() -> None:
    codes, levels = encode_factor_codes(["B", "A", "C"], levels=["C", "B", "A"])

    assert levels == ["C", "B", "A"]
    np.testing.assert_array_equal(codes, np.array([2.0, 3.0, 1.0]))


@pytest.mark.parity
def test_unseen_factor_values_match_r_na_dummy_behavior() -> None:
    result = factor_2_dummy_fr(["A", "D", "B"], levels=["A", "B", "C"])

    np.testing.assert_array_equal(result["A"], np.array([1.0, 0.0, 0.0]))
    np.testing.assert_array_equal(result["B"], np.array([0.0, 0.0, 1.0]))
    np.testing.assert_array_equal(result["C"], np.array([0.0, 0.0, 0.0]))


@pytest.mark.parity
def test_string_values_without_levels_are_rejected() -> None:
    with pytest.raises(ValueError, match="explicit levels"):
        factor_2_dummy(["A", "B"])
