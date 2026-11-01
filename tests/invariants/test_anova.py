from __future__ import annotations

import numpy as np
import pytest

from nns import nns_anova


def test_nns_anova_identical_degenerate_groups_match_r_convention() -> None:
    x = np.linspace(-2.0, 2.0, 100)
    constant = np.ones_like(x)

    result = nns_anova(constant, constant, confidence_interval=None)

    # The degree-1 ratios at a point mass are the degree-0 ratios, so
    # the CDFs are 1 and the patched R reports Certainty 0 (13.3 gave NaN).
    assert isinstance(result, dict)
    assert result["Control_CDF"] == result["Treatment_CDF"] == 1.0
    assert result["Certainty"] == 0.0
    assert result["Control"] == result["Treatment"]


def test_nns_anova_binary_output_structure_without_ci_matches_r_shape() -> None:
    x = np.linspace(-1.0, 1.0, 50)
    y = x + 0.2

    result = nns_anova(x, y, confidence_interval=None)

    assert list(result) == [
        "Control",
        "Treatment",
        "Grand_Statistic",
        "Control_CDF",
        "Treatment_CDF",
        "Certainty",
    ]


def test_nns_anova_pairwise_matrix_is_symmetric_with_unit_diagonal() -> None:
    x = np.linspace(-2.0, 2.0, 80)
    groups = [x, x + 0.2, np.sin(x)]

    result = nns_anova(groups, confidence_interval=None, pairwise=True)

    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, result.T)
    np.testing.assert_allclose(np.diag(result), np.ones(3))
    assert np.all((0.0 <= result) & (result <= 1.0))


@pytest.mark.stochastic
def test_nns_anova_robust_degenerate_inputs_do_not_crash() -> None:
    x = np.ones(20)

    result = nns_anova(x, x, robust=True, random_seed=123)

    assert isinstance(result, dict)
    assert "Robust Certainty Estimate" in result
    assert result["Robust Certainty Estimate"] == 0.0


@pytest.mark.stochastic
def test_nns_anova_robust_reproducible_with_seed() -> None:
    x = np.linspace(-2.0, 2.0, 40)
    y = x + 0.25

    first = nns_anova(x, y, robust=True, random_seed=123)
    second = nns_anova(x, y, robust=True, random_seed=123)

    assert isinstance(first, dict)
    assert isinstance(second, dict)
    assert first == second


def test_nns_anova_drops_nan_but_rejects_infinite_values() -> None:
    x = np.linspace(-2.0, 2.0, 40)
    y = x + 0.25
    reference = nns_anova(x, y, confidence_interval=None)

    assert nns_anova(np.r_[x, np.nan], y, confidence_interval=None) == reference
    for bad in (np.inf, -np.inf):
        with pytest.raises(ValueError, match="infinite"):
            nns_anova(np.r_[x, bad], y, confidence_interval=None)
        with pytest.raises(ValueError, match="infinite"):
            nns_anova([x, np.r_[y, bad]], confidence_interval=None)
