from __future__ import annotations

import numpy as np
import pytest

from pynns import nns_anova


def test_nns_anova_identical_degenerate_groups_match_r_nan_convention() -> None:
    x = np.linspace(-2.0, 2.0, 100)
    constant = np.ones_like(x)

    result = nns_anova(constant, constant, confidence_interval=None)

    assert isinstance(result, dict)
    assert np.isnan(result["Certainty"])
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
    assert np.isnan(result["Robust Certainty Estimate"])


@pytest.mark.stochastic
def test_nns_anova_robust_reproducible_with_seed() -> None:
    x = np.linspace(-2.0, 2.0, 40)
    y = x + 0.25

    first = nns_anova(x, y, robust=True, random_seed=123)
    second = nns_anova(x, y, robust=True, random_seed=123)

    assert isinstance(first, dict)
    assert isinstance(second, dict)
    assert first == second
