from __future__ import annotations

import numpy as np
import pytest

from pynns import nns_anova

from ._original import expected, r_vector


@pytest.mark.parity
def test_original_anova_certainty_and_pairwise_matrix_match_r_fixtures() -> None:
    x = r_vector("test_ANOVA.R", "x")
    y = r_vector("test_ANOVA.R", "y")
    z = r_vector("test_ANOVA.R", "z")
    values = np.column_stack((x, y, z))
    exp = expected("test_ANOVA.R")

    actual = nns_anova(values)
    assert isinstance(actual, dict)
    assert actual["Certainty"] == pytest.approx(exp["nns_anova"]["Certainty"], abs=1e-4)

    pairwise = nns_anova(values, pairwise=True)
    assert isinstance(pairwise, np.ndarray)
    np.testing.assert_allclose(pairwise, exp["nns_anova_pairwise"], atol=1e-4)
