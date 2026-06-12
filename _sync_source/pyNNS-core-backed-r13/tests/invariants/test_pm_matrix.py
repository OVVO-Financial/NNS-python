from __future__ import annotations

import numpy as np
import pytest
from _tolerances import EXACT

from pynns import pm_matrix


def test_pm_matrix_reconstructs_cov_matrix() -> None:
    variable = _variable()

    result = pm_matrix(2, 3, 0.0, variable, pop_adj=True)

    np.testing.assert_allclose(
        result["clpm"] + result["cupm"] - result["dlpm"] - result["dupm"],
        result["cov.matrix"],
        atol=EXACT,
    )


@pytest.mark.parametrize("pop_adj, ddof", [(False, 0), (True, 1)])
def test_pm_matrix_degree_one_mean_matches_numpy_covariance(pop_adj: bool, ddof: int) -> None:
    variable = _variable()

    result = pm_matrix(1, 1, "mean", variable, pop_adj=pop_adj)

    np.testing.assert_allclose(result["cov.matrix"], np.cov(variable.T, ddof=ddof), atol=EXACT)


def test_pm_matrix_clpm_and_cupm_are_symmetric() -> None:
    variable = _variable()

    result = pm_matrix(2, 2, "mean", variable, pop_adj=False)

    np.testing.assert_allclose(result["clpm"], result["clpm"].T, atol=EXACT)
    np.testing.assert_allclose(result["cupm"], result["cupm"].T, atol=EXACT)


def test_pm_matrix_clpm_and_cupm_are_positive_semidefinite() -> None:
    variable = _variable()

    result = pm_matrix(3, 3, "mean", variable, pop_adj=True)

    assert np.linalg.eigvalsh(result["clpm"]).min() > -1e-10
    assert np.linalg.eigvalsh(result["cupm"]).min() > -1e-10


def test_pm_matrix_dlpm_is_dupm_transpose() -> None:
    variable = _variable()

    result = pm_matrix(2, 3, np.array([-0.2, 0.0, 0.1, 0.3]), variable, pop_adj=True)

    np.testing.assert_allclose(result["dlpm"], result["dupm"].T, atol=EXACT)


def _variable() -> np.ndarray:
    row = np.arange(80, dtype=np.float64)[:, np.newaxis]
    col = np.arange(4, dtype=np.float64)[np.newaxis, :]
    return np.sin((row + 1.0) * (col + 1.0) / 9.0) + np.cos((row + 2.0) / (col + 4.0))
