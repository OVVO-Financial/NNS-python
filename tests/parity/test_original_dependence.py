from __future__ import annotations

import numpy as np
import pytest

from nns import nns_copula

from ._original import expected, r_vector


@pytest.mark.parity
def test_original_bivariate_copula_continuous_matches_r_fixture() -> None:
    x = r_vector("test_Copula.R", "x")
    y = r_vector("test_Copula.R", "y")
    exp = expected("test_Copula.R")

    assert nns_copula(x, y) == pytest.approx(exp["bivariate_continuous"], abs=1e-5)


@pytest.mark.parity
def test_original_bivariate_copula_discrete_matches_r_fixture() -> None:
    x = r_vector("test_Copula.R", "x")
    y = r_vector("test_Copula.R", "y")
    exp = expected("test_Copula.R")

    assert nns_copula(x, y, continuous=False) == pytest.approx(
        exp["bivariate_discrete"], abs=1e-5
    )


@pytest.mark.parity
def test_original_multivariate_copula_continuous_matches_r_fixture() -> None:
    z = _three_column_matrix()
    exp = expected("test_Copula.R")

    assert nns_copula(z) == pytest.approx(exp["multivariate_continuous"], abs=1e-5)


@pytest.mark.parity
def test_original_multivariate_copula_discrete_matches_r_fixture() -> None:
    z = _three_column_matrix()
    exp = expected("test_Copula.R")

    assert nns_copula(z, continuous=False) == pytest.approx(
        exp["multivariate_discrete"], abs=1e-5
    )


def _three_column_matrix() -> np.ndarray:
    x = r_vector("test_Copula.R", "x")
    y = r_vector("test_Copula.R", "y")
    z = r_vector("test_Copula.R", "z")
    return np.column_stack((x, y, z))
