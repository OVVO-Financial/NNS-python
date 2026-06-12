from __future__ import annotations

import numpy as np
import pytest

from nns import co_lpm, co_upm, d_lpm, d_upm, lpm, lpm_ratio, nns_cdf, pm_matrix, upm, upm_ratio

from ._original import expected, r_vector

TOL = 1e-5


@pytest.mark.parity
def test_original_partial_moment_scalars_match_r_fixtures() -> None:
    x = r_vector("test_Partial_Moments.R", "x")
    y = r_vector("test_Partial_Moments.R", "y")
    target_x = float(np.mean(x))
    target_y = float(np.mean(y))
    exp = expected("test_Partial_Moments.R")

    for degree in (0, 1, 2):
        key = str(degree)
        assert lpm(degree, target_x, x) == pytest.approx(exp["lpm"][key], abs=TOL)
        assert upm(degree, target_x, x) == pytest.approx(exp["upm"][key], abs=TOL)
        assert co_upm(degree, x, y, target_x, target_y) == pytest.approx(
            exp["co_upm"][key], abs=TOL
        )
        assert co_lpm(degree, x, y, target_x, target_y) == pytest.approx(
            exp["co_lpm"][key], abs=TOL
        )
        assert lpm_ratio(degree, target_x, x) == pytest.approx(exp["lpm_ratio"][key], abs=TOL)
        assert upm_ratio(degree, target_x, x) == pytest.approx(exp["upm_ratio"][key], abs=TOL)

    for degrees, value in exp["d_lpm"].items():
        degree_x, degree_y = (int(part) for part in degrees.split(","))
        assert d_lpm(degree_x, degree_y, x, y, target_x, target_y) == pytest.approx(value, abs=TOL)
    for degrees, value in exp["d_upm"].items():
        degree_x, degree_y = (int(part) for part in degrees.split(","))
        assert d_upm(degree_x, degree_y, x, y, target_x, target_y) == pytest.approx(value, abs=TOL)


@pytest.mark.parity
def test_original_pm_matrix_and_survival_cdf_match_r_fixtures() -> None:
    exp = expected("test_Partial_Moments.R")
    values = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 3.0]])
    target = np.mean(values, axis=0)

    np.testing.assert_allclose(
        pm_matrix(1, 1, target, values, pop_adj=True)["cov.matrix"],
        np.asarray(exp["pm_matrix_cov_pop_adj_true"]),
        atol=TOL,
    )
    np.testing.assert_allclose(
        pm_matrix(1, 1, target, values, pop_adj=False)["cov.matrix"],
        np.asarray(exp["pm_matrix_cov_pop_adj_false"]),
        atol=TOL,
    )

    cdf_values = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 2.5], dtype=np.float64)
    actual = nns_cdf(cdf_values, type="survival")
    assert isinstance(actual["Function"], dict)
    np.testing.assert_allclose(actual["Function"]["x"], exp["cdf_survival"]["x"], atol=TOL)
    np.testing.assert_allclose(actual["Function"]["S(x)"], exp["cdf_survival"]["S(x)"], atol=TOL)
    assert np.asarray(actual["target.value"]).size == 0


@pytest.mark.parity
def test_pm_matrix_optional_names_match_r_dataframe_without_changing_numbers() -> None:
    # R's test_Partial_Moments.R checks that PM.matrix on a data.frame copies the
    # frame's column names (here the default V1/V2) onto the cov.matrix dimnames,
    # while a plain matrix input yields an unnamed matrix with identical numbers.
    # The Python API is NumPy-first, so labels are exposed via an optional
    # "names" key rather than as array dimnames. This proves: (a) the numeric
    # matrices are byte-for-byte identical with or without names (parity is
    # unaffected by naming), and (b) when names are supplied they match the R
    # data-frame naming behavior.
    exp = expected("test_Partial_Moments.R")
    values = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 3.0]])
    target = np.mean(values, axis=0)

    unnamed = pm_matrix(1, 1, target, values, pop_adj=True)
    named = pm_matrix(1, 1, target, values, pop_adj=True, names=["V1", "V2"])

    assert "names" not in unnamed
    assert named["names"] == ["V1", "V2"]
    for key in ("cupm", "dupm", "dlpm", "clpm", "cov.matrix"):
        np.testing.assert_array_equal(named[key], unnamed[key])
    np.testing.assert_allclose(
        named["cov.matrix"],
        np.asarray(exp["pm_matrix_cov_pop_adj_true"]),
        atol=TOL,
    )
