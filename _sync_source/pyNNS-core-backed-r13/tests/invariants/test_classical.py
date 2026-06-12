from __future__ import annotations

import numpy as np
import pytest
from _tolerances import EXACT
from scipy import stats  # type: ignore[import-untyped]

from pynns import ecdf_pm, kurt_pm, mean_pm, skew_pm, var_pm


def test_mean_pm_matches_numpy_mean() -> None:
    x = _x()

    assert mean_pm(x) == pytest.approx(np.mean(x), abs=EXACT)


@pytest.mark.parametrize("ddof", [0, 1, 2])
def test_var_pm_matches_numpy_var(ddof: int) -> None:
    x = _x()

    assert var_pm(x, ddof=ddof) == pytest.approx(np.var(x, ddof=ddof), abs=EXACT)


def test_skew_pm_matches_scipy_biased_skew() -> None:
    x = _x()

    assert skew_pm(x) == pytest.approx(stats.skew(x, bias=True), abs=EXACT)


def test_kurt_pm_matches_scipy_biased_kurtosis() -> None:
    x = _x()

    assert kurt_pm(x) == pytest.approx(stats.kurtosis(x, fisher=True, bias=True), abs=EXACT)
    assert kurt_pm(x, excess=False) == pytest.approx(
        stats.kurtosis(x, fisher=False, bias=True),
        abs=EXACT,
    )


def test_ecdf_pm_matches_searchsorted_definition() -> None:
    x = _x()
    points = np.array([-2.0, -0.5, 0.0, 0.75, 2.0])

    expected = np.searchsorted(np.sort(x), points, side="right") / x.size

    np.testing.assert_allclose(ecdf_pm(x, points), expected, atol=EXACT)
    np.testing.assert_allclose(ecdf_pm(x), np.arange(1, x.size + 1) / x.size, atol=EXACT)


def _x() -> np.ndarray:
    return np.array([-1.5, -0.25, 0.0, 0.75, 2.0, 3.5], dtype=np.float64)
