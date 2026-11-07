from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from _r import nns
from _tolerances import EXACT

from nns import kurt_pm, mean_pm, nns_moments, skew_pm, var_pm


@pytest.mark.parity
@pytest.mark.parametrize(
    "x",
    [
        np.array([-2.0, -1.0, 0.0, 1.0, 2.0]),
        np.array([0.5, 1.5, 3.0, 4.0, 8.0, 13.0]),
        np.sin(np.arange(1, 31, dtype=np.float64) / 3.0),
    ],
)
def test_classical_moments_match_r_nns_moments(x: np.ndarray) -> None:
    expected = cast(dict[str, np.ndarray], nns("NNS.moments", x.tolist(), True))
    actual = nns_moments(x)

    assert mean_pm(x) == pytest.approx(float(expected["mean"]), abs=EXACT)
    assert var_pm(x) == pytest.approx(float(expected["variance"]), abs=EXACT)
    assert skew_pm(x) == pytest.approx(float(expected["skewness"]), abs=EXACT)
    assert kurt_pm(x) == pytest.approx(float(expected["kurtosis"]), abs=EXACT)
    assert actual["mean"] == pytest.approx(float(expected["mean"]), abs=EXACT)
    assert actual["variance"] == pytest.approx(float(expected["variance"]), abs=EXACT)
    assert actual["skewness"] == pytest.approx(float(expected["skewness"]), abs=EXACT)
    assert actual["kurtosis"] == pytest.approx(float(expected["kurtosis"]), abs=EXACT)


@pytest.mark.parity
@pytest.mark.parametrize("population", [True, False])
def test_nns_moments_public_wrapper_matches_r(population: bool) -> None:
    x = np.array([-2.0, -1.0, 0.5, 1.5, 3.0, 8.0])
    expected = cast(dict[str, np.ndarray], nns("NNS.moments", x.tolist(), population))
    actual = nns_moments(x, population=population)

    assert actual["mean"] == pytest.approx(float(expected["mean"]), abs=EXACT)
    assert actual["variance"] == pytest.approx(float(expected["variance"]), abs=EXACT)
    assert actual["skewness"] == pytest.approx(float(expected["skewness"]), abs=EXACT)
    assert actual["kurtosis"] == pytest.approx(float(expected["kurtosis"]), abs=EXACT)


@pytest.mark.parity
@pytest.mark.parametrize("population", [True, False])
@pytest.mark.parametrize(
    "x",
    [[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 4.0], [1.0, 2.0], [5.0], [3.0, 3.0, 3.0]],
)
def test_nns_moments_degenerate_input_returns_nan_like_r(
    x: list[float], population: bool
) -> None:
    # Constant input and n <= 3 used to raise ZeroDivisionError; R returns NaN.
    expected = cast(dict[str, np.ndarray], nns("NNS.moments", x, population))
    actual = nns_moments(np.asarray(x), population=population)

    for key in ("mean", "variance", "skewness", "kurtosis"):
        np.testing.assert_allclose(
            actual[key], float(np.asarray(expected[key])), atol=EXACT, equal_nan=True
        )


@pytest.mark.parity
def test_nns_moments_sample_skewness_is_adjusted_fisher_pearson() -> None:
    # Patched R: the sample skewness divides by s^3 with s the (n - 1) sample sd
    # (adjusted Fisher-Pearson G1), not by the population variance^(3/2).
    x = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    expected = cast(dict[str, np.ndarray], nns("NNS.moments", x.tolist(), False))
    actual = nns_moments(x, population=False)

    n = x.size
    z = x - x.mean()
    g1 = n / ((n - 1) * (n - 2)) * np.sum(z**3) / np.std(x, ddof=1) ** 3
    assert actual["skewness"] == pytest.approx(1.325315, abs=1e-6)
    assert actual["skewness"] == pytest.approx(g1, abs=EXACT)
    assert actual["skewness"] == pytest.approx(float(expected["skewness"]), abs=EXACT)


def test_skew_and_kurt_pm_of_constant_input_are_nan() -> None:
    x = np.full(6, 2.5)
    assert np.isnan(skew_pm(x))
    assert np.isnan(kurt_pm(x))
    assert np.isnan(kurt_pm(x, excess=False))
