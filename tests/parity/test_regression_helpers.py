from __future__ import annotations

import numpy as np
import pytest
from _r import nns
from _tolerances import EXACT, STOCHASTIC

from pynns import lpm_var, nns_mode, nns_rescale, upm_var

MODE_CASES = [
    np.array([1.0, 2.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]),
    np.array([1.0, 1.0, 2.0, 2.0, 3.0, 4.0]),
    np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0]),
    np.array([-10.0, -9.0, -8.0, 0.0, 1.0, 2.0, 2.0, 50.0]),
    np.array([5.0, 5.0, 5.0, 5.0]),
    np.array([1.2, 2.8, 3.1]),
]


@pytest.mark.parity
@pytest.mark.parametrize("values", MODE_CASES)
@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("multi", [False, True])
def test_nns_mode_matches_r(values: np.ndarray, discrete: bool, multi: bool) -> None:
    expected = nns("NNS.mode", values.tolist(), discrete, multi)
    actual = nns_mode(values, discrete=discrete, multi=multi)

    np.testing.assert_allclose(_array(actual), _array(expected), atol=EXACT)


@pytest.mark.parity
def test_nns_rescale_minmax_matches_r() -> None:
    values = np.array([-3.0, -1.0, 0.0, 2.0, 4.0, 10.0])

    expected = nns("NNS.rescale", values.tolist(), -2.0, 3.0)
    actual = nns_rescale(values, -2.0, 3.0)

    np.testing.assert_allclose(actual, _array(expected), atol=EXACT)


@pytest.mark.parity
@pytest.mark.parametrize("target_type", ["Terminal", "Discounted"])
def test_nns_rescale_riskneutral_matches_r(target_type: str) -> None:
    values = np.array([11.0, 12.0, 15.0, 20.0, 25.0])

    expected = nns("NNS.rescale", values.tolist(), 100.0, 0.05, "riskneutral", 1.25, target_type)
    actual = nns_rescale(
        values,
        100.0,
        0.05,
        "riskneutral",
        1.25,
        target_type,
    )

    np.testing.assert_allclose(actual, _array(expected), atol=EXACT)


@pytest.mark.parity
@pytest.mark.parametrize("percentile", [0.0, 0.05, 0.25, 0.5, 0.95, 1.0])
@pytest.mark.parametrize("degree", [0.0, 1.0, 2.0])
def test_lpm_var_matches_r(percentile: float, degree: float) -> None:
    values = np.array([-4.0, -2.0, -1.0, 0.0, 0.5, 2.0, 3.0, 10.0])

    expected = nns("LPM.VaR", percentile, degree, values.tolist())
    actual = lpm_var(percentile, degree, values)

    np.testing.assert_allclose(actual, _array(expected), atol=STOCHASTIC)


@pytest.mark.parity
@pytest.mark.parametrize("percentile", [0.0, 0.05, 0.25, 0.5, 0.95, 1.0])
@pytest.mark.parametrize("degree", [0.0, 1.0, 2.0])
def test_upm_var_matches_r(percentile: float, degree: float) -> None:
    values = np.array([-4.0, -2.0, -1.0, 0.0, 0.5, 2.0, 3.0, 10.0])

    expected = nns("UPM.VaR", percentile, degree, values.tolist())
    actual = upm_var(percentile, degree, values)

    np.testing.assert_allclose(actual, _array(expected), atol=STOCHASTIC)


def _array(value: object) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)
