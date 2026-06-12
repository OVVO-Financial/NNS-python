from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from _r import nns
from _tolerances import EXACT

from pynns import kurt_pm, mean_pm, nns_moments, skew_pm, var_pm


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
