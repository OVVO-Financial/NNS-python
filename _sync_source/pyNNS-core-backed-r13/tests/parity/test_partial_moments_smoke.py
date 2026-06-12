from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from _r import nns
from _tolerances import EXACT
from numpy.typing import NDArray


@pytest.mark.parity
def test_co_lpm_smoke() -> None:
    result = nns("Co.LPM", 1, [-1, 1], [-1, 1], 0, 0)

    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, np.array(0.5), atol=EXACT)


@pytest.mark.parity
def test_pm_matrix_smoke() -> None:
    result = cast(
        dict[str, NDArray[np.float64]],
        nns("PM.matrix", 1, 1, [0, 0], [[-1, -1], [1, 1]], True),
    )
    expected = cast(
        dict[str, NDArray[np.float64]],
        nns("PM.matrix", 1, 1, [0, 0], [[-1, -1], [1, 1]], True),
    )

    assert isinstance(result, dict)
    assert set(result) == {"cupm", "dupm", "dlpm", "clpm", "cov.matrix"}
    for value in result.values():
        assert isinstance(value, np.ndarray)
        assert value.shape == (2, 2)

    cupm = result["cupm"]
    clpm = result["clpm"]
    dupm = result["dupm"]
    dlpm = result["dlpm"]
    cov_matrix = result["cov.matrix"]

    expected_cupm = expected["cupm"]
    expected_dupm = expected["dupm"]
    expected_dlpm = expected["dlpm"]
    expected_clpm = expected["clpm"]
    expected_cov_matrix = expected["cov.matrix"]

    assert np.allclose(cupm, expected_cupm, atol=EXACT)
    assert np.allclose(clpm, expected_clpm, atol=EXACT)
    assert np.allclose(dupm, expected_dupm, atol=EXACT)
    assert np.allclose(dlpm, expected_dlpm, atol=EXACT)
    assert np.allclose(cov_matrix, expected_cov_matrix, atol=EXACT)
