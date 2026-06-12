from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from _r import nns
from _tolerances import EXACT

from pynns import co_lpm_nd, co_upm_nd, dpm_nd, nns_gravity


@pytest.mark.parity
@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize(
    "x",
    [
        np.array([1.0, 2.0, 3.0]),
        np.array([5.0, 5.0, 5.0, 5.0]),
        np.array([-10.0, -1.0, 0.0, 1.0, 2.0, 40.0]),
    ],
)
def test_nns_gravity_public_wrapper_matches_r(x: np.ndarray, discrete: bool) -> None:
    expected = nns("NNS.gravity", x.tolist(), discrete)

    assert nns_gravity(x, discrete=discrete) == pytest.approx(_scalar(expected), abs=EXACT)


@pytest.mark.parity
@pytest.mark.parametrize(
    ("r_name", "function"),
    [
        ("Co.LPM_nD", co_lpm_nd),
        ("Co.UPM_nD", co_upm_nd),
        ("DPM_nD", dpm_nd),
    ],
)
@pytest.mark.parametrize("degree", [0.0, 1.0, 2.0])
@pytest.mark.parametrize("norm", [False, True])
def test_nd_partial_moment_wrappers_match_r(
    r_name: str,
    function: object,
    degree: float,
    norm: bool,
) -> None:
    data = np.array(
        [
            [-1.0, 0.5, 2.0],
            [0.0, -0.5, 1.5],
            [1.0, 1.5, -1.0],
            [2.0, -2.0, 0.25],
            [3.0, 0.0, 0.75],
        ],
        dtype=np.float64,
    )
    target = np.array([0.5, 0.0, 0.5], dtype=np.float64)
    expected = nns(r_name, data.tolist(), target.tolist(), degree, norm)

    actual = cast_wrapper(function)(data, target, degree=degree, norm=norm)

    assert actual == pytest.approx(_scalar(expected), abs=EXACT)


def cast_wrapper(function: object) -> Callable[..., float]:
    assert callable(function)
    return function


def _scalar(value: object) -> float:
    return float(np.asarray(value, dtype=np.float64).reshape(-1)[0])
