from __future__ import annotations

import numpy as np

from pynns import fsd, ssd, tsd


def test_sd_antisymmetry() -> None:
    x = np.array([0.0, 0.1, 0.2, 0.3])
    y = np.array([-0.1, 0.0, 0.1, 0.2])

    assert fsd(x, y) == -fsd(y, x)
    assert ssd(x, y) == -ssd(y, x)
    assert tsd(x, y) == -tsd(y, x)


def test_fsd_implies_ssd_implies_tsd() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([0.0, 1.0, 2.0, 3.0])

    assert fsd(x, y) == 1
    assert ssd(x, y) == 1
    assert tsd(x, y) == 1


def test_self_does_not_dominate() -> None:
    x = np.array([-1.0, 0.0, 1.0, 2.0])

    assert fsd(x, x) == 0
    assert ssd(x, x) == 0
    assert tsd(x, x) == 0
