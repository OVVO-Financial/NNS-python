from __future__ import annotations

import numpy as np
import pytest

from pynns import nns_copula


def test_nns_copula_is_bounded() -> None:
    x = np.linspace(-2.0, 2.0, 200)
    y = np.sin(x)

    result = nns_copula(x, y)

    assert result >= 0.0
    assert result <= 1.0


def test_nns_copula_is_symmetric() -> None:
    x = np.linspace(-2.0, 2.0, 200)
    y = x**3

    assert nns_copula(x, y) == pytest.approx(nns_copula(y, x), abs=1e-12)


def test_nns_copula_identical_pair_is_unit() -> None:
    x = np.linspace(-2.0, 2.0, 200)

    assert nns_copula(x, x) == pytest.approx(1.0)
