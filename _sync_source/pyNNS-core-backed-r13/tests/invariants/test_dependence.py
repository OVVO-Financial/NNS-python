from __future__ import annotations

import numpy as np
import pytest
from _tolerances import EXACT

from pynns import nns_dep


def test_nns_dep_identical_has_unit_dependence() -> None:
    x = np.linspace(-1.0, 1.0, 100)

    assert nns_dep(x, x)["Dependence"] == pytest.approx(1.0, abs=EXACT)


def test_nns_dep_bounds() -> None:
    x = np.linspace(-2.0, 2.0, 200)
    y = np.sin(x)

    result = nns_dep(x, y)

    assert result["Dependence"] >= 0.0
    assert result["Dependence"] <= 1.0


def test_nns_dep_asym_bounds() -> None:
    x = np.linspace(-2.0, 2.0, 200)
    y = x**2 + 0.1 * np.sin(3.0 * x)

    result = nns_dep(x, y, asym=True)

    assert result["Dependence"] >= 0.0
    assert result["Dependence"] <= 1.0


def test_nns_dep_asym_reduces_to_symmetric_for_linear_identity() -> None:
    x = np.linspace(-2.0, 2.0, 200)
    y = 3.0 * x + 1.0

    assert nns_dep(x, y, asym=True) == pytest.approx(nns_dep(x, y), abs=EXACT)


def test_nns_dep_is_symmetric() -> None:
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    y = np.array([4.0, 1.0, 0.0, 1.0, 4.0, 9.0])

    assert nns_dep(x, y) == pytest.approx(nns_dep(y, x), abs=EXACT)


def test_nns_dep_asym_can_be_directional() -> None:
    x = np.linspace(-2.0, 2.0, 200)
    y = x**2

    assert nns_dep(x, y, asym=True) != pytest.approx(nns_dep(y, x, asym=True), abs=EXACT)
