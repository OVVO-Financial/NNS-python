from __future__ import annotations

import numpy as np
import pytest
from _tolerances import EXACT

from nns import nns_dep


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


def test_nns_dep_independence_null_is_consistent() -> None:
    # The discordant partial-moment independence anchor is 1 - 2*0.5**n = 0.5
    # for the bivariate copula (both fully-aligned orthants are concordant),
    # not 1 - 0.5**n = 0.75. With the correct anchor the dependence of
    # independent data is a *consistent* estimator: it decays toward 0 as the
    # sample grows. The old 0.75 anchor left a fixed ~0.41 floor that never
    # vanished, so a large-sample independent mean well under it (and clearly
    # below the small-sample mean) can only hold with the corrected anchor.
    rng = np.random.default_rng(0)

    def mean_independent_dep(n: int, reps: int = 8) -> float:
        vals = [
            nns_dep(rng.standard_normal(n), rng.standard_normal(n))["Dependence"]
            for _ in range(reps)
        ]
        return float(np.mean(vals))

    small = mean_independent_dep(200)
    large = mean_independent_dep(4000)

    assert large < small  # consistent: decays with sample size
    assert large < 0.35  # unreachable under the old non-vanishing ~0.41 floor
