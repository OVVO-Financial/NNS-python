"""Canonical vignette 03: Correlation and dependence.

Source of truth:
    NNS/vignettes/NNSvignette_03_Correlation_and_Dependence.Rmd
"""
from __future__ import annotations

import numpy as np

from nns import nns_copula, nns_dep


def main() -> None:
    x = np.arange(0.0, 3.01, 0.01)
    linear = nns_dep(x, 2.0 * x)
    nonlinear = nns_dep(x, x**10)

    cyclic_x = np.arange(0.0, 12.0 * np.pi, np.pi / 100.0)
    cyclic_y = np.sin(cyclic_x)
    cyclic = nns_dep(cyclic_x, cyclic_y)
    asym_xy = nns_dep(cyclic_x, cyclic_y, asym=True)["Dependence"]
    asym_yx = nns_dep(cyclic_y, cyclic_x, asym=True)["Dependence"]

    rng = np.random.default_rng(123)
    points = rng.uniform(-1.0, 1.0, size=(10000, 2))
    radius2 = np.sum(points**2, axis=1)
    ring = points[(radius2 <= 1.0) & (radius2 >= 0.95)]
    ring_dep = nns_dep(ring[:, 0], ring[:, 1])

    frame = rng.normal(size=(1000, 3))
    copula = float(nns_copula(frame, continuous=True))

    assert linear["Correlation"] > 0.99 and linear["Dependence"] > 0.99
    assert nonlinear["Dependence"] >= abs(nonlinear["Correlation"])
    assert cyclic["Dependence"] > abs(cyclic["Correlation"])
    assert 0.0 <= float(ring_dep["Dependence"]) <= 1.0
    assert 0.0 <= copula <= 1.0

    print("linear:", linear)
    print("x^10:", nonlinear)
    print("sin(x):", cyclic)
    print("asymmetric D(y|x), D(x|y):", round(float(asym_xy), 4), round(float(asym_yx), 4))
    print("ring dependence:", round(float(ring_dep["Dependence"]), 4))
    print("three-variable copula dependence:", round(copula, 4))


if __name__ == "__main__":
    main()
