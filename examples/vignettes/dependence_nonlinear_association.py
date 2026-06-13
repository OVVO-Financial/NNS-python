"""Vignette 03 — Dependence and nonlinear association.

Python translation of the R NNS "Correlation and Dependence" vignette
(``tools/NNS/vignettes/NNSvignette_03_Correlation_and_Dependence.Rmd``).

Contrasts Pearson correlation with partial-moment dependence on relationships
where the linear measure collapses, and shows ``pm_matrix`` and ``nns_copula``.

Run with::

    python examples/vignettes/dependence_nonlinear_association.py
"""

from __future__ import annotations

import numpy as np

from nns import nns_copula, nns_dep, pm_matrix


def main() -> None:
    # Perfect linear relationship: correlation and dependence both ~1.
    x = np.arange(0.0, 3.01, 0.01)
    linear = 2.0 * x
    lin = nns_dep(x, linear)
    assert lin["Correlation"] > 0.99
    assert lin["Dependence"] > 0.99

    # Deterministic nonlinear map y = sin(x): Pearson is weak, dependence high.
    xs = np.arange(0.0, 12.0 * np.pi, np.pi / 100.0)
    ys = np.sin(xs)
    sine = nns_dep(xs, ys)
    # Partial-moment dependence captures the structure the linear measure cannot:
    # it is several times the (weak) Pearson correlation.
    assert sine["Dependence"] > 3.0 * abs(sine["Correlation"])
    assert sine["Dependence"] > 0.5

    # Asymmetric dependence: D(x|y) need not equal D(y|x).
    asym_xy = nns_dep(xs, ys, asym=True)["Dependence"]
    asym_yx = nns_dep(ys, xs, asym=True)["Dependence"]

    # Partial moment matrix and copula on a 3-variable frame.
    rng = np.random.default_rng(123)
    a = rng.normal(size=1000)
    b = rng.normal(size=1000)
    c = rng.normal(size=1000)
    frame = np.column_stack((a, b, c))
    pm = pm_matrix(1, 1, "mean", frame, True, names=["a", "b", "c"])
    independent_copula = float(nns_copula(frame, continuous=True))

    print(f"linear:      r={lin['Correlation']:.4f}  dep={lin['Dependence']:.4f}")
    print(f"sine:        r={sine['Correlation']:.4f}  dep={sine['Dependence']:.4f}")
    print(f"asymmetric dependence  D(y|x)={asym_xy:.4f}  D(x|y)={asym_yx:.4f}")
    print("pm matrix keys:", sorted(pm))
    print("copula (near-independent):", round(independent_copula, 4))


if __name__ == "__main__":
    main()
