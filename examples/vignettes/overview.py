"""Vignette 00 — Overview.

Python translation of the R NNS "Overview" vignette
(``tools/NNS/vignettes/NNSvignette_01_Overview.Rmd``).

A short tour that touches each pillar of NNS: partial-moment variance, the
empirical CDF, distributional summaries, nonlinear dependence, and a partial
moment matrix. Later vignettes expand on each topic.

Run with::

    python examples/vignettes/overview.py
"""

from __future__ import annotations

import numpy as np

from nns import lpm, lpm_ratio, nns_copula, nns_dep, nns_moments, pm_matrix, upm


def main() -> None:
    rng = np.random.default_rng(42)

    # Partial-moment variance identity: (LPM2 + UPM2) * n/(n-1) == var(y).
    y = rng.normal(size=3000)
    mu = float(np.mean(y))
    n = y.size
    pm_variance = (float(lpm(2, mu, y)) + float(upm(2, mu, y))) * (n / (n - 1))
    assert np.isclose(pm_variance, float(np.var(y, ddof=1)))

    # Empirical CDF through LPM.ratio(0, t, y).
    for t in (-1.0, 0.0, 1.0):
        assert np.isclose(float(lpm_ratio(0, t, y)), float(np.mean(y <= t)))

    # Distributional summary.
    moments = nns_moments(y)

    # Nonlinear association that Pearson correlation misses (y = x**2).
    x = rng.uniform(-1.0, 1.0, size=2000)
    yq = x**2 + rng.normal(scale=0.05, size=2000)
    pearson = float(np.corrcoef(x, yq)[0, 1])
    dependence = float(nns_dep(x, yq)["Dependence"])
    assert abs(pearson) < 0.2
    assert dependence > pearson

    # Partial moment matrix and copula on a small frame.
    frame = np.column_stack((x, yq, x * yq + rng.normal(scale=0.05, size=2000)))
    pm = pm_matrix(1, 1, "mean", frame, True, names=["a", "b", "c"])
    cop = float(nns_copula(frame, continuous=True))

    print("partial-moment variance vs numpy:",
          round(pm_variance, 6), round(float(np.var(y, ddof=1)), 6))
    print("moments:", {k: round(v, 4) for k, v in moments.items()})
    print("Pearson r (near zero):", round(pearson, 4))
    print("NNS dependence:", round(dependence, 4))
    print("covariance matrix keys:", sorted(pm))
    print("multivariate copula:", round(cop, 4))


if __name__ == "__main__":
    main()
