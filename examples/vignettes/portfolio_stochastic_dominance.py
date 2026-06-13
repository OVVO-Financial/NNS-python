"""Vignette 09 — Portfolios and stochastic dominance.

Python translation of the stochastic-dominance pieces of the R NNS "Comparing
Distributions" vignette (``NNSvignette_06_Comparing_Distributions.Rmd``):
``fsd_uni``/``ssd_uni``/``tsd_uni`` pairwise tests, plus ``sd_efficient_set``
and ``nns_sd_cluster`` on a small return panel.

Run with::

    python examples/vignettes/portfolio_stochastic_dominance.py
"""

from __future__ import annotations

import numpy as np

from nns import fsd_uni, nns_sd_cluster, sd_efficient_set, ssd_uni, tsd_uni


def main() -> None:
    rng = np.random.default_rng(123)

    # A clear dominance pair: y = x + 1 dominates x by first order
    # (every realization shifted up by a constant).
    x = rng.normal(size=1000)
    y = x + 1.0
    assert fsd_uni(y, x) == 1  # y first-order dominates x
    assert fsd_uni(x, y) == 0
    # First-order dominance implies second- and third-order dominance.
    assert ssd_uni(y, x) == 1
    assert tsd_uni(y, x) == 1

    # Small monthly return panel for three assets.
    ra = rng.normal(0.005, 0.03, size=240)
    rb = rng.normal(0.003, 0.02, size=240)
    rc = rng.normal(0.006, 0.04, size=240)
    returns = np.column_stack((ra, rb, rc))

    # Efficient set: indices of assets not dominated at the chosen degree.
    efficient = sd_efficient_set(returns, degree=1)
    assert isinstance(efficient, list)
    assert all(0 <= i < returns.shape[1] for i in efficient)

    # Dominance-based clustering of the assets.
    clusters = nns_sd_cluster(returns, degree=1, names=["A", "B", "C"])
    assert "Clusters" in clusters

    print("FSD(y, x):", fsd_uni(y, x), " FSD(x, y):", fsd_uni(x, y))
    print("SSD(y, x):", ssd_uni(y, x), " TSD(y, x):", tsd_uni(y, x))
    print("efficient set (asset indices):", efficient)
    print("SD clusters:", clusters["Clusters"])


if __name__ == "__main__":
    main()
