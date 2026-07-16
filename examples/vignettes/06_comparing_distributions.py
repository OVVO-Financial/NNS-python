"""Canonical vignette 06: Comparing distributions.

Source of truth:
    NNS/vignettes/NNSvignette_06_Comparing_Distributions.Rmd

Covers NNS ANOVA certainty, stochastic superiority, pairwise stochastic
dominance, the stochastic-dominance efficient set, and SD clustering.
"""
from __future__ import annotations

import numpy as np

from nns import (
    fsd_uni,
    nns_anova,
    nns_sd_cluster,
    nns_ss,
    sd_efficient_set,
    ssd_uni,
    tsd_uni,
)


def main() -> None:
    rng = np.random.default_rng(123)
    x = rng.normal(0.0, 1.0, size=1000)
    equal_mean = rng.normal(0.0, 2.0, size=1000)
    shifted = x + 1.0

    equal = nns_anova(x, equal_mean, means_only=True, random_seed=1)
    unequal = nns_anova(x, shifted, means_only=True, random_seed=1)
    superiority = nns_ss(x, shifted)

    assert 0.0 <= float(equal["Certainty"]) <= 1.0
    assert 0.0 <= float(unequal["Certainty"]) <= 1.0
    assert fsd_uni(shifted, x) == 1
    assert ssd_uni(shifted, x) == 1
    assert tsd_uni(shifted, x) == 1

    base = [rng.normal(size=500) for _ in range(4)]
    panel = np.column_stack(
        [item for pair in ((z, z + 1.0) for z in base) for item in pair]
    )
    efficient = sd_efficient_set(panel, degree=1)
    names = [f"x{i + 1}" for i in range(panel.shape[1])]
    clusters = nns_sd_cluster(panel, degree=1, names=names)

    print("ANOVA certainty, equal means:", round(float(equal["Certainty"]), 4))
    print("ANOVA certainty, shifted means:", round(float(unequal["Certainty"]), 4))
    print("stochastic superiority:", superiority)
    print(
        "FSD/SSD/TSD shifted over base:",
        fsd_uni(shifted, x),
        ssd_uni(shifted, x),
        tsd_uni(shifted, x),
    )
    print("SD efficient set:", efficient)
    print("SD clusters:", clusters["Clusters"])


if __name__ == "__main__":
    main()
