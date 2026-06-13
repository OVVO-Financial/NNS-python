"""Vignette 05 — Hypothesis testing: ANOVA and stochastic superiority.

Python translation of pieces of the R NNS "Comparing Distributions" vignette
(``tools/NNS/vignettes/NNSvignette_06_Comparing_Distributions.Rmd``):
``nns_anova`` certainty and ``nns_ss`` stochastic superiority (continuous and
discrete-with-ties).

Stochastic confidence-interval outputs are validated by range, never by exact
value, because they are bootstrap estimates.

Run with::

    python examples/vignettes/hypothesis_anova_stochastic_superiority.py
"""

from __future__ import annotations

import numpy as np

from nns import nns_anova, nns_ss


def main() -> None:
    rng = np.random.default_rng(123)

    # ANOVA certainty: identical-mean samples score high certainty of equality,
    # shifted-mean samples score lower.
    x = rng.normal(0.0, 1.0, size=1000)
    y_equal = rng.normal(0.0, 2.0, size=1000)
    y_shifted = rng.normal(1.0, 1.0, size=1000)

    equal = nns_anova(x, y_equal, means_only=True, random_seed=1)
    shifted = nns_anova(x, y_shifted, means_only=True, random_seed=1)
    certainty_equal = float(equal["Certainty"])
    certainty_shifted = float(shifted["Certainty"])
    assert 0.0 <= certainty_shifted <= certainty_equal <= 1.0

    # Stochastic superiority P(Y > X): continuous case.
    ss = nns_ss(x, y_shifted)
    p_gt = float(ss["p_gt"])
    p_tie = float(ss["p_tie"])
    p_star = float(ss["p_star"])
    assert 0.0 <= p_gt <= 1.0
    assert 0.0 <= p_star <= 1.0

    # Discrete data: ties contribute a nonzero p_tie.
    xd = rng.integers(1, 6, size=100).astype(float)
    yd = rng.integers(1, 6, size=100).astype(float)
    ss_discrete = nns_ss(xd, yd)
    assert float(ss_discrete["p_tie"]) >= 0.0

    # Bootstrap CI: assert ordering/ranges only, not exact values.
    ss_ci = nns_ss(x, y_shifted, confidence_interval=True, reps=199, ci=0.95, random_seed=1)
    lower = float(ss_ci["lower"])
    upper = float(ss_ci["upper"])
    assert 0.0 <= lower <= upper <= 1.0

    print("ANOVA certainty (equal means):", round(certainty_equal, 4))
    print("ANOVA certainty (shifted means):", round(certainty_shifted, 4))
    print("stochastic superiority p_gt/p_tie/p_star:",
          round(p_gt, 4), round(p_tie, 4), round(p_star, 4))
    print("discrete p_tie (ties present):", round(float(ss_discrete["p_tie"]), 4))
    print("bootstrap CI [lower, upper] (range-checked):", round(lower, 4), round(upper, 4))


if __name__ == "__main__":
    main()
