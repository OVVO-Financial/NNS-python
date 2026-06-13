"""Vignette 01 — Partial moments.

Python translation of the R NNS "Partial Moments" vignette
(``tools/NNS/vignettes/NNSvignette_02_Partial_Moments.Rmd``).

Shows how the lower/upper partial moments reconstruct variance, covariance,
the empirical CDF, and value-at-risk quantiles using the public ``nns`` API.

Run with::

    python examples/vignettes/partial_moments.py
"""

from __future__ import annotations

import numpy as np

from nns import lpm, lpm_ratio, lpm_var, upm, upm_ratio, upm_var


def main() -> None:
    rng = np.random.default_rng(123)
    x = rng.normal(size=100)

    mu = float(np.mean(x))
    n = x.size

    # The mean is the balance point of first-degree partial moments around 0.
    mean_via_pm = float(upm(1, 0.0, x)) - float(lpm(1, 0.0, x))
    assert np.isclose(mean_via_pm, mu)

    # Variance decomposes into upper + lower second-degree partial moments
    # about the mean (population form); the sample form rescales by n/(n-1).
    population_variance = float(upm(2, mu, x)) + float(lpm(2, mu, x))
    sample_variance = population_variance * (n / (n - 1))
    assert np.isclose(sample_variance, float(np.var(x, ddof=1)))

    # Empirical CDF: LPM.ratio with degree 0 is the proportion of mass <= t.
    targets = np.array([-1.0, 0.0, 1.0])
    cdf_pm = np.array([float(lpm_ratio(0, t, x)) for t in targets])
    cdf_empirical = np.array([float(np.mean(x <= t)) for t in targets])
    np.testing.assert_allclose(cdf_pm, cdf_empirical)

    # upm_ratio is the complementary survival proportion.
    survival = np.array([float(upm_ratio(0, t, x)) for t in targets])
    np.testing.assert_allclose(survival, 1.0 - cdf_pm)

    # Value-at-risk quantiles: LPM.VaR(p, 0, x) == numpy quantile(x, p).
    percentiles = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    var_pm = np.array([lpm_var(p, 0.0, x) for p in percentiles])
    var_np = np.quantile(x, percentiles, method="linear")
    np.testing.assert_allclose(var_pm, var_np)

    # upm_var is the right-tail VaR, i.e. the (1 - p) quantile.
    upper_var = np.array([upm_var(p, 0.0, x) for p in percentiles])
    np.testing.assert_allclose(upper_var, np.quantile(x, 1.0 - percentiles, method="linear"))

    print("mean (partial moments):", round(mean_via_pm, 6))
    print("sample variance (PM vs numpy):",
          round(sample_variance, 6), round(float(np.var(x, ddof=1)), 6))
    print("CDF at [-1, 0, 1]:", np.round(cdf_pm, 4))
    print("VaR quantiles:", np.round(var_pm, 4))


if __name__ == "__main__":
    main()
