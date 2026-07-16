"""Canonical vignette 05: Sampling and simulation.

Source of truth:
    NNS/vignettes/NNSvignette_05_Sampling.Rmd

Combines partial-moment CDF/inverse-CDF sampling with maximum-entropy bootstrap
and dependence-targeted Monte Carlo examples.
"""
from __future__ import annotations

import numpy as np

from nns import lpm_ratio, lpm_var, nns_mc, nns_meboot


def main() -> None:
    rng = np.random.default_rng(123)
    x = rng.normal(size=100)

    targets = np.sort(x)
    empirical = np.asarray([np.mean(x <= t) for t in targets])
    pm_cdf = np.asarray([lpm_ratio(0, t, x) for t in targets], dtype=float)
    np.testing.assert_allclose(pm_cdf, empirical)

    percentiles = np.linspace(0.01, 0.99, 99)
    samples = {
        degree: np.asarray([lpm_var(p, degree, x) for p in percentiles], dtype=float)
        for degree in (0.0, 0.25, 0.5, 1.0, 2.0)
    }
    np.testing.assert_allclose(samples[0.0], np.quantile(x, percentiles, method="linear"))

    series = np.cumsum(rng.normal(scale=0.7, size=80))
    meboot = nns_meboot(series, reps=10, rho=0.95, random_seed=1)
    mc = nns_mc(series, reps=1, lower_rho=-1.0, upper_rho=1.0, by=0.5, random_seed=1)
    assert np.asarray(meboot["ensemble"]).shape == series.shape
    assert np.asarray(mc["ensemble"]).shape == series.shape

    print("CDF parity max error:", float(np.max(np.abs(pm_cdf - empirical))))
    print("inverse-CDF sample heads:")
    for degree, values in samples.items():
        print(f"  degree={degree:g}:", np.round(values[:5], 4))
    print("meboot ensemble shape:", np.asarray(meboot["ensemble"]).shape)
    print("MC rho groups:", list(mc["replicates"]))


if __name__ == "__main__":
    main()
