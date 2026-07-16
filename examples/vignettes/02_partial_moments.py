"""Canonical vignette 02: Partial moments.

Source of truth:
    NNS/vignettes/NNSvignette_02_Partial_Moments.Rmd

Covers the mean and variance identities, covariance reconstruction, empirical
CDFs, a joint CDF/Bayes identity, inverse-CDF sampling, and numerical
integration using the public Python API.
"""
from __future__ import annotations

import numpy as np

from nns import (
    co_lpm,
    co_upm,
    d_lpm,
    d_upm,
    lpm,
    lpm_ratio,
    lpm_var,
    nns_moments,
    nns_mode,
    pm_matrix,
    upm,
)


def main() -> None:
    rng = np.random.default_rng(123)
    x = rng.normal(size=100)
    y = rng.normal(size=100)
    n = x.size

    mean_pm = float(upm(1, 0.0, x) - lpm(1, 0.0, x))
    variance_pm = float(upm(2, x.mean(), x) + lpm(2, x.mean(), x)) * n / (n - 1)
    assert np.isclose(mean_pm, x.mean())
    assert np.isclose(variance_pm, x.var(ddof=1))

    covariance_pm = (
        co_lpm(1, x, y, x.mean(), y.mean())
        + co_upm(1, x, y, x.mean(), y.mean())
        - d_lpm(1, 1, x, y, x.mean(), y.mean())
        - d_upm(1, 1, x, y, x.mean(), y.mean())
    ) * n / (n - 1)
    assert np.isclose(float(covariance_pm), np.cov(x, y)[0, 1])

    pm = pm_matrix(1, 1, "mean", np.column_stack((x, y)), True, names=["x", "y"])
    reconstructed = pm["clpm"] + pm["cupm"] - pm["dlpm"] - pm["dupm"]
    np.testing.assert_allclose(reconstructed, np.cov(x, y), atol=1e-8)

    targets = np.array([0.0, 1.0])
    cdf_pm = np.array([float(lpm(0, t, x)) for t in targets])
    np.testing.assert_allclose(cdf_pm, [np.mean(x <= t) for t in targets])

    # P(X > 0 | Y > 0) = P(X > 0, Y > 0) / P(Y > 0).
    bayes = float(co_upm(0, x, y, 0.0, 0.0) / upm(0, 0.0, y))
    assert 0.0 <= bayes <= 1.0

    percentiles = np.linspace(0.05, 0.95, 19)
    samples = np.asarray([lpm_var(p, 0.0, x) for p in percentiles], dtype=float)
    recovered = np.asarray([lpm_ratio(0, t, x) for t in samples], dtype=float)
    assert samples.shape == recovered.shape

    grid = np.linspace(0.0, 1.0, 1001)
    integral = float(upm(1, 0.0, grid**2) - lpm(1, 0.0, grid**2))
    assert np.isclose(integral, 1.0 / 3.0, atol=2e-3)

    print("mean via partial moments:", round(mean_pm, 6))
    print("sample variance via partial moments:", round(variance_pm, 6))
    print("sample covariance via co-partial moments:", round(float(covariance_pm), 6))
    print("moments:", nns_moments(x))
    print("mode:", nns_mode(x))
    print("CDF at [0, 1]:", np.round(cdf_pm, 4))
    print("P(X > 0 | Y > 0):", round(bayes, 4))
    print("integral of x^2 on [0, 1]:", round(integral, 6))


if __name__ == "__main__":
    main()
