"""Vignette 02 — Descriptive and distributional tools.

Python translation of distributional pieces of the R NNS Partial Moments and
Overview vignettes: ``nns_moments``, ``nns_mode``, ``pm_matrix`` covariance
reconstruction, and a quantile table built from ``lpm_var``.

Run with::

    python examples/vignettes/descriptive_distributional_tools.py
"""

from __future__ import annotations

import numpy as np

from nns import lpm_ratio, lpm_var, nns_mode, nns_moments, pm_matrix


def main() -> None:
    rng = np.random.default_rng(123)
    x = rng.normal(size=200)
    y = rng.normal(size=200)

    # Distributional summary (population and sample forms).
    population = nns_moments(x, population=True)
    sample = nns_moments(x, population=False)
    assert set(population) == {"mean", "variance", "skewness", "kurtosis"}
    assert sample["variance"] > population["variance"]  # n/(n-1) rescaling

    # Mode estimation: continuous, and discrete-multimodal.
    continuous_mode = float(nns_mode(x))
    discrete_modes = nns_mode(
        np.array([1, 2, 2, 3, 3, 4, 4, 5], dtype=float), discrete=True, multi=True
    )

    # Covariance reconstruction from the partial moment matrix:
    # clpm + cupm - dlpm - dupm == covariance.
    pm = pm_matrix(1, 1, "mean", np.column_stack((x, y)), True, names=["x", "y"])
    reconstructed = pm["clpm"] + pm["cupm"] - pm["dlpm"] - pm["dupm"]
    np.testing.assert_allclose(reconstructed, np.cov(x, y), atol=1e-8)

    # Quantile table via lpm_var (degree 0 == empirical quantile), with the
    # round-trip CDF recovered through lpm_ratio.
    percentiles = np.arange(0.05, 0.96, 0.1)
    thresholds = np.array([lpm_var(p, 0.0, x) for p in percentiles])
    recovered_cdf = np.array([float(lpm_ratio(0, t, x)) for t in thresholds])

    print("population moments:", {k: round(v, 4) for k, v in population.items()})
    print("continuous mode:", round(continuous_mode, 4))
    print("discrete modes:", np.asarray(discrete_modes))
    print("covariance (reconstructed):\n", np.round(reconstructed, 6))
    print("quantile table (threshold -> CDF):")
    for p, t, c in zip(percentiles, thresholds, recovered_cdf, strict=True):
        print(f"  p={p:.2f}  threshold={t:+.4f}  cdf={c:.4f}")


if __name__ == "__main__":
    main()
