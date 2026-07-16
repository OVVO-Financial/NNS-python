# %% [markdown]
# # 02. Getting Started with NNS: Partial Moments
#
# This is a section-for-section Python port of
# `NNSvignette_02_Partial_Moments.Rmd`.  The R vignette remains the source of
# truth for the statistical narrative.  NumPy uses a different seeded normal
# stream than R, so synthetic draws are distributionally equivalent rather
# than numerically identical.

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples._vignette_support import empirical_cdf, output_dir, save_figure, section, show, subsection
from nns import (
    co_lpm,
    co_upm,
    d_lpm,
    d_upm,
    lpm,
    lpm_ratio,
    nns_cdf,
    nns_mode,
    nns_moments,
    pm_matrix,
    upm,
)

OUT = output_dir(__file__)


def main() -> None:
    section("Partial Moments")
    print(
        "Parsing variance into lower, upper, co- and divergent partial moments "
        "reveals distributional information unavailable from a single variance statistic."
    )

    rng = np.random.default_rng(123)
    x = rng.normal(size=100)
    y = rng.normal(size=100)
    n = x.size

    # %% [markdown]
    # ## Mean
    subsection("Mean")
    mean_standard = float(np.mean(x))
    mean_pm = float(upm(1, 0.0, x) - lpm(1, 0.0, x))
    show("NumPy mean(x)", mean_standard)
    show("UPM(1, 0, x) - LPM(1, 0, x)", mean_pm)

    # %% [markdown]
    # ## Variance
    subsection("Variance")
    sample_variance = float(np.var(x, ddof=1))
    population_variance = float(upm(2, mean_standard, x) + lpm(2, mean_standard, x))
    sample_variance_pm = population_variance * n / (n - 1)
    covariance_of_self = float(
        co_lpm(1, x, x, mean_standard, mean_standard)
        + co_upm(1, x, x, mean_standard, mean_standard)
        - d_lpm(1, 1, x, x, mean_standard, mean_standard)
        - d_upm(1, 1, x, x, mean_standard, mean_standard)
    )
    show("Sample variance (NumPy)", sample_variance)
    show("Sample variance from partial moments", sample_variance_pm)
    show("Population adjustment of sample variance", sample_variance * (n - 1) / n)
    show("Population variance from partial moments", population_variance)
    show("Variance as covariance with itself", covariance_of_self)

    # %% [markdown]
    # ## Standard Deviation
    subsection("Standard Deviation")
    show("Sample standard deviation", float(np.std(x, ddof=1)))
    show("Standard deviation from partial moments", sample_variance_pm**0.5)

    # %% [markdown]
    # ## First 4 Moments
    subsection("First 4 Moments")
    show("NNS moments (population)", nns_moments(x))
    show("NNS moments (sample)", nns_moments(x, population=False))

    # %% [markdown]
    # ## Statistical Mode of a Continuous Distribution
    subsection("Statistical Mode of a Continuous Distribution")
    show("Continuous NNS mode", nns_mode(x))
    show(
        "Discrete multiple modes",
        nns_mode(np.array([1, 2, 2, 3, 3, 4, 4, 5], dtype=float), discrete=True, multi=True),
    )

    # %% [markdown]
    # ## Covariance
    subsection("Covariance")
    covariance_standard = float(np.cov(x, y, ddof=1)[0, 1])
    covariance_pm = float(
        (
            co_lpm(1, x, y, np.mean(x), np.mean(y))
            + co_upm(1, x, y, np.mean(x), np.mean(y))
            - d_lpm(1, 1, x, y, np.mean(x), np.mean(y))
            - d_upm(1, 1, x, y, np.mean(x), np.mean(y))
        )
        * n
        / (n - 1)
    )
    show("Sample covariance (NumPy)", covariance_standard)
    show("Sample covariance from co-partial moments", covariance_pm)

    # %% [markdown]
    # ## Covariance Elements and Covariance Matrix
    subsection("Covariance Elements and Covariance Matrix")
    print("Sigma = CLPM + CUPM - DLPM - DUPM")
    cov_mtx = pm_matrix(1, 1, "mean", np.column_stack((x, y)), pop_adj=True, names=["x", "y"])
    show("PM.matrix equivalent", cov_mtx)
    reconstructed = cov_mtx["clpm"] + cov_mtx["cupm"] - cov_mtx["dlpm"] - cov_mtx["dupm"]
    show("Reassembled covariance matrix", reconstructed)
    show("Standard covariance matrix", np.cov(np.column_stack((x, y)), rowvar=False, ddof=1))

    # %% [markdown]
    # ## Pearson Correlation
    subsection("Pearson Correlation")
    sd_x = sample_variance_pm**0.5
    y_pop_var = float(upm(2, np.mean(y), y) + lpm(2, np.mean(y), y))
    sd_y = (y_pop_var * n / (n - 1)) ** 0.5
    show("Pearson correlation (NumPy)", float(np.corrcoef(x, y)[0, 1]))
    show("Pearson correlation reconstructed from partial moments", covariance_pm / (sd_x * sd_y))

    # %% [markdown]
    # ## CDFs (Discrete and Continuous)
    subsection("CDFs (Discrete and Continuous)")
    targets = np.array([0.0, 1.0])
    show("Empirical CDF at 0 and 1", np.array([np.mean(x <= target) for target in targets]))
    show("LPM degree-0 CDF at 0 and 1", lpm(0, targets, x))
    show("Joint CDF at (0, 0)", co_lpm(0, x, y, 0.0, 0.0))
    show("Vectorized joint CDF at (0,0) and (1,1)", co_lpm(0, x, y, targets, targets))

    ux = np.asarray(lpm_ratio(0, x, x), dtype=float)
    uy = np.asarray(lpm_ratio(0, y, y), dtype=float)
    show("Copula at (0.5, 0.5)", co_lpm(0, ux, uy, 0.5, 0.5))
    show("Continuous NNS.CDF at 1", nns_cdf(x, degree=1, target=1.0))
    show("Continuous NNS.CDF at 1 with mean target", nns_cdf(x, degree=1, target=float(np.mean(x))))
    show("Survival function at 1", nns_cdf(x, degree=1, target=1.0, type="survival"))

    ordered, ecdf_values = empirical_cdf(x)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.step(ordered, ecdf_values, where="post", label="empirical CDF")
    ax.scatter(ordered, lpm(0, ordered, x), s=13, label="LPM degree-0 CDF")
    ax.set_title("Empirical CDF and the degree-0 lower partial moment CDF")
    ax.set_xlabel("x")
    ax.set_ylabel("F(x)")
    ax.legend()
    save_figure(fig, OUT, "cdf_equivalence.png")

    # %% [markdown]
    # ## Numerical Integration
    subsection("Numerical Integration")
    grid = np.arange(0.0, 1.0001, 0.001)
    values = grid**2
    integral = float((upm(1, 0.0, values) - lpm(1, 0.0, values)) * (1.0 - 0.0))
    total_area = float((upm(1, 0.0, values) + lpm(1, 0.0, values)) * (1.0 - 0.0))
    show("Partial-moment approximation to integral_0^1 x^2 dx", integral)
    show("Total unsigned area", total_area)

    # %% [markdown]
    # ## Bayes' Theorem
    subsection("Bayes' Theorem")
    print("P(A > 0 | B > 0) = Co.UPM(0, A, B, 0, 0) / UPM(0, 0, B)")
    conditional_probability = float(co_upm(0, x, y, 0.0, 0.0) / upm(0, 0.0, y))
    show("P(x > 0 | y > 0)", conditional_probability)

    np.testing.assert_allclose(mean_pm, mean_standard)
    np.testing.assert_allclose(sample_variance_pm, sample_variance)
    np.testing.assert_allclose(covariance_pm, covariance_standard)
    np.testing.assert_allclose(reconstructed, np.cov(np.column_stack((x, y)), rowvar=False, ddof=1))


if __name__ == "__main__":
    main()
