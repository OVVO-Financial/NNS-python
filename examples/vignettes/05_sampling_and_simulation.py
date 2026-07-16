"""05. Getting Started with NNS: Sampling and Simulation.

Instructional Python port of:
    NNS/vignettes/NNSvignette_05_Sampling.Rmd

Run directly to print the important returned structures and write the figures to
``examples/vignettes/output/05_sampling_and_simulation``.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples._vignette_support import (
    empirical_cdf,
    fast_mode,
    note,
    output_dir,
    save_figure,
    section,
    show,
    subsection,
    table,
)
from nns import lpm_ratio, lpm_var, nns_copula, nns_mc, nns_meboot

OUT = output_dir(__file__)

DEGREES = (0.0, 0.25, 0.5, 1.0, 2.0)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ranks_a = np.argsort(np.argsort(a)).astype(float)
    ranks_b = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ranks_a, ranks_b)[0, 1])


def main() -> None:
    section("Sampling")
    print(
        "NNS offers novel sampling methods from any distribution, as well as "
        "simulation that maintains the dependence of the original variables."
    )

    # %% [markdown]
    # ## CDFs
    #
    # ### Empirical CDF
    #
    # The empirical CDF is the base construct: F(x) = P(X <= x). R uses
    # ``ecdf(x)`` and evaluates the resulting step function at 0 and 1.
    subsection("Empirical CDF")
    rng = np.random.default_rng(123)
    x = rng.normal(size=100)
    p0 = float(np.mean(x <= 0.0))
    p1 = float(np.mean(x <= 1.0))
    table(["target", "empirical CDF"], [(0.0, p0), (1.0, p1)])

    # %% [markdown]
    # ### Lower Partial Moment CDF (LPM.ratio)
    #
    # With degree 0, LPM.ratio is identical to the empirical CDF:
    # LPM(0, t, X) counts the share of observations at or below t.
    subsection("LPM.ratio degree 0 equals the empirical CDF")
    table(
        ["target", "ecdf", "LPM.ratio(0, target, x)"],
        [
            (0.0, p0, float(lpm_ratio(0, 0.0, x))),
            (1.0, p1, float(lpm_ratio(0, 1.0, x))),
        ],
    )

    targets = np.sort(x)
    lpm_cdf = np.asarray(lpm_ratio(0, targets, x), dtype=float)
    empirical = np.asarray([np.mean(x <= t) for t in targets])
    max_gap = float(np.max(np.abs(lpm_cdf - empirical)))
    print("Maximum |LPM.ratio - ecdf| over every observation:", max_gap)

    steps, probs = empirical_cdf(x)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.step(steps, probs, where="post", color="black", label="ecdf")
    ax.scatter(targets, lpm_cdf, color="red", s=18, zorder=3, label="LPM.ratio")
    ax.set_xlabel("x")
    ax.set_ylabel("F(x)")
    ax.set_title("Empirical CDF and LPM.ratio(degree = 0)")
    ax.legend(loc="upper left")
    save_figure(fig, OUT, "01_ecdf_vs_lpm_ratio.png")

    # %% [markdown]
    # ### LPM.ratio degree > 0
    #
    # Increasing the degree parameter to any positive real number generates
    # different CDFs of the initial distribution.
    subsection("LPM.ratio for degrees > 0")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.step(steps, probs, where="post", color="black", linewidth=3, label="ecdf")
    colors = plt.get_cmap("rainbow")(np.linspace(0.0, 1.0, len(DEGREES)))
    for degree, color in zip(DEGREES, colors, strict=True):
        curve = np.asarray(lpm_ratio(degree, targets, x), dtype=float)
        ax.plot(targets, curve, color=color, linewidth=2, label=f"LPM.ratio(degree = {degree:g})")
    normal_grid = np.linspace(-3.0, 3.0, 200)
    normal_cdf = 0.5 * (1.0 + np.asarray([math.erf(v / math.sqrt(2.0)) for v in normal_grid]))
    ax.plot(
        normal_grid,
        normal_cdf,
        color="black",
        linestyle=":",
        linewidth=2,
        label="N(0,1) approximation",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("F(x)")
    ax.set_title("CDF shapes from the LPM degree")
    ax.legend(loc="upper left", fontsize=8)
    save_figure(fig, OUT, "02_lpm_ratio_degrees.png")

    # %% [markdown]
    # ### Generating PDFs with LPM.VaR
    #
    # LPM.VaR provides inverse-CDF estimates: sampling percentile grids under a
    # degree produces new samples of the underlying distribution.
    subsection("Inverse CDF sampling via LPM.VaR")
    percentiles = np.linspace(0.0, 1.0, 100)
    samples = {
        degree: np.asarray([lpm_var(p, degree, x) for p in percentiles], dtype=float)
        for degree in DEGREES
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()
    axes[0].step(steps, probs, where="post", color="black", linewidth=2)
    axes[0].set_title("eCDF via LPM.ratio()")
    reference_counts, reference_edges = np.histogram(samples[0.0], bins=15)
    for ax, (degree, values), color in zip(axes[1:], samples.items(), colors, strict=True):
        ax.stairs(reference_counts, reference_edges, color="black", linewidth=2)
        ax.hist(values, bins=15, color=color, alpha=0.5)
        ax.set_title(f"Inverse CDF via LPM.VaR(degree {degree:g})")
        ax.set_xlabel("x")
        ax.set_ylabel("freq")
    save_figure(fig, OUT, "03_lpm_var_inverse_cdf.png")

    print("First 10 samples from each degree compared with the original x:")
    sorted_x = np.sort(x)
    table(
        ["original x"] + [f"degree {degree:g}" for degree in DEGREES],
        [
            tuple([float(sorted_x[i])] + [float(samples[degree][i]) for degree in DEGREES])
            for i in range(10)
        ],
    )
    print(
        "Degree 0 reproduces the empirical quantiles exactly; higher degrees "
        "concentrate samples toward the distribution center."
    )

    section("Simulation")

    # %% [markdown]
    # ## Bootstrapping (NNS.meboot)
    #
    # NNS.meboot is based on the maximum entropy bootstrap, designed for time
    # series and avoiding the IID assumption. Sampling from specified
    # correlations ensures the full spectrum of future paths is sampled;
    # typical Monte Carlo samples are restricted to [-0.3, 0.3] correlation
    # with the original data. NNS.MC is a streamlined wrapper generating one
    # replicate for each rho in a sequence.
    subsection("NNS.MC replicates across the correlation space")
    mc = nns_mc(x, reps=1, lower_rho=-1.0, upper_rho=1.0, by=0.5, random_seed=123)
    replicates = {
        name: np.asarray(mc["replicates"][name], dtype=float).ravel() for name in mc["replicates"]
    }
    show("NNS.MC returned structure", mc)

    fig, ax = plt.subplots(figsize=(9, 6))
    rho_colors = plt.get_cmap("rainbow")(np.linspace(0.0, 1.0, len(replicates)))
    for (name, path), color in zip(replicates.items(), rho_colors, strict=True):
        ax.plot(path, color=color, linewidth=1.2, label=name)
    ax.plot(x, color="black", linewidth=3, label="original x")
    ax.set_xlabel("observation")
    ax.set_title("NNS.MC replicates versus the original series")
    ax.legend(fontsize=8)
    save_figure(fig, OUT, "04_mc_replicates.png")

    print("Replicate Spearman correlations with the original x:")
    table(
        ["replicate", "Spearman correlation"],
        [(name, spearman(path, x)) for name, path in replicates.items()],
    )
    note(
        "The rho target lives on each replicate individually, and it is measured "
        "in the metric that `type` targets. The ensemble is the per-observation "
        "mean of replicates, so correlations taken on the ensemble read higher "
        "than the per-replicate rho. Calibrate rho on the replicates."
    )

    # %% [markdown]
    # ### target_drift Specification
    #
    # A target drift can be requested for the replicates with target_drift.
    subsection("target_drift specification")
    drifted = nns_mc(
        x,
        reps=1,
        lower_rho=-1.0,
        upper_rho=1.0,
        by=0.5,
        target_drift=0.05,
        random_seed=123,
    )
    drift_replicates = {
        name: np.asarray(drifted["replicates"][name], dtype=float).ravel()
        for name in drifted["replicates"]
    }
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x, color="black", linewidth=3, label="original x")
    for (name, path), color in zip(drift_replicates.items(), rho_colors, strict=True):
        ax.plot(path, color=color, linewidth=1.2, label=name)
    ax.set_xlabel("observation")
    ax.set_title("NNS.MC replicates with target_drift = 0.05")
    ax.legend(fontsize=8)
    save_figure(fig, OUT, "05_mc_replicates_target_drift.png")

    # %% [markdown]
    # ## Simulating a Multivariate Dependence Structure
    #
    # Analogous to an empirical copula transformation: (1) determine the
    # dependence structure with LPM.ratio(1, x, x); (2) generate or supply new
    # data of any distribution or length; (3) map the new data through LPM.VaR
    # at the stored positions.
    subsection("Multivariate dependence transfer")
    rng = np.random.default_rng(123)
    n = 250 if fast_mode() else 1000
    x_sim = rng.normal(size=n)
    y_sim = rng.normal(size=n)
    z_sim = rng.normal(size=n)
    original = np.column_stack((x_sim, y_sim, z_sim, x_sim))
    print("x is duplicated in the panel to avoid total independence (example only).")

    dependence_structure = np.column_stack(
        [
            np.asarray(lpm_ratio(1, original[:, j], original[:, j]), dtype=float)
            for j in range(original.shape[1])
        ]
    )
    new_data = rng.normal(10.0, 20.0, size=(original.shape[0] * 2, original.shape[1]))
    new_dep_data = np.column_stack(
        [
            np.asarray(
                [lpm_var(p, 1, new_data[:, j]) for p in dependence_structure[:, j]],
                dtype=float,
            )
            for j in range(original.shape[1])
        ]
    )

    show("head(original.data)", original[:6])
    show("head(new.dep.data)", new_dep_data[:6])
    table(
        ["panel", "NNS.copula"],
        [
            ("original.data", float(nns_copula(original))),
            ("new.dep.data", float(nns_copula(new_dep_data))),
        ],
    )
    print(
        "Similar multivariate dependence with radically different values, since "
        "N(10, 20) draws replaced the original N(0, 1) observations."
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(original[:, 0], original[:, 1], s=10, alpha=0.5)
    axes[0].set_title("original.data: x vs y")
    axes[1].scatter(new_dep_data[:, 0], new_dep_data[:, 1], s=10, alpha=0.5, color="darkorange")
    axes[1].set_title("new.dep.data: same dependence, new scale")
    for ax in axes:
        ax.set_xlabel("first variable")
        ax.set_ylabel("second variable")
    save_figure(fig, OUT, "06_dependence_transfer.png")

    # %% [markdown]
    # ## Alternative Using NNS.meboot
    #
    # To keep simulated values close to the original data, apply NNS.meboot to
    # each variable with rho = 0.95 and use the replicate ensembles.
    subsection("NNS.meboot alternative")
    reps = 10 if fast_mode() else 100
    boots = [
        nns_meboot(original[:, j], reps=reps, rho=0.95, random_seed=123)
        for j in range(original.shape[1])
    ]
    boot_matrix = np.column_stack([np.asarray(b["ensemble"], dtype=float).ravel() for b in boots])

    print("Ensemble Spearman correlations with original.data:")
    table(
        ["variable", "Spearman correlation"],
        [
            (name, spearman(boot_matrix[:, j], original[:, j]))
            for j, name in enumerate(["x", "y", "z", "x (duplicate)"])
        ],
    )
    show("head(new.boot.dep.matrix)", boot_matrix[:6])
    table(
        ["panel", "NNS.copula"],
        [
            ("original.data", float(nns_copula(original))),
            ("new.boot.dep.matrix", float(nns_copula(boot_matrix))),
        ],
    )
    print("Similar dependence with similar values.")
    note(
        "The ensemble is the mean of the replicates, so its correlation with the "
        "original series reads above the per-replicate rho of 0.95; the rho "
        "alignment is calibrated on the individual replicates."
    )
    note(
        "Stochastic draws use NumPy generators, so simulations are "
        "distributionally equivalent to R but not bit-for-bit identical draws."
    )


if __name__ == "__main__":
    main()
