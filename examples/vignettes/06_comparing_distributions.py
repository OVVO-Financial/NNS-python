"""06. Getting Started with NNS: Comparing Distributions.

Instructional Python port of:
    NNS/vignettes/NNSvignette_06_Comparing_Distributions.Rmd

Run directly to print the important returned structures and write the figures to
``examples/vignettes/output/06_comparing_distributions``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples._vignette_support import (
    empirical_cdf,
    fast_mode,
    gap,
    load_mtcars,
    note,
    output_dir,
    save_figure,
    section,
    show,
    subsection,
    table,
)
from nns import fsd, nns_anova, nns_sd_cluster, nns_ss, sd_efficient_set

OUT = output_dir(__file__)


def anova_figure(
    control: np.ndarray,
    treatment: np.ndarray,
    result: dict[str, object],
    title: str,
    filename: str,
) -> None:
    """Reconstruct R's NNS.ANOVA plot side effect from the returned statistics."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    control_stat = float(np.asarray(result["Control"], dtype=float))
    treatment_stat = float(np.asarray(result["Treatment"], dtype=float))
    grand = float(np.asarray(result["Grand_Statistic"], dtype=float))
    axes[0].bar(
        ["control", "treatment", "grand"],
        [control_stat, treatment_stat, grand],
        color=["steelblue", "darkorange", "gray"],
    )
    lower = result.get("Lower Bound Robust Certainty")
    upper = result.get("Upper Bound Robust Certainty")
    robust = result.get("Robust Certainty Estimate")
    label = f"Certainty = {float(np.asarray(result['Certainty'], dtype=float)):.4f}"
    if robust is not None:
        robust_value = float(np.asarray(robust, dtype=float))
        bounds = (float(np.asarray(lower, dtype=float)), float(np.asarray(upper, dtype=float)))
        label += f"\nrobust = {robust_value:.4f} [{bounds[0]:.4f}, {bounds[1]:.4f}]"
    axes[0].set_title(label)
    axes[0].set_ylabel("group statistic")

    for values, name, color in (
        (control, "control", "steelblue"),
        (treatment, "treatment", "darkorange"),
    ):
        ordered, probs = empirical_cdf(values)
        axes[1].step(ordered, probs, where="post", label=name, color=color)
    axes[1].axvline(grand, linestyle="--", color="gray", label="grand statistic")
    axes[1].set_xlabel("value")
    axes[1].set_ylabel("F(x)")
    axes[1].set_title(title)
    axes[1].legend()
    save_figure(fig, OUT, filename)


def main() -> None:
    section("Comparing Distributions")
    print(
        "NNS tests whether distributions came from the same population, or share "
        "the same mean or median, through NNS.ANOVA. The output is a Certainty "
        "statistic on [0, 1], with 1 representing identical distributions."
    )
    gap(
        "R's NNS.ANOVA(..., plot = TRUE) draws its diagnostics as a side effect. "
        "The Python API returns the statistics without plotting, so each figure "
        "here is reconstructed from the returned values and empirical CDFs."
    )

    # %% [markdown]
    # ## Test if Same Population
    #
    # Do automatic and manual transmissions have significantly different mpg
    # distributions per the mtcars dataset?
    subsection("Test if same population: mtcars mpg by transmission")
    mtcars = load_mtcars()
    mpg_auto = mtcars["mpg"][mtcars["am"] == 1]
    mpg_manual = mtcars["mpg"][mtcars["am"] == 0]
    same_population = nns_anova(
        mpg_manual,
        mpg_auto,
        robust=True,
        n_boot=100 if fast_mode() else 1000,
        random_seed=123,
    )
    show(
        "NNS.ANOVA(control = manual mpg, treatment = automatic mpg, robust = TRUE)",
        same_population,
    )
    anova_figure(
        mpg_manual,
        mpg_auto,
        same_population,
        "mtcars mpg by transmission",
        "01_anova_mtcars.png",
    )
    print(
        "The Certainty shows these two distributions clearly do not come from the "
        "same population. The nonparametric Mann-Whitney-Wilcoxon test agrees:"
    )
    wilcox = stats.mannwhitneyu(mpg_manual, mpg_auto, alternative="two-sided")
    table(["statistic", "p-value"], [(float(wilcox.statistic), float(wilcox.pvalue))])

    # %% [markdown]
    # ## Test if means are Equal
    #
    # Two Normal samples share a mean of zero with different standard
    # deviations; both NNS.ANOVA and the t-test should be comfortable that the
    # means are equal.
    subsection("Test if means are equal")
    rng = np.random.default_rng(123)
    x = rng.normal(0.0, 1.0, 1000)
    y = rng.normal(0.0, 2.0, 1000)
    equal_means = nns_anova(
        x,
        y,
        means_only=True,
        robust=True,
        n_boot=100 if fast_mode() else 1000,
        random_seed=123,
    )
    show("NNS.ANOVA(x, y, means.only = TRUE, robust = TRUE)", equal_means)
    anova_figure(x, y, equal_means, "Equal means, unequal variance", "02_anova_equal_means.png")
    ttest = stats.ttest_ind(x, y, equal_var=False)
    table(["t statistic", "p-value"], [(float(ttest.statistic), float(ttest.pvalue))])

    # %% [markdown]
    # ## Test if means are Unequal
    #
    # Shifting y's mean to 1 shows the sensitivity of both methods, which firmly
    # reject equal means. The effect size interval reported by NNS.ANOVA should
    # contain the specified shift of 1.
    subsection("Test if means are unequal")
    y_shifted = rng.normal(1.0, 1.0, 1000)
    unequal_means = nns_anova(
        x,
        y_shifted,
        means_only=True,
        robust=True,
        n_boot=100 if fast_mode() else 1000,
        random_seed=123,
    )
    show("NNS.ANOVA(x, y, means.only = TRUE, robust = TRUE)", unequal_means)
    anova_figure(x, y_shifted, unequal_means, "Unequal means", "03_anova_unequal_means.png")
    ttest = stats.ttest_ind(x, y_shifted, equal_var=False)
    table(["t statistic", "p-value"], [(float(ttest.statistic), float(ttest.pvalue))])
    print(
        "Effect size bounds "
        f"[{float(np.asarray(unequal_means['Effect_Size_LB'], dtype=float)):.4f}, "
        f"{float(np.asarray(unequal_means['Effect_Size_UB'], dtype=float)):.4f}] "
        "contain the specified shift of 1."
    )

    # %% [markdown]
    # ## Medians
    #
    # Setting both means.only = TRUE and medians = TRUE tests medians instead.
    subsection("Medians")
    unequal_medians = nns_anova(
        x,
        y_shifted,
        means_only=True,
        medians=True,
        robust=True,
        n_boot=100 if fast_mode() else 1000,
        random_seed=123,
    )
    show("NNS.ANOVA(x, y, means.only = TRUE, medians = TRUE, robust = TRUE)", unequal_medians)

    # %% [markdown]
    # # Stochastic Superiority
    #
    # Rather than testing equality, stochastic superiority measures the
    # probability that a random draw from one distribution exceeds a random
    # draw from another: P* = P(X > Y) + 0.5 * P(X = Y). A value of 0.5 means
    # no directional advantage.
    section("Stochastic Superiority")
    superiority = nns_ss(x, y_shifted)
    show("NNS.SS(x, y)", superiority)
    print(
        "y was generated with the higher mean, so P* for x versus y is below "
        "0.5: a draw from x is less likely to exceed a draw from y."
    )

    subsection("Bootstrap confidence interval")
    with_interval = nns_ss(
        x,
        y_shifted,
        confidence_interval=True,
        reps=99 if fast_mode() else 999,
        ci=0.95,
        random_seed=123,
    )
    interval_keys = ["p_gt", "p_tie", "p_star", "lower", "upper"]
    show(
        "NNS.SS(x, y, confidence.interval = TRUE)",
        {key: with_interval[key] for key in interval_keys if key in with_interval},
    )

    subsection("Discrete variables and ties")
    x_discrete = rng.integers(1, 6, 100).astype(float)
    y_discrete = rng.integers(1, 6, 100).astype(float)
    discrete_superiority = nns_ss(x_discrete, y_discrete)
    show("NNS.SS on discrete 1..5 samples", discrete_superiority)
    print(
        "Ties occur with positive probability, so p_tie and p_star reflect the "
        "adjustment explicitly."
    )

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for values, name, color in ((x, "x", "steelblue"), (y_shifted, "y", "darkorange")):
        ordered, probs = empirical_cdf(values)
        ax.step(ordered, probs, where="post", label=name, color=color)
    p_star = float(np.asarray(superiority["p_star"], dtype=float))
    ax.set_title(f"Stochastic superiority: P*(x > y) = {p_star:.4f}")
    ax.set_xlabel("value")
    ax.set_ylabel("F(x)")
    ax.legend()
    save_figure(fig, OUT, "04_stochastic_superiority.png")

    # %% [markdown]
    # # Stochastic Dominance
    #
    # First, second, and third degree dominance tests are available via
    # NNS.FSD, NNS.SSD and NNS.TSD. NNS.FSD correctly identifies the shift in
    # the y variable specified in the unequal means example.
    section("Stochastic Dominance")
    fsd_result = fsd(y_shifted, x)
    print("NNS.FSD(y, x):", fsd_result, "(1 indicates y first-degree dominates x)")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for values, name, color in ((x, "x", "steelblue"), (y_shifted, "y", "darkorange")):
        ordered, probs = empirical_cdf(values)
        ax.step(ordered, probs, where="post", label=name, color=color)
    ax.set_title("First-degree dominance: the y CDF sits at or right of x everywhere")
    ax.set_xlabel("value")
    ax.set_ylabel("F(x)")
    ax.legend()
    save_figure(fig, OUT, "05_stochastic_dominance.png")
    gap(
        "R's NNS.FSD plots both CDFs as a side effect and returns the degree of "
        "dominance. Python returns the indicator, so the CDF comparison figure is "
        "generated from the same inputs."
    )

    # %% [markdown]
    # ## Stochastic Dominant Efficient Sets
    #
    # x2, x4, x6, x8 dominate their preceding distributions yet do not dominate
    # one another, forming the first-degree efficient set.
    subsection("Stochastic dominant efficient sets")
    rng = np.random.default_rng(123)
    base = [rng.normal(size=1000) for _ in range(4)]
    columns: list[np.ndarray] = []
    for series in base:
        columns.extend((series, series + 1.0))
    panel = np.column_stack(columns)
    names = [f"x{i + 1}" for i in range(panel.shape[1])]
    efficient = sd_efficient_set(panel, degree=1)
    efficient_names = [names[i - 1] for i in np.asarray(efficient, dtype=int)]
    show("NNS.SD.efficient.set(cbind(x1..x8), degree = 1)", efficient_names)

    # %% [markdown]
    # ## Stochastic Dominant Clusters
    #
    # Clusters are assigned to non-dominated constituents; R also renders a
    # dendrogram from the returned linkage.
    subsection("Stochastic dominant clusters")
    clusters = nns_sd_cluster(panel, degree=1, dendrogram=True, names=names)
    show("NNS.SD.cluster(cbind(x1..x8), degree = 1, dendrogram = TRUE)", clusters["Clusters"])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    cluster_map = clusters["Clusters"]
    palette = plt.get_cmap("tab10")
    for cluster_index, (cluster_name, members) in enumerate(dict(cluster_map).items()):
        for member in np.atleast_1d(np.asarray(members, dtype=str)):
            column = names.index(str(member))
            ordered, probs = empirical_cdf(panel[:, column])
            ax.step(
                ordered,
                probs,
                where="post",
                color=palette(cluster_index),
                label=f"{member} ({cluster_name})",
            )
    ax.set_xlabel("value")
    ax.set_ylabel("F(x)")
    ax.set_title("First-degree stochastic dominance clusters")
    ax.legend(fontsize=8)
    save_figure(fig, OUT, "06_sd_clusters.png")
    note(
        "The returned Dendrogram entry carries the linkage used by R's plot; the "
        "cluster CDFs above color each member by its assigned cluster."
    )


if __name__ == "__main__":
    main()
