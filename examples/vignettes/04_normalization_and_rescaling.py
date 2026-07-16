"""04. Getting Started with NNS: Normalization and Rescaling.

This is an instructional, section-for-section Python port of
``NNSvignette_04_Normalization_and_Rescaling.Rmd``.  It preserves the R
vignette's distinction between multivariate normalization and univariate
rescaling, prints the important numerical structures, and writes equivalent
figures to ``examples/vignettes/output/04_normalization_and_rescaling``.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _vignette_support import gap, output_dir, save_figure, section, show, subsection, table
from nns import nns_norm, nns_rescale

OUT = output_dir(__file__)


def describe_columns(values: np.ndarray) -> list[tuple[str, float, float]]:
    return [
        (f"column_{index + 1}", float(np.mean(values[:, index])), float(np.std(values[:, index], ddof=1)))
        for index in range(values.shape[1])
    ]


def quantile_normalize(values: np.ndarray) -> np.ndarray:
    """Small transparent implementation used only for the vignette comparison."""
    order = np.argsort(values, axis=0)
    sorted_values = np.take_along_axis(values, order, axis=0)
    rank_means = np.mean(sorted_values, axis=1)
    output = np.empty_like(values)
    for column in range(values.shape[1]):
        output[order[:, column], column] = rank_means
    return output


def main() -> None:
    section("Overview")
    print(
        "NNS.norm rescales multiple variables to a common magnitude while preserving "
        "their individual ordering and distributional form. NNS.rescale applies a "
        "one-dimensional affine transformation to a requested range or expectation."
    )

    section("NNS.norm(): Normalize Multiple Variables")
    print(
        "For means m_j, the ratio grid is RG_ij = m_i / m_j. Linear normalization "
        "averages RG by column; nonlinear normalization weights that grid by absolute "
        "correlation for fewer than ten variables and NNS dependence otherwise."
    )

    subsection("Basic multivariate example")
    rng = np.random.default_rng(123)
    a = rng.normal(0, 1, 100)
    b = rng.normal(0, 5, 100)
    c = rng.normal(10, 1, 100)
    d = rng.normal(10, 10, 100)
    x = np.column_stack((a, b, c, d))

    linear = np.asarray(nns_norm(x, linear=True), dtype=float)
    nonlinear = np.asarray(nns_norm(x, linear=False), dtype=float)

    show("First six rows, linear normalization", linear[:6])
    table(["variable", "mean", "sample sd"], describe_columns(linear))
    print("Linear mode equalizes the column means exactly, as the R proof predicts.")

    show("First six rows, nonlinear normalization", nonlinear[:6])
    table(["variable", "mean", "sample sd"], describe_columns(nonlinear))
    print(
        "Nonlinear means differ because each is a dependence-weighted average of the "
        "original means rather than the single grand mean."
    )

    subsection("Unequal vector lengths")
    vectors = [rng.normal(0, 1, 10), rng.normal(5, 5, 5), rng.normal(10, 10, 8)]
    unequal = nns_norm(vectors)
    show("Normalized unequal-length vectors", {f"x_{i + 1}": v for i, v in enumerate(unequal)})
    print(
        "As in R, unequal lengths force the linear scaling path because a dependence "
        "matrix requires aligned observations."
    )

    subsection("Quantile normalization comparison")
    qnorm = quantile_normalize(x)
    print(
        "Quantile normalization makes every sorted column identical. NNS normalization "
        "does the opposite: it aligns magnitudes while retaining each variable's shape."
    )
    show("Sorted quantile-normalized columns (first five ranks)", np.sort(qnorm, axis=0)[:5])

    section("NNS.rescale(): Distribution Rescaling")
    subsection("Min-max scaling")
    raw = np.array([-2.5, 0.2, 1.1, 3.7, 5.0])
    scaled = nns_rescale(raw, a=5, b=10, method="minmax")
    table(["raw", "scaled"], zip(raw, scaled, strict=True))
    show("Scaled range", np.array([np.min(scaled), np.max(scaled)]))

    subsection("Risk-neutral terminal scaling")
    rng = np.random.default_rng(123)
    s0, rate, maturity = 100.0, 0.05, 1.0
    prices = s0 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, 250)))
    terminal = nns_rescale(
        prices,
        a=s0,
        b=rate,
        method="riskneutral",
        time_to_maturity=maturity,
        type="Terminal",
    )
    target = s0 * math.exp(rate * maturity)
    table(
        ["quantity", "value"],
        [
            ("mean_original", float(np.mean(prices))),
            ("mean_rescaled", float(np.mean(terminal))),
            ("target", target),
        ],
    )

    subsection("Risk-neutral discounted scaling")
    discounted = nns_rescale(
        prices,
        a=s0,
        b=rate,
        method="riskneutral",
        time_to_maturity=maturity,
        type="Discounted",
    )
    table(
        ["quantity", "value"],
        [
            ("mean_returned_series", float(np.mean(discounted))),
            ("target_discounted_mean", s0),
        ],
    )
    gap(
        "Python spells the R argument T as time_to_maturity. The statistical operation "
        "is the same; only the public parameter name differs."
    )

    section("Distribution-shape comparison")
    rng = np.random.default_rng(123)
    x1 = rng.normal(5, 2, 1000)
    x2 = rng.gamma(3, 1, 1000)
    pair = np.column_stack((x1, x2))
    pair_lin = np.asarray(nns_norm(pair, linear=True))
    pair_nonlin = np.asarray(nns_norm(pair, linear=False))
    pair_minmax = np.column_stack(
        ((pair[:, i] - pair[:, i].min()) / np.ptp(pair[:, i]) for i in range(pair.shape[1]))
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    panels = [
        (pair, "Original Variables"),
        (pair_lin, "NNS.norm(linear=True)"),
        (pair_nonlin, "NNS.norm(linear=False)"),
        (pair_minmax, "Standard Min-Max"),
    ]
    for ax, (data, title) in zip(axes.ravel(), panels, strict=True):
        shared = np.histogram_bin_edges(data.ravel(), bins=15)
        ax.hist(data[:, 0], bins=shared, alpha=0.45, label="Normal")
        ax.hist(data[:, 1], bins=shared, alpha=0.45, label="Gamma")
        ax.set_title(title)
        ax.legend()
    save_figure(fig, OUT, "normalization_comparison.png")

    section("Conceptual summary")
    print(
        "NNS.norm is multivariate and dependence-aware. NNS.rescale is univariate and "
        "range- or expectation-targeted. Both are monotone affine transformations and "
        "therefore preserve rank structure for subsequent copula and dependence work."
    )


if __name__ == "__main__":
    main()
