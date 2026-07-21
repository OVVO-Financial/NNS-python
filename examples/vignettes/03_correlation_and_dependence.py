# %% [markdown]
# # 03. Getting Started with NNS: Correlation and Dependence
#
# Section-for-section port of `NNSvignette_03_Correlation_and_Dependence.Rmd`.
# Figures are saved beside the script under `output/03_correlation_and_dependence/`.

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples._vignette_support import (
    gap,
    output_dir,
    partition_scatter,
    save_figure,
    section,
    show,
    subsection,
)
from nns import nns_copula, nns_dep, nns_part

OUT = output_dir(__file__)


def relationship(name: str, x: np.ndarray, y: np.ndarray, *, obs_req: int = 8) -> None:
    subsection(name)
    show("Pearson correlation", float(np.corrcoef(x, y)[0, 1]))
    show("NNS correlation and dependence", nns_dep(x, y))
    part = nns_part(x, y, order=3, obs_req=obs_req)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].scatter(x, y, s=12, alpha=0.65)
    axes[0].set_title(name)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    partition_scatter(axes[1], part, "NNS partial-moment partition")
    save_figure(fig, OUT, name.lower().replace(" ", "_") + ".png")


def main() -> None:
    section("Correlation and Dependence")
    print(
        "Pearson correlation measures linear association. NNS separates signed "
        "correlation from unsigned signal-to-noise dependence so nonlinear relationships "
        "need not disappear merely because their global slope is small."
    )
    gap(
        "R's NNS.part(..., Voronoi=TRUE) draws Voronoi regions internally. The Python "
        "partition API returns the same quadrant paths and regression points but has no "
        "Voronoi flag, so the figures color the returned partition memberships directly."
    )

    # %% [markdown]
    # ## Linear Equivalence
    x = np.arange(0.0, 3.001, 0.01)
    relationship("Linear Equivalence", x, 2.0 * x)

    # %% [markdown]
    # ## Nonlinear Relationship
    relationship("Nonlinear Relationship", x, x**10)

    # %% [markdown]
    # ## Cyclic Relationship
    x_cycle = np.arange(0.0, 12.0 * np.pi + np.pi / 200.0, np.pi / 100.0)
    y_cycle = np.sin(x_cycle)
    relationship("Cyclic Relationship", x_cycle, y_cycle, obs_req=0)

    # %% [markdown]
    # ## Asymmetrical Analysis
    subsection("Asymmetrical Analysis")
    show("Pearson cor(x, y)", float(np.corrcoef(x_cycle, y_cycle)[0, 1]))
    show("NNS.dep(x, y, asym=True)", nns_dep(x_cycle, y_cycle, asym=True))
    show("Pearson cor(y, x)", float(np.corrcoef(y_cycle, x_cycle)[0, 1]))
    show("NNS.dep(y, x, asym=True)", nns_dep(y_cycle, x_cycle, asym=True))

    # %% [markdown]
    # ## Dependence
    subsection("Dependence")
    rng = np.random.default_rng(123)
    cloud = rng.uniform(-1.0, 1.0, size=(10000, 2))
    radius2 = np.sum(cloud**2, axis=1)
    ring = cloud[(radius2 <= 1.0) & (radius2 >= 0.95)]
    show("NNS dependence for annulus", nns_dep(ring[:, 0], ring[:, 1]))
    ring_part = nns_part(ring[:, 0], ring[:, 1], order=3, obs_req=0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(ring[:, 0], ring[:, 1], s=9, alpha=0.6)
    axes[0].set_aspect("equal")
    axes[0].set_title("Annulus: near-zero Pearson correlation, strong dependence")
    partition_scatter(axes[1], ring_part, "NNS partition of annulus")
    axes[1].set_aspect("equal")
    save_figure(fig, OUT, "annulus_dependence.png")

    # %% [markdown]
    # # p-values for NNS.dep()
    subsection("p-values for NNS.dep()")
    gap(
        "Python nns_dep currently exposes the deterministic coefficient only; R also "
        "accepts p.value=TRUE and print.map=TRUE. This vignette reproduces that executable "
        "permutation analysis transparently rather than omitting it."
    )
    rng = np.random.default_rng(123)
    x_perm = np.arange(-5.0, 5.001, 0.1)
    y_perm = x_perm**2 + rng.normal(size=x_perm.size)
    observed = nns_dep(x_perm, y_perm)
    repetitions = 30 if __import__("os").environ.get("NNS_VIGNETTE_FAST") == "1" else 100
    null_corr = np.empty(repetitions)
    null_dep = np.empty(repetitions)
    for index in range(repetitions):
        permuted = rng.permutation(y_perm)
        result = nns_dep(x_perm, permuted)
        null_corr[index] = result["Correlation"]
        null_dep[index] = result["Dependence"]
    p_corr = (1.0 + np.sum(np.abs(null_corr) >= abs(observed["Correlation"]))) / (repetitions + 1.0)
    p_dep = (1.0 + np.sum(null_dep >= observed["Dependence"])) / (repetitions + 1.0)
    show("Observed NNS.dep", observed)
    show("Permutation p-value for correlation", p_corr)
    show("Permutation p-value for dependence", p_dep)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(null_corr, bins=15)
    axes[0].axvline(observed["Correlation"], linewidth=2)
    axes[0].set_title("Permutation null: NNS correlation")
    axes[1].hist(null_dep, bins=15)
    axes[1].axvline(observed["Dependence"], linewidth=2)
    axes[1].set_title("Permutation null: NNS dependence")
    save_figure(fig, OUT, "permutation_map.png")

    # %% [markdown]
    # # Multivariate Dependence NNS.copula()
    subsection("Multivariate Dependence NNS.copula()")
    rng = np.random.default_rng(123)
    independent = rng.normal(size=(1000, 3))
    dependence = nns_copula(independent)
    show("Three-variable NNS copula dependence", dependence)
    gap(
        "R's NNS.copula(..., plot=TRUE, independence.overlay=TRUE) renders its own map. "
        "Python returns the scalar multivariate dependence; the figure below supplies a "
        "pairwise projection and an independently simulated reference overlay."
    )
    reference = rng.normal(size=(1000, 3))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    pairs = [(0, 1), (0, 2), (1, 2)]
    for ax, (left, right) in zip(axes, pairs, strict=True):
        ax.scatter(independent[:, left], independent[:, right], s=8, alpha=0.35, label="sample")
        ax.scatter(
            reference[:, left],
            reference[:, right],
            s=8,
            alpha=0.15,
            label="independence overlay",
        )
        ax.set_title(f"X{left + 1} vs X{right + 1}")
    axes[0].legend()
    fig.suptitle(f"Multivariate NNS copula dependence = {dependence:.4f}")
    save_figure(fig, OUT, "multivariate_copula.png")


if __name__ == "__main__":
    main()
