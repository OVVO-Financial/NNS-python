"""07. Getting Started with NNS: Clustering and Regression.

Instructional Python port of:
    NNS/vignettes/NNSvignette_07_Clustering_and_Regression.Rmd

Run directly to print the important returned structures and write the figures to
``examples/vignettes/output/07_clustering_and_regression``.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples._vignette_support import (
    fast_mode,
    gap,
    load_iris,
    note,
    output_dir,
    partition_scatter,
    regression_scatter,
    save_figure,
    section,
    show,
    subsection,
    table,
)
from nns import nns_part, nns_reg, nns_stack

OUT = output_dir(__file__)


def main() -> None:
    section("Clustering and Regression")
    print(
        "NNS.part is both a partitional and hierarchical clustering method: it "
        "iteratively splits the joint distribution into partial-moment quadrants and "
        "assigns a quadrant identification at each partition. The quadrant means are "
        "the regression points reused by NNS.reg."
    )

    # %% [markdown]
    # ## NNS Partitioning NNS.part()
    #
    # R draws each order's Voronoi tessellation as a side effect of
    # ``NNS.part(..., Voronoi = TRUE)``. Python has no Voronoi rendering flag, so
    # each returned partition is drawn directly from its quadrant memberships and
    # regression points.
    subsection("NNS.part(): orders 1 through 4")
    x = np.round(np.arange(-5.0, 5.0001, 0.05), 10)
    y = x**3

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for order, ax in zip(range(1, 5), axes.ravel(), strict=True):
        part = nns_part(x, y, order=order, obs_req=0)
        partition_scatter(ax, part, f"NNS.part order = {order}")
    save_figure(fig, OUT, "01_partition_orders.png")

    final = nns_part(x, y, order=4, obs_req=0)
    show("NNS.part(x, y, order = 4, obs.req = 0) returned structure", final)
    gap(
        "R's NNS.part(..., Voronoi = TRUE) draws the tessellation itself. The Python "
        "partition result contains the same observation quadrant assignments and "
        "regression points, which these figures color directly."
    )

    # %% [markdown]
    # ### X-only Partitioning
    #
    # ``type = "XONLY"`` partitions on the x values alone, using the entire
    # bandwidth for the regression point derivation. Identifications are limited
    # to 1's and 2's (left and right of each partition) rather than the four
    # quadrant values of joint partitioning.
    subsection("X-only partitioning")
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for order, ax in zip(range(1, 5), axes.ravel(), strict=True):
        part = nns_part(x, y, order=order, type="XONLY")
        partition_scatter(ax, part, f'NNS.part type = "XONLY", order = {order}')
    save_figure(fig, OUT, "02_partition_orders_xonly.png")

    xonly = nns_part(x, y, order=4, type="XONLY")
    show('NNS.part(x, y, order = 4, type = "XONLY") returned structure', xonly)
    ids = np.unique(np.asarray(xonly["dt"]["quadrant"], dtype=str))
    print("Distinct X-only partition identifications use only 1s and 2s per level:")
    print(sorted(ids)[:8], "..." if ids.size > 8 else "")

    # %% [markdown]
    # ## Clusters Used in Regression
    #
    # The left column shows the partitions for orders 1 through 3; the right
    # column shows the corresponding NNS regression built from those clusters.
    subsection("Clusters used in regression")
    fig, axes = plt.subplots(3, 2, figsize=(11, 13))
    for order in range(1, 4):
        part = nns_part(x, y, order=order, obs_req=0, type="XONLY")
        partition_scatter(axes[order - 1][0], part, f"NNS.part order = {order}")
        reg = nns_reg(x, y, order=order)
        regression_scatter(axes[order - 1][1], x, y, reg, f"NNS.reg order = {order}")
    save_figure(fig, OUT, "03_clusters_used_in_regression.png")

    # %% [markdown]
    # # NNS Regression NNS.reg()
    #
    # ## Univariate
    subsection("Univariate NNS.reg")
    uni = nns_reg(x, y)
    fig, ax = plt.subplots(figsize=(8, 5))
    regression_scatter(ax, x, y, uni, "NNS.reg(x, y)")
    save_figure(fig, OUT, "04_univariate_regression.png")
    show("Univariate NNS.reg returned structure", uni)

    # %% [markdown]
    # ## Multivariate
    #
    # The R vignette fits f(x, y) = x^3 + 3y - y^3 - 3x over the full
    # expand.grid of x with itself using ``order = "max"``. The full 201-point
    # grid (40,401 rows) is preserved outside fast mode; CI uses a coarser grid
    # solely to reduce runtime.
    subsection("Multivariate NNS.reg")
    grid_axis = np.round(np.arange(-5.0, 5.0001, 0.25 if fast_mode() else 0.05), 10)
    grid = np.asarray(list(itertools.product(grid_axis, grid_axis)), dtype=float)
    g = grid[:, 0] ** 3 + 3.0 * grid[:, 1] - grid[:, 1] ** 3 - 3.0 * grid[:, 0]
    multi = nns_reg(grid, g, order="max")
    show("Multivariate NNS.reg returned structure", multi)
    show("Per-regressor partitions (rhs.partitions)", multi["rhs.partitions"])
    show("Regression point matrix (RPM)", multi["RPM"])

    fitted = np.asarray(multi["Fitted.xy"]["y.hat"], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(g, fitted, s=6, alpha=0.4)
    axes[0].plot([g.min(), g.max()], [g.min(), g.max()], linestyle="--", linewidth=1.5)
    axes[0].set_xlabel("observed g")
    axes[0].set_ylabel("fitted g")
    axes[0].set_title(f"Multivariate fit; R2 = {float(multi['R2']):.4f}")
    surface = axes[1].tricontourf(grid[:, 0], grid[:, 1], fitted, levels=20, cmap="viridis")
    fig.colorbar(surface, ax=axes[1])
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_title("Fitted surface for f(x, y) = x^3 + 3y - y^3 - 3x")
    save_figure(fig, OUT, "05_multivariate_regression.png")

    # %% [markdown]
    # ## Inter/Extrapolation
    #
    # ``point.est`` accepts any data of the regressors' dimension; estimates are
    # returned in ``$Point.est``.
    subsection("Inter/extrapolation with point.est")
    points = np.asarray([[-5.5, -5.5], [0.25, 0.25], [5.5, 5.5]], dtype=float)
    extrap = nns_reg(grid, g, order="max", point_est=points)
    f_true = points[:, 0] ** 3 + 3.0 * points[:, 1] - points[:, 1] ** 3 - 3.0 * points[:, 0]
    table(
        ["x", "y", "true f(x, y)", "NNS point estimate"],
        [
            (float(p[0]), float(p[1]), float(t), float(e))
            for p, t, e in zip(
                points, f_true, np.asarray(extrap["Point.est"], dtype=float), strict=True
            )
        ],
    )
    note(
        "The first row lies outside the training support, demonstrating "
        "extrapolation; the second interpolates between grid nodes."
    )

    # %% [markdown]
    # ## NNS Dimension Reduction Regression
    #
    # ``dim.red.method = "cor"`` reduces all regressors to a single dimension
    # using the returned equation.
    subsection("Dimension reduction regression")
    iris_x, iris_y, levels = load_iris()
    feature_names = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]
    dim_red = nns_reg(iris_x, iris_y, dim_red_method="cor", class_levels=levels)
    equation = dim_red["equation"]
    coefficients = np.asarray(equation["Coefficient"], dtype=float)
    show("Dimension reduction equation", equation)
    terms = " + ".join(
        f"{coefficients[i]:.3f}*{feature_names[i]}" for i in range(len(feature_names))
    )
    print(f"\nSpecies = ({terms}) / {coefficients[-1]:g}")

    # %% [markdown]
    # ### Threshold
    #
    # ``threshold = 0.75`` drops regressors whose absolute correlation falls
    # below the requested level, reducing the denominator accordingly.
    subsection("Dimension reduction with threshold")
    reduced = nns_reg(iris_x, iris_y, dim_red_method="cor", threshold=0.75, class_levels=levels)
    equation = reduced["equation"]
    coefficients = np.asarray(equation["Coefficient"], dtype=float)
    show("Thresholded dimension reduction equation", equation)
    terms = " + ".join(
        f"{coefficients[i]:.3f}*{feature_names[i]}" for i in range(len(feature_names))
    )
    print(f"\nSpecies = ({terms}) / {coefficients[-1]:g}")

    point_est = nns_reg(
        iris_x,
        iris_y,
        dim_red_method="cor",
        threshold=0.75,
        point_est=iris_x[:10],
        class_levels=levels,
    )
    show("Point estimates for iris rows 1 through 10", np.asarray(point_est["Point.est"]))

    # %% [markdown]
    # # Classification
    #
    # ``type = "CLASS"`` rounds estimates to class values. The base category of
    # the response must be 1, not 0.
    subsection("Classification")
    classified = nns_reg(iris_x, iris_y, type="CLASS", point_est=iris_x[:10], class_levels=levels)
    show(
        "Classification point estimates for iris rows 1 through 10",
        np.asarray(classified["Point.est"]),
    )
    note(
        "The response is encoded 1=setosa, 2=versicolor, 3=virginica, preserving "
        "the R base-1 contract."
    )

    # %% [markdown]
    # # Cross-Validation NNS.stack()
    #
    # NNS.stack cross-validates n.best for the multivariate regression and the
    # threshold for the dimension-reduced regression, then ensembles both. The
    # objective function is the classification rate, matching the R call
    # ``obj.fn = expression(mean(round(predicted) == actual))``.
    subsection("NNS.stack cross-validation")
    stack = nns_stack(
        iris_x,
        iris_y,
        iris_x[:10],
        type="CLASS",
        dim_red_method="cor",
        obj_fn=lambda predicted, actual: float(np.mean(np.round(predicted) == actual)),
        objective="max",
        folds=1,
        status=False,
        seed=123,
        class_levels=levels,
    )
    show("NNS.stack returned structure", stack)

    # %% [markdown]
    # # Increasing Dimensions
    #
    # Multicollinearity is not an issue for nonparametric regression, so an
    # ill-fit univariate model can duplicate its regressor, cross-validate
    # n.best, and regress in the doubled space.
    subsection("Increasing dimensions")
    rng = np.random.default_rng(123)
    x_rand = rng.normal(size=100)
    y_rand = rng.normal(size=100)
    doubled = np.column_stack((x_rand, x_rand))
    params = nns_stack(doubled, y_rand, method=1, folds=1, status=False, seed=123)
    n_best = params["NNS.reg.n.best"]
    show("Cross-validated parameters from NNS.stack(method = 1)", params)

    doubled_fit = nns_reg(
        doubled,
        y_rand,
        n_best=n_best,
        point_est=doubled,
        confidence_interval=0.95,
    )
    estimates = np.asarray(doubled_fit["Point.est"], dtype=float)
    residuals = y_rand - estimates
    pred_int = doubled_fit["pred.int"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    order_idx = np.argsort(x_rand)
    axes[0].scatter(x_rand, y_rand, s=16, alpha=0.6, label="observed")
    axes[0].plot(
        x_rand[order_idx], estimates[order_idx], linewidth=1.4, label="NNS fit (cbind(x, x))"
    )
    if isinstance(pred_int, dict) and pred_int:
        keys = list(pred_int)
        lower = np.asarray(pred_int[keys[0]], dtype=float)
        upper = np.asarray(pred_int[keys[-1]], dtype=float)
        axes[0].fill_between(
            x_rand[order_idx],
            lower[order_idx],
            upper[order_idx],
            alpha=0.2,
            label="95% interval",
        )
    axes[0].set_title(f"Duplicated regressors, n.best = {n_best:g}")
    axes[0].legend()
    axes[1].scatter(estimates, residuals, s=16, alpha=0.6)
    axes[1].axhline(0.0, linewidth=1.0, linestyle="--")
    axes[1].set_xlabel("fitted")
    axes[1].set_ylabel("residual")
    axes[1].set_title("Residual diagnostic (R residual.plot equivalent)")
    save_figure(fig, OUT, "06_increasing_dimensions.png")

    # %% [markdown]
    # # Smoothing Option
    #
    # Smoothness is not required, but ``smooth = TRUE`` applies a smoothing
    # spline to the internally generated regression points.
    subsection("Smoothing option")
    rough = nns_reg(x_rand, y_rand)
    smooth = nns_reg(x_rand, y_rand, smooth=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    regression_scatter(axes[0], x_rand, y_rand, rough, "NNS.reg(x, y)")
    regression_scatter(axes[1], x_rand, y_rand, smooth, "NNS.reg(x, y, smooth = TRUE)")
    save_figure(fig, OUT, "07_smoothing_option.png")
    show("Smoothed regression returned structure", smooth)

    # %% [markdown]
    # # Imputation
    #
    # Imputation is a direct application of nearest-neighbor regression: the
    # observed (X, y) pairs are the training set and the predictors of the
    # missing rows are point.est. With ``order = "max", n.best = 1`` each
    # missing y is filled by its closest donor, so imputations remain strictly
    # within the support of the observed data.

    # %% [markdown]
    # ## Univariate Imputation
    #
    # The increasing dimensions trick duplicates x into cbind(x, x) so the
    # distance function operates in a 2-D space, sharpening donor selection.
    subsection("Univariate imputation")
    rng = np.random.default_rng(123)
    n = 200 if fast_mode() else 400
    x_uni = np.sort(rng.uniform(-3.0, 3.0, n))
    y_uni = np.sin(x_uni) + 0.2 * x_uni**2 + rng.normal(0.0, 0.25, n)
    missing = rng.binomial(1, 0.25, n) == 1

    x2_train = np.column_stack((x_uni[~missing], x_uni[~missing]))
    x2_missing = np.column_stack((x_uni[missing], x_uni[missing]))
    imputed_uni = np.asarray(
        nns_reg(x2_train, y_uni[~missing], point_est=x2_missing, order="max", n_best=1)[
            "Point.est"
        ],
        dtype=float,
    )
    completed = y_uni.copy()
    completed[missing] = imputed_uni
    table(
        ["quantity", "value"],
        [
            ("missing rows", int(np.sum(missing))),
            (
                "imputation RMSE vs truth",
                float(np.sqrt(np.mean((imputed_uni - y_uni[missing]) ** 2))),
            ),
        ],
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        x_uni,
        y_uni,
        s=26,
        facecolors="none",
        edgecolors="steelblue",
        linewidths=1.4,
        label="Observed",
    )
    ax.scatter(
        x_uni[missing], imputed_uni, s=30, marker="s", color="red", label="Imputed (NNS 1-NN)"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("NNS 1-NN Imputation")
    ax.legend(loc="upper left", frameon=False)
    save_figure(fig, OUT, "08_univariate_imputation.png")

    # %% [markdown]
    # ## Multivariate Imputation
    #
    # The same form applies directly with the full predictor set: observed rows
    # train the model and incomplete rows are point.est.
    subsection("Multivariate imputation")
    rng = np.random.default_rng(123)
    n = 300 if fast_mode() else 800
    predictors = np.column_stack(
        (rng.normal(size=n), rng.uniform(-2.0, 2.0, n), rng.normal(0.0, 1.0, n))
    )

    def truth(p: np.ndarray) -> np.ndarray:
        return (
            1.1 * p[:, 0]
            - 0.8 * p[:, 1]
            + 0.5 * p[:, 2]
            + 0.6 * p[:, 0] * p[:, 1]
            - 0.4 * p[:, 1] * p[:, 2]
            + 0.3 * np.sin(1.3 * p[:, 0])
        )

    y_multi = truth(predictors) + rng.normal(0.0, 0.4, n)
    missing = rng.binomial(1, 0.30, n) == 1
    imputed_multi = np.asarray(
        nns_reg(
            predictors[~missing],
            y_multi[~missing],
            point_est=predictors[missing],
            order="max",
            n_best=1,
        )["Point.est"],
        dtype=float,
    )
    table(
        ["quantity", "value"],
        [
            ("missing rows", int(np.sum(missing))),
            (
                "imputation RMSE vs truth",
                float(np.sqrt(np.mean((imputed_multi - y_multi[missing]) ** 2))),
            ),
        ],
    )
    fig, ax = plt.subplots(figsize=(9, 5.5))
    index = np.arange(n)
    ax.scatter(
        index,
        y_multi,
        s=24,
        facecolors="none",
        edgecolors="steelblue",
        linewidths=1.2,
        label="Observed",
    )
    ax.scatter(
        index[missing], imputed_multi, s=26, marker="s", color="red", label="Imputed (NNS 1-NN)"
    )
    ax.set_xlabel("Observation index")
    ax.set_ylabel("y")
    ax.set_title("NNS 1-NN Multivariate Imputation")
    ax.legend(loc="upper left", frameon=False)
    save_figure(fig, OUT, "09_multivariate_imputation.png")
    note(
        "The R vignette closes with bootstrap multiple imputation pooled via "
        "Rubin's rules; that section is narrative in R (no executable chunk) and "
        "links to the NNS_MI_vs_MICE comparison, so it is summarized here rather "
        "than reinvented."
    )


if __name__ == "__main__":
    main()
