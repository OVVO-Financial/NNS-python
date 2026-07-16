"""08. Getting Started with NNS: Classification.

Instructional Python port of:
    NNS/vignettes/NNSvignette_08_Classification.Rmd

The examples use the exact R ``iris`` observations and preserve the vignette's
order: classification geometry, partitions, boosting, stacking, and parameter
interpretation. Figures are written to
``examples/vignettes/output/08_classification``.
"""
from __future__ import annotations

# %% [markdown]
# # Classification
#
# NNS classification uses the same partition-and-regression mechanism as the
# continuous model, with the response encoded as classes beginning at 1. The R
# vignette emphasizes that these are multidimensional partitions rather than a
# sequence of one-variable tree splits.

import matplotlib.pyplot as plt
import numpy as np

from examples._vignette_support import (
    fast_mode,
    gap,
    load_iris,
    note,
    output_dir,
    save_figure,
    section,
    show,
    subsection,
    table,
)
from nns import nns_boost, nns_part, nns_reg, nns_stack

OUT = output_dir(__file__)


def accuracy(predicted: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.asarray(predicted, dtype=float) == np.asarray(actual, dtype=float)))


def main() -> None:
    iris_x, iris_y, levels = load_iris()
    feature_names = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]
    test_idx = np.arange(140, 150)
    train_idx = np.arange(0, 140)
    x_train, y_train = iris_x[train_idx], iris_y[train_idx]
    x_test, y_test = iris_x[test_idx], iris_y[test_idx]

    section("Classification")
    note(
        "The response is encoded 1=setosa, 2=versicolor, 3=virginica. "
        "This preserves the R contract that the base category is 1, not 0."
    )
    table(
        ["code", "class", "training rows", "test rows"],
        [
            (code, label, int(np.sum(y_train == code)), int(np.sum(y_test == code)))
            for code, label in enumerate(levels, start=1)
        ],
    )

    # %% [markdown]
    # ## Splits versus partitions
    #
    # A tree asks one split question at a time. NNS partitions the joint support
    # through partial-moment quadrants, then uses the resulting regional centers
    # as regression points. The figure below shows the first two iris predictors
    # only so that the otherwise four-dimensional geometry can be inspected.
    subsection("Splits versus partitions")
    pair = nns_part(iris_x[:, 0], iris_x[:, 1], order=4, obs_req=0)
    labels = np.asarray(pair["dt"]["quadrant"], dtype=str)
    _, codes = np.unique(labels, return_inverse=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(iris_x[:, 0], iris_x[:, 1], c=codes, cmap="tab20", s=28, alpha=0.75)
    rp = pair["regression.points"]
    ax.scatter(rp["x"], rp["y"], marker="x", s=70, linewidths=2, label="partition centers")
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title("NNS joint partitions on two iris dimensions")
    ax.legend()
    save_figure(fig, OUT, "01_iris_joint_partitions.png")
    show("Two-dimensional partition result", pair)

    # %% [markdown]
    # ## NNS partitions used by the classifier
    #
    # The R vignette prints ``NNS.reg(... )$rhs.partitions`` for all four
    # predictors. Python currently returns the fitted reduction and equation but
    # not the internal per-predictor partition list. We therefore expose the
    # closest public diagnostic explicitly instead of pretending the object is
    # present.
    subsection("NNS partitions")
    base_fit = nns_reg(
        iris_x,
        iris_y,
        type="CLASS",
        point_est=iris_x[:10],
        residual_plot=False,
        class_levels=levels,
    )
    show("Classification regression result", base_fit)
    gap(
        "R exposes NNS.reg(... )$rhs.partitions for every predictor. The Python "
        "public result does not yet expose that internal list. The two-dimensional "
        "NNS.part result above is the supported public partition diagnostic."
    )

    # %% [markdown]
    # ## NNS.boost()
    #
    # This is the exact R holdout: rows 141--150 of iris. The same 10 epochs,
    # 10 learner trials, class balancing, and seed are used by default.
    subsection("NNS.boost")
    boost = nns_boost(
        x_train,
        y_train,
        x_test,
        type="CLASS",
        epochs=3 if fast_mode() else 10,
        learner_trials=4 if fast_mode() else 10,
        balance=True,
        status=False,
        seed=123,
        class_levels=levels,
    )
    boost_pred = np.asarray(boost["results"], dtype=float)
    show("NNS.boost result", boost)
    print(f"Boost holdout accuracy: {accuracy(boost_pred, y_test):.4f}")

    weights = boost["feature.weights"]
    frequencies = boost["feature.frequency"]
    weight_values = np.asarray(list(weights.values()), dtype=float)
    frequency_values = np.asarray(list(frequencies.values()), dtype=float)
    labels_for_plot = list(weights.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(np.arange(len(weight_values)), weight_values)
    axes[0].set_xticks(np.arange(len(weight_values)), labels_for_plot, rotation=35, ha="right")
    axes[0].set_title("Boost feature weights")
    axes[1].bar(np.arange(len(frequency_values)), frequency_values)
    axes[1].set_xticks(np.arange(len(frequency_values)), list(frequencies.keys()), rotation=35, ha="right")
    axes[1].set_title("Boost feature frequency")
    save_figure(fig, OUT, "02_boost_feature_diagnostics.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    positions = np.arange(y_test.size)
    ax.plot(positions, y_test, marker="o", label="actual class")
    ax.plot(positions, boost_pred, marker="x", linestyle="--", label="boost prediction")
    ax.set_yticks([1, 2, 3], levels)
    ax.set_xlabel("Held-out iris observation")
    ax.set_title("NNS.boost: exact R vignette holdout")
    ax.legend()
    save_figure(fig, OUT, "03_boost_holdout_predictions.png")
    note(
        "Balanced boosting is stochastic. Python reproduces R's random stream on "
        "deterministic and simple sampling paths, but the interleaved class-balance "
        "draw order is a documented remaining bit-for-bit RNG gap. The statistical "
        "procedure and returned diagnostics are the same."
    )

    # %% [markdown]
    # ## Cross-validation classification using NNS.stack()
    #
    # The R vignette next combines the full and dimension-reduced classifiers by
    # cross-validating ``n.best``, the reduction threshold, and the ensemble
    # weights. The default here preserves its five-fold specification; CI fast
    # mode uses one fold solely to reduce runtime.
    subsection("Cross-validation classification using NNS.stack")
    stack = nns_stack(
        x_train,
        y_train,
        x_test,
        type="CLASS",
        balance=True,
        folds=1 if fast_mode() else 5,
        method=(1, 2),
        status=False,
        seed=123,
    )
    show("NNS.stack result", stack)
    stack_pred = np.asarray(stack["stack"], dtype=float)
    reg_pred = np.asarray(stack["reg"], dtype=float)
    dim_pred = np.asarray(stack["dim.red"], dtype=float)
    table(
        ["component", "accuracy"],
        [
            ("full regression", accuracy(reg_pred, y_test)),
            ("dimension reduction", accuracy(dim_pred, y_test)),
            ("stack", accuracy(stack_pred, y_test)),
        ],
    )

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(positions, y_test, marker="o", linewidth=2, label="actual")
    ax.plot(positions, reg_pred, marker="x", linestyle="--", label="reg")
    ax.plot(positions, dim_pred, marker="s", linestyle=":", label="dim.red")
    ax.plot(positions, stack_pred, marker="d", linestyle="-.", label="stack")
    ax.set_yticks([1, 2, 3], levels)
    ax.set_xlabel("Held-out iris observation")
    ax.set_title("Cross-validated NNS classification components")
    ax.legend(ncol=4)
    save_figure(fig, OUT, "04_stack_holdout_predictions.png")

    # %% [markdown]
    # ## Depth, nearest-neighbor classification, and extremes
    #
    # The R vignette closes by explaining three controls: maximum partition depth,
    # ``n.best = 1`` for a donor/nearest-neighbor classifier, and ``extreme`` in
    # boosting. Each supported path is demonstrated directly.
    subsection("Depth, n.best = 1, and extreme boosting")
    nearest = nns_reg(
        x_train,
        y_train,
        type="CLASS",
        order="max",
        n_best=1,
        point_est=x_test,
        class_levels=levels,
    )
    nearest_pred = np.asarray(nearest["Point.est"], dtype=float)
    extreme = nns_boost(
        x_train,
        y_train,
        x_test,
        type="CLASS",
        depth="max",
        learner_trials=3 if fast_mode() else 10,
        epochs=2 if fast_mode() else 10,
        balance=True,
        extreme=True,
        status=False,
        seed=123,
        class_levels=levels,
    )
    extreme_pred = np.asarray(extreme["results"], dtype=float)
    table(
        ["model", "accuracy", "predictions"],
        [
            ("order=max, n.best=1", accuracy(nearest_pred, y_test), nearest_pred.astype(int)),
            ("boost extreme=True", accuracy(extreme_pred, y_test), extreme_pred.astype(int)),
        ],
    )
    show("Nearest-neighbor classification result", nearest)
    show("Extreme boost result", extreme)


if __name__ == "__main__":
    main()
