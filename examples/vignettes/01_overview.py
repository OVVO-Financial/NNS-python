"""01. Getting Started with NNS: Overview.

Instructional Python port of:
    NNS/vignettes/NNSvignette_01_Overview.Rmd

A complete hands-on curriculum for Nonlinear Nonparametric Statistics using
partial moments, preserving the R vignette's nine-section sequence. Figures are
written to ``examples/vignettes/output/01_overview``.
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
    load_mtcars,
    note,
    output_dir,
    save_figure,
    section,
    show,
    subsection,
    table,
)
from nns import (
    co_lpm,
    fsd_uni,
    lpm,
    lpm_ratio,
    lpm_var,
    nns_anova,
    nns_arma_optim,
    nns_boost,
    nns_causation,
    nns_copula,
    nns_dep,
    nns_mc,
    nns_meboot,
    nns_mode,
    nns_moments,
    nns_norm,
    nns_reg,
    nns_rescale,
    nns_sd_cluster,
    nns_seas,
    nns_ss,
    nns_stack,
    pm_matrix,
    sd_efficient_set,
    ssd_uni,
    tsd_uni,
    upm,
)

OUT = output_dir(__file__)


def main() -> None:
    section("Orientation")
    print(
        "Goal: a complete curriculum for Nonlinear Nonparametric Statistics (NNS) "
        "using partial moments. LPM(n, t, X) integrates (t - x)^n below the target "
        "t; UPM(n, t, X) integrates (x - t)^n above it. The empirical estimators "
        "replace the population CDF with the empirical one."
    )

    # %% [markdown]
    # # 1. Foundations — Partial Moments & Variance Decomposition
    #
    # Classical variance treats upside and downside symmetrically. Partial
    # moments separate them around a global target, so at t = mean(x):
    # Var(X) = UPM(2, mu, X) + LPM(2, mu, X) (with Bessel adjustment).
    section("1. Foundations — Partial Moments & Variance Decomposition")
    rng = np.random.default_rng(42)
    y = rng.normal(size=3000)
    mu = float(np.mean(y))
    n = y.size
    lower2 = float(lpm(2, mu, y))
    upper2 = float(upm(2, mu, y))
    decomposed = (lower2 + upper2) * (n / (n - 1))
    table(
        ["quantity", "value"],
        [
            ("LPM(2, mean, y)", lower2),
            ("UPM(2, mean, y)", upper2),
            ("(LPM2 + UPM2) * n/(n-1)", decomposed),
            ("var(y) sample", float(np.var(y, ddof=1))),
        ],
    )

    print("\nEmpirical CDF via LPM.ratio(0, t, y):")
    table(
        ["target", "LPM.ratio", "empirical"],
        [(t, float(lpm_ratio(0, t, y)), float(np.mean(y <= t))) for t in (-1.0, 0.0, 1.0)],
    )

    z = rng.exponential(size=3000) - 1.0
    mu_z = float(np.mean(z))
    print("\nAsymmetry on a skewed distribution (expect imbalance):")
    table(
        ["quantity", "value"],
        [("LPM(2, mean, z)", float(lpm(2, mu_z, z))), ("UPM(2, mean, z)", float(upm(2, mu_z, z)))],
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, (values, target, name) in zip(
        axes,
        (
            (y, mu, "N(0,1): symmetric partial moments"),
            (z, mu_z, "exp(1) - 1: asymmetric partial moments"),
        ),
        strict=True,
    ):
        _, edges, patches = ax.hist(values, bins=40, alpha=0.8)
        for edge, patch in zip(edges[:-1], patches, strict=False):
            patch.set_facecolor("steelblue" if edge < target else "darkorange")
        ax.axvline(target, color="black", linewidth=1.6, label="target = mean")
        ax.set_title(name)
        ax.legend()
    save_figure(fig, OUT, "01_partial_moment_decomposition.png")
    print(
        "The equality holds because deviations are measured against the global "
        "mean; LPM.ratio(0, t, x) constructs an empirical CDF from partial-moment counts."
    )

    # %% [markdown]
    # # 2. Descriptive & Distributional Tools
    section("2. Descriptive & Distributional Tools")
    subsection("Higher moments from partial moments: NNS.moments")
    show("NNS.moments(y)", nns_moments(y))

    subsection("Mode estimation: NNS.mode")
    rng_mode = np.random.default_rng(23)
    multimodal = np.concatenate(
        (rng_mode.normal(-2.0, 0.5, 1500), rng_mode.normal(2.0, 0.5, 1500))
    )
    modes = np.atleast_1d(np.asarray(nns_mode(multimodal, multi=True), dtype=float))
    show("NNS.mode(multimodal, multi = TRUE)", modes)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.hist(multimodal, bins=60, alpha=0.75)
    for mode_value in modes:
        ax.axvline(mode_value, color="red", linewidth=2)
    ax.set_title("Multimodal sample with NNS.mode(multi = TRUE) estimates")
    save_figure(fig, OUT, "02_mode_estimation.png")

    subsection("CDF tables via LPM ratios")
    quantile_probs = np.arange(0.05, 0.951, 0.1)
    qgrid = np.asarray([lpm_var(p, 0, z) for p in quantile_probs], dtype=float)
    table(
        ["threshold (LPM.VaR)", "CDF (LPM.ratio)"],
        [(float(q), float(lpm_ratio(0, q, z))) for q in qgrid],
    )

    # %% [markdown]
    # # 3. Dependence & Nonlinear Association
    #
    # Pearson r captures linear monotone relationships. U-shapes, saturation and
    # asymmetric tails can produce near-zero r despite strong dependence.
    section("3. Dependence & Nonlinear Association")
    rng_dep = np.random.default_rng(1)
    x_dep = rng_dep.uniform(-1.0, 1.0, 2000)
    y_dep = x_dep**2 + rng_dep.normal(scale=0.05, size=2000)
    pearson = float(np.corrcoef(x_dep, y_dep)[0, 1])
    dependence = float(nns_dep(x_dep, y_dep)["Dependence"])
    table(["measure", "value"], [("Pearson r", pearson), ("NNS.dep", dependence)])

    frame = np.column_stack((x_dep, y_dep, x_dep * y_dep + rng_dep.normal(scale=0.05, size=2000)))
    pm = pm_matrix(1, 1, "means", frame, True, names=["a", "b", "c"])
    show("PM.matrix(1, 1, target = 'means')", pm)
    show("NNS.copula(X, continuous = TRUE)", float(nns_copula(frame, continuous=True)))

    subsection("Copula visualization")
    gap(
        "R renders the Co.LPM copula surfaces interactively with rgl::plot3d. The "
        "same surfaces are saved here as static Matplotlib 3-D figures."
    )
    rng_cop = np.random.default_rng(123)
    grid_n = 40 if fast_mode() else 60
    x_cop = rng_cop.normal(size=100)
    y_cop = rng_cop.normal(size=100)
    grid = np.asarray(list(itertools.product(x_cop[:grid_n], y_cop[:grid_n])), dtype=float)
    raw_surface = np.asarray(co_lpm(0, x_cop, y_cop, grid[:, 0], grid[:, 1]), dtype=float)
    u_x = np.asarray(lpm_ratio(0, x_cop, x_cop), dtype=float)
    u_y = np.asarray(lpm_ratio(0, y_cop, y_cop), dtype=float)
    u_grid = np.asarray(list(itertools.product(u_x[:grid_n], u_y[:grid_n])), dtype=float)
    uniform_surface = np.asarray(co_lpm(0, u_x, u_y, u_grid[:, 0], u_grid[:, 1]), dtype=float)
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.scatter(grid[:, 0], grid[:, 1], raw_surface, c="red", s=6, alpha=0.6)
    ax.set_title("Co.LPM surface on raw margins")
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.scatter(u_grid[:, 0], u_grid[:, 1], uniform_surface, c="blue", s=6, alpha=0.6)
    ax.set_title("Co.LPM surface on uniform margins")
    save_figure(fig, OUT, "03_copula_surfaces.png")

    # %% [markdown]
    # # 4. Normalization and Rescaling
    section("4. Normalization and Rescaling")
    subsection("NNS.norm")
    rng_norm = np.random.default_rng(42)
    panel = np.column_stack(
        (
            rng_norm.normal(0.0, 1.0, 100),
            rng_norm.normal(0.0, 5.0, 100),
            rng_norm.normal(10.0, 1.0, 100),
            rng_norm.normal(10.0, 10.0, 100),
        )
    )
    linear_normalized = np.asarray(nns_norm(panel, linear=True), dtype=float)
    show("First rows of NNS.norm(X, linear = TRUE)", linear_normalized[:6])
    print(
        "Linear mode equalizes means; nonlinear mode weights each variable by its "
        "dependence with the others."
    )

    subsection("Risk-neutral rescale (pricing context)")
    prices = 100.0 + np.cumsum(rng_norm.normal(0.0, 1.0, 260))
    risk_neutral = nns_rescale(
        prices, a=100.0, b=0.03, method="riskneutral", time_to_maturity=1.0, type="Terminal"
    )
    table(
        ["quantity", "value"],
        [
            ("target 100 * exp(0.03)", 100.0 * float(np.exp(0.03))),
            ("mean of rescaled series", float(np.mean(risk_neutral))),
        ],
    )
    gap("Python spells the R argument T as time_to_maturity; the operation is identical.")

    # %% [markdown]
    # # 5. Hypothesis Testing, ANOVA & Stochastic Superiority
    #
    # Instead of distributional assumptions, groups are compared through
    # LPM-based CDFs; the output is a degree of certainty, not a p-value.
    section("5. Hypothesis Testing, ANOVA & Stochastic Superiority")
    subsection("Two-sample and multi-group NNS.ANOVA")
    rng_test = np.random.default_rng(42)
    control = rng_test.normal(0.0, 1.0, 200)
    treatment = rng_test.normal(0.35, 1.2, 180)
    show("NNS.ANOVA(control, treatment)", nns_anova(control, treatment, random_seed=123))

    groups = [
        rng_test.normal(0.0, 1.1, 150),
        rng_test.normal(0.2, 1.0, 150),
        rng_test.normal(-0.1, 0.9, 150),
    ]
    show("Multi-group NNS.ANOVA(control = list(g1, g2, g3), means.only = TRUE)",
         nns_anova(groups, means_only=True, random_seed=123))

    subsection("Stochastic superiority NNS.SS")
    rng_ss = np.random.default_rng(123)
    x_ss = rng_ss.normal(0.0, 1.0, 1000)
    y_ss = rng_ss.normal(1.0, 1.0, 1000)
    show("NNS.SS(x, y)", nns_ss(x_ss, y_ss))
    print(
        "y has the higher mean, so P* is below 0.5: a draw from x is less likely "
        "to exceed a draw from y. P* = P(X > Y) + 0.5 * P(X = Y)."
    )
    with_interval = nns_ss(
        x_ss,
        y_ss,
        confidence_interval=True,
        reps=99 if fast_mode() else 999,
        ci=0.95,
        random_seed=123,
    )
    show(
        "NNS.SS(x, y, confidence.interval = TRUE)",
        {
            key: with_interval[key]
            for key in ("p_gt", "p_tie", "p_star", "lower", "upper")
            if key in with_interval
        },
    )
    x_discrete = rng_ss.integers(1, 6, 100).astype(float)
    y_discrete = rng_ss.integers(1, 6, 100).astype(float)
    show("NNS.SS on discrete samples (explicit tie adjustment)", nns_ss(x_discrete, y_discrete))

    # %% [markdown]
    # # 6. Regression, Boosting, Stacking & Causality
    #
    # NNS.reg learns partitioned relationships using partial-moment weights —
    # linear where appropriate, nonlinear where needed.
    section("6. Regression, Boosting, Stacking & Causality")
    subsection("Nonlinear regression")
    rng_reg = np.random.default_rng(123)
    x_train = rng_reg.uniform(-2.0, 2.0, 1000)
    y_train = np.sin(np.pi * x_train) + rng_reg.normal(scale=0.2, size=1000)
    x_test = np.linspace(-2.0, 2.0, 100)
    regression = nns_reg(x_train, y_train, point_est=x_test)
    show("NNS.reg returned structure", regression)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.scatter(x_train, y_train, s=8, alpha=0.35, label="training data")
    ax.plot(
        x_test,
        np.asarray(regression["Point.est"], dtype=float),
        color="red",
        linewidth=2,
        label="NNS point estimates",
    )
    ax.plot(
        x_test,
        np.sin(np.pi * x_test),
        color="black",
        linestyle="--",
        linewidth=1.4,
        label="true sin(pi x)",
    )
    ax.set_title("NNS.reg on sin(pi x) with noise")
    ax.legend()
    save_figure(fig, OUT, "04_nonlinear_regression.png")

    subsection("Classification via boosting and stacking")
    iris_x, iris_y, levels = load_iris()
    test_idx = np.arange(140, 150)
    train_idx = np.arange(0, 140)
    boost = nns_boost(
        iris_x[train_idx],
        iris_y[train_idx],
        iris_x[test_idx],
        type="CLASS",
        epochs=3 if fast_mode() else 10,
        learner_trials=4 if fast_mode() else 10,
        status=False,
        balance=True,
        seed=123,
        class_levels=levels,
    )
    boost_accuracy = float(np.mean(np.asarray(boost["results"], dtype=float) == iris_y[test_idx]))
    show("NNS.boost feature weights", boost["feature.weights"])
    show("NNS.boost feature frequency", boost["feature.frequency"])

    stacked = nns_stack(
        iris_x[train_idx],
        iris_y[train_idx],
        iris_x[test_idx],
        type="CLASS",
        balance=True,
        folds=1 if fast_mode() else 2,
        status=False,
        seed=123,
        class_levels=levels,
    )
    stack_accuracy = float(np.mean(np.asarray(stacked["stack"], dtype=float) == iris_y[test_idx]))
    table(
        ["ensemble", "holdout accuracy (iris rows 141-150)"],
        [("NNS.boost", boost_accuracy), ("NNS.stack", stack_accuracy)],
    )

    subsection("Directional causality")
    mtcars = load_mtcars()
    hp_to_mpg = nns_causation(mtcars["hp"], mtcars["mpg"])
    mpg_to_hp = nns_causation(mtcars["mpg"], mtcars["hp"])
    show("NNS.caus(mtcars$hp, mtcars$mpg)", hp_to_mpg)
    show("NNS.caus(mtcars$mpg, mtcars$hp)", mpg_to_hp)
    print("Asymmetry in the conditional-dependence scores indicates direction.")

    # %% [markdown]
    # # 7. Time Series & Forecasting
    #
    # NNS seasonality uses the coefficient of variation instead of ACF/PACF, and
    # NNS.ARMA blends multiple seasonal periods into linear or nonlinear
    # regression forecasts.
    section("7. Time Series & Forecasting")
    rng_ts = np.random.default_rng(42)
    length = 240 if fast_mode() else 480
    horizon = 24 if fast_mode() else 48
    raw_series = np.sin(np.arange(1, length + 1) / 8.0) + rng_ts.normal(scale=0.35, size=length)
    series = (raw_series - raw_series.mean()) / raw_series.std(ddof=1)

    seasonal = nns_seas(series, plot=False)
    show(
        "NNS.seas head of all.periods",
        np.asarray(seasonal["all.periods"]["Period"], dtype=float)[:6],
    )
    periods = np.asarray(seasonal["periods"], dtype=int).ravel()
    train_n = series.size - horizon
    usable = [int(p) for p in periods if p < train_n / 4][: 2 if fast_mode() else 3]
    print("Seasonal periods validated by NNS.ARMA.optim:", usable)

    optimized = nns_arma_optim(
        series,
        h=horizon,
        seasonal_factor=usable,
        print_trace=False,
        plot=False,
    )
    show("NNS.ARMA.optim returned structure", optimized)
    forecast = np.asarray(optimized["results"], dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(series.size), series, linewidth=1.2, label="observed (scaled)")
    horizon_index = np.arange(series.size, series.size + forecast.size)
    ax.plot(horizon_index, forecast, color="red", linewidth=2, label="NNS.ARMA.optim forecast")
    lower = optimized.get("lower.pred.int")
    upper = optimized.get("upper.pred.int")
    if lower is not None and upper is not None:
        ax.fill_between(
            horizon_index,
            np.asarray(lower, dtype=float),
            np.asarray(upper, dtype=float),
            alpha=0.25,
            label="prediction interval",
        )
    ax.set_title(f"NNS.ARMA.optim forecast, h = {horizon}")
    ax.legend()
    save_figure(fig, OUT, "05_arma_optim_forecast.png")
    note(
        "R passes every detected period straight to NNS.ARMA.optim; the Python "
        "optimizer validates seasonal factors against the training span, so the "
        "detected periods are filtered to the supported range first."
    )

    # %% [markdown]
    # # 8. Simulation & Bootstrap & Risk-Neutral Rescaling
    section("8. Simulation & Bootstrap")
    subsection("Maximum entropy bootstrap NNS.meboot")
    x_ts = np.cumsum(rng_ts.normal(scale=0.7, size=350))
    meboot = nns_meboot(x_ts, reps=5, rho=1.0, random_seed=123)
    replicates = np.asarray(meboot["replicates"], dtype=float)
    show("dim(NNS.meboot replicates)", np.asarray(replicates.shape))

    subsection("Monte Carlo over the full correlation space NNS.MC")
    mc = nns_mc(x_ts, reps=5, lower_rho=-1.0, upper_rho=1.0, by=0.5, exp=1.0, random_seed=123)
    print("length(mc$ensemble):", np.asarray(mc["ensemble"]).size)
    print("names(mc$replicates):", list(mc["replicates"]))
    show("head(mc$replicates$'rho = 0')", np.asarray(mc["replicates"]["rho = 0"], dtype=float)[:6])

    fig, ax = plt.subplots(figsize=(9.5, 5))
    colors = plt.get_cmap("rainbow")(np.linspace(0.0, 1.0, len(mc["replicates"])))
    for (name, reps), color in zip(mc["replicates"].items(), colors, strict=True):
        ax.plot(np.asarray(reps, dtype=float)[:, 0], color=color, linewidth=1.0, label=name)
    ax.plot(x_ts, color="black", linewidth=2.5, label="original")
    ax.set_title("NNS.MC replicates across the correlation space")
    ax.legend(fontsize=8)
    save_figure(fig, OUT, "06_monte_carlo_replicates.png")

    # %% [markdown]
    # # 9. Portfolio & Stochastic Dominance
    #
    # Stochastic dominance orders uncertain prospects for broad classes of
    # risk-averse utilities; partial moments supply nonparametric estimators.
    section("9. Portfolio & Stochastic Dominance")
    rng_sd = np.random.default_rng(42)
    ra = rng_sd.normal(0.005, 0.03, 240)
    rb = rng_sd.normal(0.003, 0.02, 240)
    rc = rng_sd.normal(0.006, 0.04, 240)
    table(
        ["test", "result (1 = dominance)"],
        [
            ("NNS.FSD.uni(RA, RB)", fsd_uni(ra, rb)),
            ("NNS.SSD.uni(RA, RB)", ssd_uni(ra, rb)),
            ("NNS.TSD.uni(RA, RB)", tsd_uni(ra, rb)),
        ],
    )
    returns_matrix = np.column_stack((ra, rb, rc))
    names = ["A", "B", "C"]
    clusters = nns_sd_cluster(returns_matrix, degree=1, names=names)
    show("NNS.SD.cluster(Rmat, degree = 1)", clusters["Clusters"])
    efficient = sd_efficient_set(returns_matrix, degree=1)
    show(
        "NNS.SD.efficient.set(Rmat, degree = 1)",
        [names[i - 1] for i in np.asarray(efficient, dtype=int)],
    )

    section("Appendix")
    print(
        "The measure-theoretic sketch and the grouped quick reference in the R "
        "vignette are narrative; each linked topic is executable in the numbered "
        "vignettes 02 through 09 of this catalog."
    )


if __name__ == "__main__":
    main()
