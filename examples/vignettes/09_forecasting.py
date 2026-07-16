"""09. Getting Started with NNS: Forecasting.

Instructional Python port of:
    NNS/vignettes/NNSvignette_09_Forecasting.Rmd

The exact R ``AirPassengers`` series, training window, holdout, seasonal-factor
search, and optimizer sequence are preserved. Figures are written to
``examples/vignettes/output/09_forecasting``.
"""
from __future__ import annotations

# %% [markdown]
# # Forecasting
#
# NNS forecasting treats a time series as a nonlinear regression problem on
# lagged seasonal components. The R vignette moves from a fixed seasonal factor
# to validation-based selection, automatic seasonality, and finally
# ``NNS.ARMA.optim``.

import matplotlib.pyplot as plt
import numpy as np

from examples._vignette_support import (
    fast_mode,
    gap,
    load_air_passengers,
    note,
    output_dir,
    save_figure,
    section,
    show,
    subsection,
    table,
)
from nns import nns_arma, nns_arma_optim, nns_seas

OUT = output_dir(__file__)


def rmse(predicted: np.ndarray, actual: np.ndarray) -> float:
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))


def estimates(result: object) -> np.ndarray:
    if isinstance(result, dict):
        for key in ("Estimates", "results"):
            if key in result:
                return np.asarray(result[key], dtype=float)
        raise KeyError("forecast dictionary has no Estimates/results field")
    return np.asarray(result, dtype=float)


def forecast_figure(
    history: np.ndarray,
    actual: np.ndarray | None,
    predicted: np.ndarray,
    title: str,
    filename: str,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x_hist = np.arange(history.size)
    x_fc = np.arange(history.size, history.size + predicted.size)
    ax.plot(x_hist, history, label="history")
    if actual is not None:
        ax.plot(x_fc[: actual.size], actual, label="actual holdout")
    ax.plot(x_fc, predicted, label="NNS forecast", linewidth=2)
    if lower is not None and upper is not None:
        ax.fill_between(x_fc, lower, upper, alpha=0.2, label="prediction interval")
    ax.set_xlabel("Month index")
    ax.set_ylabel("Passengers")
    ax.set_title(title)
    ax.legend()
    save_figure(fig, OUT, filename)


def main() -> None:
    air = load_air_passengers()
    training_set = 100
    horizon = air.size - training_set
    training = air[:training_set]
    actual = air[training_set:]

    section("Forecasting")
    print(f"AirPassengers observations: {air.size}")
    print(f"Training observations: {training_set}; validation observations: {horizon}")

    # %% [markdown]
    # ## Linear regression
    #
    # The R vignette begins with a 12-month seasonal factor and a 44-month
    # validation horizon. ``method='lin'`` regresses each seasonal component
    # linearly before recombining the forecasts.
    subsection("Linear regression")
    linear_raw = nns_arma(
        air,
        h=horizon,
        training_set=training_set,
        method="lin",
        seasonal_factor=12,
        plot=False,
        seasonal_plot=False,
    )
    linear = estimates(linear_raw)
    linear_rmse = rmse(linear, actual)
    print(f"Linear seasonal-factor-12 RMSE: {linear_rmse:.6f}")
    show("Linear forecast", linear_raw)
    forecast_figure(training, actual, linear, "Linear NNS.ARMA, seasonal factor 12", "01_linear.png")

    # %% [markdown]
    # ## Nonlinear regression
    #
    # The same data and seasonal factor are now passed through the nonlinear NNS
    # regression engine. The comparison isolates the regression form; nothing
    # else about the experiment changes.
    subsection("Nonlinear regression")
    nonlinear_raw = nns_arma(
        air,
        h=horizon,
        training_set=training_set,
        method="nonlin",
        seasonal_factor=12,
        plot=False,
        seasonal_plot=False,
    )
    nonlinear = estimates(nonlinear_raw)
    nonlinear_rmse = rmse(nonlinear, actual)
    print(f"Nonlinear seasonal-factor-12 RMSE: {nonlinear_rmse:.6f}")
    show("Nonlinear forecast", nonlinear_raw)
    forecast_figure(
        training,
        actual,
        nonlinear,
        "Nonlinear NNS.ARMA, seasonal factor 12",
        "02_nonlinear.png",
    )
    table(
        ["method", "seasonal factor", "RMSE"],
        [("linear", 12, linear_rmse), ("nonlinear", 12, nonlinear_rmse)],
    )

    # %% [markdown]
    # ## Cross-validation of the seasonal factor
    #
    # Following the R vignette, every period from 1 through 25 is evaluated on
    # the same 44-observation holdout. CI fast mode uses a reduced grid only for
    # runtime; normal execution preserves the complete search.
    subsection("Cross-validation")
    periods = np.arange(1, 26) if not fast_mode() else np.array([1, 6, 12, 18, 24])
    rows: list[tuple[int, float]] = []
    forecasts: dict[int, np.ndarray] = {}
    for period in periods:
        raw = nns_arma(
            air,
            h=horizon,
            training_set=training_set,
            method="nonlin",
            seasonal_factor=int(period),
            plot=False,
            seasonal_plot=False,
        )
        pred = estimates(raw)
        forecasts[int(period)] = pred
        rows.append((int(period), rmse(pred, actual)))
    table(["Period", "RMSE"], rows)
    best_period, best_rmse = min(rows, key=lambda item: item[1])
    print(f"Best validated period: {best_period}; RMSE={best_rmse:.6f}")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot([row[0] for row in rows], [row[1] for row in rows], marker="o")
    ax.axvline(best_period, linestyle="--", label=f"best={best_period}")
    ax.set_xlabel("Seasonal factor")
    ax.set_ylabel("Validation RMSE")
    ax.set_title("Cross-validation of NNS.ARMA seasonal factor")
    ax.legend()
    save_figure(fig, OUT, "03_period_cross_validation.png")
    forecast_figure(
        training,
        actual,
        forecasts[best_period],
        f"Best nonlinear validation forecast, period {best_period}",
        "04_best_period_forecast.png",
    )

    # %% [markdown]
    # ## Automatic seasonality detection
    #
    # ``NNS.seas`` tests candidate component periods against the variability of
    # the original series. The R vignette constrains candidates around a modulo
    # of 12 and displays the ranked period table.
    subsection("NNS.seas")
    seasonality = nns_seas(air, modulo=12, mod_only=False, plot=False)
    show("NNS.seas result", seasonality)
    all_periods = seasonality["all.periods"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.scatter(all_periods["Period"], all_periods["Coefficient.of.Variation"], label="component CV")
    ax.axhline(
        float(all_periods["Variable.Coefficient.of.Variation"][0]),
        linestyle="--",
        label="series CV",
    )
    ax.set_xlabel("Period")
    ax.set_ylabel("Coefficient of variation")
    ax.set_title("NNS seasonality diagnostics")
    ax.legend()
    save_figure(fig, OUT, "05_seasonality.png")

    # %% [markdown]
    # ## Cross-validating combinations with NNS.ARMA.optim
    #
    # The optimizer compares linear, nonlinear, and combined forecasts over
    # candidate seasonal-factor combinations. The exact R candidate grid is
    # ``12, 18, ..., 60``; fast mode reduces it solely for CI.
    subsection("NNS.ARMA.optim")
    candidate_periods = list(range(12, 61, 6)) if not fast_mode() else [12, 18, 24]
    optim = nns_arma_optim(
        air,
        training_set=training_set,
        seasonal_factor=candidate_periods,
        obj_fn=rmse,
        objective="min",
        pred_int=0.95,
        print_trace=not fast_mode(),
        plot=False,
    )
    show("NNS.ARMA.optim validation result", optim)
    optim_pred = np.asarray(optim["results"], dtype=float)
    print(f"Optimizer holdout RMSE: {rmse(optim_pred, actual):.6f}")
    forecast_figure(
        training,
        actual,
        optim_pred,
        "NNS.ARMA.optim validation forecast",
        "06_arma_optim_validation.png",
        np.asarray(optim["lower.pred.int"], dtype=float),
        np.asarray(optim["upper.pred.int"], dtype=float),
    )

    # %% [markdown]
    # ## Extension estimates
    #
    # Once the validation specification is selected, the R vignette refits on the
    # complete series and extends it 50 months. Prediction intervals are retained
    # and displayed rather than reduced to a printed forecast vector.
    subsection("Extension estimates")
    extension_h = 12 if fast_mode() else 50
    extension = nns_arma_optim(
        air,
        h=extension_h,
        seasonal_factor=candidate_periods,
        obj_fn=rmse,
        objective="min",
        pred_int=0.95,
        print_trace=not fast_mode(),
        plot=False,
    )
    show("NNS.ARMA.optim extension result", extension)
    forecast_figure(
        air,
        None,
        np.asarray(extension["results"], dtype=float),
        f"AirPassengers extension: {extension_h} months",
        "07_extension_forecast.png",
        np.asarray(extension["lower.pred.int"], dtype=float),
        np.asarray(extension["upper.pred.int"], dtype=float),
    )

    # %% [markdown]
    # ## Parameter interpretation and multivariate forecasting
    #
    # The R vignette closes with parameter notes and points readers to NNS.VAR for
    # multivariate systems. Python exposes ``nns_var`` but the canonical R
    # forecasting vignette contains no executable VAR example to translate here.
    subsection("Parameter interpretation")
    note(
        "seasonal_factor selects lag components; method chooses linear, nonlinear, "
        "or combined regression; weights combine multiple periods; dynamic=True "
        "re-estimates periods recursively; shrink averages linear estimates toward "
        "component means; pred_int requests partial-moment prediction intervals."
    )
    gap(
        "R's NNS.ARMA seasonal.plot side effect is not separately exposed in the "
        "Python return object. The translated seasonality figure above is generated "
        "from NNS.seas diagnostics instead."
    )
    note(
        "NNS.VAR is available in Python, but the R vignette only references the "
        "multivariate method and paper; it does not provide an executable example. "
        "No synthetic VAR demonstration is inserted into the canonical port."
    )


if __name__ == "__main__":
    main()
