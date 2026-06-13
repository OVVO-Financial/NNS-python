"""Plots for ``nns_arma`` and ``nns_arma_optim`` (R: ARMA.R, ARMA_optim.R)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from nns.plotting import palette
from nns.plotting._mpl import resolve_ax

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes


def _estimates_and_bounds(
    forecast: Any,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Split an nns_arma return into (estimates, lower, upper)."""
    if isinstance(forecast, Mapping):
        est = np.asarray(forecast["Estimates"], dtype=np.float64)
        lower = upper = None
        for key, value in forecast.items():
            if key.startswith("Lower"):
                lower = np.asarray(value, dtype=np.float64)
            elif key.startswith("Upper"):
                upper = np.asarray(value, dtype=np.float64)
        return est, lower, upper
    return np.asarray(forecast, dtype=np.float64), None, None


def plot_nns_arma(
    forecast: Any,
    original: Any,
    *,
    training_set: int | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot an ``nns_arma`` forecast, faithful to R ``NNS.ARMA(..., plot = TRUE)``.

    * original series: ``steelblue`` line
    * prediction-interval band: pink at alpha ``0.5``
    * forecast line + connector segment: ``red``
    * training-set markers (filled diamonds): pure green
    """
    ax = resolve_ax(ax)
    ov = np.asarray(original, dtype=np.float64)
    n = ov.size
    start = int(training_set) if training_set is not None else n
    est, lower, upper = _estimates_and_bounds(forecast)
    h = est.size

    # x positions are 1-based to match R's plot indices.
    ov_x = np.arange(1, n + 1)
    fc_x = np.arange(start + 1, start + h + 1)

    # Original series: steelblue.
    ax.plot(ov_x, ov, color="steelblue", linewidth=2)

    # Prediction-interval band: pink at alpha 0.5.
    if lower is not None and upper is not None:
        ax.fill_between(fc_x, lower, upper, color=palette.PINK,
                        alpha=palette.CI_ALPHA_ARMA, linewidth=0.0)

    # Forecast line + connector segment back to the last observed point: red.
    ax.plot(fc_x, est, color="red", linewidth=2)
    if 1 <= start <= n:
        ax.plot([start, start + 1], [ov[start - 1], est[0]], color="red", linewidth=2)

    # Training-set markers: pure-green diamonds.
    if 1 <= start <= n:
        ax.scatter([start], [ov[start - 1]], color=palette.GREEN, marker="D")
    ax.scatter([start + h], [est[-1]], color=palette.GREEN, marker="D")

    ax.legend(["Original", f"Forecast {h} period(s)"], loc="upper left", frameon=False)
    ax.set_title("NNS.ARMA Forecast")
    return ax


def plot_nns_arma_optim(
    result: Mapping[str, Any],
    original: Any,
    *,
    ax: Axes | None = None,
) -> Axes:
    """Plot an ``nns_arma_optim`` result, faithful to R ``NNS.ARMA.optim``.

    * original series: ``steelblue`` line
    * confidence band: ``steelblue`` at alpha ``0.5``
    * predicted/model line (dashed): ``red``
    """
    ax = resolve_ax(ax)
    ov = np.asarray(original, dtype=np.float64)
    n = ov.size
    results = np.asarray(result["results"], dtype=np.float64)
    h = results.size
    ov_x = np.arange(1, n + 1)
    fc_x = np.arange(n + 1, n + h + 1)

    ax.plot(ov_x, ov, color="steelblue", linewidth=2)

    lower = result.get("lower.pred.int")
    upper = result.get("upper.pred.int")
    if lower is not None and upper is not None:
        ax.fill_between(
            fc_x,
            np.asarray(lower, dtype=np.float64),
            np.asarray(upper, dtype=np.float64),
            color="steelblue",
            alpha=palette.CI_ALPHA_ARMA,
            linewidth=0.0,
        )

    ax.plot(fc_x, results, color="red", linewidth=2, linestyle="--")
    ax.legend(["Variable", "Internal Validation"], loc="upper left", frameon=False)
    ax.set_title("NNS.ARMA Forecast")
    return ax


__all__ = ["plot_nns_arma", "plot_nns_arma_optim"]
