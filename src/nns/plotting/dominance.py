"""Plots for FSD / SSD / TSD stochastic-dominance tests (R: FSD.R, SSD.R, TSD.R).

In every case the X curve is ``red`` and the Y curve is ``steelblue`` with the
legend ordered ``[X (red), Y (steelblue)]`` -- matching the R ``col=`` usage.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from nns.core import lpm, lpm_ratio
from nns.plotting._mpl import resolve_ax

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes


def _dominance_plot(
    x: Any,
    y: Any,
    *,
    title: str,
    ylabel: str,
    curve: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ax: Axes | None,
) -> Axes:
    ax = resolve_ax(ax)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    combined = np.sort(np.concatenate([x, y]))
    lpm_x = curve(combined, x)
    lpm_y = curve(combined, y)
    ax.plot(combined, lpm_x, color="red", linewidth=3)
    ax.plot(combined, lpm_y, color="steelblue", linewidth=3)
    ax.legend(["X", "Y"], loc="upper left")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    return ax


def plot_fsd(x: Any, y: Any, *, type: str = "discrete", ax: Axes | None = None) -> Axes:
    """Plot the FSD test: cumulative distributions (X red, Y steelblue)."""
    degree = 0.0 if str(type).lower() == "discrete" else 1.0
    return _dominance_plot(
        x, y,
        title="FSD",
        ylabel="Probability of Cumulative Distribution",
        curve=lambda t, v: np.asarray(lpm_ratio(degree, t, v), dtype=np.float64),
        ax=ax,
    )


def plot_ssd(x: Any, y: Any, *, ax: Axes | None = None) -> Axes:
    """Plot the SSD test: area of cumulative distributions (X red, Y steelblue)."""
    return _dominance_plot(
        x, y,
        title="SSD",
        ylabel="Area of Cumulative Distribution",
        curve=lambda t, v: np.asarray(lpm(1.0, t, v), dtype=np.float64),
        ax=ax,
    )


def plot_tsd(x: Any, y: Any, *, ax: Axes | None = None) -> Axes:
    """Plot the TSD test: area of cumulative distributions (X red, Y steelblue)."""
    return _dominance_plot(
        x, y,
        title="TSD",
        ylabel="Area of Cumulative Distribution",
        curve=lambda t, v: np.asarray(lpm(2.0, t, v), dtype=np.float64),
        ax=ax,
    )


__all__ = ["plot_fsd", "plot_ssd", "plot_tsd"]
