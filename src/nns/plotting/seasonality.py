"""Plot for ``nns_seas`` (R: Seasonality_Test.R)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from nns.plotting._mpl import resolve_ax

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes


def plot_nns_seas(result: Mapping[str, Any], *, ax: Axes | None = None) -> Axes:
    """Plot an ``nns_seas`` result, faithful to R ``NNS.seas(..., plot = TRUE)``.

    * component-series CV points: ``steelblue``
    * best period (enlarged point) + overall-CV reference line (dashed) + label: ``red``
    """
    ax = resolve_ax(ax)
    periods = result["all.periods"]
    period = np.asarray(periods["Period"], dtype=np.float64)
    cv = np.asarray(periods["Coefficient.of.Variation"], dtype=np.float64)

    # All component-series CV points: steelblue.
    ax.scatter(period, cv, color="steelblue", marker="o")

    # Best period is the first (table keyed ascending by CV): enlarged red point.
    if period.size:
        ax.scatter([period[0]], [cv[0]], color="red", marker="o", s=80)

    overall_cv = float(np.asarray(periods["Variable.Coefficient.of.Variation"])[0])
    if np.isfinite(overall_cv):
        ax.axhline(overall_cv, color="red", linestyle="--")

    ax.set_xlabel("Period")
    ax.set_ylabel("Component Series CV")
    return ax


__all__ = ["plot_nns_seas"]
