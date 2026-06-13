"""Plot for ``nns_cdf`` / VaR overlays (R: Partial_Moments.R)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from nns.plotting import palette
from nns.plotting._mpl import resolve_ax

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes


def plot_nns_cdf(
    result: Mapping[str, Any],
    *,
    target: float | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot an ``nns_cdf`` result, faithful to R ``NNS.CDF(..., plot = TRUE)``.

    * step CDF line + points: ``steelblue``
    * VaR/target segments (dashed): ``red``
    * VaR point (at the target): pure green
    """
    ax = resolve_ax(ax)
    function = result["Function"]
    x = np.asarray(function["x"], dtype=np.float64)
    colname = next(k for k in function if k != "x")
    fx = np.asarray(function[colname], dtype=np.float64)

    # Step CDF + points: steelblue.
    ax.step(x, fx, where="post", color="steelblue", linewidth=2)
    ax.scatter(x, fx, color="steelblue", marker="o")

    target_value = np.asarray(result.get("target.value", []), dtype=np.float64).reshape(-1)
    if target is not None and target_value.size:
        pv = float(target_value[0])
        # Dashed red VaR segments down from the curve and across to the y-axis.
        ax.plot([target, target], [0.0, pv], color="red", linestyle="--", linewidth=2)
        ax.plot([float(x.min()), target], [pv, pv], color="red", linestyle="--", linewidth=2)
        # Pure-green target point.
        ax.scatter([target], [pv], color=palette.GREEN, marker="o")

    ax.set_ylabel(colname)
    return ax


__all__ = ["plot_nns_cdf"]
