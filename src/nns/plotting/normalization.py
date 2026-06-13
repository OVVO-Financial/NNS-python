"""Plot for ``nns_norm`` (R: Normalization.R)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from nns.plotting import palette
from nns.plotting._mpl import resolve_ax

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes


def _series_colors(n: int) -> list[Any]:
    """First series ``steelblue``; remaining series follow ``rainbow(n)``."""
    rainbow = [tuple(c) for c in palette.rainbow(n)]
    return ["steelblue", *rainbow][:n]


def plot_nns_norm(
    x: Any,
    *,
    chart_type: Literal["l", "b"] = "l",
    ax: Axes | None = None,
) -> Axes:
    """Plot the input series for ``NNS.norm``, faithful to Normalization.R.

    * line chart (``chart_type="l"``): first series ``steelblue``, rest ``rainbow``
    * boxplot chart (``chart_type="b"``): raw boxes ``grey``, normalized ``rainbow``
    """
    ax = resolve_ax(ax)
    data = np.asarray(x, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    n = data.shape[1]

    if chart_type == "b":
        from nns.norm import nns_norm

        normalized = np.asarray(nns_norm(data), dtype=np.float64)
        columns = [data[:, j] for j in range(n)] + [normalized[:, j] for j in range(n)]
        bp = ax.boxplot(columns, patch_artist=True)
        rainbow = [tuple(c) for c in palette.rainbow(n)]
        facecolors = [palette.GREY] * n + rainbow
        for patch, color in zip(bp["boxes"], facecolors, strict=True):
            patch.set_facecolor(color)
        return ax

    colors = _series_colors(n)
    for j in range(n):
        ax.plot(data[:, j], color=colors[j], linewidth=2)
    return ax


__all__ = ["plot_nns_norm"]
