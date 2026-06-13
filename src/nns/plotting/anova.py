"""Plot for ``nns_anova`` (R: ANOVA.R)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from nns.plotting import palette
from nns.plotting._mpl import horizontal_boxplot, resolve_ax

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes


def plot_nns_anova(
    groups: Sequence[Any],
    *,
    medians: bool = False,
    ax: Axes | None = None,
) -> Axes:
    """Plot ANOVA group boxplots, faithful to R ``NNS.ANOVA(..., plot = TRUE)``.

    * first box: ``steelblue``; remaining boxes: ``rainbow(n - 1)``
    * grand mean/median reference line (vertical): ``red``
    """
    ax = resolve_ax(ax)
    arrays = [np.asarray(g, dtype=np.float64) for g in groups]
    n = len(arrays)

    bp = horizontal_boxplot(ax, arrays, patch_artist=True)
    rest = palette.rainbow(n - 1)
    facecolors = ["steelblue", *[tuple(c) for c in rest]]
    for patch, color in zip(bp["boxes"], facecolors, strict=True):
        patch.set_facecolor(color)

    centers = [float(np.median(a)) if medians else float(np.mean(a)) for a in arrays]
    grand = float(np.mean(centers))
    ax.axvline(grand, color="red", linewidth=4)
    ax.set_title("NNS ANOVA")
    ax.set_xlabel("Grand Median" if medians else "Grand Mean")
    return ax


__all__ = ["plot_nns_anova"]
