"""Plot for ``nns_copula`` (R: Copula.R)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from nns.plotting._mpl import require_mpl, resolve_ax

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes


def _orthant_colors(data: np.ndarray) -> list[str]:
    """Color each row by orthant (Copula.R:46-48).

    * all dimensions <= their column mean -> ``red`` (lower orthant)
    * all dimensions  > their column mean -> ``green`` (upper orthant)
    * otherwise -> ``steelblue``
    """
    means = data.mean(axis=0)
    below = (data <= means).all(axis=1)
    above = (data > means).all(axis=1)
    colors = np.full(data.shape[0], "steelblue", dtype=object)
    colors[below] = "red"
    colors[above] = "#00FF00"  # R pure green (mpl "green" would be #008000)
    return colors.tolist()


def plot_nns_copula(x: Any, *, ax: Axes | None = None) -> Axes:
    """Scatter the copula input matrix with R's orthant coloring (Copula.R).

    Supports 2-D (scatter) and 3-D (``n == 3``) inputs. Lower-orthant points
    (all dimensions below their mean) are ``red``, upper-orthant points are pure
    green, and the rest are ``steelblue``.
    """
    data = np.asarray(x, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("plot_nns_copula expects a 2-D array with >= 2 columns.")
    colors = _orthant_colors(data)
    n = data.shape[1]

    if n == 3:
        if ax is None:
            plt = require_mpl()
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], color=colors)
        return ax

    ax = resolve_ax(ax)
    ax.scatter(data[:, 0], data[:, 1], color=colors)
    return ax


__all__ = ["plot_nns_copula"]
