"""Plot for ``nns_causation`` (R: Uni_Causation.R)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from nns.plotting._mpl import resolve_ax

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes


def _standardize(v: np.ndarray) -> np.ndarray:
    sd = np.std(v)
    if sd == 0:
        return np.asarray(v - np.mean(v), dtype=np.float64)
    return np.asarray((v - np.mean(v)) / sd, dtype=np.float64)


def plot_nns_causation(x: Any, y: Any, *, ax: Axes | None = None) -> Axes:
    """Plot the standardized X/Y series used by ``NNS.causation``.

    Faithful to R's first causation panel: the Y series is ``red`` and the X
    series is ``steelblue``, with the legend ordered ``[X (steelblue), Y (red)]``.
    """
    ax = resolve_ax(ax)
    xs = _standardize(np.asarray(x, dtype=np.float64))
    ys = _standardize(np.asarray(y, dtype=np.float64))
    ax.plot(xs, color="steelblue", linewidth=3, label="X")
    ax.plot(ys, color="red", linewidth=3, label="Y")
    ax.legend(loc="upper center", ncol=2)
    ax.set_ylabel("STANDARDIZED")
    return ax


__all__ = ["plot_nns_causation"]
