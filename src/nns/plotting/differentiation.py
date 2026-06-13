"""Plot for ``nns_diff`` numerical differentiation (R: Numerical_Differentiation.R)."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from nns.plotting import palette
from nns.plotting._mpl import resolve_ax

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes


def plot_nns_diff(
    f: Callable[[Any], Any],
    point: float,
    *,
    h: float | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot ``NNS.diff``'s ``f(x)`` / y-intercept-range panel (Numerical_Differentiation.R).

    Reconstructs the initial finite-step geometry (R uses ``h``, not the inferred
    step, in this panel):

    * function curve: ``azure4``
    * zero reference lines: ``grey`` (R ``grey`` == ``#BEBEBE``, not mpl gray)
    * center point ``f(point)``: pure green
    * the two finite-step bound points/segments swap ``steelblue`` <-> ``red``
      according to which secant y-intercept (``B1`` vs ``B2``) is higher.
    """
    ax = resolve_ax(ax)
    point = float(point)
    h = abs(point) * 0.1 + 0.01 if h is None else float(h)

    f_x = float(f(point))
    f_lower = float(f(point - h))  # f.x.h.lower
    f_upper = float(f(point + h))  # f.x.h.upper

    left_slope = (f_x - f_lower) / h
    right_slope = (f_upper - f_x) / h
    b1 = f_x - left_slope * point
    b2 = f_x - right_slope * point
    high_b = max(b1, b2)
    lower_color = "steelblue" if b1 == high_b else "red"
    upper_color = "red" if b1 == high_b else "steelblue"

    # Function curve: azure4 (no matplotlib name -> pinned hex).
    lo = min(point - 100 * h, point + 100 * h, 0.0)
    hi = max(point - 100 * h, point + 100 * h, 0.0)
    xs = np.linspace(lo, hi, 1000)
    ax.plot(xs, np.asarray([f(v) for v in xs], dtype=np.float64), color=palette.AZURE4, linewidth=2)

    # Zero reference lines: R grey (#BEBEBE).
    ax.axhline(0.0, color=palette.GREY)
    ax.axvline(0.0, color=palette.GREY)

    # Center point: pure green.
    ax.scatter([point], [f_x], color=palette.GREEN, marker="o")

    # Finite-step bound points (filled) and secant y-intercept markers (open).
    ax.scatter([point - h], [f_lower], color=lower_color, marker="o")
    ax.scatter([point + h], [f_upper], color=upper_color, marker="o")
    ax.scatter([0.0, 0.0], [b1, b2], facecolors="none",
               edgecolors=[lower_color, upper_color], marker="o")

    # Dashed secant segments from each y-intercept to its bound point.
    ax.plot([0.0, point - h], [b1, f_lower], color=lower_color, linestyle="--")
    ax.plot([0.0, point + h], [b2, f_upper], color=upper_color, linestyle="--")

    ax.set_ylabel("f(x)")
    ax.set_title("f(x) and initial y-intercept range")
    return ax


__all__ = ["plot_nns_diff"]
