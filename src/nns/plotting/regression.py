"""Plots for ``nns_reg`` and ``nns_part`` (R: Regression.R, Partition_Map.R)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from nns.plotting import palette
from nns.plotting._mpl import resolve_ax

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes


def _sorted_xy(x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(x)
    return x[order], y[order]


def plot_nns_reg(
    result: Mapping[str, Any],
    *,
    ax: Axes | None = None,
    ci: bool = True,
    point_est: Any = None,
) -> Axes:
    """Plot an ``nns_reg`` result, faithful to R ``NNS.reg(..., plot = TRUE)``.

    Element colors (Regression.R):

    * data scatter (open circles): ``steelblue``
    * confidence-interval band fill: pink at alpha ``0.375``
    * regression points (filled squares) + connecting line: ``red``
    * point estimates (filled diamonds) + extrapolation segments: pure green
    """
    ax = resolve_ax(ax)
    fitted = result["Fitted.xy"]
    x = np.asarray(fitted["x"], dtype=np.float64)
    y = np.asarray(fitted["y"], dtype=np.float64)

    # 1. Data scatter: open circles, steelblue.
    ax.scatter(x, y, facecolors="none", edgecolors="steelblue", marker="o")

    # 2. Confidence-interval band: pink polygon at alpha 0.375 (if available).
    if ci and "conf.int.pos" in fitted and "conf.int.neg" in fitted:
        idx = np.argsort(x)
        pos = np.asarray(fitted["conf.int.pos"], dtype=np.float64)[idx]
        neg = np.asarray(fitted["conf.int.neg"], dtype=np.float64)[idx]
        mask = np.isfinite(pos) & np.isfinite(neg)
        if mask.any():
            ax.fill_between(
                x[idx][mask],
                neg[mask],
                pos[mask],
                color=palette.PINK,
                alpha=palette.CI_ALPHA_REG,
                linewidth=0.0,
            )

    # 3. Regression points: red filled squares + red dashed connecting line.
    rp = result["regression.points"]
    rpx, rpy = _sorted_xy(rp["x"], rp["y"])
    finite = np.isfinite(rpx) & np.isfinite(rpy)
    ax.plot(rpx[finite], rpy[finite], color="red", linewidth=2, linestyle="--")
    ax.scatter(rpx[finite], rpy[finite], color="red", marker="s")

    # 4. Point estimates: pure-green filled diamonds (+ extrapolation segments).
    pe_y = np.asarray(result.get("Point.est", []), dtype=np.float64)
    if point_est is not None and pe_y.size:
        pe_x = np.asarray(point_est, dtype=np.float64).reshape(-1)
        ax.scatter(pe_x, pe_y, color=palette.GREEN, marker="D", s=80)
        if pe_x.size and rpx[finite].size:
            hi = pe_x > x.max()
            for px, py in zip(pe_x[hi], pe_y[hi], strict=True):
                ax.plot([px, rpx[finite][-1]], [py, rpy[finite][-1]],
                        color=palette.GREEN, linestyle="--")
            lo = pe_x < x.min()
            for px, py in zip(pe_x[lo], pe_y[lo], strict=True):
                ax.plot([px, rpx[finite][0]], [py, rpy[finite][0]],
                        color=palette.GREEN, linestyle="--")

    ax.set_title(f"NNS Order = {max(1, int(result.get('order', 1) or 1))}")
    return ax


def plot_nns_part(result: Mapping[str, Any], *, ax: Axes | None = None) -> Axes:
    """Plot an ``nns_part`` result, faithful to R ``NNS.part(..., plot = TRUE)``.

    Scatter is ``steelblue``; regression points (filled squares) are ``red``.
    """
    ax = resolve_ax(ax)
    dt = result["dt"]
    x = np.asarray(dt["x"], dtype=np.float64)
    y = np.asarray(dt["y"], dtype=np.float64)
    ax.scatter(x, y, color="steelblue")

    rp = result["regression.points"]
    rpx = np.asarray(rp["x"], dtype=np.float64)
    rpy = np.asarray(rp["y"], dtype=np.float64)
    ax.scatter(rpx, rpy, color="red", marker="s", linewidths=2)
    return ax


__all__ = ["plot_nns_part", "plot_nns_reg"]
