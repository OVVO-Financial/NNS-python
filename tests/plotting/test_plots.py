"""Color/element fidelity tests for nns.plotting.

These assert the *colors* and *which element they sit on* -- matching the R
``col=`` usage -- rather than doing any pixel/PDF comparison (see
``docs/plot_parity_policy.md``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.typing import NDArray

import nns
from nns.plotting import (
    plot_fsd,
    plot_nns_anova,
    plot_nns_arma,
    plot_nns_arma_optim,
    plot_nns_causation,
    plot_nns_cdf,
    plot_nns_copula,
    plot_nns_diff,
    plot_nns_norm,
    plot_nns_part,
    plot_nns_reg,
    plot_nns_seas,
    plot_ssd,
    plot_tsd,
)

STEELBLUE = "#4682b4"
RED = "#ff0000"
GREEN = "#00ff00"
PINK = "#ffc0cb"
GREY = "#bebebe"
AZURE4 = "#838b8b"


def _hex(color: Any) -> str:
    return mcolors.to_hex(color)


def line_hexes(ax: object) -> set[str]:
    return {_hex(line.get_color()) for line in ax.get_lines()}  # type: ignore[attr-defined]


def collection_face_hexes(ax: object) -> set[str]:
    out: set[str] = set()
    for coll in ax.collections:  # type: ignore[attr-defined]
        for row in coll.get_facecolor():
            if len(row):
                out.add(_hex(row))
    return out


def collection_edge_hexes(ax: object) -> set[str]:
    out: set[str] = set()
    for coll in ax.collections:  # type: ignore[attr-defined]
        for row in coll.get_edgecolor():
            if len(row):
                out.add(_hex(row))
    return out


def patch_face_hexes(ax: object) -> set[str]:
    return {_hex(p.get_facecolor()) for p in ax.patches}  # type: ignore[attr-defined]


Array = NDArray[np.float64]


@pytest.fixture(autouse=True)
def _close_figs() -> Iterator[None]:
    yield
    plt.close("all")


@pytest.fixture
def reg_xy() -> tuple[Array, Array]:
    rng = np.random.default_rng(0)
    x = np.sort(rng.normal(size=50))
    y = 2.0 * x + rng.normal(scale=0.3, size=50)
    return x, y


def test_plot_nns_reg_colors(reg_xy: tuple[Array, Array]) -> None:
    x, y = reg_xy
    point_est = np.array([3.0, -3.0])
    result = nns.nns_reg(x, y, confidence_interval=0.95, point_est=point_est)
    ax = plot_nns_reg(result, point_est=point_est)

    # steelblue open-circle scatter (edge colored, face transparent)
    assert STEELBLUE in collection_edge_hexes(ax)
    # pink CI band
    assert PINK in collection_face_hexes(ax)
    # red regression-point line + red squares
    assert RED in line_hexes(ax)
    assert RED in collection_face_hexes(ax)
    # pure-green point-estimate diamonds + extrapolation segments
    assert GREEN in collection_face_hexes(ax)
    assert GREEN in line_hexes(ax)


def test_plot_nns_part_colors(reg_xy: tuple[Array, Array]) -> None:
    x, y = reg_xy
    result = nns.nns_part(x, y)
    ax = plot_nns_part(result)
    faces = collection_face_hexes(ax)
    assert STEELBLUE in faces
    assert RED in faces


@pytest.fixture
def series() -> Array:
    rng = np.random.default_rng(1)
    return np.cumsum(rng.normal(size=60)) + 20.0


def test_plot_nns_arma_colors(series: Array) -> None:
    forecast = nns.nns_arma(series, h=6, pred_int=0.95, seasonal_factor=False)
    ax = plot_nns_arma(forecast, series)
    lines = line_hexes(ax)
    assert STEELBLUE in lines  # original series
    assert RED in lines  # forecast + connector
    faces = collection_face_hexes(ax)
    assert PINK in faces  # prediction band
    assert GREEN in faces  # training-set markers


def test_plot_nns_arma_optim_colors(series: Array) -> None:
    result = nns.nns_arma_optim(
        series[:40], h=6, seasonal_factor=[1], pred_int=0.95, print_trace=False
    )
    ax = plot_nns_arma_optim(result, series[:40])
    lines = line_hexes(ax)
    assert STEELBLUE in lines
    assert RED in lines
    # steelblue confidence band fill
    assert STEELBLUE in collection_face_hexes(ax)


def test_plot_nns_cdf_colors(series: Array) -> None:
    result = nns.nns_cdf(series, target=20.0)
    ax = plot_nns_cdf(result, target=20.0)
    assert STEELBLUE in line_hexes(ax)  # step CDF
    assert STEELBLUE in collection_face_hexes(ax)  # points
    assert RED in line_hexes(ax)  # VaR segments
    assert GREEN in collection_face_hexes(ax)  # VaR point


@pytest.mark.parametrize("fn", [plot_fsd, plot_ssd, plot_tsd])
def test_plot_dominance_colors(fn: Callable[..., Any]) -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=80)
    y = rng.normal(loc=0.5, size=80)
    ax = fn(x, y)
    lines = line_hexes(ax)
    assert RED in lines  # X curve
    assert STEELBLUE in lines  # Y curve


def test_plot_nns_anova_colors() -> None:
    rng = np.random.default_rng(4)
    groups = [rng.normal(size=30), rng.normal(loc=1, size=30), rng.normal(loc=2, size=30)]
    ax = plot_nns_anova(groups)
    # first box steelblue, grand-mean line red
    assert STEELBLUE in patch_face_hexes(ax)
    assert RED in line_hexes(ax)


def test_plot_nns_causation_colors() -> None:
    rng = np.random.default_rng(5)
    x = rng.normal(size=60)
    y = np.roll(x, 1) + rng.normal(scale=0.1, size=60)
    ax = plot_nns_causation(x, y)
    lines = line_hexes(ax)
    assert STEELBLUE in lines  # X series
    assert RED in lines  # Y series


def test_plot_nns_norm_line_colors() -> None:
    rng = np.random.default_rng(6)
    x = np.column_stack([rng.normal(size=40), rng.normal(loc=5, size=40)])
    ax = plot_nns_norm(x, chart_type="l")
    # first series steelblue
    assert STEELBLUE in line_hexes(ax)


def test_plot_nns_norm_boxplot_colors() -> None:
    rng = np.random.default_rng(7)
    x = np.column_stack([rng.normal(size=40), rng.normal(loc=5, size=40)])
    ax = plot_nns_norm(x, chart_type="b")
    # raw boxes are grey
    assert GREY in patch_face_hexes(ax)


def test_plot_nns_seas_colors(series: Array) -> None:
    result = nns.nns_seas(series, plot=False)
    ax = plot_nns_seas(result)
    assert STEELBLUE in collection_face_hexes(ax)  # component points
    assert RED in collection_face_hexes(ax)  # best-period point
    assert RED in line_hexes(ax)  # overall-CV reference line


def test_plot_nns_diff_colors() -> None:
    ax = plot_nns_diff(lambda z: z**2, 5.0)
    assert AZURE4 in line_hexes(ax)  # function curve
    assert GREY in line_hexes(ax)  # zero reference lines
    assert GREEN in collection_face_hexes(ax)  # center point
    bounds = collection_face_hexes(ax) | collection_edge_hexes(ax)
    assert STEELBLUE in bounds and RED in bounds  # swapped bound points


def test_plot_nns_copula_colors() -> None:
    rng = np.random.default_rng(8)
    data = rng.normal(size=(120, 2))
    ax = plot_nns_copula(data)
    faces = collection_face_hexes(ax)
    assert RED in faces  # lower-orthant points
    assert STEELBLUE in faces  # mixed-orthant points
