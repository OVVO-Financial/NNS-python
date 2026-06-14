"""`plot=True` on the compute functions now renders a figure as a side effect.

This validates the wiring requested over the value-only default: each function
still returns the same value it returns with ``plot=False`` (the contract the
parity suite depends on), but ``plot=True`` additionally *creates* a Matplotlib
figure via the ``nns.plotting`` layer.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest

import nns


@pytest.fixture(autouse=True)
def _close_figs() -> object:
    plt.close("all")
    yield
    plt.close("all")


def _fig_count() -> int:
    return len(plt.get_fignums())


def test_nns_reg_plot_true_creates_figure() -> None:
    rng = np.random.default_rng(0)
    x = np.sort(rng.normal(size=40))
    y = 2.0 * x + rng.normal(scale=0.3, size=40)
    base = nns.nns_reg(x, y, confidence_interval=0.95)
    plt.close("all")
    drawn = nns.nns_reg(x, y, confidence_interval=0.95, plot=True)
    assert _fig_count() > 0
    assert base["R2"] == drawn["R2"]


def test_nns_reg_residual_plot_creates_figure() -> None:
    rng = np.random.default_rng(1)
    x = np.sort(rng.normal(size=40))
    y = 2.0 * x + rng.normal(scale=0.3, size=40)
    nns.nns_reg(x, y, residual_plot=True)
    assert _fig_count() > 0


def test_nns_cdf_plot_true_creates_figure() -> None:
    rng = np.random.default_rng(2)
    v = np.cumsum(rng.normal(size=60)) + 20.0
    base = nns.nns_cdf(v, target=20.0)
    plt.close("all")
    drawn = nns.nns_cdf(v, target=20.0, plot=True)
    assert _fig_count() > 0
    assert np.allclose(np.asarray(base["target.value"]), np.asarray(drawn["target.value"]))


def test_nns_arma_plot_true_creates_figure() -> None:
    rng = np.random.default_rng(3)
    v = np.cumsum(rng.normal(size=60)) + 20.0
    drawn = nns.nns_arma(v, h=6, pred_int=0.95, seasonal_factor=False, plot=True)
    assert _fig_count() > 0
    assert "Estimates" in drawn


def test_nns_arma_optim_plot_true_creates_figure() -> None:
    rng = np.random.default_rng(4)
    v = np.cumsum(rng.normal(size=60)) + 20.0
    drawn = nns.nns_arma_optim(
        v[:40], h=6, seasonal_factor=[1], pred_int=0.95, print_trace=False, plot=True
    )
    assert _fig_count() > 0
    assert "results" in drawn


def test_nns_seas_plot_true_creates_figure() -> None:
    rng = np.random.default_rng(5)
    v = np.cumsum(rng.normal(size=60)) + 20.0
    base = nns.nns_seas(v)
    plt.close("all")
    drawn = nns.nns_seas(v, plot=True)
    assert _fig_count() > 0
    assert base["best.period"] == drawn["best.period"]


def test_nns_m_reg_plot_true_creates_figure() -> None:
    import matplotlib.colors as mcolors

    rng = np.random.default_rng(6)
    x = np.sort(rng.normal(size=40))
    y = 2.0 * x + rng.normal(scale=0.3, size=40)
    features = np.column_stack([x, x**2])
    nns.nns_m_reg(features, y, plot=True)
    assert _fig_count() > 0
    ax: Any = plt.gca()
    # R's M.reg residual plot: actual y is steelblue, fitted y.hat is a red line.
    edge_hexes = {
        mcolors.to_hex(row)
        for coll in ax.collections
        for row in coll.get_edgecolor()
        if len(row)
    }
    line_hexes = {mcolors.to_hex(line.get_color()) for line in ax.get_lines()}
    assert "#4682b4" in edge_hexes
    assert "#ff0000" in line_hexes


def test_plot_false_creates_no_figure() -> None:
    rng = np.random.default_rng(7)
    x = np.sort(rng.normal(size=40))
    y = 2.0 * x + rng.normal(scale=0.3, size=40)
    plt.close("all")
    nns.nns_reg(x, y)
    assert _fig_count() == 0
