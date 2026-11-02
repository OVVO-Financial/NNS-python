"""``plot=True`` on the remaining compute functions draws via ``nns.plotting``.

Each function must return exactly what it returns with ``plot=False`` (the
parity suite relies on that) and, with ``plot=True``, create a figure. Plots
that cannot be drawn warn instead of silently doing nothing.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest

import nns


@pytest.fixture(autouse=True)
def _close_figs() -> object:
    plt.close("all")
    yield
    plt.close("all")


def _figures() -> int:
    return len(plt.get_fignums())


def _samples() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(11)
    return rng.normal(0.5, 1.0, size=80), rng.normal(0.0, 1.0, size=60)


@pytest.mark.parametrize("name", ["fsd", "ssd", "tsd"])
def test_sd_plot_true_creates_figure(name: str) -> None:
    x, y = _samples()
    fn = getattr(nns, name)
    base = fn(x, y)
    assert _figures() == 0
    assert fn(x, y, plot=True) == base
    assert _figures() == 1


def test_fsd_continuous_plot() -> None:
    x, y = _samples()
    base = nns.fsd(x, y, type="continuous")
    assert nns.fsd(x, y, type="continuous", plot=True) == base
    assert _figures() == 1


def test_nns_part_plot_true_creates_figure() -> None:
    x, y = _samples()
    y = y[:40] + x[:40]
    x = x[:40]
    base = nns.nns_part(x, y)
    assert _figures() == 0
    drawn = nns.nns_part(x, y, plot=True)
    assert _figures() == 1
    np.testing.assert_array_equal(base["regression.points"]["y"], drawn["regression.points"]["y"])


@pytest.mark.parametrize("pairwise", [False, True])
def test_nns_anova_plot_true_creates_figure(pairwise: bool) -> None:
    x, y = _samples()
    groups = [x, y, x[:30] + 1.0]
    base = nns.nns_anova(groups, pairwise=pairwise)
    assert _figures() == 0
    drawn = nns.nns_anova(groups, pairwise=pairwise, plot=True)
    assert _figures() == 1
    np.testing.assert_array_equal(
        np.asarray(base["Certainty"] if not pairwise else base),
        np.asarray(drawn["Certainty"] if not pairwise else drawn),
    )


def test_nns_anova_binary_plot() -> None:
    x, y = _samples()
    base = nns.nns_anova(x, y, confidence_interval=None)
    assert nns.nns_anova(x, y, confidence_interval=None, plot=True) == base
    assert _figures() == 1


def test_nns_norm_plot_true_creates_two_panels() -> None:
    rng = np.random.default_rng(3)
    data = np.cumsum(rng.normal(size=(50, 3)), axis=0) + np.array([10.0, 50.0, 100.0])
    base = nns.nns_norm(data)
    assert _figures() == 0
    drawn = nns.nns_norm(data, plot=True)
    np.testing.assert_array_equal(base, drawn)
    assert _figures() == 1
    assert len(plt.gcf().axes) == 2


def test_nns_norm_plot_unequal_lengths_warns() -> None:
    rng = np.random.default_rng(4)
    series = [rng.normal(5.0, 1.0, size=30), rng.normal(10.0, 1.0, size=20)]
    with pytest.warns(UserWarning, match="unequal-length"):
        nns.nns_norm(series, plot=True)
    assert _figures() == 0


def test_nns_diff_plot_true_creates_figure() -> None:
    base = nns.nns_diff(np.sin, 1.0)
    assert _figures() == 0
    assert nns.nns_diff(np.sin, 1.0, plot=True) == base
    assert _figures() == 1


def test_nns_diff_plot_skips_display_samples_outside_the_domain() -> None:
    base = nns.nns_diff(math.log, 5.0)
    np.testing.assert_equal(nns.nns_diff(math.log, 5.0, plot=True), base)
    assert _figures() == 1


@pytest.mark.parametrize("k", [2, 3])
def test_nns_copula_plot_true_creates_figure(k: int) -> None:
    rng = np.random.default_rng(5)
    data = rng.normal(size=(40, k))
    base = nns.nns_copula(data)
    assert _figures() == 0
    assert nns.nns_copula(data, plot=True) == base
    assert _figures() == 1


def test_nns_copula_bivariate_vectors_plot() -> None:
    x, y = _samples()
    base = nns.nns_copula(x[:60], y)
    assert nns.nns_copula(x[:60], y, plot=True) == base
    assert _figures() == 1


def test_nns_copula_plot_many_columns_warns() -> None:
    data = np.random.default_rng(6).normal(size=(40, 4))
    with pytest.warns(UserWarning, match="two or three variables"):
        nns.nns_copula(data, plot=True)
    assert _figures() == 0


def _mreg_data(k: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(60, k))
    return x, x.sum(axis=1) + rng.normal(scale=0.1, size=60)


def test_multivariate_reg_plot_two_regressors_draws_3d() -> None:
    x, y = _mreg_data(2)
    base = nns.nns_reg(x, y)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        drawn = nns.nns_reg(x, y, plot=True)
    assert base["R2"] == drawn["R2"]
    assert _figures() == 1
    assert plt.gca().name == "3d"


def test_multivariate_reg_plot_many_regressors_warns() -> None:
    x, y = _mreg_data(3)
    with pytest.warns(UserWarning, match="one or two regressors"):
        nns.nns_reg(x, y, plot=True)
    assert _figures() == 0


def test_multivariate_reg_residual_plot() -> None:
    x, y = _mreg_data(3)
    nns.nns_reg(x, y, residual_plot=True)
    assert _figures() == 1


def test_sd_cluster_dendrogram_draws() -> None:
    rng = np.random.default_rng(8)
    data = rng.normal(size=(100, 5)) + np.arange(5) * 0.3
    result = nns.nns_sd_cluster(data, degree=2, dendrogram=True)
    assert "Dendrogram" in result
    assert _figures() == 1
    assert nns.nns_sd_cluster(data, degree=2)["Clusters"] == result["Clusters"]
    assert _figures() == 1


def test_sd_cluster_dendrogram_too_few_variables_warns() -> None:
    data = np.random.default_rng(9).normal(size=(50, 1))
    with pytest.warns(UserWarning, match="Not enough variables"):
        result = nns.nns_sd_cluster(data, dendrogram=True)
    assert result["Order"] is None
    assert _figures() == 0
