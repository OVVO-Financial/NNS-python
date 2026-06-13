"""Optional matplotlib plotting API for NNS, faithful to R NNS ``plot = TRUE``.

This subpackage is an **optional extra**. The NNS core stays NumPy/SciPy-only;
matplotlib is required only here and is imported lazily inside each plot
function (``pip install ovvo-nns[plot]``).

Design contract for every ``plot_*`` function:

* Accept an already-computed NNS result (or the same raw inputs) plus a keyword
  ``ax=None`` and return the matplotlib ``Axes`` (or ``Figure`` for 3-D).
* Never call ``plt.show()`` -- the caller controls display and saving.
* Be **color/element-faithful** to R (see :mod:`nns.plotting.palette`), not
  pixel-diffed.

Colors are exposed via :mod:`nns.plotting.palette`; the only same-named colors
that must *not* be trusted from matplotlib are ``green`` (-> ``#00FF00``) and
``grey`` (-> ``#BEBEBE``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nns.plotting import palette as palette

if TYPE_CHECKING:  # pragma: no cover - typing only
    from nns.plotting.anova import plot_nns_anova as plot_nns_anova
    from nns.plotting.arma import plot_nns_arma as plot_nns_arma
    from nns.plotting.arma import plot_nns_arma_optim as plot_nns_arma_optim
    from nns.plotting.causation import plot_nns_causation as plot_nns_causation
    from nns.plotting.copula import plot_nns_copula as plot_nns_copula
    from nns.plotting.differentiation import plot_nns_diff as plot_nns_diff
    from nns.plotting.dominance import plot_fsd as plot_fsd
    from nns.plotting.dominance import plot_ssd as plot_ssd
    from nns.plotting.dominance import plot_tsd as plot_tsd
    from nns.plotting.normalization import plot_nns_norm as plot_nns_norm
    from nns.plotting.partial_moments import plot_nns_cdf as plot_nns_cdf
    from nns.plotting.regression import plot_nns_part as plot_nns_part
    from nns.plotting.regression import plot_nns_reg as plot_nns_reg
    from nns.plotting.seasonality import plot_nns_seas as plot_nns_seas

_EXPORTS = {
    "plot_nns_anova": ("nns.plotting.anova", "plot_nns_anova"),
    "plot_nns_arma": ("nns.plotting.arma", "plot_nns_arma"),
    "plot_nns_arma_optim": ("nns.plotting.arma", "plot_nns_arma_optim"),
    "plot_nns_causation": ("nns.plotting.causation", "plot_nns_causation"),
    "plot_nns_copula": ("nns.plotting.copula", "plot_nns_copula"),
    "plot_nns_diff": ("nns.plotting.differentiation", "plot_nns_diff"),
    "plot_fsd": ("nns.plotting.dominance", "plot_fsd"),
    "plot_ssd": ("nns.plotting.dominance", "plot_ssd"),
    "plot_tsd": ("nns.plotting.dominance", "plot_tsd"),
    "plot_nns_norm": ("nns.plotting.normalization", "plot_nns_norm"),
    "plot_nns_cdf": ("nns.plotting.partial_moments", "plot_nns_cdf"),
    "plot_nns_part": ("nns.plotting.regression", "plot_nns_part"),
    "plot_nns_reg": ("nns.plotting.regression", "plot_nns_reg"),
    "plot_nns_seas": ("nns.plotting.seasonality", "plot_nns_seas"),
}

__all__ = sorted((*_EXPORTS, "palette"))


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module 'nns.plotting' has no attribute {name!r}")
    from importlib import import_module

    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
