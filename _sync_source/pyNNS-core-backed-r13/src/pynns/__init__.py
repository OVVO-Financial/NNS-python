from __future__ import annotations

from typing import Any

from pynns.pm_matrix import pm_matrix as pm_matrix

__version__ = "0.2.0"

_EXPORTS = {
    "FactorDesign": ("pynns.regression", "FactorDesign"),
    "causal_matrix": ("pynns.causation", "causal_matrix"),
    "co_lpm": ("pynns.co_moments", "co_lpm"),
    "co_lpm_nd": ("pynns.dependence", "co_lpm_nd"),
    "co_upm": ("pynns.co_moments", "co_upm"),
    "co_upm_nd": ("pynns.dependence", "co_upm_nd"),
    "d_lpm": ("pynns.co_moments", "d_lpm"),
    "dpm_nd": ("pynns.dependence", "dpm_nd"),
    "dy_d": ("pynns.diff", "dy_d"),
    "dy_dx": ("pynns.diff", "dy_dx"),
    "d_upm": ("pynns.co_moments", "d_upm"),
    "ecdf_pm": ("pynns.classical", "ecdf_pm"),
    "encode_factor_codes": ("pynns.categorical", "encode_factor_codes"),
    "factor_2_dummy": ("pynns.categorical", "factor_2_dummy"),
    "factor_2_dummy_fr": ("pynns.categorical", "factor_2_dummy_fr"),
    "fsd": ("pynns.stochastic_dominance", "fsd"),
    "fsd_uni": ("pynns.stochastic_dominance", "fsd_uni"),
    "kurt_pm": ("pynns.classical", "kurt_pm"),
    "lpm": ("pynns.core", "lpm"),
    "lpm_ratio": ("pynns.core", "lpm_ratio"),
    "lpm_var": ("pynns.var", "lpm_var"),
    "mean_pm": ("pynns.classical", "mean_pm"),
    "nns_anova": ("pynns.anova", "nns_anova"),
    "nns_arma": ("pynns.arma", "nns_arma"),
    "nns_arma_optim": ("pynns.arma", "nns_arma_optim"),
    "nns_boost": ("pynns.boost", "nns_boost"),
    "nns_causation": ("pynns.causation", "nns_causation"),
    "nns_cdf": ("pynns.cdf", "nns_cdf"),
    "nns_copula": ("pynns.copula", "nns_copula"),
    "nns_cor": ("pynns.dependence", "nns_cor"),
    "nns_dep": ("pynns.dependence", "nns_dep"),
    "nns_diff": ("pynns.diff", "nns_diff"),
    "nns_distance": ("pynns.distance", "nns_distance"),
    "nns_distance_bulk": ("pynns.distance", "nns_distance_bulk"),
    "nns_gravity": ("pynns.central_tendencies", "nns_gravity"),
    "nns_mode": ("pynns.central_tendencies", "nns_mode"),
    "nns_moments": ("pynns.classical", "nns_moments"),
    "nns_m_reg": ("pynns.multivariate_regression", "nns_m_reg"),
    "nns_mc": ("pynns.mc", "nns_mc"),
    "nns_meboot": ("pynns.meboot", "nns_meboot"),
    "nns_norm": ("pynns.norm", "nns_norm"),
    "nns_nowcast_panel": ("pynns.nowcast", "nns_nowcast_panel"),
    "nns_part": ("pynns.part", "nns_part"),
    "nns_reg": ("pynns.regression", "nns_reg"),
    "nns_rescale": ("pynns.central_tendencies", "nns_rescale"),
    "nns_seas": ("pynns.seasonality", "nns_seas"),
    "nns_sd_cluster": ("pynns.stochastic_dominance", "nns_sd_cluster"),
    "nns_stack": ("pynns.stack", "nns_stack"),
    "nns_ss": ("pynns.stochastic_superiority", "nns_ss"),
    "nns_var": ("pynns.var", "nns_var"),
    "prepare_factor_predictors": ("pynns.regression", "prepare_factor_predictors"),
    "sd_efficient_set": ("pynns.stochastic_dominance", "sd_efficient_set"),
    "skew_pm": ("pynns.classical", "skew_pm"),
    "ssd": ("pynns.stochastic_dominance", "ssd"),
    "ssd_uni": ("pynns.stochastic_dominance", "ssd_uni"),
    "tsd": ("pynns.stochastic_dominance", "tsd"),
    "tsd_uni": ("pynns.stochastic_dominance", "tsd_uni"),
    "upm": ("pynns.core", "upm"),
    "upm_ratio": ("pynns.core", "upm_ratio"),
    "upm_var": ("pynns.var", "upm_var"),
    "var_pm": ("pynns.classical", "var_pm"),
}

__all__ = sorted((*_EXPORTS, "pm_matrix"))


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module 'pynns' has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    from importlib import import_module

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
