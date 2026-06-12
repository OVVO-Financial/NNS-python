from __future__ import annotations

from typing import Any

from nns.pm_matrix import pm_matrix as pm_matrix

__version__ = "1.0.0a0"

_EXPORTS = {
    "FactorDesign": ("nns.regression", "FactorDesign"),
    "causal_matrix": ("nns.causation", "causal_matrix"),
    "co_lpm": ("nns.co_moments", "co_lpm"),
    "co_lpm_nd": ("nns.dependence", "co_lpm_nd"),
    "co_upm": ("nns.co_moments", "co_upm"),
    "co_upm_nd": ("nns.dependence", "co_upm_nd"),
    "d_lpm": ("nns.co_moments", "d_lpm"),
    "dpm_nd": ("nns.dependence", "dpm_nd"),
    "dy_d": ("nns.diff", "dy_d"),
    "dy_dx": ("nns.diff", "dy_dx"),
    "d_upm": ("nns.co_moments", "d_upm"),
    "ecdf_pm": ("nns.classical", "ecdf_pm"),
    "encode_factor_codes": ("nns.categorical", "encode_factor_codes"),
    "factor_2_dummy": ("nns.categorical", "factor_2_dummy"),
    "factor_2_dummy_fr": ("nns.categorical", "factor_2_dummy_fr"),
    "fsd": ("nns.stochastic_dominance", "fsd"),
    "fsd_uni": ("nns.stochastic_dominance", "fsd_uni"),
    "kurt_pm": ("nns.classical", "kurt_pm"),
    "lpm": ("nns.core", "lpm"),
    "lpm_ratio": ("nns.core", "lpm_ratio"),
    "lpm_var": ("nns.var", "lpm_var"),
    "mean_pm": ("nns.classical", "mean_pm"),
    "nns_anova": ("nns.anova", "nns_anova"),
    "nns_arma": ("nns.arma", "nns_arma"),
    "nns_arma_optim": ("nns.arma", "nns_arma_optim"),
    "nns_boost": ("nns.boost", "nns_boost"),
    "nns_causation": ("nns.causation", "nns_causation"),
    "nns_cdf": ("nns.cdf", "nns_cdf"),
    "nns_copula": ("nns.copula", "nns_copula"),
    "nns_cor": ("nns.dependence", "nns_cor"),
    "nns_dep": ("nns.dependence", "nns_dep"),
    "nns_diff": ("nns.diff", "nns_diff"),
    "nns_distance": ("nns.distance", "nns_distance"),
    "nns_distance_bulk": ("nns.distance", "nns_distance_bulk"),
    "nns_gravity": ("nns.central_tendencies", "nns_gravity"),
    "nns_mode": ("nns.central_tendencies", "nns_mode"),
    "nns_moments": ("nns.classical", "nns_moments"),
    "nns_m_reg": ("nns.multivariate_regression", "nns_m_reg"),
    "nns_mc": ("nns.mc", "nns_mc"),
    "nns_meboot": ("nns.meboot", "nns_meboot"),
    "nns_norm": ("nns.norm", "nns_norm"),
    "nns_nowcast_panel": ("nns.nowcast", "nns_nowcast_panel"),
    "nns_part": ("nns.part", "nns_part"),
    "nns_reg": ("nns.regression", "nns_reg"),
    "nns_rescale": ("nns.central_tendencies", "nns_rescale"),
    "nns_seas": ("nns.seasonality", "nns_seas"),
    "nns_sd_cluster": ("nns.stochastic_dominance", "nns_sd_cluster"),
    "nns_stack": ("nns.stack", "nns_stack"),
    "nns_ss": ("nns.stochastic_superiority", "nns_ss"),
    "nns_var": ("nns.var", "nns_var"),
    "prepare_factor_predictors": ("nns.regression", "prepare_factor_predictors"),
    "sd_efficient_set": ("nns.stochastic_dominance", "sd_efficient_set"),
    "skew_pm": ("nns.classical", "skew_pm"),
    "ssd": ("nns.stochastic_dominance", "ssd"),
    "ssd_uni": ("nns.stochastic_dominance", "ssd_uni"),
    "tsd": ("nns.stochastic_dominance", "tsd"),
    "tsd_uni": ("nns.stochastic_dominance", "tsd_uni"),
    "upm": ("nns.core", "upm"),
    "upm_ratio": ("nns.core", "upm_ratio"),
    "upm_var": ("nns.var", "upm_var"),
    "var_pm": ("nns.classical", "var_pm"),
}

__all__ = sorted((*_EXPORTS, "pm_matrix"))


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module 'nns' has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    from importlib import import_module

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
