"""R 14.0 directional association (NNS.caus / Uni.caus).

Ground truth: NNS@7b07a837 R/Uni_Causation.R, R/Causal_matrix.R.
Each pair is ranked to its own empirical lower-tail copula coordinate;
the raw scores combine variance explained within equal-frequency cells and
unfloored mean-of-partition-level dependence (not NNS.dep's maximum).
"""
from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from nns._dep_r14 import dep_levels_r14
from nns._rmean import r_mean, r_sum

Array = NDArray[np.float64]


def _ranks(v: Array) -> Array:
    """R LPM.ratio(0, v, v): inclusive empirical CDF of every observation."""
    sorted_values = np.sort(v)
    return np.searchsorted(sorted_values, v, side="right").astype(np.float64) / v.size


def _variance(v: Array) -> float:
    """R .nns_caus_dev: (LPM(2, mean(v), v) + UPM(2, mean(v), v)) * m / (m - 1).

    Evaluated as R does: R's mean(), then each partial moment as LPM_C/UPM_C
    accumulate it (sequential double sums of the squared deviations on its
    side of the mean, divided by m), then the factor applied left to right.
    """
    m = int(v.size)
    if m < 2:
        return float("nan")
    mu = r_mean(v)
    below = mu - v
    above = v - mu
    lpm = float(np.cumsum(np.where(below >= 0.0, below * below, 0.0))[-1]) / m
    upm = float(np.cumsum(np.where(above > 0.0, above * above, 0.0))[-1]) / m
    return (lpm + upm) * m / (m - 1)


def _explained(cause_u: Array, effect_u: Array) -> float:
    """R .nns_caus_P: conditional share of effect variance resolved by cause."""
    n = int(cause_u.size)
    if n < 3:
        return 0.0
    groups = np.maximum(1, np.ceil(cause_u * max(2, math.floor(math.sqrt(n))))).astype(int)
    total = _variance(effect_u)
    if not math.isfinite(total) or total <= 0.0:
        return 0.0
    weighted: list[float] = []
    denominator = 0
    for group in np.unique(groups):  # split(): cells in increasing order
        sample = effect_u[groups == group]
        d = _variance(sample)
        if math.isfinite(d):
            weighted.append(sample.size * d)
            denominator += sample.size
    if denominator == 0:
        return 0.0
    # R: sum(w[ok] * d[ok]) / sum(w[ok]), with sum()'s long-double accumulator.
    within = r_sum(np.asarray(weighted, dtype=np.float64)) / denominator
    return min(1.0, max(0.0, 1.0 - within / total))


def _rho(cause_u: Array, effect_u: Array) -> float:
    """R .nns_caus_rho: average, not maximum, of unfloored NNS.dep levels."""
    try:
        value = r_mean(np.asarray(dep_levels_r14(cause_u, effect_u), dtype=np.float64))
        return value if math.isfinite(value) else 0.0
    except (ValueError, ArithmeticError, OverflowError):
        # Mirrors R's tryCatch(error=function(e) 0).
        return 0.0


def raw_scores(x: Array, y: Array, tau: int = 0) -> tuple[float, float]:
    """R .nns_caus_raw: paired, equally lagged directional raw scores."""
    n = int(x.size)
    k = n - tau
    x_lag, y_lag = _ranks(x[:k]), _ranks(y[:k])
    x_lead, y_lead = _ranks(x[tau:]), _ranks(y[tau:])
    p_xy = _explained(x_lag, y_lead)
    p_yx = _explained(y_lag, x_lead)
    rho_xy = _rho(x_lag, y_lead)
    rho_yx = _rho(y_lag, x_lead)
    raw_xy = (p_xy + max(rho_xy - rho_yx, 0.0)) / 2.0
    raw_yx = (p_yx + max(rho_yx - rho_xy, 0.0)) / 2.0
    return raw_xy, raw_yx


def index(x: Array, y: Array, tau: int = 0) -> tuple[float, float]:
    """R .nns_caus_index: complementary directional indices, neutral 0.5."""
    xy, yx = raw_scores(x, y, tau)
    k = 1.0 / math.sqrt(float(x.size - tau))
    forward = (xy + k) / (xy + yx + 2.0 * k)
    return forward, 1.0 - forward


def predictor_weight(response: Array, predictor: Array) -> float:
    """R Uni.caus(response, predictor, tau): aligned raw predictor->response."""
    n = min(response.size, predictor.size)
    return raw_scores(predictor[:n], response[:n], 0)[0]
