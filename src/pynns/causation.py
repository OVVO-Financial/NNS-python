from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from pynns.core import lpm_ratio, upm_ratio
from pynns.dependence import (
    _as_pair,
    _copula_degree0_unsigned,
    _copula_signed,
    _directional_dep,
    _finite_or_zero,
    _gravity,
    _is_constant,
    _is_discrete_case,
    _xonly_partition,
)
from pynns.norm import nns_norm
from pynns.seasonality import nns_seas

CausationResult = dict[str, float]


def nns_causation(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    tau: int | str = 0,
) -> CausationResult:
    """Return R's default bivariate NNS.caus vector as a dict."""
    x_values, y_values = _as_pair(x, y)
    if tau == "ts":
        x_tau, y_tau = _ts_tau_values(x_values, y_values)
        causation_x_given_y = _uni_caus(x_values, y_values, y_tau)
        causation_y_given_x = _uni_caus(y_values, x_values, x_tau)
    else:
        tau_value = _tau_value(tau)
        causation_x_given_y = _uni_caus(x_values, y_values, tau_value)
        causation_y_given_x = _uni_caus(y_values, x_values, tau_value)
    if not math.isfinite(causation_x_given_y):
        causation_x_given_y = 0.0
    if not math.isfinite(causation_y_given_x):
        causation_y_given_x = 0.0

    eps = np.finfo(np.float64).eps
    result: CausationResult = {
        "Causation.x.given.y": causation_x_given_y,
        "Causation.y.given.x": causation_y_given_x,
    }
    if abs(causation_y_given_x) >= abs(causation_x_given_y):
        net = math.copysign(
            math.log((abs(causation_y_given_x) + eps) / (abs(causation_x_given_y) + eps)),
            causation_y_given_x,
        )
        result["C(x--->y)"] = _cap_inf100(net)
    else:
        net = math.copysign(
            math.log((abs(causation_x_given_y) + eps) / (abs(causation_y_given_x) + eps)),
            causation_x_given_y,
        )
        result["C(y--->x)"] = _cap_inf100(net)
    return result


def causal_matrix(
    x: NDArray[np.float64],
    tau: int | str = 0,
) -> NDArray[np.float64]:
    """Return R's NNS.caus.matrix antisymmetric net-causation matrix."""
    values = _as_matrix(x)
    if tau != "ts":
        tau = _tau_value(tau)
    n_variables = values.shape[1]
    causes = np.zeros((n_variables, n_variables), dtype=np.float64)

    for i in range(n_variables - 1):
        for j in range(i + 1, n_variables):
            cp = nns_causation(values[:, i], values[:, j], tau=tau)
            third_key = next(key for key in cp if key.startswith("C("))
            net_value = cp[third_key]
            if third_key == "C(x--->y)":
                val_ij = net_value
            elif third_key == "C(y--->x)":
                val_ij = -net_value
            else:
                val_ij = net_value
            causes[i, j] = -val_ij
            causes[j, i] = val_ij

    causes[~np.isfinite(causes)] = 0.0
    return causes


def _uni_caus(x: NDArray[np.float64], y: NDArray[np.float64], tau: int) -> float:
    x_norm_tau, y_norm_tau = _tau_normalized(x, y, tau)
    x_norm_to_y, y_norm_to_x = nns_norm(np.column_stack((x_norm_tau, y_norm_tau))).T

    p_x_given_y = 1.0 - (
        float(lpm_ratio(1.0, float(np.min(y_norm_to_x)), x_norm_to_y))
        + float(upm_ratio(1.0, float(np.max(y_norm_to_x)), x_norm_to_y))
    )

    rho_x_y = _asym_dep(y_norm_to_x, x_norm_to_y)
    rho_y_x = _asym_dep(x_norm_to_y, y_norm_to_x)
    return float(np.mean([p_x_given_y * rho_x_y, max(0.0, rho_x_y - rho_y_x)]))


def _tau_normalized(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    tau: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if tau <= 0:
        return x, y

    min_length = min(x.size, y.size)
    x_vectors = []
    y_vectors = []
    for i in range(tau + 1):
        start = tau - i
        end = min_length - i
        x_vectors.append(x[start:end])
        y_vectors.append(y[start:end])

    x_tau = np.column_stack(x_vectors)
    y_tau = np.column_stack(y_vectors)
    return nns_norm(x_tau)[:, 0], nns_norm(y_tau)[:, 0]


def _asym_dep(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    if _is_constant(x) or _is_constant(y):
        return 0.0

    obs_req = max(8, x.size // 8)
    quadrants = _xonly_partition(x, obs_req)
    global_cop = _finite_or_zero(_copula_signed(x, y))
    _, dep_xy = _directional_dep(x, y, quadrants, global_cop)

    if _is_discrete_case(x, y):
        disc_cop = _copula_degree0_unsigned(x, y)
        if not math.isfinite(disc_cop):
            disc_cop = dep_xy
        dep_xy = _gravity(np.array([dep_xy, disc_cop], dtype=np.float64))
    return dep_xy


def _tau_value(tau: int | str) -> int:
    if tau == "cs":
        return 0
    tau_value = int(tau)
    if tau_value < 0:
        raise ValueError("tau must be non-negative.")
    return tau_value


def _ts_tau_values(x: NDArray[np.float64], y: NDArray[np.float64]) -> tuple[int, int]:
    limit = math.sqrt(float(x.size))
    x_tau = _first_period_at_or_below_limit(x, limit)
    y_tau = _first_period_at_or_below_limit(y, limit)
    return x_tau, y_tau


def _first_period_at_or_below_limit(values: NDArray[np.float64], limit: float) -> int:
    periods = np.asarray(nns_seas(values, plot=False)["periods"], dtype=np.int64)
    eligible = periods[periods <= limit]
    if eligible.size == 0:
        raise ValueError("tau='ts' did not find an eligible seasonal period.")
    return int(eligible[0])


def _cap_inf100(value: float, cap: float = 100.0) -> float:
    if math.isinf(value):
        return math.copysign(cap, value)
    if abs(value) > cap:
        return math.copysign(cap, value)
    return value


def _as_matrix(x: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("x must be 2D.")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("x must be non-empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError("x must contain only finite values.")
    return values
