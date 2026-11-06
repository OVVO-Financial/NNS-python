from __future__ import annotations

import math
import numbers
from typing import Any

import numpy as np
from numpy.typing import NDArray

from nns._caus_r14 import _rho, index, predictor_weight
from nns._helpers import _as_matrix, _as_pair
from nns.dependence import _column_labels, _is_constant, _labelled_matrix
from nns.seasonality import nns_seas

CausationResult = dict[str, float]


def nns_causation(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    tau: int | str = 0,
) -> CausationResult:
    """R 14.0 NNS.caus: two complementary directional association indices.

    The index measures lagged association, not causal identification.
    The return fields are C(x--->y) and C(y--->x), sum to one, and
    have neutral value 0.5. R 13.3's three-component log ratio is obsolete.
    """
    x_values, y_values = _as_pair(x, y)
    if _is_constant(x_values) or _is_constant(y_values):
        raise ValueError("x and y must not be constant: causation is undefined for a variable with no variation.")
    lag = _ts_tau_lag(x_values, y_values) if tau == "ts" else _tau_value(tau, x_values.size)
    xy, yx = index(x_values, y_values, lag)
    return {"C(x--->y)": float(xy), "C(y--->x)": float(yx)}


def causal_matrix(x: NDArray[np.float64], tau: int | str = 0) -> Any:
    """R14 NNS.caus.matrix, with C(i->j) + C(j->i)=1 off diagonal."""
    labels = _column_labels(x)
    values = _as_matrix(x, "x")
    if not (isinstance(tau, str) and tau == "ts"):
        _tau_value(tau, values.shape[0])
    nvars = values.shape[1]
    causes = np.zeros((nvars, nvars), dtype=np.float64)
    for i in range(nvars - 1):
        for j in range(i + 1, nvars):
            directions = nns_causation(values[:, i], values[:, j], tau=tau)
            causes[i, j] = directions["C(x--->y)"]
            causes[j, i] = directions["C(y--->x)"]
    causes[~np.isfinite(causes)] = 0.0
    return _labelled_matrix(causes, labels)


def _uni_caus(x: NDArray[np.float64], y: NDArray[np.float64], tau: int) -> float:
    """R Uni.caus(response=x, predictor=y, tau): aligned raw predictor weight.

    R14 explicitly uses tau=0 for this *regression* coefficient, because it
    must describe the aligned observations fitted by NNS.reg / NNS.stack.
    """
    return predictor_weight(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))


def _asym_dep(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """R .nns_caus_rho: mean of unfloored dependence levels of y on x."""
    return _rho(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))


def _asym_dep_python(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """Same R14 directional level average, without an obsolete native path."""
    return _asym_dep(x, y)


def _tau_value(tau: object, n: int | None = None) -> int:
    """Validate a non-"ts" tau like R NNS.caus: "cs" or an integer in [0, n - 3]."""
    if isinstance(tau, str) and tau == "cs":
        return 0
    message = 'tau must be "cs", "ts" or a single non-negative integer smaller than len(x) - 2.'
    if isinstance(tau, bool) or not isinstance(tau, numbers.Real):
        raise ValueError(message)
    tau_float = float(tau)
    if not math.isfinite(tau_float) or tau_float < 0 or tau_float != round(tau_float):
        raise ValueError(message)
    if n is not None and tau_float >= n - 2:
        raise ValueError(message)
    return int(tau_float)



def _ts_tau_lag(x: NDArray[np.float64], y: NDArray[np.float64]) -> int:
    """R .nns_caus_ts_lag: one common rounded mean of eligible periods."""
    limit = math.sqrt(float(x.size))
    px = _first_period_at_or_below_limit(x, limit)
    py = _first_period_at_or_below_limit(y, limit)
    return min(int(round((px + py) / 2.0)), int(x.size) - 3)


def _first_period_at_or_below_limit(values: NDArray[np.float64], limit: float) -> int:
    periods = np.asarray(nns_seas(values, plot=False)["periods"], dtype=np.int64)
    eligible = periods[periods <= limit]
    if eligible.size == 0:
        raise ValueError(
            'tau="ts": NNS.seas found no seasonal period <= sqrt(len(x)); supply a numeric tau instead.'
        )
    return int(eligible[0])
