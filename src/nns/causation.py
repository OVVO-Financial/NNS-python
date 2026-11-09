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


_PERMUTE_CHOICES = ("auto", "y", "x", "both", "shift")


def nns_causation(
    x: Any,
    y: NDArray[np.float64] | None = None,
    tau: int | str = 0,
    *,
    p_value: bool = False,
    nperm: int = 100,
    permute: str = "auto",
    seed: int | None = None,
    conf_int: float = 0.95,
) -> Any:
    """R 14.0 NNS.caus: two complementary directional association indices.

    The index measures lagged association, not causal identification.
    The return fields are C(x--->y) and C(y--->x), sum to one, and
    have neutral value 0.5. R 13.3's three-component log ratio is obsolete.
    With ``y=None`` and a matrix-like ``x`` this is :func:`causal_matrix`, as in R.

    ``p_value=True`` adds R's permutation test of the signed index
    ``C(x--->y) - C(y--->x)`` and returns ``{"causation": ..., "p.value": {...}}``
    with R's fields (``two.sided``, ``x_causes_y``, ``y_causes_x``,
    ``null_distribution``, ``observed_signed``, ``permute``, ``nperm``,
    ``lower_CI``, ``upper_CI``). ``permute="auto"`` circularly shifts ``y`` for a
    positive lag (the series keep their own dynamics) and shuffles it otherwise;
    ``"y"``, ``"x"``, ``"both"`` and ``"shift"`` choose explicitly. Every
    permutation is scored at the lag resolved on the observed series. With
    ``seed = s`` the draws are R's after ``set.seed(s)``.
    """
    permute = _match_permute(permute)
    if y is None:
        return causal_matrix(
            x, tau=tau, p_value=p_value, nperm=nperm, conf_int=conf_int, seed=seed,
            permute=permute,
        )
    cp = _caus_core(x, y, tau)
    if not p_value:
        return cp
    return _caus_p_value(cp, x, y, tau, int(nperm), permute, seed, conf_int)


def _match_permute(permute: str) -> str:
    """R's ``match.arg(permute)``: an exact choice or a unique prefix of one."""
    if permute in _PERMUTE_CHOICES:
        return permute
    matches = [choice for choice in _PERMUTE_CHOICES if choice.startswith(str(permute))]
    if len(matches) != 1:
        choices = ", ".join(f"'{choice}'" for choice in _PERMUTE_CHOICES)
        raise ValueError(f"'arg' should be one of {choices}")
    return matches[0]


def _caus_p_value(
    cp: CausationResult,
    x: Any,
    y: Any,
    tau: int | str,
    nperm: int,
    permute: str,
    seed: int | None,
    conf_int: float,
) -> dict[str, Any]:
    """The permutation null of R ``NNS.caus(p.value = TRUE)``."""
    from nns._rrng import r_rng
    from nns.var import lpm_var, upm_var

    rng = r_rng(seed)
    x_values, y_values = _as_pair(x, y)
    observed = float(cp["C(x--->y)"] - cp["C(y--->x)"])
    # One numeric lag for the observed statistic and every permutation (#96).
    if isinstance(tau, str) and tau == "ts":
        lag = _ts_tau_lag(x_values, y_values)
    elif isinstance(tau, str) and tau == "cs":
        lag = 0
    else:
        lag = int(tau)
    if permute == "auto":
        permute = "shift" if lag > 0 else "y"
    offsets: list[int] = []
    if permute == "shift":
        offsets = _shift_draw(y_values.size, lag, nperm, rng)
        nperm = len(offsets)
    null = np.empty(nperm, dtype=np.float64)
    for b in range(nperm):
        x_perm, y_perm = x_values, y_values
        if permute == "y":
            y_perm = rng.sample(y_values, y_values.size)
        if permute == "x":
            x_perm = rng.sample(x_values, x_values.size)
        if permute == "both":
            x_perm = rng.sample(x_values, x_values.size)
            y_perm = rng.sample(y_values, y_values.size)
        if permute == "shift":
            y_perm = _shift(y_values, offsets[b])
        cp_perm = _caus_core(x_perm, y_perm, lag)
        null[b] = cp_perm["C(x--->y)"] - cp_perm["C(y--->x)"]
    alpha = (1.0 - conf_int) / 2.0
    lower = float(lpm_var(alpha, 0.0, null))
    upper = float(upm_var(alpha, 0.0, null))
    return {
        "causation": cp,
        "p.value": {
            "two.sided": (1 + int(np.count_nonzero(np.abs(null) >= abs(observed)))) / (1 + nperm),
            "x_causes_y": (1 + int(np.count_nonzero(null >= observed))) / (1 + nperm),
            "y_causes_x": (1 + int(np.count_nonzero(null <= observed))) / (1 + nperm),
            "null_distribution": null,
            "observed_signed": observed,
            "permute": permute,
            "nperm": nperm,
            "lower_CI": 0.0 if math.isnan(lower) else lower,
            "upper_CI": 0.0 if math.isnan(upper) else upper,
        },
    }


def _shift_offsets(n: int, tau: int) -> list[int]:
    """R ``.nns_caus_shift_offsets``: shifts at least max(tau + 1, n / 10) from alignment."""
    m = max(int(tau) + 1, math.ceil(n / 10))
    if n - 2 * m >= 0:
        return list(range(m, n - m + 1))
    return list(range(1, n))


def _shift_draw(n: int, tau: int, nperm: int, rng: Any) -> list[int]:
    """R ``.nns_caus_shift_draw``: distinct offsets, ``sample.int``-drawn when too many."""
    offsets = _shift_offsets(n, tau)
    if len(offsets) <= nperm:
        return offsets
    return [offsets[int(i) - 1] for i in rng.sample_int(len(offsets), nperm)]


def _shift(values: NDArray[np.float64], k: int) -> NDArray[np.float64]:
    """R ``.nns_caus_shift``: ``c(v[(k + 1):n], v[1:k])``."""
    return np.concatenate((values[k:], values[:k]))


def _caus_core(x: Any, y: Any, tau: int | str) -> CausationResult:
    """R ``NNS.caus_core`` for a pair of variables."""
    x_values, y_values = _as_pair(x, y)
    if _is_constant(x_values) or _is_constant(y_values):
        raise ValueError(
            "x and y must not be constant: causation is undefined for a variable with no variation."
        )
    lag = _ts_tau_lag(x_values, y_values) if tau == "ts" else _tau_value(tau, x_values.size)
    xy, yx = index(x_values, y_values, lag)
    return {"C(x--->y)": float(xy), "C(y--->x)": float(yx)}


def causal_matrix(
    x: NDArray[np.float64],
    tau: int | str = 0,
    *,
    p_value: bool = False,
    nperm: int = 100,
    conf_int: float = 0.95,
    seed: int | None = None,
    permute: str = "auto",
) -> Any:
    """R14 NNS.caus.matrix, with C(i->j) + C(j->i)=1 off diagonal.

    ``p_value=True`` returns R's ``{"causality", "lower_CI", "upper_CI",
    "p.value_matrix"}``: each index's permutation null at the pair's observed
    lag, its partial-moment interval and the two-sided p-value of
    ``C(i--->j) - C(j--->i)``. ``permute="auto"`` circularly shifts the second
    column of each pair for time series (a positive lag) and shuffles every
    column otherwise; ``"x"``, ``"y"`` and ``"shift"`` permute within each pair.
    With ``seed = s`` the draws are R's after ``set.seed(s)``.
    """
    permute = _match_permute(permute)
    labels = _column_labels(x)
    values = _as_matrix(x, "x")
    if not (isinstance(tau, str) and tau == "ts"):
        _tau_value(tau, values.shape[0])
    nvars = values.shape[1]
    causes = np.zeros((nvars, nvars), dtype=np.float64)
    for i in range(nvars - 1):
        for j in range(i + 1, nvars):
            directions = _caus_core(values[:, i], values[:, j], tau)
            causes[i, j] = directions["C(x--->y)"]
            causes[j, i] = directions["C(y--->x)"]
    causes[~np.isfinite(causes)] = 0.0
    if not p_value:
        return _labelled_matrix(causes, labels)
    return _causal_matrix_p_value(
        values, causes, labels, tau, int(nperm), conf_int, seed, permute
    )


def _causal_matrix_p_value(
    values: NDArray[np.float64],
    causes: NDArray[np.float64],
    labels: list[Any] | None,
    tau: int | str,
    nperm: int,
    conf_int: float,
    seed: int | None,
    permute: str,
) -> dict[str, Any]:
    """The permutation intervals and p-values of R ``NNS.caus.matrix(p.value = TRUE)``."""
    from nns._rrng import r_rng
    from nns.var import lpm_var, upm_var

    rng = r_rng(seed)
    nrow, nvars = values.shape
    pairs = [(i, j) for i in range(nvars - 1) for j in range(i + 1, nvars)]  # combn order
    pair_tau = []
    for i, j in pairs:
        if isinstance(tau, str) and tau == "ts":
            pair_tau.append(_ts_tau_lag(values[:, i], values[:, j]))
        elif isinstance(tau, str) and tau == "cs":
            pair_tau.append(0)
        else:
            pair_tau.append(int(tau))
    if permute == "auto":
        permute = "shift" if any(lag > 0 for lag in pair_tau) else "both"

    def null_row(xi: NDArray[np.float64], xj: NDArray[np.float64], lag: int) -> tuple[float, float]:
        cp = _caus_core(xi, xj, lag)
        return cp["C(x--->y)"], cp["C(x--->y)"] - cp["C(y--->x)"]

    null_rows: list[list[tuple[float, float]]] = [[] for _ in pairs]
    if permute != "both":
        # Permute within each pair: shifting every column independently would
        # leave pairs whose two shifts nearly coincide still aligned.
        for k, (i, j) in enumerate(pairs):
            draws = (
                _shift_draw(nrow, pair_tau[k], nperm, rng)
                if permute == "shift"
                else list(range(1, nperm + 1))
            )
            for s in draws:
                xi, xj = values[:, i], values[:, j]
                if permute == "shift":
                    xj = _shift(values[:, j], s)
                if permute == "x":
                    xi = rng.sample(values[:, i])
                if permute == "y":
                    xj = rng.sample(values[:, j])
                null_rows[k].append(null_row(xi, xj, pair_tau[k]))
    else:
        for _ in range(nperm):
            shuffled = [rng.sample(values[:, j]) for j in range(nvars)]
            for k, (i, j) in enumerate(pairs):
                null_rows[k].append(null_row(shuffled[i], shuffled[j], pair_tau[k]))

    alpha = (1.0 - conf_int) / 2.0
    lower = np.zeros((nvars, nvars), dtype=np.float64)
    upper = np.zeros((nvars, nvars), dtype=np.float64)
    p_matrix = np.zeros((nvars, nvars), dtype=np.float64)
    for k, (i, j) in enumerate(pairs):
        rows = np.asarray(null_rows[k], dtype=np.float64).reshape(-1, 2)
        null_ij = rows[:, 0]
        lower[i, j] = float(lpm_var(alpha, 0.0, null_ij))
        upper[i, j] = float(upm_var(alpha, 0.0, null_ij))
        lower[j, i] = float(lpm_var(alpha, 0.0, 1.0 - null_ij))
        upper[j, i] = float(upm_var(alpha, 0.0, 1.0 - null_ij))
        observed = causes[i, j] - causes[j, i]
        null_signed = rows[:, 1]
        p = (1 + int(np.count_nonzero(np.abs(null_signed) >= abs(observed)))) / (
            1 + null_signed.size
        )
        p_matrix[i, j] = p_matrix[j, i] = p
    for matrix in (lower, upper, p_matrix):
        np.fill_diagonal(matrix, 0.0)
        matrix[np.isnan(matrix)] = 0.0
    return {
        "causality": _labelled_matrix(causes, labels),
        "lower_CI": _labelled_matrix(lower, labels),
        "upper_CI": _labelled_matrix(upper, labels),
        "p.value_matrix": _labelled_matrix(p_matrix, labels),
    }


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
    return min(round((px + py) / 2.0), int(x.size) - 3)


def _first_period_at_or_below_limit(values: NDArray[np.float64], limit: float) -> int:
    periods = np.asarray(nns_seas(values, plot=False)["periods"], dtype=np.int64)
    eligible = periods[periods <= limit]
    if eligible.size == 0:
        raise ValueError(
            'tau="ts": NNS.seas found no seasonal period <= sqrt(len(x)); '
            "supply a numeric tau instead."
        )
    return int(eligible[0])
