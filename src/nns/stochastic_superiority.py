from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from nns._native import native_fn
from nns._rrng import RRNG, r_rng
from nns.meboot import _nns_meboot
from nns.var import lpm_var, upm_var


def nns_ss(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    confidence_interval: bool = False,
    reps: int = 999,
    ci: float = 0.95,
    rho: float = 1.0,
    random_seed: int | None = None,
) -> dict[str, object]:
    """Stochastic superiority matching R's NNS.SS.

    With ``confidence_interval=True`` the bounds come from ``reps`` maximum-entropy
    bootstrap replicates of each group; with ``random_seed = s`` they are R's
    after ``set.seed(s)``, the replicates of ``x`` and then of ``y`` drawn from
    one stream as R draws them.
    """
    x_values = _omit_nan_numeric(x)
    y_values = _omit_nan_numeric(y)
    if x_values.size == 0 or y_values.size == 0:
        raise ValueError("x and y must both contain at least one non-missing value.")
    if not isinstance(confidence_interval, (bool, np.bool_)):
        raise ValueError("confidence_interval must be a single TRUE/FALSE value.")

    empirical = _stoch_superiority(x_values, y_values)
    if not confidence_interval:
        return dict(empirical)

    if reps < 2:
        raise ValueError("reps must be a single number >= 2.")
    if ci <= 0.0 or ci >= 1.0:
        raise ValueError("ci must be a single number in (0, 1).")

    rng = r_rng(random_seed)
    x_reps = _bootstrap(x_values, int(reps), rho, rng)
    y_reps = _bootstrap(y_values, int(reps), rho, rng)
    boot_vals = np.empty(int(reps), dtype=np.float64)
    for index in range(int(reps)):
        boot_vals[index] = _stoch_superiority(x_reps[:, index], y_reps[:, index])["p_star"]

    alpha = (1.0 - float(ci)) / 2.0
    return {
        **empirical,
        "lower": lpm_var(alpha, 0.0, boot_vals),
        "upper": upm_var(alpha, 0.0, boot_vals),
        "ci": float(ci),
        "reps": int(reps),
        "boot_vals": boot_vals,
    }


def _stoch_superiority(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> dict[str, float]:
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    if xs.size == 0 or ys.size == 0:
        raise ValueError("x and y must both have positive length.")

    native_ss = native_fn("stochastic_superiority")
    if native_ss is not None:
        return dict(native_ss(np.ascontiguousarray(xs), np.ascontiguousarray(ys)))

    xs = np.sort(xs)
    ys = np.sort(ys)
    left = np.searchsorted(ys, xs, side="left")
    right = np.searchsorted(ys, xs, side="right")
    less_count = int(np.sum(left))
    tie_count = int(np.sum(right - left))

    # R divides the counts in long double before rounding to double.
    denominator = np.longdouble(xs.size) * np.longdouble(ys.size)
    p_gt = float(np.longdouble(less_count) / denominator)
    p_tie = float(np.longdouble(tie_count) / denominator)
    return {
        "p_gt": p_gt,
        "p_tie": p_tie,
        "p_star": p_gt + 0.5 * p_tie,
    }


def _omit_nan_numeric(x: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(x, dtype=np.float64).reshape(-1)
    return np.asarray(values[~np.isnan(values)], dtype=np.float64)


def _bootstrap(
    values: NDArray[np.float64], reps: int, rho: float | NDArray[np.float64], rng: RRNG
) -> NDArray[np.float64]:
    """R: ``NNS.meboot(x = g, reps = reps, rho = rho)["replicates", ]$replicates``."""
    # A constant group's bootstrap is the constant itself.
    if np.all(values == values[0]):
        return np.tile(values[:, np.newaxis], (1, reps))
    boot = _nns_meboot(
        values, reps, rho, "spearman", True, None, None, 0.10, None, None,
        True, True, True, False, False, 6, True, rng,
    )
    # With several rho values every call draws, and $replicates takes the first.
    result = boot[0] if isinstance(boot, list) else boot
    replicates = np.asarray(result.get("replicates"), dtype=np.float64)
    if replicates.ndim != 2:
        raise ValueError("NNS.meboot result does not contain a replicate matrix.")
    return replicates
