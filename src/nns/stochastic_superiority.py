from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from nns._native import nnscore
from nns.meboot import nns_meboot
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
    """Stochastic superiority matching R's NNS.SS."""
    x_values = _omit_nan_numeric(x)
    y_values = _omit_nan_numeric(y)
    if x_values.size == 0 or y_values.size == 0:
        raise ValueError("x and y must both contain at least one non-missing value.")
    if not isinstance(confidence_interval, bool):
        raise ValueError("confidence_interval must be a single TRUE/FALSE value.")

    empirical = _stoch_superiority(x_values, y_values)
    if not confidence_interval:
        return dict(empirical)

    if reps < 2:
        raise ValueError("reps must be a single number >= 2.")
    if ci <= 0.0 or ci >= 1.0:
        raise ValueError("ci must be a single number in (0, 1).")

    x_seed: int | None = None
    y_seed: int | None = None
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
        x_seed, y_seed = [int(seed) for seed in rng.integers(0, np.iinfo(np.int32).max, size=2)]

    x_boot = nns_meboot(x_values, reps=reps, rho=rho, random_seed=x_seed)
    y_boot = nns_meboot(y_values, reps=reps, rho=rho, random_seed=y_seed)
    if not isinstance(x_boot, dict) or not isinstance(y_boot, dict):
        raise ValueError("NNS.meboot returned an unexpected vectorized result.")

    x_reps = _replicate_matrix(x_boot)
    y_reps = _replicate_matrix(y_boot)
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

    native = nnscore()
    if native is not None and hasattr(native, "stochastic_superiority"):
        return dict(
            native.stochastic_superiority(np.ascontiguousarray(xs), np.ascontiguousarray(ys))
        )

    xs = np.sort(xs)
    ys = np.sort(ys)
    left = np.searchsorted(ys, xs, side="left")
    right = np.searchsorted(ys, xs, side="right")
    less_count = int(np.sum(left))
    tie_count = int(np.sum(right - left))

    denominator = float(xs.size * ys.size)
    p_gt = float(less_count / denominator)
    p_tie = float(tie_count / denominator)
    return {
        "p_gt": p_gt,
        "p_tie": p_tie,
        "p_star": p_gt + 0.5 * p_tie,
    }


def _omit_nan_numeric(x: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(x, dtype=np.float64).reshape(-1)
    return np.asarray(values[~np.isnan(values)], dtype=np.float64)


def _replicate_matrix(result: dict[str, Any]) -> NDArray[np.float64]:
    replicates = np.asarray(result.get("replicates"), dtype=np.float64)
    if replicates.ndim != 2:
        raise ValueError("NNS.meboot result does not contain a replicate matrix.")
    return replicates
