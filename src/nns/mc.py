from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
from numpy.typing import NDArray

from nns.meboot import nns_meboot


def nns_mc(
    x: np.ndarray,
    reps: int = 30,
    lower_rho: float = -1.0,
    upper_rho: float = 1.0,
    by: float = 0.01,
    exp: float = 1.0,
    type: str = "spearman",
    drift: bool = True,
    target_drift: float | list[float] | np.ndarray | None = None,
    target_drift_scale: float | list[float] | np.ndarray | None = None,
    xmin: float | None = None,
    xmax: float | None = None,
    random_seed: int | None = None,
    **kwargs: Any,
) -> dict[str, object]:
    """Monte Carlo sampling over NNS.meboot's rho space.

    A vector ``target_drift`` / ``target_drift_scale`` is recycled along the rho grid
    as in R (``mapply``); it cannot be longer than the grid. As in R, a given
    ``target_drift`` takes precedence and ``target_drift_scale`` is then ignored.
    """
    exp_rhos = _generate_mc_rhos(lower_rho, upper_rho, by, exp)
    if target_drift is not None:
        target_drift_scale = None
    for name, value in (("target_drift", target_drift), ("target_drift_scale", target_drift_scale)):
        if value is not None and np.asarray(value).size > exp_rhos.size:
            raise ValueError(f"{name} must not have more elements than the rho grid.")
    meboot_result = nns_meboot(
        x=np.asarray(x, dtype=np.float64),
        reps=reps,
        rho=exp_rhos,
        type=type,
        drift=drift,
        target_drift=target_drift,
        target_drift_scale=target_drift_scale,
        xmin=xmin,
        xmax=xmax,
        random_seed=random_seed,
        **kwargs,
    )

    if isinstance(meboot_result, dict):
        result_list = [meboot_result]
    else:
        result_list = meboot_result

    replicates: OrderedDict[str, NDArray[np.float64]] = OrderedDict()
    matrices: list[NDArray[np.float64]] = []
    for rho_value, result in zip(exp_rhos, result_list, strict=True):
        matrix = np.asarray(result["replicates"], dtype=np.float64)
        replicates[f"rho = {_format_r_number(rho_value)}"] = matrix
        matrices.append(matrix)

    if not matrices:
        raise ValueError("rho grid must contain at least one value.")

    ensemble = np.mean(np.column_stack(matrices), axis=1)
    return {"ensemble": ensemble, "replicates": replicates}


def _generate_mc_rhos(
    lower_rho: float,
    upper_rho: float,
    by: float,
    exp: float,
) -> NDArray[np.float64]:
    if by == 0.0:
        raise ValueError("'by' must be non-zero.")
    rhos = _r_seq(lower_rho, upper_rho, by)
    # Warp each side of 0 (or of the bound nearest 0) within its own span, so the grid
    # stays inside the bounds.
    if exp != 1.0:
        pivot = min(max(0.0, lower_rho), upper_rho)
        low = rhos <= pivot
        if pivot > lower_rho:
            w = ((pivot - rhos[low]) / (pivot - lower_rho)) ** exp
            rhos[low] = (1.0 - w) * pivot + w * lower_rho
        if upper_rho > pivot:
            w = ((rhos[~low] - pivot) / (upper_rho - pivot)) ** (1.0 / exp)
            rhos[~low] = (1.0 - w) * pivot + w * upper_rho
    return np.asarray(rhos[::-1], dtype=np.float64)


def _r_seq(start: float, stop: float, step: float) -> NDArray[np.float64]:
    span = stop - start
    if span == 0.0:
        return np.array([start], dtype=np.float64)
    if span * step < 0.0:
        return np.array([], dtype=np.float64)
    # R's seq.default: a fuzz of 1e-10 on the count, and the end clamped to stop.
    count = int(np.floor(span / step + 1e-10)) + 1
    values = start + step * np.arange(count, dtype=np.float64)
    return np.minimum(values, stop) if step > 0.0 else np.maximum(values, stop)


def _format_r_number(value: float) -> str:
    if value == 0.0:
        value = 0.0
    return f"{value:.15g}"
