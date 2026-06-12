from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pynns.meboot import nns_meboot


def nns_mc(
    x: np.ndarray,
    reps: int = 30,
    lower_rho: float = -1.0,
    upper_rho: float = 1.0,
    by: float = 0.01,
    exp: float = 1.0,
    type: str = "spearman",
    drift: bool = True,
    target_drift: float | None = None,
    target_drift_scale: float | None = None,
    xmin: float | None = None,
    xmax: float | None = None,
    random_seed: int | None = None,
    **kwargs: Any,
) -> dict[str, object]:
    """Monte Carlo sampling over NNS.meboot's rho space."""
    exp_rhos = _generate_mc_rhos(lower_rho, upper_rho, by, exp)
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
    neg_rhos = np.abs(rhos[rhos <= 0.0])
    pos_rhos = rhos[rhos > 0.0]
    exp_rhos = np.concatenate((-(neg_rhos**exp), pos_rhos ** (1.0 / exp)))[::-1]
    return np.asarray(exp_rhos, dtype=np.float64)


def _r_seq(start: float, stop: float, step: float) -> NDArray[np.float64]:
    span = stop - start
    if span == 0.0:
        return np.array([start], dtype=np.float64)
    if span * step < 0.0:
        return np.array([], dtype=np.float64)
    count = int(np.floor(span / step + 1e-12)) + 1
    values = start + step * np.arange(count, dtype=np.float64)
    if values.size and ((step > 0.0 and values[-1] > stop) or (step < 0.0 and values[-1] < stop)):
        values = values[:-1]
    return values


def _format_r_number(value: float) -> str:
    if value == 0.0:
        value = 0.0
    return f"{value:.15g}"
