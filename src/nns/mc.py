from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any

import numpy as np
from numpy.typing import NDArray

from nns._rmean import r_pow, r_row_means, r_seq_by
from nns._rrng import r_rng
from nns.meboot import _nns_meboot


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
    """Monte Carlo sampling over NNS.meboot's rho space, matching R's NNS.MC.

    With ``random_seed = s`` the replicates are R's after ``set.seed(s)`` (see
    :func:`nns.nns_meboot`). A vector ``target_drift`` / ``target_drift_scale`` is
    recycled along the rho grid as in R (``mapply``); it cannot be longer than the
    grid. As in R, a given ``target_drift`` takes precedence and
    ``target_drift_scale`` is then ignored. Extra keyword arguments go to
    ``nns_meboot``.
    """
    if not _scalar_finite(exp) or float(exp) <= 0.0:
        raise ValueError("exp must be a single finite number > 0.")
    if (
        not _scalar_finite(lower_rho)
        or not _scalar_finite(upper_rho)
        or float(lower_rho) < -1.0
        or float(upper_rho) > 1.0
        or float(lower_rho) > float(upper_rho)
    ):
        raise ValueError("lower_rho and upper_rho must satisfy -1 <= lower_rho <= upper_rho <= 1.")
    if not _scalar_finite(by) or float(by) <= 0.0:
        raise ValueError("by must be a single finite number > 0.")

    exp_rhos = _generate_mc_rhos(float(lower_rho), float(upper_rho), float(by), float(exp))
    if target_drift is not None:
        target_drift_scale = None
    for name, value in (("target_drift", target_drift), ("target_drift_scale", target_drift_scale)):
        if value is not None and np.asarray(value).size > exp_rhos.size:
            raise ValueError(f"{name} must not have more elements than the rho grid.")
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("x must be a 1D numeric vector.")
    meboot_args: dict[str, Any] = {
        "trim": 0.10,
        "reachbnd": True,
        "expand_sd": True,
        "force_clt": True,
        "scl_adjustment": False,
        "sym": False,
        "digits": 6,
        "expand_residual_sd": True,
    }
    kwargs.pop("elaps", None)
    unknown = set(kwargs) - set(meboot_args)
    if unknown:
        raise TypeError(f"unexpected NNS.meboot argument(s): {', '.join(sorted(unknown))}")
    meboot_args.update(kwargs)
    meboot_result = _nns_meboot(
        values,
        reps,
        exp_rhos,
        type,
        True if target_drift is not None or target_drift_scale is not None else drift,
        target_drift,
        target_drift_scale,
        meboot_args["trim"],
        xmin,
        xmax,
        meboot_args["reachbnd"],
        meboot_args["expand_sd"],
        meboot_args["force_clt"],
        meboot_args["scl_adjustment"],
        meboot_args["sym"],
        meboot_args["digits"],
        meboot_args["expand_residual_sd"],
        r_rng(random_seed),
    )

    result_list = [meboot_result] if isinstance(meboot_result, dict) else meboot_result
    replicates: OrderedDict[str, NDArray[np.float64]] = OrderedDict()
    matrices: list[NDArray[np.float64]] = []
    for rho_value, result in zip(exp_rhos, result_list, strict=True):
        matrix = np.asarray(result["replicates"], dtype=np.float64)
        replicates[f"rho = {_format_r_number(float(rho_value))}"] = matrix
        matrices.append(matrix)

    if not matrices:
        raise ValueError("rho grid must contain at least one value.")

    # R: rowMeans(do.call(cbind, replicates)), accumulated in long double.
    ensemble = r_row_means(np.column_stack(matrices))
    return {"ensemble": ensemble, "replicates": replicates}


def _scalar_finite(value: object) -> bool:
    """R's ``is.numeric(z) && length(z) == 1L && is.finite(z)``."""
    if isinstance(value, (bool, np.bool_)):
        return False
    array = np.asarray(value)
    if array.size != 1 or not np.issubdtype(array.dtype, np.number):
        return False
    return bool(np.isfinite(array.reshape(-1)[0]))


def _generate_mc_rhos(
    lower_rho: float,
    upper_rho: float,
    by: float,
    exp: float,
) -> NDArray[np.float64]:
    rhos = np.array(r_seq_by(lower_rho, upper_rho, by), dtype=np.float64)
    # Warp each side of 0 (or of the bound nearest 0) within its own span, so the grid
    # stays inside the bounds.
    if exp != 1.0:
        pivot = min(max(0.0, lower_rho), upper_rho)
        low = rhos <= pivot
        if pivot > lower_rho:
            w = r_pow((pivot - rhos[low]) / (pivot - lower_rho), exp)
            rhos[low] = (1.0 - w) * pivot + w * lower_rho
        if upper_rho > pivot:
            w = r_pow((rhos[~low] - pivot) / (upper_rho - pivot), 1.0 / exp)
            rhos[~low] = (1.0 - w) * pivot + w * upper_rho
    return np.asarray(rhos[::-1], dtype=np.float64)


def _format_r_number(value: float) -> str:
    """R's ``as.character()`` of a double: 15 significant digits, in fixed or
    scientific notation, whichever is narrower (fixed on a tie)."""
    if value == 0.0:
        return "0"
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Inf" if value > 0 else "-Inf"
    mantissa, exponent_text = f"{value:.14e}".split("e")
    exponent = int(exponent_text)
    digits = mantissa.lstrip("-").replace(".", "").rstrip("0")
    nsig = max(1, len(digits))
    neg = 1 if value < 0 else 0
    rgt = max(0, nsig - exponent - 1)
    width_fixed = neg + (exponent + 1 if exponent >= 0 else 1) + (rgt + 1 if rgt > 0 else 0)
    width_sci = neg + (nsig + 1 if nsig > 1 else 1) + (4 if abs(exponent) < 100 else 5)
    if width_fixed <= width_sci:
        return f"{value:.{rgt}f}"
    sign = "+" if exponent >= 0 else "-"
    return f"{float(mantissa):.{nsig - 1}f}e{sign}{abs(exponent):02d}"
