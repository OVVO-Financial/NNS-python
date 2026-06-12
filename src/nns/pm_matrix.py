from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from nns._native import nnscore
from nns.core import _as_degree

Target: TypeAlias = float | None | Literal["mean"] | NDArray[np.float64]
PMMatrixResult: TypeAlias = dict[str, Any]


def pm_matrix(
    lpm_degree: float,
    upm_degree: float,
    target: Target,
    variable: NDArray[np.float64],
    pop_adj: bool,
    norm: bool = False,
    names: Sequence[str] | None = None,
) -> PMMatrixResult:
    """Return the partial-moment covariance decomposition matrices.

    The numeric matrices are always plain row/column-major NumPy arrays
    (NumPy-first behavior). R's ``PM.matrix`` additionally copies the input
    data-frame's column names onto the result matrices' row/column dimnames.
    NumPy arrays do not carry dimension labels, so that naming behavior is an
    intentional divergence. When the optional ``names`` argument is supplied
    (matching the column count), the labels R would attach are echoed back under
    a ``"names"`` key so callers can build a labeled structure if they want one;
    the numeric arrays are byte-for-byte identical whether or not ``names`` is
    given.
    """
    lpm_degree = _as_degree(lpm_degree)
    upm_degree = _as_degree(upm_degree)
    values = _as_matrix(variable)
    targets = _as_target(target, values)
    resolved_names = _resolve_names(names, values.shape[1])

    observations = values.shape[0]

    native = nnscore()
    if native is not None and hasattr(native, "pm_matrix"):
        native_result = native.pm_matrix(
            lpm_degree,
            upm_degree,
            np.ascontiguousarray(targets),
            np.ascontiguousarray(np.ravel(values, order="F")),
            observations,
            values.shape[1],
            pop_adj,
            norm,
        )
        dim = int(native_result["dim"])
        result: PMMatrixResult = {
            "cupm": np.asarray(native_result["cupm"], dtype=np.float64).reshape(
                (dim, dim), order="F"
            ),
            "dupm": np.asarray(native_result["dupm"], dtype=np.float64).reshape(
                (dim, dim), order="F"
            ),
            "dlpm": np.asarray(native_result["dlpm"], dtype=np.float64).reshape(
                (dim, dim), order="F"
            ),
            "clpm": np.asarray(native_result["clpm"], dtype=np.float64).reshape(
                (dim, dim), order="F"
            ),
            "cov.matrix": np.asarray(native_result["cov.matrix"], dtype=np.float64).reshape(
                (dim, dim), order="F"
            ),
        }
        if resolved_names is not None:
            result["names"] = resolved_names
        return result

    dev_lower = _lower_deviation(values, targets, lpm_degree)
    dev_upper = _upper_deviation(values, targets, upm_degree)

    clpm = (dev_lower.T @ dev_lower) / observations
    cupm = (dev_upper.T @ dev_upper) / observations
    dlpm = (dev_upper.T @ dev_lower) / observations
    dupm = (dev_lower.T @ dev_upper) / observations

    adjust = observations / (observations - 1) if observations > 1 else 1.0
    should_adjust = pop_adj and observations > 1 and lpm_degree > 0 and upm_degree > 0
    if should_adjust:
        clpm *= adjust
        cupm *= adjust
        dlpm *= adjust
        dupm *= adjust

    if norm:
        total = cupm + dupm + dlpm + clpm
        np.divide(cupm, total, out=cupm, where=total > 0.0)
        np.divide(dupm, total, out=dupm, where=total > 0.0)
        np.divide(dlpm, total, out=dlpm, where=total > 0.0)
        np.divide(clpm, total, out=clpm, where=total > 0.0)
        cupm[total <= 0.0] = 0.0
        dupm[total <= 0.0] = 0.0
        dlpm[total <= 0.0] = 0.0
        clpm[total <= 0.0] = 0.0

    cov_matrix = cupm + clpm - dupm - dlpm
    result = {
        "cupm": cupm,
        "dupm": dupm,
        "dlpm": dlpm,
        "clpm": clpm,
        "cov.matrix": cov_matrix,
    }
    if resolved_names is not None:
        result["names"] = resolved_names
    return result


def _resolve_names(names: Sequence[str] | None, n_cols: int) -> list[str] | None:
    if names is None:
        return None
    resolved = [str(name) for name in names]
    if len(resolved) != n_cols:
        raise ValueError("names length must match the number of variable columns.")
    return resolved


def _lower_deviation(
    values: NDArray[np.float64],
    targets: NDArray[np.float64],
    degree: float,
) -> NDArray[np.float64]:
    if degree == 0:
        return (values <= targets[np.newaxis, :]).astype(np.float64)
    return np.maximum(0.0, targets[np.newaxis, :] - values) ** degree


def _upper_deviation(
    values: NDArray[np.float64],
    targets: NDArray[np.float64],
    degree: float,
) -> NDArray[np.float64]:
    if degree == 0:
        return (values > targets[np.newaxis, :]).astype(np.float64)
    return np.maximum(0.0, values - targets[np.newaxis, :]) ** degree


def _as_matrix(variable: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(variable, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("variable must be 2D.")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("variable must be non-empty.")
    return values


def _as_target(target: Target, variable: NDArray[np.float64]) -> NDArray[np.float64]:
    if target is None or isinstance(target, str):
        return cast(NDArray[np.float64], np.mean(variable, axis=0))

    targets = np.asarray(target, dtype=np.float64)
    if targets.ndim == 0:
        return np.full(variable.shape[1], float(targets), dtype=np.float64)
    if targets.ndim != 1:
        raise ValueError("target must be 1D.")
    if targets.size != variable.shape[1]:
        raise ValueError("variable matrix cols != target vector length.")
    return targets
