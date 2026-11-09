from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from nns._native import native_fn
from nns.core import _as_degree, _require_finite
from nns.dependence import _r_pm_terms

Target: TypeAlias = float | Literal["mean"] | NDArray[np.float64] | None
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

    native_pm_matrix = native_fn("pm_matrix")
    if native_pm_matrix is not None:
        native_result = native_pm_matrix(
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
            "cupm": _native_matrix(native_result["cupm"], dim),
            "dupm": _native_matrix(native_result["dupm"], dim),
            "dlpm": _native_matrix(native_result["dlpm"], dim),
            "clpm": _native_matrix(native_result["clpm"], dim),
            "cov.matrix": _native_matrix(native_result["cov.matrix"], dim),
        }
        if resolved_names is not None:
            result["names"] = resolved_names
        return result

    # R's PMMatrix_CPv: deviation components first, then per-pair sequential
    # double sums scaled by 1 / rows (a multiplication), the population
    # adjustment, and the upper triangle mirrored (DUPM/DLPM crossed).
    dev_lower = _lower_deviation(values, targets, lpm_degree)
    dev_upper = _upper_deviation(values, targets, upm_degree)

    cupm = _sequential_cross(dev_upper, dev_upper)
    clpm = _sequential_cross(dev_lower, dev_lower)
    dupm = _sequential_cross(dev_lower, dev_upper)
    dlpm = _sequential_cross(dev_upper, dev_lower)

    inv_rows = 1.0 / float(observations)
    cupm *= inv_rows
    clpm *= inv_rows
    dupm *= inv_rows
    dlpm *= inv_rows

    if pop_adj and observations > 1 and lpm_degree > 0 and upm_degree > 0:
        adjust = float(observations) / float(observations - 1)
        cupm *= adjust
        clpm *= adjust
        dupm *= adjust
        dlpm *= adjust

    if norm:
        total = cupm + dupm + dlpm + clpm
        positive = total > 0.0
        safe_total = np.where(positive, total, 1.0)
        cupm = np.where(positive, cupm / safe_total, 0.0)
        dupm = np.where(positive, dupm / safe_total, 0.0)
        dlpm = np.where(positive, dlpm / safe_total, 0.0)
        clpm = np.where(positive, clpm / safe_total, 0.0)

    cov_matrix = cupm + clpm - dupm - dlpm
    # R fills only the upper triangle and mirrors it: CUPM, CLPM and the
    # covariance plainly, DUPM and DLPM crossed (each from the other's transpose).
    upper = np.triu(np.ones(cupm.shape, dtype=bool))
    cupm, clpm, dupm, dlpm, cov_matrix = (
        np.where(upper, cupm, cupm.T),
        np.where(upper, clpm, clpm.T),
        np.where(upper, dupm, dlpm.T),
        np.where(upper, dlpm, dupm.T),
        np.where(upper, cov_matrix, cov_matrix.T),
    )
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


def _sequential_cross(
    left: NDArray[np.float64], right: NDArray[np.float64]
) -> NDArray[np.float64]:
    """``t(left) %*% right`` summed row by row in double, as R's C++ loop does."""
    rows, cols = left.shape
    total = np.zeros((cols, cols), dtype=np.float64)
    block = max(1, 2_000_000 // max(1, cols * cols))
    for start in range(0, rows, block):
        stop = start + block
        products = left[start:stop, :, np.newaxis] * right[start:stop, np.newaxis, :]
        total = np.cumsum(np.concatenate((total[np.newaxis], products)), axis=0)[-1]
    return total


def _native_matrix(value: Any, dim: int) -> NDArray[np.float64]:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape == (dim, dim):
        return matrix
    return cast(NDArray[np.float64], matrix.reshape((dim, dim), order="F"))


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
    return _r_pm_terms(np.maximum(0.0, targets[np.newaxis, :] - values), degree)


def _upper_deviation(
    values: NDArray[np.float64],
    targets: NDArray[np.float64],
    degree: float,
) -> NDArray[np.float64]:
    if degree == 0:
        return (values > targets[np.newaxis, :]).astype(np.float64)
    return _r_pm_terms(np.maximum(0.0, values - targets[np.newaxis, :]), degree)


def _as_matrix(variable: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(variable, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("variable must be 2D.")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("variable must be non-empty.")
    # As R's PM.matrix: NaN would be counted in n but in no quadrant.
    _require_finite(values, "variable")
    return values


def _as_target(target: Target, variable: NDArray[np.float64]) -> NDArray[np.float64]:
    if target is None or isinstance(target, str):
        # Rcpp sugar colMeans: a sequential double sum divided by the row count.
        return cast(NDArray[np.float64], np.cumsum(variable, axis=0)[-1] / variable.shape[0])

    targets = np.asarray(target, dtype=np.float64)
    if targets.ndim == 0:
        targets = np.full(variable.shape[1], float(targets), dtype=np.float64)
    if targets.ndim != 1:
        raise ValueError("target must be 1D.")
    if targets.size != variable.shape[1]:
        raise ValueError("variable matrix cols != target vector length.")
    _require_finite(targets, "target")
    return targets
