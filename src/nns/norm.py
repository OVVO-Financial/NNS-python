from __future__ import annotations

from collections.abc import Sequence
from typing import cast, overload

import numpy as np
from numpy.typing import NDArray

from nns._helpers import _as_matrix
from nns.dependence import nns_dep


@overload
def nns_norm(x: NDArray[np.float64], linear: bool = ...) -> NDArray[np.float64]: ...


@overload
def nns_norm(
    x: Sequence[NDArray[np.float64]],
    linear: bool = ...,
) -> NDArray[np.float64] | list[NDArray[np.float64]]: ...


def nns_norm(
    x: NDArray[np.float64] | Sequence[NDArray[np.float64]],
    linear: bool = False,
) -> NDArray[np.float64] | list[NDArray[np.float64]]:
    """Normalize variables following R's NNS.norm scaling.

    Two input conventions are supported, matching R's ``NNS.norm(X, ...)``:

    * A 2-D array whose columns are variables. Returns the scaled 2-D array.
    * A list or tuple of 1-D arrays, one per variable (R's list input; the
      elements are variables/columns, not observation rows). Equal-length
      vectors are column-stacked and normalized through the matrix path,
      returning a 2-D array — mirroring R, where ``mapply`` simplifies the
      equal-length list result to a matrix. Unequal-length vectors force
      ``linear=True`` exactly as R does (dependence-based scale factors need
      aligned columns) and return a list of scaled arrays, one per input
      vector.
    """
    if isinstance(x, np.ndarray):
        values = _as_matrix(x, "x")
    else:
        series = [_as_vector(item, index) for index, item in enumerate(x)]
        if not series:
            raise ValueError("x must be non-empty.")
        if len({item.size for item in series}) > 1:
            return _norm_unequal_series(series)
        values = _as_matrix(np.column_stack(series), "x")
    means = np.mean(values.astype(np.longdouble), axis=0).astype(np.float64)
    means = means.copy()
    means[means == 0.0] = 1e-10
    ratio_grid = means[:, np.newaxis] * (1.0 / means[np.newaxis, :])

    if linear:
        scales = np.mean(ratio_grid, axis=0)
    else:
        scale_factor = _scale_factor(values)
        scales = np.mean(ratio_grid * scale_factor, axis=0)

    # Annotated assignment instead of cast(): under mypy targeting 3.12+ the
    # numpy stubs type this product precisely, making an explicit cast redundant
    # (warn_redundant_casts), while on 3.11 the stubs return Any and the
    # annotation still narrows it without a no-any-return error.
    scaled: NDArray[np.float64] = values * scales[np.newaxis, :]
    return scaled


def _norm_unequal_series(series: list[NDArray[np.float64]]) -> list[NDArray[np.float64]]:
    means = np.array([float(np.mean(item)) for item in series])
    means[means == 0.0] = 1e-10
    ratio_grid = means[:, np.newaxis] * (1.0 / means[np.newaxis, :])
    scales = np.mean(ratio_grid, axis=0)
    return [item * scale for item, scale in zip(series, scales, strict=True)]


def _scale_factor(values: NDArray[np.float64]) -> NDArray[np.float64]:
    if values.shape[1] < 10:
        return cast(NDArray[np.float64], np.abs(np.corrcoef(values, rowvar=False)))

    n_variables = values.shape[1]
    deps = np.eye(n_variables, dtype=np.float64)
    for i in range(n_variables - 1):
        for j in range(i + 1, n_variables):
            dep = nns_dep(values[:, i], values[:, j])["Dependence"]
            deps[i, j] = dep
            deps[j, i] = dep
    return deps


def _as_vector(x: NDArray[np.float64], index: int) -> NDArray[np.float64]:
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"x[{index}] must be a 1D numeric vector.")
    if values.size == 0:
        raise ValueError(f"x[{index}] must be non-empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"x[{index}] must contain only finite values.")
    return values
