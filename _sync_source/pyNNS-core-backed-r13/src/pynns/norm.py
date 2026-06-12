from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from pynns.dependence import nns_dep


def nns_norm(x: NDArray[np.float64], linear: bool = False) -> NDArray[np.float64]:
    """Normalize a numeric matrix following R's NNS.norm scaling."""
    values = _as_matrix(x)
    means = np.mean(values, axis=0)
    means = means.copy()
    means[means == 0.0] = 1e-10
    ratio_grid = means[:, np.newaxis] * (1.0 / means[np.newaxis, :])

    if linear:
        scales = np.mean(ratio_grid, axis=0)
    else:
        scale_factor = _scale_factor(values)
        scales = np.mean(ratio_grid * scale_factor, axis=0)

    return cast(NDArray[np.float64], values * scales[np.newaxis, :])


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


def _as_matrix(x: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("x must be 2D.")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("x must be non-empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError("x must contain only finite values.")
    return values
