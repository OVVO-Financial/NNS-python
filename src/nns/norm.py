from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any, overload

import numpy as np
from numpy.typing import NDArray

from nns._helpers import _as_matrix
from nns._rmean import r_col_means, r_cor_matrix, r_mean
from nns.dependence import nns_dep


@overload
def nns_norm(
    x: NDArray[np.float64], linear: bool = ..., *, plot: bool = ...
) -> NDArray[np.float64]: ...


@overload
def nns_norm(
    x: Sequence[NDArray[np.float64]],
    linear: bool = ...,
    *,
    plot: bool = ...,
) -> NDArray[np.float64] | list[NDArray[np.float64]]: ...


@overload
def nns_norm(x: Any, linear: bool = ..., *, plot: bool = ...) -> Any: ...


def nns_norm(
    x: NDArray[np.float64] | Sequence[NDArray[np.float64]] | Any,
    linear: bool = False,
    *,
    plot: bool = False,
) -> NDArray[np.float64] | list[NDArray[np.float64]] | Any:
    """Normalize variables following R's NNS.norm scaling.

    Three input conventions are supported, matching R's ``NNS.norm(X, ...)``:

    * A 2-D array whose columns are variables. Returns the scaled 2-D array.
    * A pandas DataFrame (R's data.frame input). Returns a DataFrame with the
      same index and R's ``"<column> Normalized"`` column names.
    * A list or tuple of 1-D arrays, one per variable (R's list input; the
      elements are variables/columns, not observation rows). Equal-length
      vectors are column-stacked and normalized through the matrix path,
      returning a 2-D array — mirroring R, where ``mapply`` simplifies the
      equal-length list result to a matrix. Unequal-length vectors force
      ``linear=True`` exactly as R does (dependence-based scale factors need
      aligned columns) and return a list of scaled arrays, one per input
      vector.

    ``plot=True`` also draws R's ``chart.type = "l"`` figure: the raw series
    above the normalized ones (``nns.plotting.plot_nns_norm``). It is not drawn
    for unequal-length lists (R draws no chart for list input); a warning says so.
    """
    if not isinstance(x, np.ndarray) and hasattr(x, "columns") and hasattr(x, "to_numpy"):
        # Iterating a DataFrame yields its column labels, not its columns.
        frame: Any = x
        values = _as_matrix(np.ascontiguousarray(frame.to_numpy(dtype=np.float64)), "x")
        normalized = _norm_matrix(values, linear)
        if plot:
            _plot_norm(values, normalized)
        return type(frame)(
            normalized,
            index=frame.index,
            columns=[f"{name} Normalized" for name in frame.columns],
        )
    if isinstance(x, np.ndarray):
        values = _as_matrix(x, "x")
    else:
        series = [_as_vector(item, index) for index, item in enumerate(x)]
        if not series:
            raise ValueError("x must be non-empty.")
        if len({item.size for item in series}) > 1:
            if plot:
                warnings.warn(
                    "plot=True is not supported for unequal-length inputs; nothing is drawn.",
                    UserWarning,
                    stacklevel=2,
                )
            return _norm_unequal_series(series)
        values = _as_matrix(np.column_stack(series), "x")
        normalized = _norm_matrix(values, linear, list_input=True)
        if plot:
            _plot_norm(values, normalized)
        return normalized
    normalized = _norm_matrix(values, linear)
    if plot:
        _plot_norm(values, normalized)
    return normalized


def _plot_norm(values: NDArray[np.float64], normalized: NDArray[np.float64]) -> None:
    """Draw R's ``chart.type = "l"`` panels: raw series on top, normalized below."""
    from nns.plotting._mpl import require_mpl
    from nns.plotting.normalization import plot_nns_norm

    _, (top, bottom) = require_mpl().subplots(2, 1, sharex=True)
    plot_nns_norm(values, ax=top)
    plot_nns_norm(normalized, ax=bottom)


def _norm_matrix(
    values: NDArray[np.float64], linear: bool, list_input: bool = False
) -> NDArray[np.float64]:
    # R: m <- colMeans(X) (list input: sapply(X, mean)); RG <- m %o% (1 / m);
    # scales <- colMeans(RG [* scale.factor]).
    if list_input:
        means = np.array([r_mean(values[:, j]) for j in range(values.shape[1])])
    else:
        means = r_col_means(values)
    means[means == 0.0] = 1e-10
    ratio_grid = means[:, np.newaxis] * (1.0 / means[np.newaxis, :])

    if linear:
        scales = r_col_means(ratio_grid)
    else:
        scale_factor = _scale_factor(values)
        scales = r_col_means(ratio_grid * scale_factor)

    # Annotated assignment instead of cast(): under mypy targeting 3.12+ the
    # numpy stubs type this product precisely, making an explicit cast redundant
    # (warn_redundant_casts), while on 3.11 the stubs return Any and the
    # annotation still narrows it without a no-any-return error.
    scaled: NDArray[np.float64] = values * scales[np.newaxis, :]
    return scaled


def _norm_unequal_series(series: list[NDArray[np.float64]]) -> list[NDArray[np.float64]]:
    means = np.array([r_mean(item) for item in series])  # R: sapply(X, mean)
    means[means == 0.0] = 1e-10
    ratio_grid = means[:, np.newaxis] * (1.0 / means[np.newaxis, :])
    scales = r_col_means(ratio_grid)
    return [item * scale for item, scale in zip(series, scales, strict=True)]


def _scale_factor(values: NDArray[np.float64]) -> NDArray[np.float64]:
    if values.shape[1] < 10:
        return np.abs(r_cor_matrix(values))  # R: abs(cor(X_mat))

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
