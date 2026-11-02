from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict, cast, overload

import numpy as np
from numpy.typing import NDArray

from nns._helpers import _is_fcl
from nns.categorical import factor_2_dummy_fr
from nns.part import NoiseReduction

Order = int | Literal["max"] | None

if TYPE_CHECKING:
    from nns.multivariate_regression import MRegResult

# Result shapes mirror installed R NNS.reg output names, including dotted keys,
# hence the functional TypedDict syntax.

class RegPoints(TypedDict):
    x: NDArray[np.float64]
    y: NDArray[np.float64]
"""Consolidated regression points; also the whole result when ``multivariate_call=True``."""

DerivativeTable = TypedDict(
    "DerivativeTable",
    {
        "Coefficient": NDArray[np.float64],
        "X.Lower.Range": NDArray[np.float64],
        "X.Upper.Range": NDArray[np.float64],
    },
)
"""Piecewise slope per x-interval, as in R's ``$derivative`` table."""

RegPredInt = TypedDict(
    "RegPredInt",
    {
        "pred.int.neg": NDArray[np.float64],
        "pred.int.pos": NDArray[np.float64],
    },
)
"""Prediction interval bounds for the requested ``point_est`` rows."""

RegFitted = TypedDict(
    "RegFitted",
    {
        "x": NDArray[np.float64],
        "y": NDArray[np.float64],
        "y.hat": NDArray[np.float64],
        "NNS.ID": NDArray[np.str_],
        "gradient": NDArray[np.float64],
        "residuals": NDArray[np.float64],
        # Only the unused legacy implementation below fills this column.
        "standard.errors": NotRequired[NDArray[np.float64]],
        "conf.int.pos": NotRequired[NDArray[np.float64]],
        "conf.int.neg": NotRequired[NDArray[np.float64]],
    },
)
"""Per-observation fit table (R's ``$Fitted.xy``).

The ``conf.int.*`` columns are added only when ``confidence_interval`` is set.
"""

class RegEquation(TypedDict):
    Variable: NDArray[np.str_]
    Coefficient: NDArray[np.float64]
"""Dimension-reduction synthetic-regressor weights (R's ``$equation``)."""

class RegXStar(TypedDict):
    x: NDArray[np.float64]
"""Synthetic dimension-reduction regressor (R's ``$x.star``)."""

RegResult = TypedDict(
    "RegResult",
    {
        "R2": float | None,
        "SE": float | None,
        "Prediction.Accuracy": float | None,
        "equation": "RegEquation | None",
        "x.star": "RegXStar | None",
        "derivative": DerivativeTable,
        "Point.est": "NDArray[np.float64] | None",
        "pred.int": "RegPredInt | None",
        "regression.points": RegPoints,
        "Fitted.xy": "RegFitted | None",
        "class.levels": NotRequired["list[str] | None"],
    },
)
"""Univariate ``nns_reg`` result.

``equation`` and ``x.star`` are populated only on the dimension-reduction path;
``Prediction.Accuracy`` only in class mode; ``pred.int`` only when
``confidence_interval`` is set (one row per ``point_est`` row); ``Point.est``
is ``None`` unless ``point_est`` is given; ``R2``, ``SE`` and ``Fitted.xy`` are
``None`` with ``point_only=True``. ``class.levels`` is the label behind each
class code (classification and categorical responses), else ``None``.
"""


@dataclass(frozen=True)
class FactorDesign:
    """Numeric design matrix produced from categorical predictor columns."""

    x: NDArray[np.float64]
    point_est: NDArray[np.float64] | None
    feature_names: tuple[str, ...]


@overload
def nns_reg(
    x: NDArray[Any],
    y: NDArray[Any],
    *,
    factor_2_dummy: bool = ...,
    order: Order = ...,
    dim_red_method: object | None = ...,
    tau: object | None = ...,
    type: str | None = ...,
    point_est: NDArray[np.float64] | float | None = ...,
    return_values: bool = ...,
    plot: bool = ...,
    plot_regions: bool = ...,
    residual_plot: bool = ...,
    confidence_interval: float | None = ...,
    threshold: float = ...,
    n_best: object | None = ...,
    smooth: bool = ...,
    noise_reduction: NoiseReduction = ...,
    dist: str | None = ...,
    ncores: int | None = ...,
    point_only: bool = ...,
    multivariate_call: Literal[True],
    class_levels: list[object] | None = ...,
    factor_levels: Sequence[object] | Sequence[Sequence[object] | None] | None = ...,
) -> RegPoints: ...


@overload
def nns_reg(
    x: NDArray[Any],
    y: NDArray[Any],
    *,
    factor_2_dummy: bool = ...,
    order: Order = ...,
    dim_red_method: object | None = ...,
    tau: object | None = ...,
    type: str | None = ...,
    point_est: NDArray[np.float64] | float | None = ...,
    return_values: bool = ...,
    plot: bool = ...,
    plot_regions: bool = ...,
    residual_plot: bool = ...,
    confidence_interval: float | None = ...,
    threshold: float = ...,
    n_best: object | None = ...,
    smooth: bool = ...,
    noise_reduction: NoiseReduction = ...,
    dist: str | None = ...,
    ncores: int | None = ...,
    point_only: bool = ...,
    multivariate_call: Literal[False] = ...,
    class_levels: list[object] | None = ...,
    factor_levels: Sequence[object] | Sequence[Sequence[object] | None] | None = ...,
) -> RegResult | MRegResult: ...


def nns_reg(
    x: NDArray[Any],
    y: NDArray[Any],
    *,
    factor_2_dummy: bool = True,
    order: Order = None,
    dim_red_method: object | None = None,
    tau: object | None = None,
    type: str | None = None,
    point_est: NDArray[np.float64] | float | None = None,
    return_values: bool = True,
    plot: bool = False,
    plot_regions: bool = False,
    residual_plot: bool = False,
    confidence_interval: float | None = None,
    threshold: float = 0.0,
    n_best: object | None = None,
    smooth: bool = False,
    noise_reduction: NoiseReduction = "off",
    dist: str | None = None,
    ncores: int | None = None,
    point_only: bool = False,
    multivariate_call: bool = False,
    class_levels: list[object] | None = None,
    factor_levels: Sequence[object] | Sequence[Sequence[object] | None] | None = None,
) -> Any:
    """Repaired NNS.reg port with the original bounded Racine-Hastie R2.

    pandas inputs keep their column names and types: [point_est] columns are
    matched to the training columns by name (reordered when the names are the
    same set, positional when they share none, an error otherwise), string and
    categorical columns are factors, and a pandas categorical response or
    predictor is coded in its declared category order. ``class_levels`` gives
    the response levels explicitly (codes ``1..K`` in that order; it implies
    classification) and ``factor_levels`` the predictor levels: a mapping from
    column name to levels, one entry per column (``None`` for numeric
    columns), or a flat level list for a single predictor. Plotting arguments
    are handled by the public wrapper; ``return_values`` and ``ncores`` are
    accepted for API compatibility and ignored.
    """
    from nns._reg_engine import nns_reg_engine

    del return_values, plot, plot_regions, residual_plot, ncores
    return nns_reg_engine(
        x,
        y,
        factor_2_dummy=factor_2_dummy,
        order=order,
        dim_red_method=dim_red_method,
        tau=tau,
        type=type,
        point_est=point_est,
        confidence_interval=confidence_interval,
        threshold=threshold,
        n_best=n_best,
        smooth=smooth,
        noise_reduction=cast(str, noise_reduction),
        dist=dist,
        point_only=point_only,
        multivariate_call=multivariate_call,
        class_levels=class_levels,
        factor_levels=factor_levels,
    )


def _maybe_render_reg(
    result: Mapping[str, object],
    *,
    plot: bool,
    plot_regions: bool,
    residual_plot: bool,
    point_est: NDArray[np.float64] | float | None,
) -> None:
    """Render the NNS.reg figure(s) as a side effect when a plot flag is set.

    Plotting is decoupled from computation: this only fires for the standard
    univariate result (one that carries ``Fitted.xy``) so the value-only return
    contract is unchanged. ``plot``/``plot_regions`` draw the regression figure;
    ``residual_plot`` draws a residual scatter.
    """
    if not (plot or plot_regions or residual_plot):
        return
    fitted = result.get("Fitted.xy") if isinstance(result, dict) else None
    if not isinstance(fitted, dict) or "x" not in fitted:
        return
    if plot or plot_regions:
        from nns.plotting.regression import plot_nns_reg

        plot_nns_reg(result, point_est=point_est)
    if residual_plot:
        from nns.plotting._mpl import resolve_ax

        xs = np.asarray(fitted["x"], dtype=np.float64)
        residuals = np.asarray(fitted.get("residuals", []), dtype=np.float64)
        if residuals.size and residuals.size == xs.size:
            ax = resolve_ax(None)
            ax.scatter(xs, residuals, color="steelblue")
            ax.axhline(0.0, color="red")
            ax.set_title("NNS Residual Plot")


def prepare_factor_predictors(
    x: NDArray[Any],
    *,
    point_est: NDArray[Any] | float | None = None,
    factor_levels: Sequence[object] | Sequence[Sequence[object] | None] | None = None,
    names: str | Sequence[str] | None = None,
) -> FactorDesign:
    """Return a numeric factor-expanded design matrix for regression APIs.

    This exposes the same full-rank factor expansion used internally by
    ``nns_reg(..., factor_2_dummy=True)``. The returned arrays are always
    two-dimensional so they can be passed directly to ``nns_m_reg``.
    """
    train, points, feature_names = _expand_factor_predictors_with_names(
        x,
        point_est,
        factor_levels=factor_levels,
        names=names,
    )
    return FactorDesign(
        x=_as_factor_design_matrix(train, "x"),
        point_est=None if points is None else _as_factor_design_matrix(points, "point_est"),
        feature_names=tuple(feature_names),
    )


def _expand_factor_predictors_with_names(
    x: NDArray[Any],
    point_est: NDArray[Any] | float | None,
    *,
    factor_levels: Sequence[object] | Sequence[Sequence[object] | None] | None,
    names: str | Sequence[str] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None, list[str]]:
    x_array = np.asarray(x)
    point_array = None if point_est is None else np.asarray(point_est)
    if x_array.ndim == 0:
        x_array = x_array.reshape(1)
    if x_array.ndim == 1:
        prefixes = _factor_column_prefixes(names, 1)
        combined = (
            x_array
            if point_array is None
            else np.concatenate((x_array.reshape(-1), point_array.reshape(-1)))
        )
        levels = _levels_for_column(factor_levels, 0, x_array.ndim)
        expanded, names = _dummy_matrix_for_column(
            combined,
            levels=levels,
            prefix="x" if prefixes is None else prefixes[0],
        )
        n_train = x_array.shape[0]
        train = expanded[:n_train]
        points = None if point_array is None else expanded[n_train:]
        if train.shape[1] == 1:
            return train[:, 0], None if points is None else points[:, 0], names
        return train, points, names

    if x_array.ndim != 2:
        raise ValueError("x must be a vector or 2D matrix.")
    prefixes = _factor_column_prefixes(names, x_array.shape[1])
    if point_array is not None:
        if point_array.ndim == 1:
            point_array = point_array.reshape(1, -1)
        if point_array.ndim != 2:
            raise ValueError("point_est must be a vector or 2D matrix.")
        if point_array.shape[1] != x_array.shape[1]:
            raise ValueError("point_est must have the same column count as x.")

    train_blocks: list[NDArray[np.float64]] = []
    point_blocks: list[NDArray[np.float64]] = []
    variable_names: list[str] = []
    for col in range(x_array.shape[1]):
        column = x_array[:, col]
        if point_array is None:
            combined = column
        else:
            combined = np.concatenate((column, point_array[:, col]))
        levels = _levels_for_column(factor_levels, col, x_array.ndim)
        expanded, column_names = _dummy_matrix_for_column(
            combined,
            levels=levels,
            prefix=f"X{col + 1}" if prefixes is None else prefixes[col],
        )
        train_blocks.append(expanded[: x_array.shape[0]])
        variable_names.extend(column_names)
        if point_array is not None:
            point_blocks.append(expanded[x_array.shape[0] :])
    train_matrix = np.column_stack(train_blocks)
    point_matrix = None if point_array is None else np.column_stack(point_blocks)
    return train_matrix, point_matrix, variable_names


def _as_factor_design_matrix(values: NDArray[np.float64], label: str) -> NDArray[np.float64]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        return matrix.reshape(-1, 1)
    if matrix.ndim != 2:
        raise ValueError(f"{label} expansion must produce a vector or 2D matrix.")
    return matrix


def _factor_column_prefixes(
    names: str | Sequence[str] | None,
    n_columns: int,
) -> tuple[str, ...] | None:
    if names is None:
        return None
    if isinstance(names, str):
        names = (names,)
    if len(names) != n_columns:
        raise ValueError("names must provide one name for every x column.")
    prefixes = tuple(str(name) for name in names)
    if any(name == "" for name in prefixes):
        raise ValueError("names must contain non-empty values.")
    return prefixes


def _dummy_matrix_for_column(
    values: NDArray[Any],
    *,
    levels: Sequence[object] | None,
    prefix: str,
) -> tuple[NDArray[np.float64], list[str]]:
    if levels is None:
        try:
            numeric = np.asarray(values, dtype=np.float64).reshape(-1, 1)
        except (TypeError, ValueError):
            if not _is_fcl(values):
                raise
        else:
            return numeric, [prefix]
    block = factor_2_dummy_fr(values, levels=levels)
    columns = [np.asarray(column, dtype=np.float64).reshape(-1) for column in block.values()]
    names = [prefix if name == "x" else f"{prefix}_{name}" for name in block]
    return np.column_stack(columns), names


def _levels_for_column(
    factor_levels: Sequence[object] | Sequence[Sequence[object] | None] | None,
    column: int,
    x_ndim: int,
) -> Sequence[object] | None:
    if factor_levels is None:
        return None
    if x_ndim == 1:
        return factor_levels
    if column >= len(factor_levels):
        raise ValueError("factor_levels must provide levels for every x column.")
    levels = cast(Sequence[Sequence[object] | None], factor_levels)[column]
    return levels


def _normalize_type(type_value: str | None) -> str | None:
    if type_value is None:
        return None
    normalized = type_value.lower()
    if normalized != "class":
        raise ValueError("type must be 'class' when provided.")
    return normalized


def _r2(y: NDArray[np.float64], y_hat: NDArray[np.float64]) -> float:
    """Racine-Hastie within-sample goodness-of-fit, bounded in [0, 1]."""
    y_centered = y - float(np.mean(y))
    y_hat_centered = y_hat - float(np.mean(y_hat))
    denominator = float(np.sum(y_centered**2) * np.sum(y_hat_centered**2))
    if denominator == 0.0:
        return 1.0 if np.array_equal(y, y_hat) else 0.0
    numerator = float(np.sum(y_centered * y_hat_centered) ** 2)
    return float(np.clip(numerator / denominator, 0.0, 1.0))
