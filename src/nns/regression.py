from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from nns._helpers import _fast_lm, _is_fcl
from nns.categorical import encode_factor_codes, factor_2_dummy_fr
from nns.causation import _uni_caus
from nns.central_tendencies import nns_mode
from nns.copula import _copula
from nns.dependence import _gravity, nns_dep
from nns.part import NoiseReduction, nns_part
from nns.smoothing import r_smooth_spline_fixed_spar
from nns.var import lpm_var, upm_var

Order = int | Literal["max"] | None


@dataclass(frozen=True)
class FactorDesign:
    """Numeric design matrix produced from categorical predictor columns."""

    x: NDArray[np.float64]
    point_est: NDArray[np.float64] | None
    feature_names: tuple[str, ...]


def nns_reg(
    x: NDArray[Any],
    y: NDArray[Any],
    *,
    factor_2_dummy: bool = False,
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
    dist: str = "L2",
    ncores: int | None = None,
    point_only: bool = False,
    multivariate_call: bool = False,
    class_levels: list[object] | None = None,
    factor_levels: Sequence[object] | Sequence[Sequence[object] | None] | None = None,
) -> dict[str, Any]:
    """Univariate numeric port of R's NNS.reg."""
    del return_values, plot, plot_regions, residual_plot, ncores

    if dim_red_method is not None:
        return _nns_reg_dimred(
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
            noise_reduction=noise_reduction,
            dist=dist,
            point_only=point_only,
            multivariate_call=multivariate_call,
            class_levels=class_levels,
            factor_levels=factor_levels,
        )

    type_value = _normalize_type(type)
    if type_value == "class":
        noise_reduction = "mode_class"

    x_for_dispatch: NDArray[Any] | NDArray[np.float64] = np.asarray(x)
    point_for_dispatch = point_est
    if factor_2_dummy:
        x_for_dispatch, point_for_dispatch = _expand_factor_predictors(
            x,
            point_est,
            factor_levels=factor_levels,
        )

    if np.asarray(x_for_dispatch).ndim == 2:
        from nns.multivariate_regression import nns_m_reg

        y_matrix_values, _ = _prepare_y_values(y, type_value=type_value, class_levels=class_levels)
        dispatch_n_best = n_best
        if type_value == "class" and dispatch_n_best is None:
            dispatch_n_best = 1
        return nns_m_reg(
            np.asarray(x_for_dispatch, dtype=np.float64),
            y_matrix_values,
            factor_2_dummy=False,
            order=order,
            n_best=cast(Any, dispatch_n_best),
            type=type_value,
            point_est=None
            if point_for_dispatch is None
            else np.asarray(point_for_dispatch, dtype=np.float64),
            point_only=point_only,
            noise_reduction=noise_reduction,
            dist=dist,
            confidence_interval=confidence_interval,
            class_levels=class_levels,
        )

    del tau, threshold, n_best, dist
    x_values, y_values = _validate_univariate_inputs(
        x_for_dispatch,
        y,
        False if factor_2_dummy else factor_2_dummy,
        type_value=type_value,
        class_levels=class_levels,
    )
    class_mode = type_value == "class" or _should_auto_classify(y_values)
    if class_mode:
        noise_reduction = "mode_class"
    _reject_deferred_paths(
        point_est=point_for_dispatch,
        confidence_interval=confidence_interval,
        smooth=smooth,
        multivariate_call=multivariate_call,
    )
    noise = _validate_noise_reduction(noise_reduction)
    point_values = _as_point_est(point_for_dispatch)
    return _nns_reg_univariate_core(
        x_values,
        y_values,
        order=order,
        noise=noise,
        point_values=point_values,
        confidence_interval=confidence_interval,
        multivariate_call=multivariate_call,
        class_mode=class_mode,
        smooth=smooth,
        equation=None,
        x_star=None,
    )


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


def _nns_reg_univariate_core(
    x_values: NDArray[np.float64],
    y_values: NDArray[np.float64],
    *,
    order: Order,
    noise: NoiseReduction,
    point_values: NDArray[np.float64] | None,
    confidence_interval: float | None,
    multivariate_call: bool,
    class_mode: bool,
    equation: dict[str, NDArray[np.float64] | NDArray[np.str_]] | None,
    x_star: dict[str, NDArray[np.float64]] | None,
    smooth: bool = False,
) -> dict[str, Any]:

    dependence = _regression_dependence(x_values, y_values)
    dep_order = _dep_reduced_order(dependence, order, y_values.size)
    part_map = _partition_for_regression(x_values, y_values, dependence, dep_order, order, noise)
    nns_ids = part_map["dt"]["quadrant"].astype(str)

    rp = part_map["regression.points"]
    rp_x, rp_y = _initial_regression_points(rp["x"], rp["y"], x_values)
    if not class_mode:
        rp_x, rp_y = _add_central_point(rp_x, rp_y, x_values, y_values)
    rp_x, rp_y = _add_endpoint_points(
        rp_x,
        rp_y,
        x_values,
        y_values,
        dependence,
        class_mode=class_mode,
    )
    rp_x = np.minimum(np.max(x_values), np.maximum(np.min(x_values), rp_x))
    rp_y = np.minimum(np.max(y_values), np.maximum(np.min(y_values), rp_y))

    spline_fit = None
    smooth_condition = smooth and rp_x.size >= 4 and not isinstance(order, str)
    if smooth_condition:
        spline_fit = r_smooth_spline_fixed_spar(
            rp_x,
            rp_y,
            spar=(dependence + 0.5) / 2.0,
        )
        smooth_rp_y = spline_fit.predict(rp_x)
        # R derives smooth slopes before clamping returned regression points.
        coeff_rp_y = smooth_rp_y.copy()
        rp_y = np.minimum(np.max(y_values), np.maximum(np.min(y_values), smooth_rp_y))
    else:
        coeff_rp_y = rp_y.copy()

    if class_mode:
        rp_y = _round_clamp_classes(rp_y, y_values)

    if multivariate_call:
        return {"x": rp_x, "y": rp_y}

    coeff = _coefficients(rp_x, coeff_rp_y, x_values, y_values)
    if smooth_condition and spline_fit is not None:
        order_idx = np.argsort(x_values, kind="mergesort")
        estimate = np.empty_like(x_values, dtype=np.float64)
        estimate[order_idx] = spline_fit.predict(x_values[order_idx])
    else:
        estimate = _fitted_values(x_values, y_values, rp_x, rp_y, coeff, order)
    if class_mode:
        estimate = _round_clamp_classes(estimate, y_values)

    if point_values is None:
        point_est_y = np.array([], dtype=np.float64)
    elif smooth_condition and spline_fit is not None:
        point_est_y = spline_fit.predict(point_values)
        point_est_y = _extrapolate_points(point_values, point_est_y, x_values, y_values, coeff)
        if class_mode:
            point_est_y = _round_clamp_classes(point_est_y, y_values)
    else:
        point_est_y = _predict_points(point_values, x_values, y_values, rp_x, rp_y, coeff)
        if class_mode:
            point_est_y = _round_clamp_classes(point_est_y, y_values)

    if isinstance(order, str):
        rp_out_x, rp_out_y = _consolidate_points(part_map["dt"]["x"], part_map["dt"]["y"])
    elif np.unique(x_values).size <= 1 and rp_x.size == 1:
        rp_out_x = np.repeat(rp_x, 3)
        rp_out_y = np.repeat(rp_y, 3)
    else:
        rp_out_x, rp_out_y = rp_x, rp_y

    fitted = _fitted_table(x_values, y_values, estimate, nns_ids, coeff)
    pred_int = _apply_univariate_intervals(
        fitted,
        point_values,
        confidence_interval=confidence_interval,
        class_mode=class_mode,
    )
    se = float(math.sqrt(float(np.sum((estimate - y_values) ** 2)) / (y_values.size - 1)))
    r2 = _r2(y_values, estimate)
    prediction_accuracy = (
        float((y_values.size - np.sum(np.abs(np.round(estimate) - y_values) > 0.0)) / y_values.size)
        if class_mode
        else None
    )

    return {
        "R2": r2,
        "SE": se,
        "Prediction.Accuracy": prediction_accuracy,
        "equation": equation,
        "x.star": x_star,
        "derivative": {
            "Coefficient": coeff["Coefficient"],
            "X.Lower.Range": coeff["X.Lower.Range"],
            "X.Upper.Range": coeff["X.Upper.Range"],
        },
        "Point.est": point_est_y,
        "pred.int": pred_int,
        "regression.points": {"x": rp_out_x, "y": rp_out_y},
        "Fitted.xy": fitted,
    }


def _validate_univariate_inputs(
    x: NDArray[Any],
    y: NDArray[Any],
    factor_2_dummy: bool,
    *,
    type_value: str | None,
    class_levels: list[object] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if factor_2_dummy and (_is_fcl(x) or _is_fcl(y)):
        raise ValueError("non-numeric univariate inputs must be expanded before validation.")
    x_values = np.asarray(x, dtype=np.float64)
    y_values, _ = _prepare_y_values(y, type_value=type_value, class_levels=class_levels)
    if x_values.ndim != 1 or y_values.ndim != 1:
        raise ValueError("univariate validation requires 1D x and y.")
    if x_values.size == 0:
        raise ValueError("x and y must be non-empty.")
    if x_values.size != y_values.size:
        raise ValueError("x and y must have the same length.")
    if not np.all(np.isfinite(x_values)) or not np.all(np.isfinite(y_values)):
        raise ValueError("x and y must contain only finite values.")
    return x_values, y_values


def _expand_factor_predictors(
    x: NDArray[Any],
    point_est: NDArray[Any] | float | None,
    *,
    factor_levels: Sequence[object] | Sequence[Sequence[object] | None] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    train, points, _ = _expand_factor_predictors_with_names(
        x,
        point_est,
        factor_levels=factor_levels,
    )
    return train, points


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


def _prepare_y_values(
    y: NDArray[Any],
    *,
    type_value: str | None,
    class_levels: list[object] | None,
) -> tuple[NDArray[np.float64], list[object] | None]:
    y_array = np.asarray(y)
    if y_array.ndim != 1:
        y_array = y_array.reshape(-1)
    if class_levels is not None:
        return encode_factor_codes(y_array, levels=class_levels)
    if y_array.dtype.kind in {"U", "S", "O"}:
        if type_value == "class":
            raise ValueError(
                "raw string/object class labels require class_levels to reproduce R factor codes."
            )
    return np.asarray(y_array, dtype=np.float64).reshape(-1), None


def _should_auto_classify(y: NDArray[np.float64]) -> bool:
    if y.size == 0:
        return False
    if not np.all(np.isclose(y, np.round(y), rtol=0.0, atol=1e-12)):
        return False
    return np.unique(y).size < math.sqrt(y.size)


def _round_clamp_classes(
    values: NDArray[np.float64],
    y: NDArray[np.float64],
) -> NDArray[np.float64]:
    rounded = np.where(values % 1.0 < 0.5, np.floor(values), np.ceil(values))
    return np.minimum(float(np.max(y)), np.maximum(float(np.min(y)), rounded)).astype(np.float64)


def _nns_reg_dimred(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    factor_2_dummy: bool,
    order: Order,
    dim_red_method: object,
    tau: object | None,
    type: str | None,
    point_est: NDArray[np.float64] | float | None,
    confidence_interval: float | None,
    threshold: float,
    n_best: object | None,
    smooth: bool,
    noise_reduction: NoiseReduction,
    dist: str,
    point_only: bool,
    multivariate_call: bool,
    class_levels: list[object] | None = None,
    factor_levels: Sequence[object] | Sequence[Sequence[object] | None] | None = None,
) -> dict[str, Any]:
    del n_best
    if factor_2_dummy:
        x, point_est, variable_names = _expand_factor_predictors_with_names(
            x,
            point_est,
            factor_levels=factor_levels,
        )
    else:
        variable_names = None
    type_value = _normalize_type(type)
    x_matrix, y_values = _validate_dimred_inputs(
        x,
        y,
        type_value=type_value,
        class_levels=class_levels,
    )
    class_mode = type_value == "class" or _should_auto_classify(y_values)
    if class_mode:
        noise_reduction = "mode_class"
    point_matrix = _as_dimred_point_est(point_est, x_matrix.shape[1])
    noise = _validate_noise_reduction(noise_reduction)
    projection = _dimred_projection(
        x_matrix,
        y_values,
        dim_red_method=dim_red_method,
        tau=tau,
        threshold=threshold,
        point_est=point_matrix,
        dist=dist,
        variable_names=variable_names,
    )
    dimred_order = _dimred_order(projection.x_star, y_values, order)
    result = _nns_reg_univariate_core(
        projection.x_star,
        y_values,
        order=dimred_order,
        noise=noise,
        point_values=projection.point_est,
        confidence_interval=confidence_interval,
        multivariate_call=multivariate_call,
        class_mode=class_mode,
        smooth=smooth,
        equation=projection.equation,
        x_star={"x": projection.x_star},
    )
    if point_only:
        return result
    return result


class _DimredProjection:
    def __init__(
        self,
        x_star: NDArray[np.float64],
        point_est: NDArray[np.float64] | None,
        equation: dict[str, NDArray[np.float64] | NDArray[np.str_]],
    ) -> None:
        self.x_star = x_star
        self.point_est = point_est
        self.equation = equation


def _validate_dimred_inputs(
    x: NDArray[np.float64],
    y: NDArray[Any],
    *,
    type_value: str | None = None,
    class_levels: list[object] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    x_values = np.asarray(x, dtype=np.float64)
    y_values, _ = _prepare_y_values(y, type_value=type_value, class_levels=class_levels)
    if x_values.ndim != 2:
        raise ValueError("dim_red_method requires a 2D numeric x matrix.")
    if x_values.shape[0] == 0 or x_values.shape[1] == 0:
        raise ValueError("x must be non-empty.")
    if x_values.shape[0] != y_values.size:
        raise ValueError("x and y must have the same row count.")
    if not np.all(np.isfinite(x_values)) or not np.all(np.isfinite(y_values)):
        raise ValueError("x and y must contain only finite values.")
    return x_values, y_values


def _as_dimred_point_est(
    point_est: NDArray[np.float64] | float | None,
    n_cols: int,
) -> NDArray[np.float64] | None:
    if point_est is None:
        return None
    values = np.asarray(point_est, dtype=np.float64)
    if values.ndim == 0:
        values = values.reshape(1, 1)
    elif values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2:
        raise ValueError("point_est must be a vector or 2D matrix.")
    if values.shape[1] != n_cols:
        raise ValueError("point_est must have the same column count as x.")
    if not np.all(np.isfinite(values)):
        raise ValueError("point_est must contain only finite values.")
    return values


def _dimred_projection(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    dim_red_method: object,
    tau: object | None,
    threshold: float,
    point_est: NDArray[np.float64] | None,
    dist: str,
    variable_names: Sequence[str] | None = None,
) -> _DimredProjection:
    coef = _dimred_coefficients(x, y, dim_red_method=dim_red_method, tau=tau)
    if coef.size != x.shape[1]:
        raise ValueError("numeric dim_red_method must have one coefficient per x column.")
    preserved = coef.copy()
    coef = coef.copy()
    coef[np.abs(coef) < threshold] = 0.0

    norm_x = _r_minmax_columns(x, zero_guard=False)
    x_star_matrix = norm_x * coef[np.newaxis, :]
    x_star_matrix[~np.isfinite(x_star_matrix)] = 0.0
    if np.all(x_star_matrix == 0.0):
        x_star_matrix = x.copy()
        coef[coef == 0.0] = preserved[coef == 0.0]

    active_count = int(np.sum(np.abs(coef) > 0.0))
    if active_count == 0:
        active_count = 1
    x_star = np.sum(x_star_matrix / active_count, axis=1)
    point_star = (
        None
        if point_est is None
        else _project_dimred_points(point_est, x, coef, active_count, dist=dist)
    )
    denominator = float(np.sum(dim_red_method)) if isinstance(dim_red_method, np.ndarray) else None
    if denominator is None and isinstance(dim_red_method, (list, tuple)):
        try:
            denominator = float(np.sum(np.asarray(dim_red_method, dtype=np.float64)))
        except (TypeError, ValueError):
            denominator = None
    if denominator is None:
        denominator = float(active_count)
    names = (
        [f"X{index + 1}" for index in range(x.shape[1])]
        if variable_names is None
        else list(variable_names)
    )
    if len(names) != x.shape[1]:
        raise ValueError("variable_names must match the number of x columns.")
    equation = {
        "Variable": np.asarray([*names, "DENOMINATOR"]),
        "Coefficient": np.concatenate((coef, np.array([denominator], dtype=np.float64))),
    }
    return _DimredProjection(x_star=x_star, point_est=point_star, equation=equation)


def _dimred_coefficients(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    dim_red_method: object,
    tau: object | None,
) -> NDArray[np.float64]:
    if isinstance(dim_red_method, str):
        method = dim_red_method.lower()
        if method == "cor":
            return _spearman_coefficients(x, y)
        if method == "nns.dep":
            return np.asarray(
                [nns_dep(x[:, col], y, asym=True)["Dependence"] for col in range(x.shape[1])]
            )
        if method == "nns.caus":
            tau_value = _dimred_tau(tau)
            return np.asarray([_uni_caus(y, x[:, col], tau_value) for col in range(x.shape[1])])
        if method == "all":
            tau_value = _dimred_tau(tau)
            caus = np.asarray([_uni_caus(y, x[:, col], tau_value) for col in range(x.shape[1])])
            dep = np.asarray(
                [nns_dep(x[:, col], y, asym=True)["Dependence"] for col in range(x.shape[1])]
            )
            cor = _spearman_coefficients(x, y)
            equal = np.ones(x.shape[1], dtype=np.float64)
            stacked = np.column_stack((caus, dep, cor, equal))
            return np.asarray([float(nns_mode(row)) for row in stacked], dtype=np.float64)
        if method == "equal":
            return np.ones(x.shape[1], dtype=np.float64)
        raise ValueError(
            "dim_red_method must be one of 'cor', 'NNS.dep', 'NNS.caus', 'all', 'equal', "
            "or a numeric vector."
        )
    coef = np.asarray(dim_red_method, dtype=np.float64).reshape(-1)
    coef[~np.isfinite(coef)] = 0.0
    return coef


def _dimred_tau(tau: object | None) -> int:
    if tau is None or tau == "cs":
        return 0
    if tau == "ts":
        # R's NNS.reg dim-red branch calls internal Uni.caus directly, where
        # tau="ts" is a fixed lag of 3 rather than the public NNS.caus
        # NNS.seas-derived lag path.
        return 3
    tau_value = int(cast(Any, tau))
    if tau_value < 0:
        raise ValueError("tau must be non-negative.")
    return tau_value


def _spearman_coefficients(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    y_rank = _rank_average(y)
    out = np.empty(x.shape[1], dtype=np.float64)
    for col in range(x.shape[1]):
        out[col] = _pearson(_rank_average(x[:, col]), y_rank)
    out[~np.isfinite(out)] = 0.0
    return out


def _rank_average(values: NDArray[np.float64]) -> NDArray[np.float64]:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def _pearson(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    x_centered = x - float(np.mean(x))
    y_centered = y - float(np.mean(y))
    denom = math.sqrt(float(np.sum(x_centered**2) * np.sum(y_centered**2)))
    if denom == 0.0:
        return 0.0
    return float(np.sum(x_centered * y_centered) / denom)


def _r_minmax_columns(values: NDArray[np.float64], *, zero_guard: bool) -> NDArray[np.float64]:
    vmin = np.min(values, axis=0)
    vmax = np.max(values, axis=0)
    denom = vmax - vmin
    if zero_guard:
        denom = np.where(denom == 0.0, 1.0, denom)
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = (values - vmin[np.newaxis, :]) / denom[np.newaxis, :]
    return np.asarray(scaled, dtype=np.float64)


def _project_dimred_points(
    point_est: NDArray[np.float64],
    x: NDArray[np.float64],
    coef: NDArray[np.float64],
    active_count: int,
    *,
    dist: str,
) -> NDArray[np.float64]:
    joint = np.vstack((point_est, x))
    if dist.lower() != "factor":
        joint = _r_minmax_columns(joint, zero_guard=True)
    point_norm = joint[: point_est.shape[0]]
    return np.asarray(point_norm @ coef / active_count, dtype=np.float64)


def _dimred_order(x_star: NDArray[np.float64], y: NDArray[np.float64], order: Order) -> Order:
    if order == "max":
        return "max"
    if order is None:
        dependence = _regression_dependence(x_star, y)
        computed = max(1, math.floor(dependence * 10.0))
    else:
        computed = max(1, _round_half_up(float(order)))
    if y.size < 100:
        computed = _round_half_up(max(1.0, computed / 2.0))
    return max(1, computed)


def _reject_deferred_paths(
    *,
    point_est: NDArray[np.float64] | float | None,
    confidence_interval: float | None,
    smooth: bool,
    multivariate_call: bool,
    allow_smooth_fallback: bool = False,
) -> None:
    del point_est, confidence_interval, smooth, multivariate_call, allow_smooth_fallback


def _validate_noise_reduction(value: str) -> NoiseReduction:
    noise = value.lower()
    if noise not in {"off", "mean", "median", "mode", "mode_class"}:
        raise ValueError(
            "noise_reduction must be one of 'mean', 'median', 'mode', 'mode_class', 'off'."
        )
    return cast(NoiseReduction, noise)


def _as_point_est(point_est: NDArray[np.float64] | float | None) -> NDArray[np.float64] | None:
    if point_est is None:
        return None
    values = np.asarray(point_est, dtype=np.float64)
    if values.ndim == 0:
        values = values.reshape(1)
    if values.ndim != 1:
        values = values.reshape(-1, order="F")
    if not np.all(np.isfinite(values)):
        raise ValueError("point_est must contain only finite values.")
    return values


def _regression_dependence(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    dep = nns_dep(x, y, asym=True)["Dependence"]
    try:
        scaled = np.column_stack(
            (
                _rescale_01(x),
                _rescale_01(x),
                _rescale_01(y),
            )
        )
        dep = float(np.mean(np.array([dep, _nns_copula_matrix(scaled)], dtype=np.float64)))
    except (ValueError, FloatingPointError):
        dep = float(dep)
    if not math.isfinite(dep):
        dep = 0.1
    return dep


def _rescale_01(values: NDArray[np.float64]) -> NDArray[np.float64]:
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmax == vmin:
        return np.zeros_like(values, dtype=np.float64)
    return (values - vmin) / (vmax - vmin)


def _nns_copula_matrix(values: NDArray[np.float64]) -> float:
    target = cast(NDArray[np.float64], np.mean(values, axis=0))
    return _copula(values, target, continuous=True)


def _dep_reduced_order(dependence: float, order: Order, n: int) -> int | Literal["max"]:
    if order == "max":
        return "max"
    if order is None:
        rounded_dep = math.floor(dependence * 10.0)
        if n < 100:
            rounded_dep = math.floor(rounded_dep / 2.0)
        return max(1, rounded_dep)
    if isinstance(order, bool) or not isinstance(order, int):
        raise TypeError("order must be an integer, 'max', or None.")
    return max(1, _round_half_up(float(order)))


def _round_half_up(value: float) -> int:
    floor = math.floor(value)
    return floor if value - floor < 0.5 else math.ceil(value)


def _partition_for_regression(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    dependence: float,
    dep_order: int | Literal["max"],
    requested_order: Order,
    noise: NoiseReduction,
) -> dict[str, Any]:
    if dependence == 1.0 or dep_order == "max":
        if requested_order is None or dep_order == "max":
            return _max_order_part_map(x, y)
        return cast(dict[str, Any], nns_part(x, y, order=int(dep_order), obs_req=0))
    return cast(
        dict[str, Any],
        nns_part(
            x,
            y,
            noise_reduction=noise,
            order=int(dep_order),
            type="XONLY",
            obs_req=0,
            min_obs_stop=True,
        ),
    )


def _max_order_part_map(x: NDArray[np.float64], y: NDArray[np.float64]) -> dict[str, Any]:
    quadrants = np.full(x.size, "q", dtype=str)
    seed_map = nns_part(x, y, order=1, obs_req=0)
    return {
        "order": x.size,
        "dt": {"x": x.copy(), "y": y.copy(), "quadrant": quadrants, "prior.quadrant": quadrants},
        "regression.points": seed_map["regression.points"],
    }


def _initial_regression_points(
    point_x: NDArray[np.float64],
    point_y: NDArray[np.float64],
    x: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    clamped_x = np.minimum(float(np.max(x)), np.maximum(point_x, float(np.min(x))))
    return _consolidate_points(clamped_x, point_y)


def _add_central_point(
    rp_x: NDArray[np.float64],
    rp_y: NDArray[np.float64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    n_points = rp_x.size
    row_positions = np.arange(1, n_points + 1)
    rows = np.array(
        [math.floor(np.median(row_positions)), math.ceil(np.median(row_positions))],
        dtype=np.int64,
    )
    central_x_values = rp_x[rows - 1]
    if np.unique(rows).size > 1:
        mask = (x >= central_x_values[0]) & (x <= central_x_values[1])
        central_y = _gravity(y[mask])
    else:
        central_y = float(rp_y[rows[0] - 1])
    central_x = _gravity(central_x_values)
    return _consolidate_points(
        np.concatenate((rp_x, np.array([central_x], dtype=np.float64))),
        np.concatenate((rp_y, np.array([central_y], dtype=np.float64))),
    )


def _add_endpoint_points(
    rp_x: NDArray[np.float64],
    rp_y: NDArray[np.float64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    dependence: float,
    *,
    class_mode: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if dependence >= 1.0 and not class_mode:
        min_y = float(y[np.flatnonzero(x == np.min(x))[0]])
        max_y = float(y[np.flatnonzero(x == np.max(x))[0]])
    else:
        min_y = _endpoint_y(x, y, rp_x, low=True, dependence=dependence, class_mode=class_mode)
        max_y = _endpoint_y(x, y, rp_x, low=False, dependence=dependence, class_mode=class_mode)
    return _consolidate_points(
        np.concatenate((rp_x, np.array([float(np.min(x)), float(np.max(x))]))),
        np.concatenate((rp_y, np.array([min_y, max_y]))),
    )


def _endpoint_y(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    rp_x: NDArray[np.float64],
    *,
    low: bool,
    dependence: float,
    class_mode: bool,
) -> float:
    boundary = float(np.min(x) if low else np.max(x))
    reg_range = float(np.min(rp_x) if low else np.max(rp_x))
    mid_range = float(np.mean([boundary, reg_range]))
    boundary_mask = x <= reg_range if low else x >= reg_range
    mid_mask = x <= mid_range if low else x >= mid_range
    y_boundary = y[boundary_mask]
    if class_mode:
        return float(nns_mode(y_boundary, discrete=True))
    y_mid = y[mid_mask]
    x_mid = x[mid_mask]
    unique_x_mid = np.unique(x_mid).size

    if unique_x_mid > 1 and y_boundary.size > 5:
        if dependence < 0.95 and y_boundary.size > 1 and y_mid.size > 1:
            fit_boundary = _edge_lm_fit(x[boundary_mask], y_boundary, low=low)
            fit_mid = _edge_lm_fit(x_mid, y_mid, low=low)
            return float(
                (fit_boundary * y_boundary.size + fit_mid * y_mid.size)
                / (y_boundary.size + y_mid.size)
            )
        boundary_values = y[x == boundary]
        return float(np.mean(np.unique(boundary_values)))

    return float(np.mean(np.unique([_gravity(y[x == boundary])])))


def _edge_lm_fit(x: NDArray[np.float64], y: NDArray[np.float64], *, low: bool) -> float:
    intercept, slope = _fast_lm(x, y)
    edge_x = float(np.min(x) if low else np.max(x))
    return intercept + slope * edge_x


def _consolidate_points(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    finite = np.isfinite(x) & np.isfinite(y)
    x_values = x[finite].astype(np.float64)
    y_values = y[finite].astype(np.float64)
    order = np.lexsort((y_values, x_values))
    x_values = x_values[order]
    y_values = y_values[order]
    unique_x, inverse = np.unique(x_values, return_inverse=True)
    out_y = np.empty(unique_x.size, dtype=np.float64)
    for idx in range(unique_x.size):
        out_y[idx] = _gravity(y_values[inverse == idx])
    return unique_x, out_y


def _coefficients(
    rp_x: NDArray[np.float64],
    rp_y: NDArray[np.float64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    if rp_x.size > 1:
        rise = np.diff(rp_y)
        run = np.diff(rp_x)
    else:
        rise = np.array([float(np.max(y) - np.min(y))], dtype=np.float64)
        run_value = float(np.max(x) - np.min(x))
        if run_value == 0.0:
            run_value = 1.0
        run = np.array([run_value], dtype=np.float64)
        rp_x = np.repeat(rp_x, 3)
        rp_y = np.repeat(rp_y, 3)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        coef = rise / run
    lower = rp_x[:-1] if rp_x.size > 1 else np.array([float(np.unique(rp_x)[0])])
    upper = rp_x[1:] if rp_x.size > 1 else np.array([float(np.unique(rp_x)[0])])
    if np.unique(upper).size <= 1:
        collapsed = np.asarray(np.unique(upper), dtype=np.float64)
        coef = np.zeros_like(collapsed, dtype=np.float64)
        lower = collapsed
        upper = collapsed
    coef = np.where(np.isposinf(coef), 1.0, coef)
    coef = np.where(np.isfinite(coef), coef, 0.0)
    matrix = np.column_stack((coef, lower, upper))
    _, first = np.unique(matrix, axis=0, return_index=True)
    unique = matrix[np.sort(first)]
    return {
        "Coefficient": unique[:, 0],
        "X.Lower.Range": unique[:, 1],
        "X.Upper.Range": unique[:, 2],
    }


def _fitted_values(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    rp_x: NDArray[np.float64],
    rp_y: NDArray[np.float64],
    coeff: dict[str, NDArray[np.float64]],
    order: Order,
) -> NDArray[np.float64]:
    if (order is not None and _is_fcl(order)) or (
        order is not None and not isinstance(order, str) and order >= y.size
    ):
        return y.copy()
    reg_idx = _find_interval(x, rp_x, rightmost_closed=False)
    coef_idx = _find_interval(x, coeff["X.Lower.Range"], rightmost_closed=False)
    return (x - rp_x[reg_idx]) * coeff["Coefficient"][coef_idx] + rp_y[reg_idx]


def _predict_points(
    point_est: NDArray[np.float64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    rp_x: NDArray[np.float64],
    rp_y: NDArray[np.float64],
    coeff: dict[str, NDArray[np.float64]],
) -> NDArray[np.float64]:
    reg_idx = _find_interval(point_est, rp_x, rightmost_closed=True)
    coef_idx = _find_interval(point_est, coeff["X.Lower.Range"], rightmost_closed=True)
    out = (point_est - rp_x[reg_idx]) * coeff["Coefficient"][coef_idx] + rp_y[reg_idx]
    if np.any((point_est > np.max(x)) | (point_est < np.min(x))):
        _, first = np.unique(coeff["Coefficient"], return_index=True)
        unique_coef = coeff["Coefficient"][np.sort(first)]
        upper_slope = float(np.mean(unique_coef[-2:]))
        lower_slope = float(np.mean(unique_coef[:2]))
        upper_mask = point_est > np.max(x)
        lower_mask = point_est < np.min(x)
        if np.any(upper_mask):
            out[upper_mask] = (
                point_est[upper_mask] - float(np.max(x))
            ) * upper_slope + _boundary_y(
                x,
                y,
                low=False,
            )
        if np.any(lower_mask):
            out[lower_mask] = (
                point_est[lower_mask] - float(np.min(x))
            ) * lower_slope + _boundary_y(
                x,
                y,
                low=True,
            )
    return out.astype(np.float64)


def _extrapolate_points(
    point_est: NDArray[np.float64],
    point_est_y: NDArray[np.float64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    coeff: dict[str, NDArray[np.float64]],
) -> NDArray[np.float64]:
    out = point_est_y.astype(np.float64).copy()
    if not np.any((point_est > np.max(x)) | (point_est < np.min(x))):
        return out
    _, first = np.unique(coeff["Coefficient"], return_index=True)
    unique_coef = coeff["Coefficient"][np.sort(first)]
    upper_slope = float(np.mean(unique_coef[-2:]))
    lower_slope = float(np.mean(unique_coef[:2]))
    upper_mask = point_est > np.max(x)
    lower_mask = point_est < np.min(x)
    if np.any(upper_mask):
        out[upper_mask] = (point_est[upper_mask] - float(np.max(x))) * upper_slope + _boundary_y(
            x,
            y,
            low=False,
        )
    if np.any(lower_mask):
        out[lower_mask] = (point_est[lower_mask] - float(np.min(x))) * lower_slope + _boundary_y(
            x,
            y,
            low=True,
        )
    return out


def _boundary_y(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    low: bool,
) -> float:
    index = int(np.argmin(x) if low else np.argmax(x))
    return float(nns_mode(np.asarray([y[index]], dtype=np.float64)))


def _find_interval(
    values: NDArray[np.float64],
    breaks: NDArray[np.float64],
    *,
    rightmost_closed: bool,
) -> NDArray[np.int64]:
    idx = np.searchsorted(breaks, values, side="right")
    if rightmost_closed:
        idx = np.where(values == breaks[-1], breaks.size, idx)
    idx = idx - 1
    return np.clip(idx, 0, breaks.size - 1).astype(np.int64)


def _fitted_table(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    estimate: NDArray[np.float64],
    nns_ids: NDArray[np.str_],
    coeff: dict[str, NDArray[np.float64]],
) -> dict[str, NDArray[np.float64] | NDArray[np.str_]]:
    y_hat = estimate.copy()
    if np.any(~np.isfinite(y_hat)):
        replacement = _gravity(y_hat[np.isfinite(y_hat)])
        y_hat[~np.isfinite(y_hat)] = replacement
    gradient_idx = _find_interval(x, coeff["X.Lower.Range"], rightmost_closed=False)
    gradient = coeff["Coefficient"][gradient_idx]
    residuals = y_hat - y
    standard_errors = np.empty_like(y_hat)
    for grad in np.unique(gradient):
        mask = gradient == grad
        denom = max(1, int(np.sum(mask)) - 1)
        standard_errors[mask] = math.sqrt(float(np.sum((y_hat[mask] - y[mask]) ** 2)) / denom)
    return {
        "x": x.copy(),
        "y": y.copy(),
        "y.hat": y_hat,
        "NNS.ID": nns_ids,
        "gradient": gradient,
        "residuals": residuals,
        "standard.errors": standard_errors,
    }


def _apply_univariate_intervals(
    fitted: dict[str, NDArray[np.float64] | NDArray[np.str_]],
    point_values: NDArray[np.float64] | None,
    *,
    confidence_interval: float | None,
    class_mode: bool = False,
) -> dict[str, NDArray[np.float64]] | None:
    if confidence_interval is None:
        return None

    alpha = (1.0 - float(confidence_interval)) / 2.0
    y_hat = cast(NDArray[np.float64], fitted["y.hat"])
    y = cast(NDArray[np.float64], fitted["y"])
    residuals = cast(NDArray[np.float64], fitted["residuals"])
    gradient = cast(NDArray[np.float64], fitted["gradient"])

    conf_pos = np.empty_like(y_hat)
    conf_neg = np.empty_like(y_hat)
    pred_pos = np.empty_like(y_hat)
    pred_neg = np.empty_like(y_hat)
    for grad in np.unique(gradient):
        mask = gradient == grad
        residual_var = abs(upm_var(alpha, 1.0, residuals[mask]))
        conf_pos[mask] = y_hat[mask] + residual_var
        conf_neg[mask] = y_hat[mask] - residual_var
        pred_pos[mask] = upm_var(alpha, 0.0, y[mask])
        pred_neg[mask] = lpm_var(alpha, 0.0, y[mask])

    fitted["conf.int.pos"] = conf_pos
    fitted["conf.int.neg"] = conf_neg

    if point_values is None:
        return None

    order = np.argsort(cast(NDArray[np.float64], fitted["x"]), kind="mergesort")
    sorted_x = cast(NDArray[np.float64], fitted["x"])[order]
    row_indices: list[int] = []
    for point in point_values:
        close = np.flatnonzero(np.isclose(sorted_x, point, rtol=1e-12, atol=1e-12))
        if close.size:
            row_indices.append(int(close[-1]))
            continue
        interval_index = int(np.searchsorted(sorted_x, point, side="right"))
        if interval_index > 0:
            row_indices.append(min(interval_index - 1, sorted_x.size - 1))
    if not row_indices:
        return {
            "pred.int.neg": np.array([], dtype=np.float64),
            "pred.int.pos": np.array([], dtype=np.float64),
        }
    selected = np.asarray(row_indices, dtype=np.int64)
    pred_int = {
        "pred.int.neg": pred_neg[order][selected],
        "pred.int.pos": pred_pos[order][selected],
    }
    if class_mode:
        return {key: _round_class_interval(values) for key, values in pred_int.items()}
    return pred_int


def _round_class_interval(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.where(values % 1.0 < 0.5, np.floor(values), np.ceil(values)).astype(np.float64)


def _r2(y: NDArray[np.float64], y_hat: NDArray[np.float64]) -> float:
    y_mean = float(np.mean(y))
    numerator = float(np.sum((y - y_mean) * (y_hat - y_mean)) ** 2)
    denominator = float(np.sum((y - y_mean) ** 2) * np.sum((y_hat - y_mean) ** 2))
    return numerator / denominator if denominator > 0.0 else float("nan")
