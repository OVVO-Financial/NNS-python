"""Repaired NNS regression engine.

Python port of the audited R implementation (R/Regression.R,
R/Multivariate_Regression.R at the NNS 13.1 Beta repair revision):

- one consistent prediction rule for fitted values and point estimates,
- real range-normalized NNS/L1/L2/FACTOR distance dispatch,
- training-fitted encodings and dimension-reduction normalization
  (batch-independent predictions),
- restricted automatic classification (factor/character/logical or exact
  binary 0/1 responses only),
- R's data-frame input rules: column names kept and [point.est] matched by
  name, categorical columns coded in their declared level order,
- bounded Racine-Hastie ``R2`` (squared correlation of fitted and actual),
- ``order = "max"`` with fitted values and predictions from the same rule,
- one prediction-interval row per point (quantile type 8 residual bands),
- strict argument validation mirroring the R messages.

The multivariate prediction rule is vectorized with NumPy/SciPy and matches
R's native kernel (``NNS_mreg_predict_cpp``) to floating-point tolerance.
"""

from __future__ import annotations

import functools
import math
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from nns._labels import (
    Columns,
    as_float,
    codes_for,
    declared_codes,
    declared_levels,
    is_logical,
    is_text,
    level_keys,
    match_columns,
    point_codes,
    r_make_names,
    r_make_unique,
    resolve_factor_levels,
    response_values,
    split_columns,
)
from nns._native import native_fn
from nns._rmean import r_mean
from nns.central_tendencies import nns_gravity, nns_mode
from nns.dependence import nns_dep

Order = int | Literal["max"] | None
NBest = int | Literal["all"] | None

_EPS_INV = 1e-12
# NNS-core MregDistance codes (R's dist argument of the M.reg kernels).
_MREG_DIST_CODES = {"NNS": 0, "L2": 1, "L1": 2, "FACTOR": 3}


# --------------------------------------------------------------------------
# Validation helpers (mirroring .nns_reg_validate_* / .nns_reg_scalar_logical)
# --------------------------------------------------------------------------

def _scalar_logical(value: Any, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"[{name}] must be a single TRUE or FALSE value.")
    return bool(value)


def _validate_order(order: Any) -> Order:
    if order is None:
        return None
    if isinstance(order, str):
        if order.lower() == "max":
            return "max"
        raise ValueError("[order] must be NULL, 'max', or a positive integer.")
    if isinstance(order, bool) or not isinstance(order, (int, np.integer, float, np.floating)):
        raise ValueError("[order] must be NULL, 'max', or a positive integer.")
    value = float(order)
    if not math.isfinite(value) or value < 1 or value != math.floor(value):
        raise ValueError("[order] must be NULL, 'max', or a positive integer.")
    return int(value)


def _validate_nbest(n_best: Any) -> NBest:
    if n_best is None:
        return None
    if isinstance(n_best, str):
        if n_best.lower() == "all":
            return "all"
        raise ValueError("[n.best] must be NULL, 'all', or a positive integer.")
    if isinstance(n_best, bool) or not isinstance(n_best, (int, np.integer, float, np.floating)):
        raise ValueError("[n.best] must be NULL, 'all', or a positive integer.")
    value = float(n_best)
    if not math.isfinite(value) or value < 1 or value != math.floor(value):
        raise ValueError("[n.best] must be NULL, 'all', or a positive integer.")
    return int(value)


def _validate_dist(dist: Any) -> str:
    if dist is None:
        return "NNS"
    if not isinstance(dist, str):
        raise ValueError("[dist] must be one of None, 'NNS', 'L1', 'L2', or 'FACTOR'.")
    value = dist.upper()
    if value not in {"NNS", "L1", "L2", "FACTOR"}:
        raise ValueError("[dist] must be one of None, 'NNS', 'L1', 'L2', or 'FACTOR'.")
    return value


def _validate_noise(noise_reduction: Any) -> str:
    if not isinstance(noise_reduction, str):
        raise ValueError("[noise.reduction] must be one of 'mean', 'median', 'mode', or 'off'.")
    value = noise_reduction.lower()
    if value not in {"mean", "median", "mode", "off"}:
        raise ValueError("[noise.reduction] must be one of 'mean', 'median', 'mode', or 'off'.")
    return value


def _validate_ci(confidence_interval: Any) -> float | None:
    if confidence_interval is None:
        return None
    value = float(confidence_interval)
    if not math.isfinite(value) or value <= 0.0 or value >= 1.0:
        raise ValueError(
            "[confidence.interval] must be NULL or a scalar strictly between 0 and 1."
        )
    return value


# --------------------------------------------------------------------------
# Response / task resolution (.nns_reg_response_vector / .nns_reg_type)
# --------------------------------------------------------------------------

def _response_vector(y: Any) -> tuple[NDArray[Any], list[Any] | None]:
    """R ``.nns_reg_response_vector``: the values plus any declared categories."""
    return response_values(y, "[y] must be a vector or one-column object.")


def _r_number_label(value: float) -> str:
    """A numeric class code as R's ``as.character`` prints it (15 significant digits)."""
    return f"{float(value):.15g}"


def _task(
    type_value: str | None,
    y: NDArray[Any],
    categories: list[Any] | None = None,
    class_levels: Any = None,
) -> dict[str, Any]:
    """R ``.nns_reg_type``.

    A factor response (pandas categorical, or explicit ``class_levels``) is
    coded in its declared level order; a character response in sorted order
    (R ``factor(y)``); everything else is numeric. An object array of numbers
    (``df.to_numpy()`` of a mixed frame) is numeric, not a set of labels.
    """
    if type_value is not None:
        if not isinstance(type_value, str):
            raise ValueError("[type] must be NULL, 'REG', 'CLASS', or 'XONLY'.")
        type_value = type_value.lower()
        if type_value not in {"reg", "class", "xonly"}:
            raise ValueError("[type] must be NULL, 'REG', 'CLASS', or 'XONLY'.")
        # R .nns_reg_type explicitly permits REG to override automatic
        # binary (0/1) classification, but rejects a nonnumeric response.
        if type_value == "reg" and (
            categories is not None or class_levels is not None
            or y.dtype.kind == "b" or is_text(y)
        ):
            raise ValueError("[type = 'REG'] requires a numeric response.")

    declared = class_levels if class_levels is not None else categories
    what = "class_levels" if class_levels is not None else "y categories"
    if declared is not None:
        levels = declared_levels(declared, what)
        y_numeric, _, unknown = declared_codes(y, levels, "[y] contains missing values.")
        if unknown:
            raise ValueError(
                f"[y] contains values absent from {what}: " + ", ".join(unknown)
            )
        class_values = np.unique(y_numeric)
        class_levels_out: list[str] | None = levels
        auto_class = True
    elif y.dtype.kind == "b":
        auto_class = True
        y_numeric = y.astype(np.float64)
        class_values = np.unique(y_numeric)
        class_levels_out = [_r_number_label(v) for v in class_values]
    elif is_text(y):
        auto_class = True
        keys = level_keys(y, "[y] contains missing values.")
        levels = sorted(set(keys))
        y_numeric = codes_for(keys, levels)
        class_values = np.unique(y_numeric)
        class_levels_out = levels
    else:
        y_numeric = as_float(y)
        unique = np.unique(y_numeric)
        auto_class = unique.size == 2 and set(unique.tolist()) == {0.0, 1.0}
        class_values = None
        class_levels_out = None

    is_class = type_value == "class" or (type_value is None and auto_class)
    is_xonly = type_value == "xonly"
    if class_values is None and is_class:
        class_values = np.unique(y_numeric)
        class_levels_out = [_r_number_label(v) for v in class_values]

    return {
        "is_class": is_class,
        "is_xonly": is_xonly,
        "y": y_numeric,
        "class_values": class_values,
        "class_levels": class_levels_out,
    }


# --------------------------------------------------------------------------
# Predictor encoding (.nns_reg_encode_predictors, numeric/object columns)
# --------------------------------------------------------------------------

def _train_columns(x: Any) -> Columns:
    if np.ndim(x) == 0:
        raise ValueError("[x] must contain at least one predictor.")
    if np.ndim(x) > 2:
        raise ValueError("[x] must be a vector, matrix, or data frame.")
    cols = split_columns(x)
    if not cols.values:
        raise ValueError("[x] must contain at least one predictor.")
    return cols


def _raw_column_names(cols: Columns) -> list[str]:
    """R ``.nns_reg_as_frame`` names (``check.names = FALSE``, ``make.unique``).

    These are the names [point.est] columns are matched against.
    """
    p = len(cols.values)
    if cols.labels is not None:
        names = list(cols.labels)
    elif cols.vector:
        names = ["x"]
    else:
        names = [f"V{j + 1}" for j in range(p)]
    names = [nm if nm != "" else f"x{j + 1}" for j, nm in enumerate(names)]
    return r_make_unique(names, ".")


def _input_column_names(x: Any, ncol: int) -> list[str]:
    """Column names for [x], mirroring R's .nns_reg_as_frame default naming.

    A pandas frame contributes its own column labels (kept as they are, then
    made unique with R's ``make.unique``); a bare matrix/array gets
    R's ``as.data.frame`` defaults ``V1, V2, ...``. Blank labels fall back to
    ``x{i}`` as in R.
    """
    names: list[str] | None = None
    cols = getattr(x, "columns", None)
    if cols is not None:
        try:
            names = [str(c) for c in list(cols)]
        except TypeError:
            names = None
    if names is None or len(names) != ncol:
        # R names a bare vector's column "x" and a matrix's columns V1, V2, ...
        if ncol == 1 and np.ndim(x) < 2:
            names = ["x"]
        else:
            names = [f"V{j + 1}" for j in range(ncol)]
    names = [
        nm if nm not in ("", "None") and nm is not None else f"x{j + 1}"
        for j, nm in enumerate(names)
    ]
    # R keeps the raw labels (check.names = FALSE) and only de-duplicates them.
    return r_make_unique(names, ".")


def _prepare_points(point_est: Any, train_names: list[str]) -> list[NDArray[Any]] | None:
    """R ``.nns_reg_prepare_points``: [point.est] columns in training order.

    Labelled columns (a data frame, or a named row such as ``df.iloc[0]``) are
    matched to the training predictors by name: the same names are reordered,
    names disjoint from the training names are taken positionally, and a
    partial overlap is an error. A single predictor is always positional.
    """
    if point_est is None:
        return None
    p = len(train_names)
    cols = split_columns(point_est)
    if cols.vector:
        values = cols.values[0]
        if p == 1:
            return [values]
        if values.size != p:
            raise ValueError(f"A vector [point.est] must have exactly {p} values.")
        row = [values[j : j + 1] for j in range(p)]
        supplied = cols.row_labels
        if supplied is not None and all(nm != "" for nm in supplied):
            ordering = match_columns(r_make_unique(supplied, "."), train_names, "point.est")
            if ordering is not None:
                row = [row[j] for j in ordering]
        return row
    if len(cols.values) != p:
        raise ValueError(f"[point.est] must contain exactly {p} predictor columns.")
    out = list(cols.values)
    supplied = cols.labels
    if p > 1 and supplied is not None and all(nm != "" for nm in supplied):
        ordering = match_columns(r_make_unique(supplied, "."), train_names, "point.est")
        if ordering is not None:
            out = [out[j] for j in ordering]
    return out


def _encode_predictors(
    x: Any,
    point_est: Any,
    factor_2_dummy: bool,
    factor_levels: Any = None,
    names: list[str] | None = None,
) -> dict[str, Any]:
    """R ``.nns_reg_encode_predictors``.

    A column is categorical when it is a pandas categorical, has declared
    ``factor_levels``, or holds labels (strings, including numeric-looking
    strings). Levels are the declared levels in their order; undeclared label
    columns use sorted levels. (R uses first-appearance order for character
    columns, which makes its fit depend on the row order; the sorted order is
    R's ``factor()`` order.) bool columns are factors with levels FALSE/TRUE,
    as R treats logicals. Missing values in a categorical column are an
    error, as in R.
    """
    cols = _train_columns(x)
    p = len(cols.values)
    raw_names = _raw_column_names(cols)
    # Explicit names only prefix the outputs; point_est still matches the training columns.
    column_names = _input_column_names(x, p) if names is None else r_make_unique(names, ".")
    points = _prepare_points(point_est, raw_names)
    declared = resolve_factor_levels(factor_levels, raw_names, cols.labels)

    n = cols.values[0].shape[0]
    if any(z.shape[0] != n for z in cols.values):
        raise ValueError("[x] columns must all have the same length.")

    train_parts: list[NDArray[np.float64]] = []
    point_parts: list[NDArray[np.float64]] | None = None if points is None else []
    metadata: list[dict[str, Any]] = []
    encoded_names: list[str] = []

    for j in range(p):
        nm = column_names[j]
        raw = raw_names[j]
        z = cols.values[j]
        zp = None if points is None else points[j]
        level_decl = declared[j] if declared[j] is not None else cols.categories[j]
        categorical = level_decl is not None or is_text(z) or is_logical(z)

        if categorical:
            missing = f"Predictor '{raw}' contains missing values."
            by_code = False
            if level_decl is None:
                keys_train = level_keys(z, missing)
                levels = sorted(set(keys_train))
                codes_train = codes_for(keys_train, levels)
            else:
                levels = declared_levels(level_decl, f"factor_levels for '{raw}'")
                codes_train, by_code, absent = declared_codes(z, levels, missing)
                if absent:
                    raise ValueError(
                        f"Predictor '{raw}' contains values absent from its levels: "
                        + ", ".join(absent)
                    )
            codes_point = None
            if zp is not None:
                codes_point, unseen = point_codes(
                    zp, levels, by_code,
                    f"[point.est] predictor '{raw}' contains missing values.",
                )
                if unseen:
                    raise ValueError(
                        f"[point.est] predictor {j + 1} contains unseen level(s): "
                        + ", ".join(unseen)
                    )
            if factor_2_dummy:
                train_parts.append(
                    np.column_stack(
                        [(codes_train == k + 1).astype(np.float64) for k in range(len(levels))]
                    )
                )
                encoded_names.extend(
                    f"{nm}_{sanitized}" for sanitized in r_make_names(levels)
                )
                if codes_point is not None and point_parts is not None:
                    point_parts.append(
                        np.column_stack(
                            [(codes_point == k + 1).astype(np.float64)
                             for k in range(len(levels))]
                        )
                    )
            else:
                train_parts.append(codes_train.reshape(-1, 1))
                encoded_names.append(nm)
                if codes_point is not None and point_parts is not None:
                    point_parts.append(codes_point.reshape(-1, 1))
            metadata.append({"categorical": True, "levels": levels})
        else:
            tr = as_float(z).reshape(-1, 1)
            if not np.all(np.isfinite(tr)):
                raise ValueError(
                    f"Predictor {j + 1} must contain only finite numeric values."
                )
            train_parts.append(tr)
            encoded_names.append(nm)
            if zp is not None and point_parts is not None:
                pt = as_float(zp).reshape(-1, 1)
                if not np.all(np.isfinite(pt)):
                    raise ValueError(
                        f"[point.est] predictor {j + 1} must contain only finite numeric values."
                    )
                point_parts.append(pt)
            metadata.append({"categorical": False, "levels": None})

    x_matrix = np.hstack(train_parts).astype(np.float64)
    point_matrix = None if point_parts is None else np.hstack(point_parts).astype(np.float64)
    return {
        "x": x_matrix,
        "point_est": point_matrix,
        "metadata": metadata,
        "encoded_names": encoded_names,
    }


# --------------------------------------------------------------------------
# Shared reducers / class snapping (.nns_reg_reduce_value / .nns_reg_snap_class)
# --------------------------------------------------------------------------

def _reduce_value(z: NDArray[np.float64], noise_reduction: str, is_class: bool = False) -> float:
    z = z[np.isfinite(z)]
    if z.size == 0:
        return float("nan")
    if is_class:
        values, counts = np.unique(z, return_counts=True)
        return float(values[int(np.argmax(counts))])
    if noise_reduction == "mean":
        return r_mean(z)  # R's mean(): exact for tied values, unlike np.mean
    if noise_reduction == "median":
        return float(np.median(z))
    if noise_reduction == "mode":
        # Numeric cells: NNS.mode(z, discrete = FALSE, multi = FALSE), as R's
        # .nns_reg_reduce_value and NNS-core's mreg_setup; class
        # responses use the exact-frequency mode above.
        return float(cast(float, nns_mode(z, discrete=False, multi=False)))
    return float(nns_gravity(z))


def _snap_class(
    values: NDArray[np.float64], class_values: NDArray[np.float64] | None
) -> NDArray[np.float64]:
    if class_values is None or class_values.size == 0:
        return values
    out = np.empty_like(values, dtype=np.float64)
    finite = np.isfinite(values)
    out[~finite] = np.nan
    if np.any(finite):
        idx = np.argmin(
            np.abs(values[finite, None] - class_values[None, :]), axis=1
        )
        out[finite] = class_values[idx]
    return out


# --------------------------------------------------------------------------
# Dependence-driven default order (.nns_reg_dependence / .nns_reg_default_order)
# --------------------------------------------------------------------------

def _rescale_01(values: NDArray[np.float64]) -> NDArray[np.float64]:
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmax == vmin:
        return np.zeros_like(values, dtype=np.float64)
    return (values - vmin) / (vmax - vmin)


def _dependence(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return 0.1
    try:
        # R14 Regression.R .nns_reg_dependence deliberately uses the
        # finest-cell/local statistic, *not* public NNS.dep's max-level score.
        from nns._dep_r14 import dep_local_r14

        d1 = float(dep_local_r14(x, y, asym=True))
    except Exception:
        d1 = float("nan")
    try:
        from nns.copula import _copula, _r_col_means

        m = np.column_stack((_rescale_01(x), _rescale_01(x), _rescale_01(y)))
        reflected = m.copy()
        reflected[:, 2] = 1.0 - reflected[:, 2]
        # R14 takes both orientations to make the depth independent of the
        # sign convention for the response.
        d2_forward = float(_copula(m, _r_col_means(m), continuous=True))
        d2_reverse = float(_copula(reflected, _r_col_means(reflected), continuous=True))
        d2 = 0.5 * (d2_forward + d2_reverse)
    except Exception:
        d2 = float("nan")
    parts = [v for v in (d1, d2) if math.isfinite(v)]
    d = sum(parts) / len(parts) if parts else float("nan")
    if not math.isfinite(d):
        d = 0.1
    return min(1.0, max(0.0, d))


def _default_order(x: NDArray[np.float64], y: NDArray[np.float64]) -> int:
    dep = _dependence(x, y)
    ord_value = max(1, math.floor(dep * 10 + 0.5))
    if y.size < 100:
        ord_value = max(1, math.floor(ord_value / 2))
    return ord_value


# --------------------------------------------------------------------------
# Univariate regression points (.nns_reg_build_points)
# --------------------------------------------------------------------------

def _group_reduce_sorted(
    keys: NDArray[np.float64],
    values: NDArray[np.float64],
    noise_reduction: str,
    is_class: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Group values by key exactly as R's ``split(values, keys)`` does.

    R splits on ``as.character(keys)`` (15 significant digits), so keys that
    agree to 15 digits share a group and the returned key is the string
    round-trip of the original value. Replicated here for parity. Each group
    is reduced with ``_reduce_value`` (via the exact vectorised ``_reduce_cells``).
    """
    order = np.argsort(keys, kind="stable")
    keys_sorted = keys[order]
    values_sorted = np.asarray(values[order], dtype=np.float64)
    # "+ 0.0": R prints -0 as "0", so -0.0 and 0.0 keys form one group.
    key_strings = [f"{v + 0.0:.15g}" for v in keys_sorted]
    if not key_strings:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    labels = np.asarray(key_strings)
    starts = np.flatnonzero(np.concatenate(([True], labels[1:] != labels[:-1])))
    counts = np.diff(np.append(starts, labels.size))
    out_keys = np.asarray([float(key_strings[i]) for i in starts], dtype=np.float64)
    reduced = _reduce_cells(values_sorted, starts, counts, noise_reduction, is_class)
    return out_keys, reduced


def _build_points(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    order: Order,
    noise_reduction: str,
    is_class: bool,
) -> dict[str, NDArray[np.float64]]:
    from nns.part import nns_part

    def reducer(z: NDArray[np.float64]) -> float:
        return _reduce_value(z, noise_reduction, is_class)

    if order == "max":
        xs, ys = _group_reduce_sorted(x, y, noise_reduction, is_class)
        rp_x, rp_y = xs, ys
    else:
        ord_value = _default_order(x, y) if order is None else int(order)
        nr = "mode_class" if is_class else noise_reduction
        part = nns_part(
            x,
            y,
            type="XONLY",
            order=ord_value,
            obs_req=0,
            min_obs_stop=True,
            noise_reduction=cast(Any, nr),
        )
        rp_x = np.asarray(part["regression.points"]["x"], dtype=np.float64)
        rp_y = np.asarray(part["regression.points"]["y"], dtype=np.float64)

    keep = np.isfinite(rp_x) & np.isfinite(rp_y)
    rp_x, rp_y = rp_x[keep], rp_y[keep]
    if rp_x.size == 0:
        raise ValueError("NNS regression produced no finite regression points.")
    order_idx = np.argsort(rp_x, kind="stable")
    rp_x, rp_y = rp_x[order_idx], rp_y[order_idx]

    if np.unique(rp_x).size != rp_x.size:
        rp_x, rp_y = _group_reduce_sorted(rp_x, rp_y, noise_reduction, is_class)

    xmin = float(np.min(x))
    xmax = float(np.max(x))
    ymin = reducer(y[x == xmin])
    ymax = reducer(y[x == xmax])
    rp_x = np.append(rp_x, [xmin, xmax])
    rp_y = np.append(rp_y, [ymin, ymax])
    order_idx = np.argsort(rp_x, kind="stable")
    rp_x, rp_y = rp_x[order_idx], rp_y[order_idx]
    if np.unique(rp_x).size != rp_x.size:
        rp_x, rp_y = _group_reduce_sorted(rp_x, rp_y, noise_reduction, is_class)

    return {"x": rp_x, "y": rp_y}


def _derivative(rp: dict[str, NDArray[np.float64]]) -> dict[str, NDArray[np.float64]]:
    x, y = rp["x"], rp["y"]
    if x.size < 2:
        return {
            "Coefficient": np.asarray([0.0]),
            "X.Lower.Range": x[:1].copy(),
            "X.Upper.Range": x[:1].copy(),
        }
    run = np.diff(x)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        slope = np.where(run == 0.0, 0.0, np.diff(y) / run)
    return {
        "Coefficient": slope,
        "X.Lower.Range": x[:-1].copy(),
        "X.Upper.Range": x[1:].copy(),
    }


def _nknots_smspl(n: int) -> int:
    """R's ``stats:::.nknots.smspl``: number of inner knots for ``n`` unique x."""
    if n < 50:
        return n
    a1, a2, a3, a4 = (math.log2(v) for v in (50.0, 100.0, 140.0, 200.0))
    if n < 200:
        value = 2.0 ** (a1 + (a2 - a1) * (n - 50) / 150)
    elif n < 800:
        value = 2.0 ** (a2 + (a3 - a2) * (n - 200) / 600)
    elif n < 3200:
        value = 2.0 ** (a3 + (a4 - a3) * (n - 800) / 2400)
    else:
        value = 200.0 + (n - 3200) ** 0.2
    return math.trunc(value)


def _smooth_spline_predictor(
    rx: NDArray[np.float64], ry: NDArray[np.float64], spar: float
) -> Any:
    """Port of R's ``stats::smooth.spline(x, y, spar)`` with default arguments.

    R ties the smoothing parameter to dependence via an explicit ``spar`` rather
    than delegating to GCV. This reproduces R's penalized cubic B-spline fit:
    group x values closer than ``tol = 1e-6 * IQR(x)``, scale x to [0, 1], place
    clamped cubic knots at ``.nknots.smspl(nx)`` of the unique abscissas (all of
    them when ``nx < 50``), form the weighted design cross-product and the
    second-derivative penalty Sigma exactly as Fortran ``sgram`` does (linear
    interpolation of B'' on each knot interval, with its ``.3330`` constant),
    then solve ``(X'WX + lambda * Sigma) c = X'W y`` with
    ``lambda = ratio * 256^(3*spar - 1)`` and ``ratio`` from R's partial
    diagonal trace. Returns a callable evaluating the fit (with R's linear
    extrapolation beyond the data range), or ``None`` when R would refuse to fit.
    """
    from scipy.interpolate import BSpline  # type: ignore[import-untyped]
    from scipy.linalg import solveh_banded  # type: ignore[import-untyped]

    x_all = np.asarray(rx, dtype=np.float64)
    y_all = np.asarray(ry, dtype=np.float64)
    q25, q75 = np.quantile(x_all, [0.25, 0.75])
    tol = 1e-6 * float(q75 - q25)
    if not math.isfinite(tol) or tol <= 0.0:
        return None
    # R: xx <- round((x - mean(x)) / tol); one group per distinct rounded value,
    # represented by its smallest x, with the group mean of y and weight = size.
    rounded = np.round((x_all - float(np.mean(x_all))) / tol)
    order = np.argsort(x_all, kind="mergesort")
    xs = x_all[order]
    ys = y_all[order]
    rs = rounded[order]
    is_start = np.concatenate(([True], rs[:-1] < rs[1:]))
    nx = int(np.count_nonzero(is_start))
    if nx < 4:
        return None
    ux = xs[is_start]
    group = np.cumsum(is_start) - 1
    wbar = np.bincount(group).astype(np.float64)
    ybar = np.bincount(group, weights=ys) / wbar

    x0 = ux[0]
    r_ux = ux[-1] - ux[0]
    if r_ux <= 0.0:
        return None
    xbar = (ux - x0) / r_ux
    nknots = _nknots_smspl(nx)
    # R: xbar[seq.int(1, nx, length.out = nknots)] (fractional indices truncate).
    inner = xbar[np.trunc(np.linspace(1.0, float(nx), nknots)).astype(np.intp) - 1]
    knot = np.concatenate(([xbar[0]] * 3, inner, [xbar[-1]] * 3))
    nk = nknots + 2
    k = 3

    design = BSpline.design_matrix(xbar, knot, k).toarray()
    xtwx = design.T @ (wbar[:, None] * design)

    # Fortran sgram: on each knot interval B'' is linear; with y1 = B''(left)
    # and y2 = B''(right) - y1, Sigma += w * (y1 y1' + (y2 y1' + y1 y2')/2
    # + .3330 y2 y2'), where .3330 (not 1/3) is R's constant.
    d2 = np.asarray(BSpline(knot, np.eye(nk), k)(inner, nu=2), dtype=np.float64)
    y1 = d2[:-1]
    y2 = d2[1:] - y1
    width = np.diff(inner)[:, None]
    cross = (width * y2).T @ y1
    sigma = (
        (width * y1).T @ y1
        + 0.5 * (cross + cross.T)
        + 0.333 * ((width * y2).T @ y2)
    )

    diag_h = np.diag(xtwx)
    diag_s = np.diag(sigma)
    denom = float(diag_s[2 : nk - 3].sum())
    if denom <= 0.0:
        return None
    ratio = float(diag_h[2 : nk - 3].sum()) / denom
    lam = ratio * 256.0 ** (3.0 * spar - 1.0)

    # R (sslvrg) solves the 4-band system with a banded Cholesky (LINPACK dpbfa).
    system = xtwx + lam * sigma
    banded = np.zeros((k + 1, nk), dtype=np.float64)
    for offset in range(k + 1):
        banded[k - offset, offset:] = np.diagonal(system, offset)
    try:
        coef = solveh_banded(banded, design.T @ (wbar * ybar))
    except np.linalg.LinAlgError:
        return None
    spline = BSpline(knot, coef, k, extrapolate=False)
    edge_lo = float(spline(0.0))
    slope_lo = float(spline(0.0, nu=1))
    edge_hi = float(spline(1.0))
    slope_hi = float(spline(1.0, nu=1))

    def predict(xout: NDArray[np.float64]) -> NDArray[np.float64]:
        scaled = (np.asarray(xout, dtype=np.float64) - x0) / r_ux
        out = np.asarray(spline(np.clip(scaled, 0.0, 1.0)), dtype=np.float64)
        lo = scaled < 0.0
        hi = scaled > 1.0
        if np.any(lo):
            out[lo] = edge_lo + scaled[lo] * slope_lo
        if np.any(hi):
            out[hi] = edge_hi + (scaled[hi] - 1.0) * slope_hi
        return out

    return predict


def _predict_univariate(
    xout: NDArray[np.float64],
    rp: dict[str, NDArray[np.float64]],
    smooth: bool,
    is_class: bool,
    class_values: NDArray[np.float64] | None,
    smooth_fit: Any = None,
) -> NDArray[np.float64]:
    if xout.size == 0:
        return np.asarray([], dtype=np.float64)
    rx, ry = rp["x"], rp["y"]
    if rx.size == 1:
        pred = np.full(xout.size, ry[0], dtype=np.float64)
    elif smooth and smooth_fit is not None and not is_class:
        pred = np.asarray(smooth_fit(xout), dtype=np.float64)
    elif smooth and rx.size >= 4 and not is_class:
        # Degenerate fit (e.g. < 4 unique points): fall back to interpolation.
        pred = np.interp(xout, rx, ry)
    else:
        pred = np.interp(xout, rx, ry)
        left = xout < rx[0]
        right = xout > rx[-1]
        if np.any(left):
            slope = (ry[1] - ry[0]) / (rx[1] - rx[0])
            pred[left] = ry[0] + (xout[left] - rx[0]) * slope
        if np.any(right):
            slope = (ry[-1] - ry[-2]) / (rx[-1] - rx[-2])
            pred[right] = ry[-1] + (xout[right] - rx[-1]) * slope
    if is_class:
        pred = _snap_class(pred, class_values)
    return np.asarray(pred, dtype=np.float64)


# --------------------------------------------------------------------------
# Metrics and intervals (.nns_reg_r2 / .nns_reg_intervals)
# --------------------------------------------------------------------------

def _r2(actual: NDArray[np.float64], predicted: NDArray[np.float64]) -> float:
    """Racine-Hastie within-sample goodness-of-fit, bounded in [0, 1]."""
    actual_centered = actual - float(np.mean(actual))
    predicted_centered = predicted - float(np.mean(predicted))
    denominator = float(np.sum(actual_centered**2) * np.sum(predicted_centered**2))
    if denominator == 0.0:
        return 1.0 if _r_all_equal(actual, predicted) else 0.0
    numerator = float(np.sum(actual_centered * predicted_centered) ** 2)
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def _r_all_equal(
    target: NDArray[np.float64], current: NDArray[np.float64], tolerance: float = 1.5e-8
) -> bool:
    """R's ``isTRUE(all.equal(target, current))`` for two numeric vectors.

    Missing values must sit at the same positions. Over the elements that
    differ, the mean absolute difference is taken relative to their mean
    ``|target|`` when that exceeds ``tolerance`` and absolute otherwise.
    """
    if target.shape != current.shape:
        return False
    missing = np.isnan(target)
    if not np.array_equal(missing, np.isnan(current)):
        return False
    differ = ~missing & (target != current)
    if not np.any(differ):
        return True
    t = target[differ]
    with np.errstate(invalid="ignore", over="ignore"):
        xy = float(np.mean(np.abs(t - current[differ])))
        xn = float(np.mean(np.abs(t)))
    if math.isfinite(xn) and xn > tolerance:
        xy /= xn
    return not math.isnan(xy) and xy <= tolerance


def _intervals(
    actual: NDArray[np.float64],
    fitted: NDArray[np.float64],
    point_pred: NDArray[np.float64] | None,
    confidence_interval: float | None,
    is_class: bool,
    class_values: NDArray[np.float64] | None,
) -> dict[str, Any]:
    if confidence_interval is None:
        return {"conf_lower": None, "conf_upper": None, "pred_int": None}
    alpha = 1.0 - confidence_interval
    errors = actual - fitted
    q_lo, q_hi = np.quantile(
        errors, [alpha / 2.0, 1.0 - alpha / 2.0], method="median_unbiased"
    )
    conf_lower = fitted + q_lo
    conf_upper = fitted + q_hi

    pred_int = None
    if point_pred is not None:
        lower = point_pred + q_lo
        upper = point_pred + q_hi
        if is_class:
            lower = _snap_class(lower, class_values)
            upper = _snap_class(upper, class_values)
            lo = np.minimum(lower, upper)
            hi = np.maximum(lower, upper)
            lower, upper = lo, hi
        pred_int = {"pred.int.neg": lower, "pred.int.pos": upper}
    return {"conf_lower": conf_lower, "conf_upper": conf_upper, "pred_int": pred_int}


# --------------------------------------------------------------------------
# Dimension reduction (.nns_reg_dimred_coefficients / .nns_reg_dimreduce)
# --------------------------------------------------------------------------

def _dimred_coefficients(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    dim_red_method: Any,
    tau: Any,
    threshold: Any,
) -> NDArray[np.float64]:
    p = x.shape[1]
    if isinstance(dim_red_method, str):
        method = dim_red_method.lower()
        if method not in {"cor", "nns.dep", "nns.caus", "all", "equal"}:
            raise ValueError("Unsupported [dim.red.method].")
    elif isinstance(dim_red_method, (list, tuple, np.ndarray)):
        coef_vec = np.asarray(dim_red_method, dtype=np.float64).reshape(-1)
        if coef_vec.size != p or not np.all(np.isfinite(coef_vec)):
            raise ValueError(
                f"A numeric [dim.red.method] must contain exactly {p} finite coefficients."
            )
        method = "numeric"
    else:
        raise ValueError(
            "[dim.red.method] must be NULL, a supported method, or a numeric vector."
        )

    threshold_value = float(threshold)
    if not math.isfinite(threshold_value) or threshold_value < 0:
        raise ValueError("[threshold] must be a single finite nonnegative number.")

    def cor_coef() -> NDArray[np.float64]:
        from scipy import stats  # type: ignore[import-untyped]

        out = np.zeros(p, dtype=np.float64)
        for j in range(p):
            with np.errstate(all="ignore"):
                rho = stats.spearmanr(x[:, j], y).statistic
            out[j] = rho if math.isfinite(rho) else 0.0
        return out

    def dep_coef() -> NDArray[np.float64]:
        out = np.zeros(p, dtype=np.float64)
        for j in range(p):
            try:
                z = float(nns_dep(x[:, j], y, asym=True)["Dependence"])
            except Exception:
                z = 0.0
            out[j] = z if math.isfinite(z) else 0.0
        return out

    def caus_coef() -> NDArray[np.float64]:
        if tau is None:
            tau_use = "cs"
        else:
            if not isinstance(tau, str) or tau.lower() not in {"cs", "ts"}:
                raise ValueError("[tau] must be NULL, 'cs', or 'ts'.")
            tau_use = tau.lower()
        from nns.causation import _uni_caus

        # The dim-reduction weights call R's Uni.caus() directly, which maps
        # tau="cs" -> 0 and tau="ts" -> a fixed lag of 3 (it does not run the
        # seasonal-period search that NNS.caus_core uses).
        tau_lag = 3 if tau_use == "ts" else 0
        out = np.zeros(p, dtype=np.float64)
        for j in range(p):
            try:
                z = float(_uni_caus(y, x[:, j], tau_lag))
            except Exception:
                z = 0.0
            out[j] = z if math.isfinite(z) else 0.0
        return out

    if method == "cor":
        coef = cor_coef()
    elif method == "nns.dep":
        coef = dep_coef()
    elif method == "nns.caus":
        coef = caus_coef()
    elif method == "all":
        coef = np.mean(
            np.column_stack((caus_coef(), dep_coef(), cor_coef(), np.ones(p))), axis=1
        )
    elif method == "equal":
        coef = np.ones(p, dtype=np.float64)
    else:
        coef = coef_vec

    preserved = coef.copy()
    coef = coef.copy()
    coef[np.abs(coef) < threshold_value] = 0.0
    if not np.any(np.abs(coef) > 0):
        coef = preserved
        if not np.any(np.abs(coef) > 0):
            coef = np.ones(p, dtype=np.float64)
    return coef


def _dimreduce(
    x: NDArray[np.float64],
    point_est: NDArray[np.float64] | None,
    y: NDArray[np.float64],
    dim_red_method: Any,
    tau: Any,
    threshold: Any,
    variable_names: list[str] | None = None,
) -> dict[str, Any]:
    coef = _dimred_coefficients(x, y, dim_red_method, tau, threshold)
    mins = np.min(x, axis=0)
    maxs = np.max(x, axis=0)
    ranges = maxs - mins

    def normalize(m: NDArray[np.float64]) -> NDArray[np.float64]:
        out = np.full_like(m, 0.5, dtype=np.float64)
        active = ranges > 0
        if np.any(active):
            out[:, active] = (m[:, active] - mins[active]) / ranges[active]
        return out

    nx = normalize(x)
    np_points = None if point_est is None else normalize(point_est)
    denominator = int(np.sum(np.abs(coef) > 0))
    if denominator < 1:
        raise ValueError("Dimension reduction retained no predictors.")

    x_star = np.asarray(nx @ coef / denominator, dtype=np.float64)
    point_star = (
        None if np_points is None else np.asarray(np_points @ coef / denominator, dtype=np.float64)
    )
    names = (
        list(variable_names)
        if variable_names is not None and len(variable_names) == x.shape[1]
        else [f"V{j + 1}" for j in range(x.shape[1])]
    )
    equation = {
        "Variable": [*names, "DENOMINATOR"],
        "Coefficient": np.append(coef, float(denominator)),
    }
    return {"x_star": x_star, "point_star": point_star, "equation": equation}


# --------------------------------------------------------------------------
# Multivariate prediction rule (.nns_mreg_* helpers, vectorized)
# --------------------------------------------------------------------------

def _mreg_weighted_mode(values: NDArray[np.float64], weights: NDArray[np.float64]) -> float:
    keep = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    values, weights = values[keep], weights[keep]
    if values.size == 0:
        return float("nan")
    unique, inverse = np.unique(values, return_inverse=True)
    totals = np.zeros(unique.size, dtype=np.float64)
    np.add.at(totals, inverse, weights)
    winners = unique[totals == totals.max()]
    return float(winners.min())


def _normalize_weights_rows(w: NDArray[np.float64]) -> NDArray[np.float64]:
    w = np.where(np.isfinite(w) & (w >= 0), w, 0.0)
    sums = w.sum(axis=1, keepdims=True)
    k = w.shape[1]
    uniform = np.full_like(w, 1.0 / k)
    with np.errstate(invalid="ignore", divide="ignore"):
        scaled = w / sums
    return np.where(sums > 0, scaled, uniform)


def _mreg_ensemble_weights(dk: NDArray[np.float64]) -> NDArray[np.float64]:
    """Vectorized eight-component ensemble weights for (m, k) sorted distances."""
    from scipy import stats  # type: ignore[import-untyped]

    m, k = dk.shape
    if k == 1:
        return np.ones((m, 1), dtype=np.float64)

    total = np.full((m, k), 1.0 / k, dtype=np.float64)  # uniform
    total += _normalize_weights_rows(stats.t.pdf(dk, df=k))
    total += _normalize_weights_rows(1.0 / np.maximum(dk, _EPS_INV))
    for row in _rank_weight_rows(k):
        total += row

    dist_sd = np.std(dk, axis=1, ddof=1, keepdims=True)
    valid_sd = np.isfinite(dist_sd) & (dist_sd > 0)
    if np.any(valid_sd):
        with np.errstate(all="ignore"):
            normal = stats.norm.pdf(dk, loc=0.0, scale=np.where(valid_sd, dist_sd, 1.0))
        normal = np.where(valid_sd, normal, 0.0)
        total += np.where(valid_sd, _normalize_weights_rows(normal), 0.0)

    dist_var = dist_sd ** 2
    valid_var = np.isfinite(dist_var) & (dist_var > 0)
    if np.any(valid_var):
        with np.errstate(all="ignore"):
            rbf = np.exp(-dk / (2.0 * np.where(valid_var, dist_var, 1.0)))
        rbf = np.where(valid_var, rbf, 0.0)
        total += np.where(valid_var, _normalize_weights_rows(rbf), 0.0)

    return _normalize_weights_rows(total)


@functools.lru_cache(maxsize=512)
def _rank_weight_rows(k: int) -> tuple[NDArray[np.float64], ...]:
    """Normalised rank-only ensemble components for k neighbours, in the order
    they are added: exponential, reversed lognormal (when the rank sd is
    positive) and inverse-square rank. They do not depend on the distances, so
    they are computed once per k rather than per query chunk and per k of a
    path; each row is normalised exactly as a row of the broadcast matrix was.
    """
    from scipy import stats

    ranks = np.arange(1, k + 1, dtype=np.float64)
    rows = [_normalize_weights_rows(stats.expon.pdf(ranks, scale=k)[None, :])[0]]
    rank_sd = float(np.std(ranks, ddof=1))
    if math.isfinite(rank_sd) and rank_sd > 0:
        lognormal_row = np.abs(stats.lognorm.logpdf(ranks, s=rank_sd, scale=1.0))
        rows.append(_normalize_weights_rows(lognormal_row[None, :])[0, ::-1].copy())
    rows.append(_normalize_weights_rows((ranks ** -2.0)[None, :])[0])
    for row in rows:
        row.flags.writeable = False
    return tuple(rows)


def _mreg_distances(
    rpm_x: NDArray[np.float64],
    xtest: NDArray[np.float64],
    dist: str,
    mins: NDArray[np.float64],
    maxs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """(m, n) distance matrix under the repaired metric.

    Accumulated one predictor column at a time in (m, n) buffers instead of
    materialising (m, n, p) temporaries. The per-element arithmetic is
    unchanged and the columns are summed sequentially in column order, which
    is the order R's kernel uses and the order NumPy used on the former
    broadcast tensor (its predictor axis had the largest stride, so the
    reduction ran column by column), so results are bit-for-bit the same.
    """
    m, n = xtest.shape[0], rpm_x.shape[0]
    if dist == "FACTOR":
        # Mismatch counts are small integers, so their sum is exact in any order.
        out = np.zeros((m, n), dtype=np.float64)
        for j in range(rpm_x.shape[1]):
            out += xtest[:, j, None] != rpm_x[None, :, j]
        out /= rpm_x.shape[1]
        return out
    ranges = maxs - mins
    active = np.flatnonzero(np.isfinite(ranges) & (ranges > 0))
    if active.size == 0:
        return np.zeros((m, n), dtype=np.float64)

    out = np.empty((m, n), dtype=np.float64)
    buf = np.empty_like(out) if active.size > 1 else out
    sq = np.empty_like(out) if dist == "NNS" else out
    for a, j in enumerate(active):
        z = out if a == 0 else buf
        np.subtract(xtest[:, j, None], rpm_x[None, :, j], out=z)
        z /= ranges[j]
        if dist == "NNS":
            np.multiply(z, z, out=sq)
            np.abs(z, out=z)
            z += sq
        elif dist == "L1":
            np.abs(z, out=z)
        else:
            np.multiply(z, z, out=z)
        if a:
            out += z
    if dist == "L2":
        np.sqrt(out, out=out)
    return out


def _stable_topk(d: NDArray[np.float64], k: int) -> NDArray[np.intp]:
    """Indices of the k nearest RPM rows per query, in stable (distance, index)
    order - identical to ``np.argsort(d, axis=1, kind="stable")[:, :k]`` but via
    partial selection: O(n + k log k) per row instead of a full O(n log n) sort.

    Exactness (including heavy distance ties, e.g. the duplicated 1-D synthetic
    coordinates fed by dy.d_) is preserved: candidates are every row within the
    k-th smallest distance (``<=`` the threshold, so all boundary ties are
    captured), then a stable sort keeps original-index order among equal
    distances before truncating to k.
    """
    n = d.shape[1]
    if k >= n:
        return np.argsort(d, axis=1, kind="stable")
    part = np.argpartition(d, k - 1, axis=1)
    kth = np.take_along_axis(d, part[:, k - 1 : k], axis=1)
    counts = np.count_nonzero(d <= kth, axis=1)
    # Rows with exactly k candidates have no tie at the k-th distance, so the
    # first k partitioned indices are exactly the candidates: sort them by
    # index, then stable-sort by distance.
    sel = np.sort(part[:, :k], axis=1)
    order = np.argsort(np.take_along_axis(d, sel, axis=1), axis=1, kind="stable")
    out = np.take_along_axis(sel, order, axis=1)
    tied = np.flatnonzero(counts != k)
    if tied.size:
        out[tied] = _stable_topk_ties(d[tied], kth[tied, 0], counts[tied], k)
    return out


def _stable_topk_ties(
    d: NDArray[np.float64], kth: NDArray[np.float64], counts: NDArray[np.intp], k: int
) -> NDArray[np.intp]:
    """``_stable_topk`` for rows with ties at (or NaN before) the k-th distance."""
    c = d.shape[0]
    if np.any(counts < k):  # NaN distances: keep the original per-row behaviour
        out = np.empty((c, k), dtype=np.intp)
        for i in range(c):
            row = d[i]
            cand = np.flatnonzero(row <= kth[i])
            order = np.argsort(row[cand], kind="stable")
            out[i] = cand[order][:k]
        return out
    # Candidates come out of nonzero() row-major with ascending column index,
    # and lexsort is stable, so sorting by (row, distance) leaves equal
    # distances in original-index order.
    rows, cols = np.nonzero(d <= kth[:, None])
    cols = cols[np.lexsort((d[rows, cols], rows))]
    starts = np.cumsum(counts) - counts
    return cast(NDArray[np.intp], cols[starts[:, None] + np.arange(k)])


def _nearest_tie_values(
    d: NDArray[np.float64], rpm_yhat: NDArray[np.float64], is_class: bool
) -> NDArray[np.float64]:
    """k = 1 rule per query row: aggregate the y.hat of every RPM row at the
    exact minimum distance (gravity, or the unit-weight mode in class mode).

    A unique nearest row (the usual case) is resolved without a reducer call:
    the gravity or mode of one finite value is that value (gravity's np.median
    maps -0.0 to +0.0, mirrored by ``+ 0.0``) and NaN when it is non-finite.
    """
    hit = d == d.min(axis=1, keepdims=True)
    nhit = hit.sum(axis=1)
    out = np.empty(d.shape[0], dtype=np.float64)
    one = nhit == 1
    v = rpm_yhat[np.argmax(hit[one], axis=1)]
    out[one] = np.where(np.isfinite(v), v if is_class else v + 0.0, np.nan)
    for i in np.flatnonzero(~one):
        tied = rpm_yhat[hit[i]]
        if is_class:
            out[i] = _mreg_weighted_mode(tied, np.ones(tied.size))
        else:
            out[i] = float(nns_gravity(tied[np.isfinite(tied)]))
    return out


def _mreg_weighted_mode_rows(
    values: NDArray[np.float64], weights: NDArray[np.float64]
) -> NDArray[np.float64]:
    """``_mreg_weighted_mode`` applied to every row of (c, k) arrays.

    Per-row class totals are accumulated with one ``np.add.at`` over the
    row-major entries, i.e. in the same order (and from the same 0.0) as the
    per-row version, so ties between classes resolve identically.
    """
    c = values.shape[0]
    out = np.full(c, np.nan, dtype=np.float64)
    keep = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    rows, cols = np.nonzero(keep)
    if rows.size == 0:
        return out
    kept = values[rows, cols]
    classes = np.unique(kept)
    code = np.searchsorted(classes, kept)
    totals = np.zeros((c, classes.size), dtype=np.float64)
    np.add.at(totals, (rows, code), weights[rows, cols])
    present = np.zeros((c, classes.size), dtype=bool)
    present[rows, code] = True
    totals = np.where(present, totals, -np.inf)
    best = totals.max(axis=1, keepdims=True)
    winner = np.argmax(totals == best, axis=1)  # smallest tied class value
    has = present.any(axis=1)
    out[has] = classes[winner[has]]
    return out


def _mreg_predict(
    xtest: NDArray[np.float64] | None,
    rpm_x: NDArray[np.float64],
    rpm_yhat: NDArray[np.float64],
    k: int,
    dist: str,
    mins: NDArray[np.float64],
    maxs: NDArray[np.float64],
    is_class: bool,
) -> NDArray[np.float64] | None:
    if xtest is None:
        return None
    xtest = np.asarray(xtest, dtype=np.float64)
    if xtest.ndim == 1:
        xtest = xtest.reshape(1, -1)
    if xtest.shape[0] == 0:
        return np.asarray([], dtype=np.float64)
    if xtest.shape[1] != mins.size:
        raise ValueError("Prediction data and the fitted RPM have incompatible dimensions.")
    if not np.all(np.isfinite(xtest)):
        raise ValueError("[point.est] contains missing or nonfinite values.")

    n = rpm_x.shape[0]
    k = min(int(k), n)
    native = native_fn("mreg_predict")
    if native is not None and k >= 1:
        # R's NNS_mreg_predict_v2_cpp, ported to NNS-core; equal to the NumPy
        # code below to floating-point rounding.
        return np.asarray(
            native(*_native_mreg_args(rpm_x, rpm_yhat, xtest, mins, maxs), k,
                   _MREG_DIST_CODES[dist], *_native_mreg_ranges(mins, maxs), is_class),
            dtype=np.float64,
        )
    out = np.empty(xtest.shape[0], dtype=np.float64)

    chunk = max(1, int(1_000_000 // max(n, 1)))
    for start in range(0, xtest.shape[0], chunk):
        rows = slice(start, min(start + chunk, xtest.shape[0]))
        d = _mreg_distances(rpm_x, xtest[rows], dist, mins, maxs)

        if k == 1:
            out[rows] = _nearest_tie_values(d, rpm_yhat, is_class)
            continue

        idx = _stable_topk(d, k)
        dk = np.take_along_axis(d, idx, axis=1)
        yk = rpm_yhat[idx]
        w = _mreg_ensemble_weights(dk)
        if is_class:
            out[rows] = _mreg_weighted_mode_rows(yk, w)
        else:
            out[rows] = np.sum(yk * w, axis=1)

    return out


def _mreg_predict_path(
    xtest: NDArray[np.float64],
    rpm_x: NDArray[np.float64],
    rpm_yhat: NDArray[np.float64],
    kmax: int,
    dist: str,
    mins: NDArray[np.float64],
    maxs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Regression predictions for every k = 1..kmax under the same rule as
    ``_mreg_predict`` (R's NNS_mreg_predict_path_cpp)."""
    xtest = np.asarray(xtest, dtype=np.float64)
    if xtest.ndim == 1:
        xtest = xtest.reshape(1, -1)
    n = rpm_x.shape[0]
    kmax = min(int(kmax), n)
    m = xtest.shape[0]
    native = native_fn("mreg_predict_path")
    if native is not None and kmax >= 1:
        # R's NNS_mreg_predict_path_v2_cpp, ported to NNS-core.
        return np.asarray(
            native(*_native_mreg_args(rpm_x, rpm_yhat, xtest, mins, maxs), kmax,
                   _MREG_DIST_CODES[dist], *_native_mreg_ranges(mins, maxs), False),
            dtype=np.float64,
        )
    out = np.empty((m, kmax), dtype=np.float64)

    # Stream the queries in row-chunks so the query-by-RPM distance matrix is
    # never materialised in full, and select only the kmax nearest neighbours
    # (partial selection) rather than sorting every RPM row. Results are
    # byte-for-byte identical to a full stable sort of all rows.
    chunk = max(1, int(1_000_000 // max(n, 1)))
    for start in range(0, m, chunk):
        rows = slice(start, min(start + chunk, m))
        d = _mreg_distances(rpm_x, xtest[rows], dist, mins, maxs)
        idx = _stable_topk(d, kmax)
        d_sorted = np.take_along_axis(d, idx, axis=1)
        y_sorted = rpm_yhat[idx]

        # k = 1 aggregates all exact nearest-distance ties.
        out[rows, 0] = _nearest_tie_values(d, rpm_yhat, False)

        for k in range(2, kmax + 1):
            w = _mreg_ensemble_weights(d_sorted[:, :k])
            out[rows, k - 1] = np.sum(y_sorted[:, :k] * w, axis=1)
    return out


def _native_mreg_args(
    rpm_x: NDArray[np.float64],
    rpm_yhat: NDArray[np.float64],
    xtest: NDArray[np.float64],
    mins: NDArray[np.float64],
    maxs: NDArray[np.float64],
) -> tuple[Any, ...]:
    """Leading (rpm_x, n, p, yhat, xtest, m) arguments of the native M.reg kernels."""
    rpm = np.asarray(rpm_x, dtype=np.float64)
    test = np.asarray(xtest, dtype=np.float64)
    p = rpm.shape[1] if rpm.ndim == 2 else -1
    if test.ndim != 2 or not p == test.shape[1] == mins.size == maxs.size:
        raise ValueError("Prediction data and the fitted RPM have incompatible dimensions.")
    return (
        np.ascontiguousarray(rpm.ravel(order="F")),
        rpm.shape[0],
        rpm.shape[1],
        np.ascontiguousarray(rpm_yhat, dtype=np.float64),
        np.ascontiguousarray(test.ravel(order="F")),
        test.shape[0],
    )


def _native_mreg_ranges(
    mins: NDArray[np.float64], maxs: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    return (
        np.ascontiguousarray(mins, dtype=np.float64),
        np.ascontiguousarray(maxs, dtype=np.float64),
    )


def _find_interval(
    x: NDArray[np.float64], boundaries: NDArray[np.float64]
) -> NDArray[np.int64]:
    """R findInterval(x, v, left.open = FALSE, rightmost.closed = TRUE)."""
    idx = np.searchsorted(boundaries, x, side="right").astype(np.int64)
    if boundaries.size:
        idx[x == boundaries[-1]] = boundaries.size - 1
    return idx


def _mreg_partition_matrix(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    order: Order,
    noise_reduction: str,
    is_class: bool,
) -> dict[str, Any]:
    p = x.shape[1]
    boundaries: list[NDArray[np.float64]] = []
    if order == "max":
        for j in range(p):
            boundaries.append(np.unique(x[:, j]))
    else:
        for j in range(p):
            rp = _build_points(x[:, j], y, order, noise_reduction, is_class)
            b = np.unique(rp["x"])
            if b.size == 0:
                b = np.unique(x[:, j])
            boundaries.append(b)

    max_len = max(b.size for b in boundaries)
    rhs = np.full((max_len, p), np.nan, dtype=np.float64)
    for j, b in enumerate(boundaries):
        rhs[: b.size, j] = b

    if order == "max":
        # order = "max" is a rank problem, not an interval problem: use each
        # coordinate's exact 1-based rank (R's match) so the maximum value keeps
        # its own ID rather than folding into the preceding interval.
        id_parts = [
            np.searchsorted(boundaries[j], x[:, j]).astype(np.int64) + 1 for j in range(p)
        ]
    else:
        id_parts = [_find_interval(x[:, j], boundaries[j]) for j in range(p)]
    ids = np.asarray(
    [".".join(str(int(part[i])) for part in id_parts) for i in range(x.shape[0])]
    )
    return {"rhs": rhs, "ids": ids, "boundaries": boundaries}


def _mreg_build_rpm(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    ids: NDArray[Any],
    noise_reduction: str,
    is_class: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return (rpm_x, rpm_yhat), rows ordered by interval ID (R split order)."""
    order = np.argsort(ids, kind="stable")
    if np.unique(ids).size == ids.size:
        return x[order].astype(np.float64), y[order].astype(np.float64)

    ids_sorted = ids[order]
    x_sorted = np.asarray(x[order], dtype=np.float64)
    y_sorted = np.asarray(y[order], dtype=np.float64)
    _, starts = np.unique(ids_sorted, return_index=True)
    starts = np.sort(starts)
    counts = np.diff(np.append(starts, ids_sorted.size))
    rpm_x = np.column_stack(
        [
            _reduce_cells(x_sorted[:, j], starts, counts, noise_reduction, False)
            for j in range(x.shape[1])
        ]
    )
    rpm_yhat = _reduce_cells(y_sorted, starts, counts, noise_reduction, is_class)
    return rpm_x, rpm_yhat


def _reduce_cells(
    values: NDArray[np.float64],
    starts: NDArray[np.intp],
    counts: NDArray[np.intp],
    noise_reduction: str,
    is_class: bool,
) -> NDArray[np.float64]:
    """``_reduce_value`` over each cell ``values[s:s+c]``, vectorised where exact.

    One- to three-member cells (most cells of a fine partition) have closed
    forms that reproduce ``_reduce_value`` bit for bit: a lone finite value is
    returned as is, and the median/gravity of two or three finite values is
    ``(lo + hi) / 2`` or the middle value. NumPy's mean/median accumulate from
    +0.0, so ``+ 0.0`` reproduces their -0.0 -> +0.0 normalisation. Larger
    cells, non-finite members and the mode/class reducers use the reducer.
    """
    first = values[starts]
    out = np.empty(starts.size, dtype=np.float64)
    pending = np.ones(starts.size, dtype=bool)

    single = counts == 1
    if is_class or noise_reduction == "mode":
        out[single] = np.where(np.isfinite(first[single]), first[single], np.nan)
    else:
        out[single] = np.where(np.isfinite(first[single]), first[single] + 0.0, np.nan)
    pending[single] = False

    if not is_class and noise_reduction in ("off", "median"):
        for size in (2, 3):
            cells = np.flatnonzero(counts == size)
            if cells.size == 0:
                continue
            block = values[starts[cells, None] + np.arange(size)]
            exact = np.all(np.isfinite(block), axis=1)
            cells, block = cells[exact], np.sort(block[exact], axis=1)
            if size == 2:
                out[cells] = (block[:, 0] + block[:, 1]) / 2.0 + 0.0
            else:
                out[cells] = block[:, 1] + 0.0
            pending[cells] = False

    for g in np.flatnonzero(pending):
        s = starts[g]
        out[g] = _reduce_value(values[s : s + counts[g]], noise_reduction, is_class)
    return out


def _mreg_prepare(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    order: Order,
    noise_reduction: str,
    is_class: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Partition + RPM only (R's .nns_mreg_prepare_model): (rpm_x, rpm_yhat)."""
    partition = _mreg_partition_matrix(x, y, order, noise_reduction, is_class)
    return _mreg_build_rpm(x, y, partition["ids"], noise_reduction, is_class)


def _mreg_default_nbest(
    x: NDArray[np.float64], y: NDArray[np.float64], rpm_rows: int
) -> int:
    try:
        from nns.copula import nns_copula

        dep = float(nns_copula(np.column_stack((x, y))))
    except Exception:
        dep = float("nan")
    if not math.isfinite(dep):
        dep = 0.5
    k = max(1, math.floor((1.0 - dep) * math.sqrt(x.shape[0])))
    return min(k, rpm_rows)


def _m_reg(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    order: Order,
    n_best: NBest,
    is_class: bool,
    point_est: NDArray[np.float64] | None,
    point_only: bool,
    noise_reduction: str,
    dist: str,
    confidence_interval: float | None,
    class_values: NDArray[np.float64] | None,
    class_levels: list[str] | None,
    variable_names: list[str] | None = None,
    categorical: bool = False,
) -> dict[str, Any]:
    if x.shape[0] != y.size:
        raise ValueError(f"[X_n] has {x.shape[0]} rows but [Y] has {y.size} values.")
    if x.shape[1] < 2:
        raise ValueError("NNS.M.reg requires at least two encoded predictors.")
    if y.size < 2 or not np.all(np.isfinite(y)):
        raise ValueError("[Y] must contain at least two finite values.")

    mins = np.min(x, axis=0)
    maxs = np.max(x, axis=0)

    # R 14.0 NNS.M.reg: fully categorical designs are exact joint-cell
    # models, not continuous multivariate neighbor models.  Factor dummies
    # retain their original categorical status after numeric encoding.
    if categorical:
        order = "max"
    partition = _mreg_partition_matrix(x, y, order, noise_reduction, is_class)
    rpm_x, rpm_yhat = _mreg_build_rpm(x, y, partition["ids"], noise_reduction, is_class)
    if rpm_x.shape[0] == 0:
        raise ValueError("NNS.M.reg produced an empty RPM.")

    if n_best is None and (categorical or order == "max"):
        k = 1
    elif n_best is None:
        k = _mreg_default_nbest(x, y, rpm_x.shape[0])
    elif n_best == "all":
        k = rpm_x.shape[0]
    else:
        k = min(int(n_best), rpm_x.shape[0])
    k = max(1, k)

    if point_only and confidence_interval is None:
        # Fitted values are only reported (R2, Fitted.xy) or used (interval
        # residual quantiles) when point.only is FALSE or an interval is
        # requested; skip the n x nrow(RPM) training-row search otherwise.
        fitted_pred = np.full(y.size, np.nan)
    elif order == "max" and k == 1:
        # R's exact limit shortcut (Multivariate_Regression.R): with order =
        # "max" every distinct training row is its own regression point and
        # duplicate rows share one, so the k = 1 rule returns that point's
        # y.hat. Map rows to their RPM row (RPM rows are in sorted-ID order)
        # instead of an n x nrow(RPM) distance search. "+ 0.0" mirrors the
        # gravity of a single tied value (-0.0 -> +0.0) in regression mode.
        _, rpm_index = np.unique(partition["ids"], return_inverse=True)
        fitted_pred = np.asarray(rpm_yhat[rpm_index.reshape(-1)], dtype=np.float64)
        if not is_class:
            fitted_pred = fitted_pred + 0.0
    else:
        fitted_pred = cast(
            NDArray[np.float64],
            _mreg_predict(x, rpm_x, rpm_yhat, k, dist, mins, maxs, is_class),
        )
    point_pred = _mreg_predict(point_est, rpm_x, rpm_yhat, k, dist, mins, maxs, is_class)

    if is_class:
        observed = np.unique(y)
        fitted_pred = _snap_class(fitted_pred, observed)
        if point_pred is not None:
            point_pred = _snap_class(point_pred, observed)

    metric: float | None = None
    if not point_only:
        metric = float(np.mean(fitted_pred == y)) if is_class else _r2(y, fitted_pred)
    intervals = _intervals(
        y, fitted_pred, point_pred, confidence_interval, is_class,
        np.unique(y) if is_class else None,
    )

    names = (
        list(variable_names)
        if variable_names is not None and len(variable_names) == x.shape[1]
        else [f"V{j + 1}" for j in range(x.shape[1])]
    )

    rpm_dict: dict[str, NDArray[np.float64]] = {
        names[j]: rpm_x[:, j] for j in range(rpm_x.shape[1])
    }
    rpm_dict["y.hat"] = rpm_yhat

    # rhs.partitions is R's data.frame of per-column interval boundaries, padded
    # with NaN to the longest column and keyed by the encoded predictor names.
    rhs = partition["rhs"]
    rhs_dict: dict[str, NDArray[np.float64]] = {
        names[j]: rhs[:, j] for j in range(rhs.shape[1])
    }

    fitted_xy: dict[str, Any] = {names[j]: x[:, j] for j in range(x.shape[1])}
    fitted_xy.update(
        {
            "y": y,
            "y.hat": fitted_pred,
            "NNS.ID": partition["ids"],
            "residuals": fitted_pred - y,
        }
    )
    if intervals["conf_lower"] is not None:
        fitted_xy["conf.int.neg"] = intervals["conf_lower"]
        fitted_xy["conf.int.pos"] = intervals["conf_upper"]

    out: dict[str, Any] = {
        "R2": metric,
        "rhs.partitions": rhs_dict,
        "RPM": rpm_dict,
        "Point.est": point_pred,
        "pred.int": intervals["pred_int"],
        "Fitted.xy": None if point_only else fitted_xy,
        "n.best": k,
        "dist": dist,
        "class.levels": class_levels,
    }
    return out


# --------------------------------------------------------------------------
# Main entry point (NNS.reg)
# --------------------------------------------------------------------------

def nns_reg_engine(
    x: Any,
    y: Any,
    *,
    factor_2_dummy: bool = True,
    order: Any = None,
    dim_red_method: Any = None,
    tau: Any = None,
    type: str | None = None,
    point_est: Any = None,
    return_values: bool = True,
    plot: bool = False,
    residual_plot: bool = False,
    confidence_interval: Any = None,
    threshold: Any = 0.0,
    n_best: Any = None,
    smooth: bool = False,
    noise_reduction: str = "off",
    dist: str | None = None,
    point_only: bool = False,
    multivariate_call: bool = False,
    class_levels: Any = None,
    factor_levels: Any = None,
    multivariate_only: bool = False,
) -> dict[str, Any]:
    """Port of R ``NNS.reg`` (and, with ``multivariate_only``, ``NNS.M.reg``).

    ``class_levels`` declares the response levels (codes ``1..K`` in that
    order, as for an R factor) and implies classification; ``factor_levels``
    declares predictor levels (see :func:`nns._labels.resolve_factor_levels`).
    """
    factor_2_dummy = _scalar_logical(factor_2_dummy, "factor.2.dummy")
    smooth = _scalar_logical(smooth, "smooth")
    point_only = _scalar_logical(point_only, "point.only")
    multivariate_call = _scalar_logical(multivariate_call, "multivariate.call")
    order = _validate_order(order)
    n_best = _validate_nbest(n_best)
    dist = _validate_dist(dist)
    noise_reduction = _validate_noise(noise_reduction)
    confidence_interval = _validate_ci(confidence_interval)

    y_raw, y_categories = _response_vector(y)
    if y_raw.size < 2:
        raise ValueError("[y] must contain at least two observations.")
    task = _task(type, y_raw, y_categories, class_levels)
    y_numeric = cast(NDArray[np.float64], task["y"])
    if not np.all(np.isfinite(y_numeric)):
        raise ValueError("[y] must contain finite values.")

    encoded = _encode_predictors(x, point_est, factor_2_dummy, factor_levels)
    ex = cast(NDArray[np.float64], encoded["x"])
    ep = cast(NDArray[np.float64] | None, encoded["point_est"])
    if ex.shape[0] != y_numeric.size:
        raise ValueError(
            f"[x] has {ex.shape[0]} rows but [y] has {y_numeric.size} values."
        )
    if multivariate_only and ex.shape[1] < 2:
        raise ValueError("NNS.M.reg requires at least two encoded predictors.")
    if task["is_xonly"] and ex.shape[1] != 1:
        raise ValueError("[type = 'XONLY'] is only valid for a single encoded predictor.")

    # R14 nominal classes are probability-valued indicators, never numeric
    # regression on arbitrary class codes. This branch precedes NNS.M.reg and
    # the former univariate class interpolation. NNS.M.reg's standalone
    # `multivariate_only` mode retains its own separate public contract.
    if task["is_class"] and not multivariate_only and not multivariate_call:
        from nns._class_r14 import classify

        return classify(
            encoded, task, order=order, n_best=n_best, dist=dist,
            dim_red_method=dim_red_method, tau=tau, threshold=threshold,
            point_only=point_only, confidence_interval=confidence_interval,
        )

    # Full multivariate regression.
    if ex.shape[1] > 1 and dim_red_method is None:
        return _m_reg(
            ex,
            y_numeric,
            order=order,
            n_best=n_best,
            is_class=task["is_class"],
            point_est=ep,
            point_only=point_only,
            noise_reduction=noise_reduction,
            dist=dist,
            confidence_interval=confidence_interval,
            class_values=task["class_values"],
            class_levels=task["class_levels"],
            variable_names=encoded.get("encoded_names"),
            categorical=all(meta["categorical"] for meta in encoded["metadata"]),
        )

    synthetic_equation = None
    x_star = None
    if ex.shape[1] > 1:
        dr = _dimreduce(
            ex, ep, y_numeric, dim_red_method, tau, threshold,
            variable_names=encoded.get("encoded_names"),
        )
        ux = dr["x_star"]
        up = dr["point_star"]
        synthetic_equation = dr["equation"]
        x_star = {"x": ux}
    else:
        ux = ex[:, 0].astype(np.float64)
        up = None if ep is None else ep[:, 0].astype(np.float64)

    rp = _build_points(ux, y_numeric, order, noise_reduction, task["is_class"])

    if multivariate_call:
        return {"x": rp["x"], "y": rp["y"]}

    # R ties the smoothing spline's spar to dependence and fits it once, reusing
    # the fit for both the training and point predictions.
    smooth_fit = None
    if smooth and rp["x"].size >= 4 and not task["is_class"]:
        dependence = _dependence(ux, y_numeric)
        spar = (dependence + 0.5) / 2.0
        smooth_fit = _smooth_spline_predictor(rp["x"], rp["y"], spar)

    fitted_pred = _predict_univariate(
        ux, rp, smooth, task["is_class"], task["class_values"], smooth_fit
    )
    point_pred = (
        None
        if up is None
        else _predict_univariate(
            up, rp, smooth, task["is_class"], task["class_values"], smooth_fit
        )
    )

    derivative = _derivative(rp)
    ids = np.clip(_find_interval(ux, rp["x"]), 1, rp["x"].size)
    grad_idx = np.clip(
        _find_interval(ux, derivative["X.Lower.Range"]), 1, derivative["Coefficient"].size
    )

    metric = (
        float(np.mean(fitted_pred == y_numeric))
        if task["is_class"]
        else _r2(y_numeric, fitted_pred)
    )
    se = float(np.sqrt(np.mean((fitted_pred - y_numeric) ** 2)))
    intervals = _intervals(
        y_numeric, fitted_pred, point_pred, confidence_interval,
        task["is_class"], task["class_values"],
    )

    fitted_xy: dict[str, Any] = {
        "x": ux,
        "y": y_numeric,
        "y.hat": fitted_pred,
        "NNS.ID": ids,
        "gradient": derivative["Coefficient"][grad_idx - 1],
        "residuals": fitted_pred - y_numeric,
    }
    if intervals["conf_lower"] is not None:
        fitted_xy["conf.int.neg"] = intervals["conf_lower"]
        fitted_xy["conf.int.pos"] = intervals["conf_upper"]

    return {
        "R2": None if point_only else metric,
        "SE": None if point_only else se,
        "Prediction.Accuracy": (metric if task["is_class"] and not point_only else None),
        "equation": synthetic_equation,
        "x.star": x_star,
        "derivative": derivative,
        "Point.est": point_pred,
        "pred.int": intervals["pred_int"],
        "regression.points": {"x": rp["x"], "y": rp["y"]},
        "Fitted.xy": None if point_only else fitted_xy,
        "class.levels": task["class_levels"],
    }
