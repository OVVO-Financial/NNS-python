"""Repaired NNS regression engine.

Python port of the audited R implementation (R/Regression.R,
R/Multivariate_Regression.R at the NNS 13.1 Beta repair revision):

- one consistent prediction rule for fitted values and point estimates,
- real range-normalized L1/L2/FACTOR distance dispatch,
- training-fitted encodings and dimension-reduction normalization
  (batch-independent predictions),
- restricted automatic classification (factor/character/logical or exact
  binary 0/1 responses only),
- predictive ``R2 = 1 - SSE/SST``,
- ``order = "max"`` with fitted values and predictions from the same rule,
- one prediction-interval row per point (quantile type 8 residual bands),
- strict argument validation mirroring the R messages.

The multivariate prediction rule is vectorized with NumPy/SciPy and matches
R's native kernel (``NNS_mreg_predict_cpp``) to floating-point tolerance.
"""

from __future__ import annotations

import math
import re
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from nns.central_tendencies import nns_gravity, nns_mode
from nns.dependence import nns_dep

Order = int | Literal["max"] | None
NBest = int | Literal["all"] | None

_EPS_INV = 1e-12


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
    if not isinstance(dist, str):
        raise ValueError("[dist] must be one of 'L1', 'L2', or 'FACTOR'.")
    value = dist.upper()
    if value not in {"L1", "L2", "FACTOR"}:
        raise ValueError("[dist] must be one of 'L1', 'L2', or 'FACTOR'.")
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

def _response_vector(y: Any) -> NDArray[Any]:
    values = np.asarray(y)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1:
        raise ValueError("[y] must be a vector or one-column object.")
    return values


def _task(type_value: str | None, y: NDArray[Any]) -> dict[str, Any]:
    if type_value is not None:
        if not isinstance(type_value, str):
            raise ValueError("[type] must be NULL, 'CLASS', or 'XONLY'.")
        type_value = type_value.lower()
        if type_value not in {"class", "xonly"}:
            raise ValueError("[type] must be NULL, 'CLASS', or 'XONLY'.")

    categorical = y.dtype.kind in {"U", "S", "O", "b"}
    if categorical:
        auto_class = True
    else:
        numeric = y.astype(np.float64)
        unique = np.unique(numeric)
        auto_class = unique.size == 2 and set(unique.tolist()) == {0.0, 1.0}

    is_class = type_value == "class" or (type_value is None and auto_class)
    is_xonly = type_value == "xonly"

    if y.dtype.kind == "b":
        y_numeric = y.astype(np.float64)
        class_values = np.unique(y_numeric)
        class_levels = [str(v) for v in class_values]
    elif categorical:
        as_str = np.asarray([str(v) for v in y.tolist()])
        levels = sorted(set(as_str.tolist()))
        codes = np.asarray([levels.index(v) + 1 for v in as_str], dtype=np.float64)
        y_numeric = codes
        class_values = np.unique(codes)
        class_levels = levels
    else:
        y_numeric = y.astype(np.float64)
        class_values = np.unique(y_numeric) if is_class else None
        class_levels = [str(v) for v in class_values] if is_class else None

    return {
        "is_class": is_class,
        "is_xonly": is_xonly,
        "y": y_numeric,
        "class_values": class_values,
        "class_levels": class_levels,
    }


# --------------------------------------------------------------------------
# Predictor encoding (.nns_reg_encode_predictors, numeric/object columns)
# --------------------------------------------------------------------------

def _as_train_matrix(x: Any) -> NDArray[Any]:
    values = np.asarray(x)
    if values.ndim == 0:
        raise ValueError("[x] must contain at least one predictor.")
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise ValueError("[x] must be a vector, matrix, or data frame.")
    return values


def _make_names(labels: list[str]) -> list[str]:
    """Mimic R's make.names(): sanitise to syntactically valid, unique names."""
    out: list[str] = []
    for raw in labels:
        s = re.sub(r"[^0-9A-Za-z._]", ".", str(raw))
        if s == "" or not re.match(r"^[A-Za-z.]", s) or re.match(r"^\.[0-9]", s):
            s = "X" + s
        out.append(s)
    # make.unique: append .1, .2, ... to duplicates
    seen: dict[str, int] = {}
    unique: list[str] = []
    for s in out:
        if s in seen:
            seen[s] += 1
            unique.append(f"{s}.{seen[s]}")
        else:
            seen[s] = 0
            unique.append(s)
    return unique


def _input_column_names(x: Any, ncol: int) -> list[str]:
    """Column names for [x], mirroring R's .nns_reg_as_frame default naming.

    A pandas frame contributes its own column labels; a bare matrix/array gets
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
    return _make_names(names)


def _prepare_points(point_est: Any, p: int) -> NDArray[Any] | None:
    if point_est is None:
        return None
    values = np.asarray(point_est)
    if values.ndim == 0:
        values = values.reshape(1, 1)
    if values.ndim == 1:
        if p == 1:
            values = values.reshape(-1, 1)
        elif values.size == p:
            values = values.reshape(1, -1)
        else:
            raise ValueError(f"A vector [point.est] must have exactly {p} values.")
    if values.ndim != 2 or values.shape[1] != p:
        raise ValueError(f"[point.est] must contain exactly {p} predictor columns.")
    return values


def _encode_predictors(
    x: Any,
    point_est: Any,
    factor_2_dummy: bool,
) -> dict[str, Any]:
    train = _as_train_matrix(x)
    points = _prepare_points(point_est, train.shape[1])
    column_names = _input_column_names(x, train.shape[1])

    train_parts: list[NDArray[np.float64]] = []
    point_parts: list[NDArray[np.float64]] | None = None if points is None else []
    metadata: list[dict[str, Any]] = []
    encoded_names: list[str] = []

    for j in range(train.shape[1]):
        nm = column_names[j]
        z = train[:, j]
        zp = None if points is None else points[:, j]
        categorical = z.dtype.kind in {"U", "S", "O", "b"} and not _numeric_like(z)

        if categorical:
            values_train = np.asarray([str(v) for v in z.tolist()])
            levels = sorted(set(values_train.tolist()))
            if zp is not None:
                values_point = np.asarray([str(v) for v in zp.tolist()])
                unseen = sorted(set(values_point.tolist()) - set(levels))
                if unseen:
                    raise ValueError(
                        f"[point.est] predictor {j + 1} contains unseen level(s): "
                        + ", ".join(unseen)
                    )
            if factor_2_dummy:
                tr = np.column_stack(
                    [(values_train == level).astype(np.float64) for level in levels]
                )
                train_parts.append(tr)
                encoded_names.extend(
                    f"{nm}_{sanitized}" for sanitized in _make_names(list(levels))
                )
                if zp is not None and point_parts is not None:
                    pt = np.column_stack(
                        [(values_point == level).astype(np.float64) for level in levels]
                    )
                    point_parts.append(pt)
            else:
                codes = np.asarray(
                    [levels.index(v) + 1 for v in values_train.tolist()], dtype=np.float64
                ).reshape(-1, 1)
                train_parts.append(codes)
                encoded_names.append(nm)
                if zp is not None and point_parts is not None:
                    pcodes = np.asarray(
                        [levels.index(v) + 1 for v in values_point.tolist()],
                        dtype=np.float64,
                    ).reshape(-1, 1)
                    point_parts.append(pcodes)
            metadata.append({"categorical": True, "levels": levels})
        else:
            tr = np.asarray(z, dtype=np.float64).reshape(-1, 1)
            if not np.all(np.isfinite(tr)):
                raise ValueError(
                    f"Predictor {j + 1} must contain only finite numeric values."
                )
            train_parts.append(tr)
            encoded_names.append(nm)
            if zp is not None and point_parts is not None:
                pt = np.asarray(zp, dtype=np.float64).reshape(-1, 1)
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


def _numeric_like(z: NDArray[Any]) -> bool:
    if z.dtype.kind in {"f", "i", "u"}:
        return True
    try:
        np.asarray(z, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return True


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
        return float(np.mean(z))
    if noise_reduction == "median":
        return float(np.median(z))
    if noise_reduction == "mode":
        return float(cast(float, nns_mode(z)))
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
        d1 = float(nns_dep(x, y, asym=True)["Dependence"])
    except Exception:
        d1 = float("nan")
    try:
        from nns.copula import _copula

        m = np.column_stack((_rescale_01(x), _rescale_01(x), _rescale_01(y)))
        target = cast(NDArray[np.float64], np.mean(m, axis=0))
        d2 = float(_copula(m, target, continuous=True))
    except Exception:
        d2 = float("nan")
    parts = [v for v in (d1, d2) if math.isfinite(v)]
    d = sum(parts) / len(parts) if parts else float("nan")
    if not math.isfinite(d):
        d = 0.1
    return min(1.0, max(0.0, d))


def _default_order(x: NDArray[np.float64], y: NDArray[np.float64]) -> int:
    dep = _dependence(x, y)
    ord_value = max(1, int(math.floor(dep * 10 + 0.5)))
    if y.size < 100:
        ord_value = max(1, int(math.floor(ord_value / 2)))
    return ord_value


# --------------------------------------------------------------------------
# Univariate regression points (.nns_reg_build_points)
# --------------------------------------------------------------------------

def _group_reduce_sorted(
    keys: NDArray[np.float64],
    values: NDArray[np.float64],
    reducer: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Group values by key exactly as R's ``split(values, keys)`` does.

    R splits on ``as.character(keys)`` (15 significant digits), so keys that
    agree to 15 digits share a group and the returned key is the string
    round-trip of the original value. Replicated here for parity.
    """
    order = np.argsort(keys, kind="stable")
    keys_sorted = keys[order]
    values_sorted = values[order]
    key_strings = [f"{v:.15g}" for v in keys_sorted]

    out_keys: list[float] = []
    reduced: list[float] = []
    start = 0
    for i in range(1, len(key_strings) + 1):
        if i == len(key_strings) or key_strings[i] != key_strings[start]:
            out_keys.append(float(key_strings[start]))
            reduced.append(reducer(values_sorted[start:i]))
            start = i
    return np.asarray(out_keys, dtype=np.float64), np.asarray(reduced, dtype=np.float64)


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
        xs, ys = _group_reduce_sorted(x, y, reducer)
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
        rp_x, rp_y = _group_reduce_sorted(rp_x, rp_y, reducer)

    xmin = float(np.min(x))
    xmax = float(np.max(x))
    ymin = reducer(y[x == xmin])
    ymax = reducer(y[x == xmax])
    rp_x = np.append(rp_x, [xmin, xmax])
    rp_y = np.append(rp_y, [ymin, ymax])
    order_idx = np.argsort(rp_x, kind="stable")
    rp_x, rp_y = rp_x[order_idx], rp_y[order_idx]
    if np.unique(rp_x).size != rp_x.size:
        rp_x, rp_y = _group_reduce_sorted(rp_x, rp_y, reducer)

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


def _smooth_spline_predictor(
    rx: NDArray[np.float64], ry: NDArray[np.float64], spar: float
) -> Any:
    """Faithful port of R's ``stats::smooth.spline(x, y, spar)``.

    R ties the smoothing parameter to dependence via an explicit ``spar`` rather
    than delegating to GCV. This reproduces R's penalized cubic B-spline fit:
    scale x to [0, 1], place clamped cubic knots at the unique abscissas, form
    the weighted design cross-product and the second-derivative penalty (Sigma),
    then solve ``(X'WX + lambda * Sigma) c = X'W y`` with
    ``lambda = ratio * 256^(3*spar - 1)`` and ``ratio`` from R's partial diagonal
    trace. Returns a callable evaluating the fit (with R's linear extrapolation
    beyond the data range), or ``None`` when the fit is degenerate.
    """
    from scipy.interpolate import BSpline  # type: ignore[import-untyped]
    from scipy.linalg import solve  # type: ignore[import-untyped]

    order = np.argsort(rx, kind="mergesort")
    xs = np.asarray(rx, dtype=np.float64)[order]
    ys = np.asarray(ry, dtype=np.float64)[order]
    ux, inverse = np.unique(xs, return_inverse=True)
    nx = ux.size
    if nx < 4:
        return None
    wbar = np.bincount(inverse).astype(np.float64)
    ybar = np.bincount(inverse, weights=ys) / wbar

    x0 = ux[0]
    r_ux = ux[-1] - ux[0]
    if r_ux <= 0.0:
        return None
    xbar = (ux - x0) / r_ux
    knot = np.concatenate(([xbar[0]] * 3, xbar, [xbar[-1]] * 3))
    nk = nx + 2
    k = 3

    design = BSpline.design_matrix(xbar, knot, k).toarray()
    xtwx = design.T @ (wbar[:, None] * design)

    sigma = np.zeros((nk, nk), dtype=np.float64)
    gl_x, gl_w = np.polynomial.legendre.leggauss(4)
    unique_knots = np.unique(knot)
    basis = [BSpline(knot, np.eye(nk)[j], k) for j in range(nk)]
    for a, b in zip(unique_knots[:-1], unique_knots[1:]):
        if b <= a:
            continue
        mid = 0.5 * (a + b)
        half = 0.5 * (b - a)
        pts = mid + half * gl_x
        wts = half * gl_w
        d2 = np.column_stack([basis[j](pts, nu=2) for j in range(nk)])
        sigma += (d2 * wts[:, None]).T @ d2

    diag_h = np.diag(xtwx)
    diag_s = np.diag(sigma)
    denom = float(diag_s[2 : nk - 3].sum())
    if denom <= 0.0:
        return None
    ratio = float(diag_h[2 : nk - 3].sum()) / denom
    lam = ratio * 256.0 ** (3.0 * spar - 1.0)

    try:
        coef = solve(xtwx + lam * sigma, design.T @ (wbar * ybar), assume_a="sym")
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
    sse = float(np.sum((actual - predicted) ** 2))
    sst = float(np.sum((actual - np.mean(actual)) ** 2))
    if sst == 0.0:
        return 1.0 if sse == 0.0 else 0.0
    return 1.0 - sse / sst


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
        "Variable": names + ["DENOMINATOR"],
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
    ranks = np.arange(1, k + 1, dtype=np.float64)

    total = np.full((m, k), 1.0 / k, dtype=np.float64)  # uniform
    total += _normalize_weights_rows(stats.t.pdf(dk, df=k))
    total += _normalize_weights_rows(1.0 / np.maximum(dk, _EPS_INV))

    exponential = _normalize_weights_rows(
        np.broadcast_to(stats.expon.pdf(ranks, scale=k), (m, k)).copy()
    )
    total += exponential

    rank_sd = float(np.std(ranks, ddof=1))
    if math.isfinite(rank_sd) and rank_sd > 0:
        lognormal_row = np.abs(stats.lognorm.logpdf(ranks, s=rank_sd, scale=1.0))
        lognormal = _normalize_weights_rows(
            np.broadcast_to(lognormal_row, (m, k)).copy()
        )[:, ::-1]
        total += lognormal

    total += _normalize_weights_rows(np.broadcast_to(ranks ** -2.0, (m, k)).copy())

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


def _mreg_distances(
    rpm_x: NDArray[np.float64],
    xtest: NDArray[np.float64],
    dist: str,
    mins: NDArray[np.float64],
    maxs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """(m, n) distance matrix under the repaired metric."""
    if dist == "FACTOR":
        return np.mean(
            rpm_x[None, :, :] != xtest[:, None, :], axis=2, dtype=np.float64
        )
    ranges = maxs - mins
    active = np.isfinite(ranges) & (ranges > 0)
    if not np.any(active):
        return np.zeros((xtest.shape[0], rpm_x.shape[0]), dtype=np.float64)
    z = (xtest[:, None, active] - rpm_x[None, :, active]) / ranges[active]
    if dist == "L1":
        return np.sum(np.abs(z), axis=2)
    return np.sqrt(np.sum(z * z, axis=2))


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
    out = np.empty(xtest.shape[0], dtype=np.float64)

    chunk = max(1, int(4_000_000 // max(n, 1)))
    for start in range(0, xtest.shape[0], chunk):
        rows = slice(start, min(start + chunk, xtest.shape[0]))
        d = _mreg_distances(rpm_x, xtest[rows], dist, mins, maxs)

        if k == 1:
            dmin = d.min(axis=1, keepdims=True)
            for i in range(d.shape[0]):
                tied = rpm_yhat[d[i] == dmin[i, 0]]
                if is_class:
                    out[start + i] = _mreg_weighted_mode(tied, np.ones(tied.size))
                else:
                    finite_tied = tied[np.isfinite(tied)]
                    out[start + i] = float(nns_gravity(finite_tied))
            continue

        idx = np.argsort(d, axis=1, kind="stable")[:, :k]
        dk = np.take_along_axis(d, idx, axis=1)
        yk = rpm_yhat[idx]
        w = _mreg_ensemble_weights(dk)
        if is_class:
            for i in range(dk.shape[0]):
                out[start + i] = _mreg_weighted_mode(yk[i], w[i])
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
    out = np.empty((m, kmax), dtype=np.float64)

    d = _mreg_distances(rpm_x, xtest, dist, mins, maxs)
    idx = np.argsort(d, axis=1, kind="stable")
    d_sorted = np.take_along_axis(d, idx, axis=1)
    y_sorted = rpm_yhat[idx]

    # k = 1 aggregates all exact nearest-distance ties.
    dmin = d.min(axis=1, keepdims=True)
    for i in range(m):
        tied = rpm_yhat[d[i] == dmin[i, 0]]
        finite_tied = tied[np.isfinite(tied)]
        out[i, 0] = float(nns_gravity(finite_tied))

    for k in range(2, kmax + 1):
        w = _mreg_ensemble_weights(d_sorted[:, :k])
        out[:, k - 1] = np.sum(y_sorted[:, :k] * w, axis=1)
    return out


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
    x_sorted = x[order]
    y_sorted = y[order]
    _, starts = np.unique(ids_sorted, return_index=True)
    starts = np.sort(starts)
    ends = np.append(starts[1:], ids_sorted.size)
    rows = []
    yhat = []
    for s, e in zip(starts, ends, strict=True):
        rows.append(
            [
                _reduce_value(x_sorted[s:e, j], noise_reduction, False)
                for j in range(x.shape[1])
            ]
        )
        yhat.append(_reduce_value(y_sorted[s:e], noise_reduction, is_class))
    return np.asarray(rows, dtype=np.float64), np.asarray(yhat, dtype=np.float64)


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
    k = max(1, int(math.floor((1.0 - dep) * math.sqrt(x.shape[0]))))
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
) -> dict[str, Any]:
    if x.shape[0] != y.size:
        raise ValueError(f"[X_n] has {x.shape[0]} rows but [Y] has {y.size} values.")
    if x.shape[1] < 2:
        raise ValueError("NNS.M.reg requires at least two encoded predictors.")
    if y.size < 2 or not np.all(np.isfinite(y)):
        raise ValueError("[Y] must contain at least two finite values.")

    mins = np.min(x, axis=0)
    maxs = np.max(x, axis=0)

    partition = _mreg_partition_matrix(x, y, order, noise_reduction, is_class)
    rpm_x, rpm_yhat = _mreg_build_rpm(x, y, partition["ids"], noise_reduction, is_class)
    if rpm_x.shape[0] == 0:
        raise ValueError("NNS.M.reg produced an empty RPM.")

    if n_best is None and order == "max":
        k = 1
    elif n_best is None:
        k = _mreg_default_nbest(x, y, rpm_x.shape[0])
    elif n_best == "all":
        k = rpm_x.shape[0]
    else:
        k = min(int(n_best), rpm_x.shape[0])
    k = max(1, k)

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
        "R2": None if point_only else metric,
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
    dist: str = "L2",
    point_only: bool = False,
    multivariate_call: bool = False,
) -> dict[str, Any]:
    factor_2_dummy = _scalar_logical(factor_2_dummy, "factor.2.dummy")
    smooth = _scalar_logical(smooth, "smooth")
    point_only = _scalar_logical(point_only, "point.only")
    multivariate_call = _scalar_logical(multivariate_call, "multivariate.call")
    order = _validate_order(order)
    n_best = _validate_nbest(n_best)
    dist = _validate_dist(dist)
    noise_reduction = _validate_noise(noise_reduction)
    confidence_interval = _validate_ci(confidence_interval)

    y_raw = _response_vector(y)
    if y_raw.size < 2:
        raise ValueError("[y] must contain at least two observations.")
    task = _task(type, y_raw)
    y_numeric = cast(NDArray[np.float64], task["y"])
    if not np.all(np.isfinite(y_numeric)):
        raise ValueError("[y] must contain finite values.")

    encoded = _encode_predictors(x, point_est, factor_2_dummy)
    ex = cast(NDArray[np.float64], encoded["x"])
    ep = cast(NDArray[np.float64] | None, encoded["point_est"])
    if ex.shape[0] != y_numeric.size:
        raise ValueError(
            f"[x] has {ex.shape[0]} rows but [y] has {y_numeric.size} values."
        )
    if task["is_xonly"] and ex.shape[1] != 1:
        raise ValueError("[type = 'XONLY'] is only valid for a single encoded predictor.")

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
