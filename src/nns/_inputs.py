"""R 14.2's input layer (``R/inputs.R``) for NNS Python inputs.

R NNS 14.2 passes every data argument through a few shared helpers, which fix
the inputs a function accepts and the message it stops with. These are their
Python counterparts. Python inputs stand for R types as follows: a pandas
``Categorical`` or categorical ``Series`` is a factor, a ``DataFrame`` a
data.frame, a 2-D array a matrix, a string array a character vector and a
boolean array a logical vector. Python stays stricter than R in two documented
places: it does not flatten matrices with several columns into one vector, and
empty inputs raise (see docs/conventions.md).
"""

from __future__ import annotations

from numbers import Real
from typing import Any

import numpy as np
from numpy.typing import NDArray

from nns._helpers import _has_factor_column, _is_factor

F = NDArray[np.float64]

MISSING = "You have some missing values, please address."


def is_frame(value: Any) -> bool:
    return hasattr(value, "columns") and hasattr(value, "iloc")


def has_factor(value: Any) -> bool:
    """R ``.nns_has_factor``: a factor, or a data frame with a factor column."""
    return _is_factor(value) or _has_factor_column(value)


def r_typeof(value: Any) -> str:
    """R's ``typeof()`` of the vector a Python input stands for."""
    if _is_factor(value):
        return "integer"
    if is_frame(value):
        return "list"
    arr = np.asarray(value)
    kind = arr.dtype.kind
    if kind == "b":
        return "logical"
    if kind in "iu":
        return "integer"
    if kind in "fmM":
        return "double"
    if kind == "c":
        return "complex"
    if kind in "US":
        return "character"
    if kind == "O":
        items = [v for v in arr.ravel().tolist() if v is not None]
        if any(isinstance(v, (str, bytes)) for v in items):
            return "character"
        if items and all(isinstance(v, (bool, np.bool_)) for v in items):
            return "logical"
        if all(isinstance(v, (Real, np.number)) for v in items):
            return "double"
        return "list"
    return "double"


def is_numeric(value: Any) -> bool:
    """R's ``is.numeric()``: integer or double vectors, not factors or data frames."""
    return not has_factor(value) and r_typeof(value) in ("integer", "double")


def no_factor(value: Any, arg: str) -> None:
    """R ``.nns_no_factor``."""
    if has_factor(value):
        raise ValueError(f"`{arg}` must be numeric, not a factor.")


def column(value: Any) -> Any:
    """A one-column data frame or 2-D array as its column; anything else unchanged."""
    if is_frame(value) and value.shape[1] == 1:
        return value.iloc[:, 0]
    if not is_frame(value) and not _is_factor(value):
        arr = np.asarray(value)
        if arr.ndim == 2 and arr.shape[1] == 1:
            return arr[:, 0]
    return value


def ncol(value: Any) -> int:
    """R's ``NCOL()``."""
    if is_frame(value):
        return int(value.shape[1])
    if _is_factor(value):
        return 1
    arr = np.asarray(value)
    return int(arr.shape[1]) if arr.ndim == 2 else 1


def as_numeric(value: Any) -> F:
    """R's ``as.numeric()`` of a numeric vector or matrix (a matrix by columns)."""
    if is_frame(value):
        value = value.to_numpy()
    if _is_factor(value):
        return np.asarray(getattr(value, "codes", None), dtype=np.float64) + 1.0
    return np.asarray(value, dtype=np.float64).ravel(order="F")


def native_vector(value: Any, arg: str) -> F:
    """R ``.nns_native_sample``'s type rules: numeric vectors and data frames pass;
    character and logical vectors and factors are rejected with R's messages.

    A one-column data frame or 2-D array is read as its column. Wider inputs come
    back 2-D for the caller to reject (R would flatten them).
    """
    if is_frame(value):
        if any(r_typeof(value[c]) == "character" for c in value.columns):
            raise ValueError("Not compatible with requested type: [type=character; target=double].")
    elif not _is_factor(value) and r_typeof(value) not in ("double", "integer"):
        raise ValueError(f"{arg} should be numeric vector, or data table")
    no_factor(value, arg)
    picked = column(value)
    if is_frame(picked):
        picked = picked.to_numpy()
    return np.asarray(picked, dtype=np.float64)


def var_sample(value: Any, arg: str) -> F:
    """R ``.nns_var_sample``: missing values dropped, factors rejected."""
    no_factor(value, arg)
    values = np.asarray(column(value), dtype=np.float64).ravel(order="F")
    values = values[~np.isnan(values)]
    if values.size == 0:
        raise ValueError("x must contain at least one non-NA numeric value.")
    return values


def series_values(value: Any, arg: str) -> Any:
    """R ``.nns_series_values``: one column, no factor, no missing values."""
    if ncol(value) != 1 and np.ndim(value) == 2:
        raise ValueError(f"`{arg}` must have exactly one column.")
    no_factor(value, arg)
    if r_typeof(column(value)) in ("double", "integer") and np.isnan(
        np.asarray(column(value), dtype=np.float64)
    ).any():
        raise ValueError(MISSING)
    return value


def seasonal_series(value: Any, arg: str) -> F:
    """R ``.nns_seasonal_series`` (``NNS.seas``)."""
    if not is_numeric(value):
        raise ValueError("Variable must be numeric")
    series_values(value, arg)
    values = np.asarray(column(value), dtype=np.float64).ravel()
    if np.isinf(values).any():
        raise ValueError("Infinite values not allowed")
    return values


def bootstrap_series(value: Any, arg: str) -> Any:
    """R ``.nns_bootstrap_series`` (``NNS.meboot``, ``NNS.MC``): one column, no
    factor, no missing values. The column comes back unconverted."""
    value = column(value)
    series_values(value, arg)
    return value


def superiority_sample(value: Any, arg: str) -> F:
    """R ``.nns_superiority_sample`` (``NNS.SS``)."""
    no_factor(value, arg)
    raw = np.asarray(column(value)).ravel()
    if r_typeof(raw) == "character":
        out = np.full(raw.size, np.nan)
        for i, item in enumerate(raw.tolist()):
            try:
                out[i] = float(item)
            except (TypeError, ValueError):
                out[i] = np.nan
        keep = np.array([item is not None and item == item for item in raw.tolist()])
        values = out[keep]
    else:
        values = np.asarray(raw, dtype=np.float64)
        values = values[~np.isnan(values)]
    if np.isnan(values).any():
        raise ValueError(f"`{arg}` must contain only numeric values.")
    return np.asarray(values, dtype=np.float64)


def dependence_column(value: Any, arg: str) -> F:
    """R ``.nns_dependence_column``: factors by their numeric labels."""
    value = column(value)
    if _is_factor(value):
        categories = np.asarray(getattr(value, "categories", []), dtype=object)
        codes = np.asarray(getattr(value, "codes", []))
        try:
            labels = np.asarray(categories, dtype=np.float64)
        except (TypeError, ValueError):
            return codes.astype(np.float64) + 1.0
        return np.where(codes < 0, np.nan, labels[codes])
    if is_frame(value):
        raise ValueError(
            f"{arg} must be a vector or single-column object in the bivariate path."
        )
    return np.asarray(value, dtype=np.float64)


def dependence_pair(x: Any, y: Any) -> tuple[F, F]:
    """R ``.nns_dependence_pair`` (``NNS.dep``). A matrix with several columns is
    rejected rather than flattened, as everywhere in NNS Python."""
    x_values = dependence_column(x, "x")
    y_values = dependence_column(y, "y")
    if x_values.ndim != 1 or y_values.ndim != 1:
        raise ValueError("x and y must be 1D.")
    if np.isnan(x_values).any():
        raise ValueError("x has missing values, please address.")
    if np.isnan(y_values).any():
        raise ValueError("y has missing values, please address.")
    if np.isinf(x_values).any():
        raise ValueError("x must be finite.")
    if np.isinf(y_values).any():
        raise ValueError("y must be finite.")
    if x_values.size != y_values.size:
        raise ValueError("x and y must have the same length.")
    if x_values.size == 0:
        raise ValueError("x and y must be non-empty.")
    return x_values, y_values


def partition_column(value: Any, arg: str) -> Any:
    """R ``.nns_column``."""
    if is_frame(value) or (not _is_factor(value) and np.ndim(value) == 2):
        if ncol(value) != 1:
            raise ValueError(f"[{arg}] must be a vector or one-column object.")
        return column(value)
    return value


def partition_pair(x: Any, y: Any) -> tuple[F, F]:
    """R ``.nns_partition_pair`` (``NNS.part``): complete pairs only."""
    x = partition_column(x, "x")
    y = partition_column(y, "y")
    if np.size(x) != np.size(y):
        raise ValueError("[x] and [y] must have the same length.")
    if np.size(x) == 0:
        raise ValueError("[x] and [y] must not be empty.")
    x_values = as_numeric(x) if _is_factor(x) else np.asarray(x, dtype=np.float64).ravel()
    no_factor(y, "y")
    y_values = np.asarray(y, dtype=np.float64).ravel()
    keep = ~(np.isnan(x_values) | np.isnan(y_values))
    if not keep.any():
        raise ValueError("[x] and [y] must contain at least one complete (non-missing) pair.")
    return x_values[keep], y_values[keep]


def _has_missing(value: Any) -> bool:
    """R's ``anyNA()`` of a vector, factor or data frame."""
    if _is_factor(value):
        return bool((np.asarray(getattr(value, "codes", [])) < 0).any())
    if is_frame(value):
        return bool(value.isna().to_numpy().any())
    arr = np.asarray(value)
    if arr.dtype.kind in "fc":
        return bool(np.isnan(arr).any())
    if arr.dtype.kind == "O":
        return any(v is None or (isinstance(v, float) and v != v) for v in arr.ravel().tolist())
    return False


def causation_pair(x: Any, y: Any) -> tuple[F, F]:
    """R ``.nns_causation_inputs`` for two variables (``factor.2.dummy = FALSE``)
    up to its constant and ``tau`` checks."""
    if _has_missing(x) or _has_missing(y):
        raise ValueError(MISSING)
    x = column(x)
    y = column(y)
    if _is_factor(y):
        y = as_numeric(y)
    if ncol(x) > 1 or ncol(y) > 1:
        raise ValueError(
            "x and y must each be a single variable (factor.2.dummy = TRUE turns a factor "
            "with k levels into k - 1 columns)."
        )
    if np.size(x) != np.size(y):
        raise ValueError("x and y must have the same length.")
    x_values = as_numeric(x) if _is_factor(x) else np.asarray(x, dtype=np.float64).ravel()
    y_values = np.asarray(y, dtype=np.float64).ravel()
    if np.isinf(x_values).any() or np.isinf(y_values).any():
        raise ValueError("x and y must be finite.")
    return x_values, y_values
