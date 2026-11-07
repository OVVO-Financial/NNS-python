"""Column, name and level handling shared by the regression/stack/boost ports.

R's ``NNS.reg``, ``NNS.M.reg``, ``NNS.stack`` and ``NNS.boost`` work on data
frames: columns keep their names and their type (numeric, character,
factor), and prediction columns are matched to the training columns by name.
``np.asarray`` on a pandas frame loses all of that, so the ports split their
inputs column by column here first.

pandas is optional: frames, series and categoricals are recognised by duck
typing and never imported.
"""

from __future__ import annotations

import math
import numbers
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Columns:
    """An input split into 1-D columns.

    ``labels`` holds the column labels of a frame (``None`` for arrays),
    ``row_labels`` the index labels of a 1-D series (R's named vector), and
    ``categories`` the declared categories of pandas categorical columns.
    """

    values: list[NDArray[Any]]
    categories: list[list[Any] | None]
    labels: list[str] | None
    row_labels: list[str] | None
    vector: bool
    ordered: list[bool] | None = None


def _categories_of(obj: Any) -> list[Any] | None:
    dtype = getattr(obj, "dtype", None)
    if getattr(dtype, "name", None) != "category":
        return None
    categories = getattr(dtype, "categories", None)
    if categories is None:
        return None
    return list(categories)


def _label(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def split_columns(x: Any) -> Columns:
    """Split ``x`` (frame, series, categorical, array or nested list) into columns."""
    columns = getattr(x, "columns", None)
    if columns is not None and hasattr(x, "iloc") and getattr(x, "ndim", 0) == 2:
        values = [np.asarray(x.iloc[:, j]) for j in range(x.shape[1])]
        categories = [_categories_of(x.iloc[:, j]) for j in range(x.shape[1])]
        return Columns(
            values=values,
            categories=categories,
            labels=[_label(c) for c in list(columns)],
            row_labels=None,
            vector=False,
            ordered=[
                bool(getattr(x.iloc[:, j].dtype, "ordered", False)) for j in range(x.shape[1])
            ],
        )

    categories_1d = _categories_of(x)
    index = getattr(x, "index", None)
    if categories_1d is not None or (index is not None and getattr(x, "ndim", 0) == 1):
        arr = np.asarray(x)
        if arr.ndim == 1:
            row_labels = None if index is None else [_label(i) for i in list(index)]
            return Columns(
                values=[arr],
                categories=[categories_1d],
                labels=None,
                row_labels=row_labels,
                vector=True,
                ordered=[bool(getattr(getattr(x, "dtype", None), "ordered", False))],
            )

    arr = np.asarray(x)
    if arr.ndim == 0:
        return Columns(
            values=[arr.reshape(1)],
            categories=[None],
            labels=None,
            row_labels=None,
            vector=True,
        )
    if arr.ndim == 1:
        return Columns(values=[arr], categories=[None], labels=None, row_labels=None, vector=True)
    if arr.ndim != 2:
        raise ValueError("Expected a vector, matrix, or data frame.")
    return Columns(
        values=[arr[:, j] for j in range(arr.shape[1])],
        categories=[None] * arr.shape[1],
        labels=None,
        row_labels=None,
        vector=False,
    )


def response_values(
    y: Any, message: str = "[y] must be a vector or one-column object."
) -> tuple[NDArray[Any], list[Any] | None]:
    """A response as a 1-D array plus its declared categories (R ``y[[1]]``)."""
    if np.ndim(y) > 2:
        raise ValueError(message)
    cols = split_columns(y)
    if len(cols.values) != 1:
        raise ValueError(message)
    return cols.values[0], cols.categories[0]


# --------------------------------------------------------------------------
# Value typing
# --------------------------------------------------------------------------


def is_missing(value: Any) -> bool:
    """R ``is.na`` for one element: None, NaN, pandas NA / NaT."""
    if value is None:
        return True
    if isinstance(value, (float, np.floating)):
        return bool(math.isnan(value))
    return type(value).__name__ in {"NAType", "NaTType"}


def _is_number(value: Any) -> bool:
    # Decimal is registered only as numbers.Number, not numbers.Real.
    return isinstance(value, (numbers.Real, Decimal, np.bool_)) and not isinstance(value, str)


def is_text(values: NDArray[Any]) -> bool:
    """Whether a column holds labels (R character) rather than numbers.

    String dtypes are labels even when they look numeric ("1", "10"), as in R.
    An object column is numeric only when every non-missing element is a real
    number (as from ``df.to_numpy()`` of a mixed frame); any string or other
    object makes it a label column.
    """
    kind = values.dtype.kind
    if kind in {"U", "S"}:
        return True
    if kind != "O":
        return False
    return any(not is_missing(v) and not _is_number(v) for v in values.tolist())


def is_logical(values: NDArray[Any]) -> bool:
    """Whether a column is logical (R ``is.logical``): bool dtype, or an object
    column whose non-missing elements are all booleans."""
    kind = values.dtype.kind
    if kind == "b":
        return True
    if kind != "O":
        return False
    present = [v for v in values.tolist() if not is_missing(v)]
    return bool(present) and all(isinstance(v, (bool, np.bool_)) for v in present)


def as_float(values: NDArray[Any]) -> NDArray[np.float64]:
    """Numeric conversion where missing or unparsable elements become NaN."""
    try:
        return np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        pass
    out = np.empty(values.shape[0], dtype=np.float64)
    for i, v in enumerate(values.tolist()):
        if is_missing(v):
            out[i] = np.nan
            continue
        try:
            out[i] = float(v)
        except (TypeError, ValueError):
            out[i] = np.nan
    return out


def level_key(value: Any) -> str:
    """The label of one value, as R's ``as.character`` would print it.

    Integral numbers print without a decimal part, so the values ``1``, ``1.0``
    and the declared level ``"1"`` are the same level.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return str(value).upper()
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        v = float(value)
        if math.isfinite(v) and v.is_integer():
            return str(int(v))
        return repr(v)
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def level_keys(values: NDArray[Any], missing_message: str) -> list[str]:
    raw = values.tolist()
    if any(is_missing(v) for v in raw):
        raise ValueError(missing_message)
    if values.dtype.kind == "U":
        return [str(v) for v in raw]
    return [level_key(v) for v in raw]


def declared_levels(levels: Sequence[Any], what: str) -> list[str]:
    """Normalise a user / pandas level list to unique label keys."""
    if isinstance(levels, (str, bytes)):
        raise ValueError(f"[{what}] must be a sequence of levels, not a single string.")
    keys = [level_key(level) for level in levels]
    if len(set(keys)) != len(keys):
        raise ValueError(f"[{what}] contains duplicate levels.")
    if not keys:
        raise ValueError(f"[{what}] must contain at least one level.")
    return keys


def codes_for(keys: list[str], levels: list[str]) -> NDArray[np.float64]:
    position = {level: i + 1.0 for i, level in enumerate(levels)}
    return np.asarray([position.get(k, np.nan) for k in keys], dtype=np.float64)


def _as_codes(values: NDArray[Any], k: int) -> NDArray[np.float64] | None:
    """``values`` as 1-based codes into ``k`` levels, or None if they are not."""
    if values.dtype.kind == "b" or is_text(values):
        return None
    num = as_float(values)
    if num.size == 0 or not np.all(np.isfinite(num)):
        return None
    if np.any(num != np.round(num)) or num.min() < 1 or num.max() > k:
        return None
    return num


def declared_codes(
    values: NDArray[Any], levels: list[str], missing_message: str
) -> tuple[NDArray[np.float64], bool, list[str]]:
    """Code ``values`` against declared ``levels`` (R ``factor(values, levels)``).

    Values are matched to the levels as labels. When they are not labels but
    numeric codes ``1..K`` (an already integer-coded factor, e.g. ``y`` coded
    1, 2, 3 with ``class_levels=["setosa", "versicolor", "virginica"]``), the
    codes are kept and the levels label them. Returns the codes, whether the
    values were taken as codes, and the values matching neither (sorted).
    """
    keys = level_keys(values, missing_message)
    codes = codes_for(keys, levels)
    if np.all(np.isfinite(codes)):
        return codes, False, []
    as_codes = _as_codes(values, len(levels))
    if as_codes is not None:
        return as_codes, True, []
    absent = sorted({k for k, c in zip(keys, codes.tolist(), strict=True) if not c == c})
    return codes, False, absent


def point_codes(
    values: NDArray[Any], levels: list[str], by_code: bool, missing_message: str
) -> tuple[NDArray[np.float64], list[str]]:
    """Code prediction values the way the training column was coded."""
    keys = level_keys(values, missing_message)
    if by_code:
        as_codes = _as_codes(values, len(levels))
        if as_codes is not None:
            return as_codes, []
        codes = np.full(len(keys), np.nan)
    else:
        codes = codes_for(keys, levels)
    unseen = sorted({k for k, c in zip(keys, codes.tolist(), strict=True) if not c == c})
    return codes, unseen


def canonical_predictor_codes(
    levels: list[str],
    train: NDArray[np.float64],
    point: NDArray[np.float64] | None,
    *,
    ordered: bool = False,
) -> tuple[list[str], NDArray[np.float64], NDArray[np.float64] | None]:
    """R14 sorts unordered predictor levels, but preserves ordered factors.

    Recode already-integerized factors against their original levels before
    changing column order. Response-class level mappings are not canonicalized.
    """
    if ordered or levels == sorted(levels):
        return levels, train, point
    canonical = sorted(levels)
    lookup = np.asarray([canonical.index(level) + 1.0 for level in levels])
    return (
        canonical,
        lookup[train.astype(np.intp) - 1],
        None if point is None else lookup[point.astype(np.intp) - 1],
    )


# --------------------------------------------------------------------------
# R name helpers
# --------------------------------------------------------------------------

_RESERVED = {
    "if",
    "else",
    "repeat",
    "while",
    "function",
    "for",
    "next",
    "break",
    "TRUE",
    "FALSE",
    "NULL",
    "Inf",
    "NaN",
    "NA",
    "NA_integer_",
    "NA_real_",
    "NA_character_",
    "NA_complex_",
    "in",
}


def r_make_unique(names: Sequence[str], sep: str = ".") -> list[str]:
    """R ``make.unique``: later duplicates get the smallest free ``<sep>k``."""
    out = list(names)
    used = set(out)
    counters: dict[str, int] = {}
    seen: set[str] = set()
    for i, name in enumerate(names):
        if name not in seen:
            seen.add(name)
            continue
        k = counters.get(name, 0)
        while True:
            k += 1
            candidate = f"{name}{sep}{k}"
            if candidate not in used:
                break
        counters[name] = k
        used.add(candidate)
        out[i] = candidate
    return out


def r_make_names(names: Sequence[str], unique: bool = True) -> list[str]:
    """R ``make.names`` (ASCII letters; invalid characters become ".")."""
    out: list[str] = []
    for raw in names:
        s = re.sub(r"[^0-9A-Za-z._]", ".", str(raw))
        if s == "" or not re.match(r"^[A-Za-z.]", s) or re.match(r"^\.[0-9]", s):
            s = "X" + s
        if s in _RESERVED:
            s = s + "."
        out.append(s)
    return r_make_unique(out, ".") if unique else out


def match_columns(supplied: list[str], train: list[str], label: str) -> list[int] | None:
    """R ``.nns_match_predictor_names``.

    The same set of names gives the permutation into training order; no names
    in common means positional matching (``None``); a partial overlap is an
    error.
    """
    if set(supplied) == set(train) and len(set(supplied)) == len(supplied):
        position = {name: i for i, name in enumerate(supplied)}
        return [position[name] for name in train]
    if not any(name in train for name in supplied):
        return None
    missing = [name for name in train if name not in supplied]
    extra = [name for name in supplied if name not in train]
    raise ValueError(
        f"[{label}] columns must exactly match the training predictors, or share no "
        "names with them to be matched positionally. "
        f"Missing: {', '.join(missing) if missing else 'none'}; "
        f"extra: {', '.join(extra) if extra else 'none'}."
    )


def resolve_factor_levels(
    factor_levels: Any, names: list[str], labels: list[str] | None
) -> list[list[Any] | None]:
    """Per-column declared levels from a ``factor_levels`` argument.

    Accepted forms: a mapping from column name to levels; a sequence with one
    entry per predictor column (``None`` for columns without declared levels);
    or, when there is a single predictor, a flat sequence of its levels.
    """
    p = len(names)
    if factor_levels is None:
        return [None] * p
    if isinstance(factor_levels, Mapping):
        out: list[list[Any] | None] = [None] * p
        lookup: dict[str, int] = {}
        for j, name in enumerate(names):
            lookup.setdefault(name, j)
        if labels is not None:
            for j, name in enumerate(labels):
                lookup.setdefault(name, j)
        for key, levels in factor_levels.items():
            idx = lookup.get(str(key))
            if idx is None:
                raise ValueError(f"[factor_levels] names unknown predictor '{key}'.")
            out[idx] = None if levels is None else list(levels)
        return out
    if isinstance(factor_levels, (str, bytes)):
        raise ValueError("[factor_levels] must be a sequence of levels, not a single string.")
    entries = list(factor_levels)

    def is_level_list(v: Any) -> bool:
        return v is None or (
            not isinstance(v, (str, bytes)) and isinstance(v, (Sequence, np.ndarray))
        )

    if p == 1 and not all(is_level_list(v) for v in entries):
        return [entries]
    if len(entries) != p or not all(is_level_list(v) for v in entries):
        raise ValueError(
            f"[factor_levels] must give one level list (or None) per predictor: "
            f"expected {p} entries."
        )
    return [None if v is None else list(v) for v in entries]
