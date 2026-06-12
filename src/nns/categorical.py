from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray


def encode_factor_codes(
    values: NDArray[Any] | Sequence[Any],
    *,
    levels: Sequence[Any] | None = None,
) -> tuple[NDArray[np.float64], list[Any]]:
    """Encode values as R-style 1-based factor codes.

    NumPy arrays do not carry R factor level metadata. Pass ``levels`` when
    reproducing an R factor with an explicit level order.
    """
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError("values must be 1D.")

    resolved_levels = _resolve_levels(arr, levels)
    level_to_code = {level: index + 1.0 for index, level in enumerate(resolved_levels)}
    codes = np.empty(arr.size, dtype=np.float64)
    for index, value in enumerate(arr.tolist()):
        key = _normalize_bool(value)
        codes[index] = level_to_code.get(key, np.nan)
    return codes, resolved_levels


def factor_2_dummy(
    values: NDArray[Any] | Sequence[Any],
    *,
    levels: Sequence[Any] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Return R ``factor_2_dummy`` columns.

    Explicit levels reproduce R factor behavior. Without levels, numeric and
    logical inputs follow R's non-factor fallback and are returned as one
    numeric column named ``"x"``.
    """
    arr = np.asarray(values)
    if levels is None:
        return {"x": _as_numeric_fallback(arr)}

    codes, resolved_levels = encode_factor_codes(arr, levels=levels)
    present = np.unique(codes[np.isfinite(codes)]).size
    if present <= 1:
        return {"x": codes}

    return {
        str(level): (codes == float(index + 1)).astype(np.float64)
        for index, level in enumerate(resolved_levels[1:], start=1)
    }


def factor_2_dummy_fr(
    values: NDArray[Any] | Sequence[Any],
    *,
    levels: Sequence[Any] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Return R ``factor_2_dummy_FR`` full-rank columns."""
    arr = np.asarray(values)
    if levels is None:
        return {"x": _as_numeric_fallback(arr)}

    codes, resolved_levels = encode_factor_codes(arr, levels=levels)
    present = np.unique(codes[np.isfinite(codes)]).size
    if present <= 1:
        return {"x": codes}

    return {
        str(level): (codes == float(index + 1)).astype(np.float64)
        for index, level in enumerate(resolved_levels)
    }


def _down_sample_rows(
    x: NDArray[np.float64],
    y_codes: NDArray[np.float64],
    *,
    classes: NDArray[np.float64],
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """R ``downSample`` row selection: class groups down to minority count."""
    x_values, y_values, observed = _sampling_inputs(x, y_codes, classes)
    per_class = [np.flatnonzero(y_values == class_code) for class_code in observed]
    target = min(indices.size for indices in per_class)
    picked = [
        indices[rng.choice(indices.size, size=target, replace=False)] for indices in per_class
    ]
    rows = np.concatenate(picked) if picked else np.empty(0, dtype=np.int64)
    return x_values[rows].copy(), y_values[rows].copy()


def _up_sample_rows(
    x: NDArray[np.float64],
    y_codes: NDArray[np.float64],
    *,
    classes: NDArray[np.float64],
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """R ``upSample`` row selection: class groups up to majority count."""
    x_values, y_values, observed = _sampling_inputs(x, y_codes, classes)
    per_class = [np.flatnonzero(y_values == class_code) for class_code in observed]
    target = max(indices.size for indices in per_class)
    picked = [indices[rng.choice(indices.size, size=target, replace=True)] for indices in per_class]
    rows = np.concatenate(picked) if picked else np.empty(0, dtype=np.int64)
    return x_values[rows].copy(), y_values[rows].copy()


def _balance_class_training(
    x: NDArray[np.float64],
    y_codes: NDArray[np.float64],
    *,
    classes: NDArray[np.float64],
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """R stack/boost balance layout: ``rbind(downSample(...), upSample(...))``."""
    down_x, down_y = _down_sample_rows(x, y_codes, classes=classes, rng=rng)
    up_x, up_y = _up_sample_rows(x, y_codes, classes=classes, rng=rng)
    return np.vstack((down_x, up_x)), np.concatenate((down_y, up_y))


def _dense_factor_codes(
    values: NDArray[Any] | Sequence[Any],
    *,
    levels: Sequence[Any] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return R ``as.numeric(factor(values))`` codes and observed class order."""
    if levels is not None:
        codes, resolved = encode_factor_codes(values, levels=levels)
        classes = np.arange(1, len(resolved) + 1, dtype=np.float64)
        return codes, classes

    arr = np.asarray(values)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if arr.dtype.kind in {"U", "S", "O"}:
        raise ValueError("string/object values require explicit levels to mimic R factors.")
    numeric = np.asarray(arr, dtype=np.float64)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("class values must contain only finite values.")
    observed = np.unique(numeric)
    mapping = {float(value): float(index + 1) for index, value in enumerate(observed)}
    codes = np.asarray([mapping[float(value)] for value in numeric], dtype=np.float64)
    classes = np.arange(1, observed.size + 1, dtype=np.float64)
    return codes, classes


def _resolve_levels(arr: NDArray[Any], levels: Sequence[Any] | None) -> list[Any]:
    if levels is not None:
        resolved = [_normalize_bool(level) for level in levels]
        if len(resolved) == 0:
            raise ValueError("levels must be non-empty.")
        if len(set(resolved)) != len(resolved):
            raise ValueError("levels must be unique.")
        return resolved

    if arr.dtype.kind in {"U", "S", "O"}:
        raise ValueError("string/object values require explicit levels to mimic R factors.")
    unique_values: list[Any] = []
    for value in arr.tolist():
        key = _normalize_bool(value)
        if key not in unique_values:
            unique_values.append(key)
    return unique_values


def _as_numeric_fallback(arr: NDArray[Any]) -> NDArray[np.float64]:
    if arr.ndim != 1:
        raise ValueError("values must be 1D.")
    if arr.dtype.kind in {"U", "S", "O"}:
        raise ValueError("string/object values require explicit levels to mimic R factors.")
    return np.asarray(arr, dtype=np.float64)


def _normalize_bool(value: Any) -> Any:
    if isinstance(value, bool | np.bool_):
        return bool(value)
    return value


def _sampling_inputs(
    x: NDArray[np.float64],
    y_codes: NDArray[np.float64],
    classes: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    x_values = np.asarray(x, dtype=np.float64)
    if x_values.ndim != 2:
        raise ValueError("x must be a 2D matrix.")
    y_values = np.asarray(y_codes, dtype=np.float64).reshape(-1)
    if x_values.shape[0] != y_values.size:
        raise ValueError("x and y_codes must have the same row count.")
    class_values = np.asarray(classes, dtype=np.float64).reshape(-1)
    if class_values.size == 0:
        raise ValueError("classes must be non-empty.")
    observed = np.asarray(
        [class_code for class_code in class_values if np.any(y_values == class_code)],
        dtype=np.float64,
    )
    if observed.size == 0:
        raise ValueError("no non-empty classes.")
    return x_values, y_values, observed
