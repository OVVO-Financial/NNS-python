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
