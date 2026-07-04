from __future__ import annotations

import importlib
import importlib.util
from typing import Any

_NNSCORE_SPEC = importlib.util.find_spec("nns._nnscore")

try:
    _nnscore = importlib.import_module("nns._nnscore") if _NNSCORE_SPEC is not None else None
except (ImportError, OSError):
    _nnscore = None


def native_fn(name: str) -> Any | None:
    """Return a callable from the NNS-core extension, or None when unavailable."""
    if _nnscore is None:
        return None
    return getattr(_nnscore, name, None)
