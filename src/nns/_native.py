from __future__ import annotations

import importlib
import importlib.util
from typing import Any, cast

_NNSCORE_SPEC = importlib.util.find_spec("nns._nnscore")

try:
    _nnscore = importlib.import_module("nns._nnscore") if _NNSCORE_SPEC is not None else None
except (ImportError, OSError):
    _nnscore = None


def nnscore() -> Any | None:
    """Return the optional private NNS-core extension module when available."""
    return cast(Any | None, _nnscore)


def native_fn(name: str) -> Any | None:
    """Return a callable from the NNS-core extension, or None when unavailable."""
    if _nnscore is None:
        return None
    return getattr(_nnscore, name, None)
