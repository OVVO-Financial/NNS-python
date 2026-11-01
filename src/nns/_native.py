from __future__ import annotations

import functools
import importlib
import importlib.util
import inspect
import numbers
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, ParamSpec, TypeVar

_P = ParamSpec("_P")
_R = TypeVar("_R")

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


def set_num_threads(n: int | None) -> None:
    """Set the default number of threads used by the native NNS-core kernels.

    ``n=None`` (or ``n <= 0``) restores the automatic choice: the
    ``NNS_NUM_THREADS`` environment variable when it is a positive integer, else
    the number of hardware threads. Small inputs always run serially, and results
    never depend on the thread count. A function's ``ncores`` argument overrides
    this setting for that call. Without the native extension this does nothing.
    """
    value = 0 if n is None else max(_integer(n, "n"), 0)
    setter = native_fn("set_num_threads")
    if setter is not None:
        setter(value)


def get_num_threads() -> int:
    """Return the default number of native threads (1 without the extension)."""
    getter = native_fn("get_num_threads")
    return int(getter()) if getter is not None else 1


@contextmanager
def native_threads(ncores: Any) -> Iterator[None]:
    """Use at most ``ncores`` native threads for the calls made inside the block.

    This is R's ``ncores`` argument: ``None`` keeps the default
    (:func:`set_num_threads`), and ``1`` runs every native kernel serially. The
    setting is per Python thread, so concurrent calls with different ``ncores``
    do not interfere.
    """
    if ncores is None:
        yield
        return
    value = _integer(ncores, "ncores")
    if value < 1:
        raise ValueError("[ncores] must be NULL or a positive integer.")
    setter = native_fn("_set_call_threads")
    if setter is None:
        yield
        return
    previous = setter(value)
    try:
        yield
    finally:
        setter(previous)


def honours_ncores(func: Callable[_P, _R]) -> Callable[_P, _R]:
    """Run ``func`` under :func:`native_threads` with its ``ncores`` argument."""
    signature = inspect.signature(func)
    default = signature.parameters["ncores"].default

    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        if "ncores" in kwargs:
            ncores = kwargs["ncores"]
        else:
            ncores = signature.bind_partial(*args, **kwargs).arguments.get("ncores", default)
        with native_threads(ncores):
            return func(*args, **kwargs)

    return wrapper


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"[{name}] must be NULL or a positive integer.")
    number = float(value)
    if not number.is_integer():
        raise ValueError(f"[{name}] must be NULL or a positive integer.")
    return int(number)
