"""matplotlib is lazily imported; absence raises a clear, actionable error."""

from __future__ import annotations

import builtins

import pytest

import nns.plotting._mpl as mpl_helper


def test_require_mpl_returns_pyplot() -> None:
    plt = mpl_helper.require_mpl()
    assert hasattr(plt, "subplots")


def test_require_mpl_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("matplotlib"):
            raise ImportError("No module named 'matplotlib'")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"ovvo-nns\[plot\]"):
        mpl_helper.require_mpl()


def test_importing_nns_core_does_not_import_matplotlib() -> None:
    # The core package must not pull matplotlib in at import time.
    import importlib

    src = importlib.import_module("nns").__file__ or ""
    with open(src, encoding="utf-8") as handle:
        text = handle.read()
    assert "matplotlib" not in text
