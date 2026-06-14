from __future__ import annotations

import importlib

import pytest

import nns

_REMOVED_NOWCAST_NAMES = ("nns_nowcast", "nns_nowcast_panel")


@pytest.mark.parametrize("name", _REMOVED_NOWCAST_NAMES)
def test_nowcast_names_are_not_public(name: str) -> None:
    assert name not in nns.__all__
    with pytest.raises(AttributeError):
        nns.__getattr__(name)


def test_nowcast_modules_are_gone() -> None:
    for module in ("nns.nowcast", "nns.providers", "nns.providers.nowcast"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module)
