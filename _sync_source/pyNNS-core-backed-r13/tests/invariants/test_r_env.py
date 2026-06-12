from __future__ import annotations

import os
from pathlib import Path

import _r
import pytest


def test_r_env_preserves_existing_r_libs_user(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("R_LIBS_USER", "custom-library")

    assert _r._r_env()["R_LIBS_USER"] == "custom-library"


def test_r_env_sets_linux_style_default_only_on_non_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("R_LIBS_USER", raising=False)
    monkeypatch.setattr(os, "name", "posix")

    assert _r._r_env()["R_LIBS_USER"] == str(Path.home() / "R" / "library")


def test_r_env_preserves_absent_r_libs_user_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("R_LIBS_USER", raising=False)
    monkeypatch.setattr(os, "name", "nt")

    env = _r._r_env()

    assert "R_LIBS_USER" not in env
