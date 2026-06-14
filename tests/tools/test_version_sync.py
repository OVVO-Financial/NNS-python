from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _pyproject_version() -> str:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def _package_version() -> str:
    text = (REPO_ROOT / "src" / "nns" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert match is not None, "could not find __version__ in src/nns/__init__.py"
    return match.group(1)


def test_package_version_matches_pyproject() -> None:
    assert _package_version() == _pyproject_version()
