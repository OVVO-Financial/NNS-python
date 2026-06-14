from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check_version_consistency.py"


def _load_module() -> object:
    spec = importlib.util.spec_from_file_location("check_version_consistency", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_version_is_consistent_everywhere() -> None:
    """pyproject, nns.__version__, and the README current-version row must agree.

    The README is the PyPI project description, so a stale version row there
    shows the wrong version on the package page. This guards every version bump.
    """
    module = _load_module()
    _version, problems = module.check(REPO_ROOT)  # type: ignore[attr-defined]
    assert problems == [], problems
