from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "sync" / "nns_source.json"
API_MAP = REPO_ROOT / "sync" / "r_api_map.json"

REQUIRED_MANIFEST_KEYS = {
    "r_repo",
    "r_commit",
    "r_version",
    "r_src_tree_hash",
    "core_repo",
    "core_commit",
    "python_repo",
    "python_commit",
    "vendored_core_path",
    "vendored_r_path",
    "vendored_r_tarball",
    "r_cache_path",
    "notes",
}

# Python module paths that are intentionally mapped but may not exist as a
# standalone module in this repository. Partial-moment primitives (LPM/UPM)
# live in src/nns/core.py and src/nns/var.py rather than a partial_moments.py.
ALLOWED_MISSING_MODULES = {
    "src/nns/partial_moments.py",
}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def test_manifest_has_required_keys() -> None:
    manifest = _load(MANIFEST)
    assert isinstance(manifest, dict)
    missing = REQUIRED_MANIFEST_KEYS - set(manifest)
    assert not missing, f"sync/nns_source.json missing keys: {sorted(missing)}"
    assert manifest["r_repo"] == "OVVO-Financial/NNS"
    assert manifest["core_repo"] == "OVVO-Financial/NNS-core"
    assert manifest["python_repo"] == "OVVO-Financial/NNS-python"


def test_api_map_is_valid_json() -> None:
    api_map = _load(API_MAP)
    assert isinstance(api_map, dict)
    assert api_map, "sync/r_api_map.json must not be empty"


def test_mapped_python_modules_exist_or_allowed_missing() -> None:
    api_map = _load(API_MAP)
    problems: list[str] = []
    for r_file, entry in api_map.items():
        for module in entry.get("python_modules", []):
            if module in ALLOWED_MISSING_MODULES:
                continue
            if not (REPO_ROOT / module).exists():
                problems.append(f"{r_file} -> {module}")
    assert not problems, f"mapped python_modules do not exist: {problems}"


def test_description_requires_fresh_cache() -> None:
    api_map = _load(API_MAP)
    assert api_map["DESCRIPTION"].get("requires_fresh_cache") is True


def test_namespace_requires_export_review() -> None:
    api_map = _load(API_MAP)
    assert api_map["NAMESPACE"].get("requires_export_review") is True
