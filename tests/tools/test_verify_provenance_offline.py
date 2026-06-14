from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "verify_release_provenance.py"
MANIFEST = REPO_ROOT / "sync" / "nns_source.json"


def _load_module() -> object:
    spec = importlib.util.spec_from_file_location("verify_release_provenance", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_recorded_r_src_tree_hash_matches_vendored_tree() -> None:
    """The committed r_src_tree_hash must match the vendored tools/NNS/src tree.

    This is the offline half of the release provenance verifier; it catches any
    edit to the vendored R snapshot that forgets to update the manifest.
    """
    module = _load_module()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    problems = module.check_offline(manifest, REPO_ROOT)  # type: ignore[attr-defined]
    assert problems == [], problems
