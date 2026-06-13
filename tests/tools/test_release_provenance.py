from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check_release_provenance.py"
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _project_version() -> str:
    import tomllib

    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def _write_manifest(path: Path, **overrides: object) -> None:
    manifest: dict[str, object] = {
        "r_repo": "OVVO-Financial/NNS",
        "r_commit": "unknown",
        "r_version": "13.0",
        "core_commit": "unknown",
    }
    manifest.update(overrides)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_allow_unknown_passes_with_placeholder_provenance() -> None:
    result = _run(["--allow-unknown"])
    assert result.returncode == 0, result.stdout + result.stderr


def test_unknown_provenance_fails_real_release() -> None:
    result = _run([])
    assert result.returncode != 0
    assert "provenance" in (result.stdout + result.stderr).lower()


def test_tag_mismatch_fails(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, r_commit="abc123", core_commit="def456")
    result = _run(["--tag", "v9.9.9", "--manifest", str(manifest)])
    assert result.returncode != 0
    assert "does not match" in (result.stdout + result.stderr)


def test_matching_tag_and_full_provenance_passes(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, r_commit="abc123", core_commit="def456")
    result = _run(["--tag", f"v{_project_version()}", "--manifest", str(manifest)])
    assert result.returncode == 0, result.stdout + result.stderr
