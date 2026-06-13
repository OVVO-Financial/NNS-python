from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "plan_r_api_parity_review.py"
API_MAP = REPO_ROOT / "sync" / "r_api_map.json"


def _run_plan(changed: list[str], tmp_path: Path) -> dict[str, Any]:
    changed_files = tmp_path / "changed_files.json"
    changed_files.write_text(json.dumps(changed), encoding="utf-8")
    out_md = tmp_path / "inspection.md"
    out_json = tmp_path / "plan.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--changed-files-json",
            str(changed_files),
            "--map",
            str(API_MAP),
            "--out",
            str(out_md),
            "--json-out",
            str(out_json),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert out_md.exists()
    plan: dict[str, Any] = json.loads(out_json.read_text(encoding="utf-8"))
    return plan


def test_mapped_arma_change(tmp_path: Path) -> None:
    plan = _run_plan(["R/ARMA.R"], tmp_path)
    assert "src/nns/arma.py" in plan["affected_python_modules"]
    assert "NNS.ARMA" in plan["cache_scope"]
    assert plan["has_unmapped_r_files"] is False
    assert plan["unmapped_r_files"] == []


def test_unmapped_r_file_is_reported(tmp_path: Path) -> None:
    plan = _run_plan(["R/NewFunction.R"], tmp_path)
    assert plan["has_unmapped_r_files"] is True
    assert "R/NewFunction.R" in plan["unmapped_r_files"]
