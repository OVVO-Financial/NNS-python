from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PLAN_JSON = Path("sync/last_r_api_plan.json")
INSPECTION_MD = Path("sync/last_r_api_inspection.md")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r-checkout", type=Path, required=False)
    parser.add_argument("--r-commit", required=True)
    parser.add_argument("--r-version", required=True)
    parser.add_argument("--description-changed", default="false")
    parser.add_argument("--changed-files-json", type=Path, required=True)
    parser.add_argument("--map", type=Path, default=Path("sync/r_api_map.json"))
    parser.add_argument("--out", type=Path, default=INSPECTION_MD)
    parser.add_argument("--json-out", type=Path, default=PLAN_JSON)
    args = parser.parse_args()

    # Delegate planning to the planning script so both stay in sync.
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("plan_r_api_parity_review.py")),
        "--changed-files-json",
        str(args.changed_files_json),
        "--map",
        str(args.map),
        "--out",
        str(args.out),
        "--json-out",
        str(args.json_out),
    ]
    print("+ " + " ".join(cmd))
    completed = subprocess.run(cmd)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

    plan = load_json(args.json_out)
    description_changed = str(args.description_changed).lower() in {"1", "true", "yes"}
    requires_fresh_cache = bool(plan.get("requires_fresh_cache")) or description_changed
    requires_export_review = bool(plan.get("requires_export_review"))
    has_unmapped = bool(plan.get("has_unmapped_r_files"))

    lines = [
        "# R API update inspection",
        "",
        "- R repo: `OVVO-Financial/NNS`",
        f"- R commit: `{args.r_commit}`",
        f"- R version: `{args.r_version}`",
        f"- DESCRIPTION changed: `{description_changed}`",
        "",
        "## Changed files",
        "",
    ]
    changed = plan.get("changed_files", [])
    lines.extend(f"- `{file}`" for file in changed) if changed else lines.append("- None")

    lines.extend(["", "## Affected Python modules", ""])
    modules = plan.get("affected_python_modules", [])
    lines.extend(f"- `{m}`" for m in modules) if modules else lines.append("- None mapped")

    lines.extend(["", "## Mapped parity tests", ""])
    tests = plan.get("parity_tests", [])
    lines.extend(f"- `{t}`" for t in tests) if tests else lines.append("- None mapped")

    lines.extend(["", "## Cache scope", ""])
    scope = plan.get("cache_scope", [])
    lines.extend(f"- `{s}`" for s in scope) if scope else lines.append("- None mapped")

    lines.extend(
        [
            "",
            "## Required actions",
            "",
            f"- Fresh cache regeneration required: `{requires_fresh_cache}`",
            f"- Export review required: `{requires_export_review}`",
            f"- Unmapped R files require manual review: `{has_unmapped}`",
        ]
    )
    if has_unmapped:
        lines.extend(["", "## Unmapped R files", ""])
        lines.extend(f"- `{file}`" for file in plan.get("unmapped_r_files", []))
    if plan.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {w}" for w in plan["warnings"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
