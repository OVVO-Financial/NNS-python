from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _bullets(items: list[str]) -> list[str]:
    if not items:
        return ["- None mapped"]
    return [f"- `{item}`" for item in items]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--changed-files-json", type=Path, required=True)
    parser.add_argument("--map", type=Path, default=Path("sync/r_api_map.json"))
    parser.add_argument("--out", type=Path, default=Path("sync/last_r_api_inspection.md"))
    parser.add_argument("--json-out", type=Path, default=Path("sync/last_r_api_plan.json"))
    args = parser.parse_args()

    changed = load_json(args.changed_files_json)
    api_map = load_json(args.map)

    affected_modules: set[str] = set()
    tests: set[str] = set()
    cache_scope: set[str] = set()
    unmapped: list[str] = []
    warnings: list[str] = []
    requires_fresh_cache = False
    requires_export_review = False

    for file in changed:
        entry = api_map.get(file)
        if entry is None and file.startswith("R/") and file.endswith(".R"):
            unmapped.append(file)
            continue
        if entry is None:
            continue
        affected_modules.update(entry.get("python_modules", []))
        tests.update(entry.get("parity_tests", []))
        cache_scope.update(entry.get("cache_scope", []))
        requires_fresh_cache = requires_fresh_cache or bool(entry.get("requires_fresh_cache"))
        requires_export_review = requires_export_review or bool(
            entry.get("requires_export_review")
        )

    for test in sorted(tests):
        if not Path(test).exists():
            warnings.append(f"mapped test path does not exist: {test}")

    plan = {
        "changed_files": changed,
        "affected_python_modules": sorted(affected_modules),
        "parity_tests": sorted(tests),
        "cache_scope": sorted(cache_scope),
        "requires_fresh_cache": requires_fresh_cache,
        "requires_export_review": requires_export_review,
        "has_unmapped_r_files": bool(unmapped),
        "unmapped_r_files": unmapped,
        "warnings": warnings,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# R API parity review plan",
        "",
        "## Changed files",
        "",
    ]
    lines.extend(_bullets(list(changed)))
    lines.extend(["", "## Affected Python modules", ""])
    lines.extend(_bullets(sorted(affected_modules)))
    lines.extend(["", "## Parity tests to run", ""])
    lines.extend(_bullets(sorted(tests)))
    lines.extend(["", "## Cache scope", ""])
    lines.extend(_bullets(sorted(cache_scope)))
    lines.extend(
        [
            "",
            "## Required actions",
            "",
            f"- Fresh cache required: `{requires_fresh_cache}`",
            f"- Export review required: `{requires_export_review}`",
            f"- Unmapped R files present: `{bool(unmapped)}`",
        ]
    )
    if unmapped:
        lines.extend(["", "## Unmapped R files", ""])
        lines.extend(f"- `{file}`" for file in unmapped)
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
