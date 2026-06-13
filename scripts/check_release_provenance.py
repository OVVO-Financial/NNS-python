"""Release-readiness provenance gate for NNS-python.

Validates that a release is traceable before it is published:

* the git tag (when given) matches the `pyproject.toml` project version, and
* `sync/nns_source.json` records the R and NNS-core commits this release is
  built from (not the `unknown` placeholders).

Used by `.github/workflows/release.yml`. For dry runs (TestPyPI / artifact-only
builds) pass `--allow-unknown` to permit placeholder provenance.

Exits non-zero with a clear message when a real release is not traceable.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

PLACEHOLDERS = {"", "unknown", None}


def project_version(pyproject: Path) -> str:
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="", help="release tag, e.g. v1.0.0")
    parser.add_argument("--manifest", type=Path, default=Path("sync/nns_source.json"))
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    parser.add_argument(
        "--allow-unknown",
        action="store_true",
        help="permit placeholder R/core provenance (dry runs / TestPyPI)",
    )
    args = parser.parse_args()

    problems: list[str] = []

    version = project_version(args.pyproject)
    tag = args.tag.lstrip("v").strip()
    if tag and tag != version:
        problems.append(
            f"tag '{args.tag}' does not match pyproject version '{version}' "
            f"(expected tag 'v{version}')"
        )

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not args.allow_unknown:
        for field in ("r_commit", "core_commit"):
            if manifest.get(field) in PLACEHOLDERS:
                problems.append(
                    f"sync/nns_source.json '{field}' is unset/placeholder "
                    f"('{manifest.get(field)}'); a real release must record it "
                    "so the published version is traceable to R + NNS-core"
                )
        if manifest.get("r_version") in PLACEHOLDERS:
            problems.append("sync/nns_source.json 'r_version' is unset")

    if problems:
        print("Release provenance check FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1

    print(f"Release provenance OK (version {version}, tag '{args.tag or '(none)'}').")
    if args.allow_unknown:
        print("  (placeholder R/core provenance allowed for this dry run)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
