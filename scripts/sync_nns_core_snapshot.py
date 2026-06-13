from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

EXCLUDES = {
    ".git",
    "build",
    ".cache",
    ".pytest_cache",
    "__pycache__",
}


def ignore(_directory: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        if name in EXCLUDES:
            ignored.add(name)
        if name.endswith((".pyc", ".pyo", "~")):
            ignored.add(name)
    return ignored


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-checkout", type=Path, required=True)
    parser.add_argument("--core-commit", required=True)
    parser.add_argument("--r-repo", required=True)
    parser.add_argument("--r-commit", required=True)
    parser.add_argument("--r-version", required=True)
    parser.add_argument("--r-src-tree-hash", required=True)
    parser.add_argument("--manifest", type=Path, default=Path("sync/nns_source.json"))
    args = parser.parse_args()

    core = args.core_checkout.resolve()
    if not (core / "CMakeLists.txt").is_file():
        raise SystemExit(f"{core} does not look like NNS-core: missing CMakeLists.txt")

    dest = Path("extern/NNS-core")
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(core, dest, ignore=ignore)

    if not (dest / "CMakeLists.txt").is_file():
        raise SystemExit(f"copied NNS-core is invalid: missing {dest / 'CMakeLists.txt'}")

    manifest = {
        "r_repo": args.r_repo,
        "r_commit": args.r_commit,
        "r_version": args.r_version,
        "r_src_tree_hash": args.r_src_tree_hash,
        "core_repo": "OVVO-Financial/NNS-core",
        "core_commit": args.core_commit,
        "python_repo": "OVVO-Financial/NNS-python",
        "python_commit": None,
        "vendored_core_path": "extern/NNS-core",
        "vendored_r_path": "tools/NNS",
        "vendored_r_tarball": f"tools/NNS_{args.r_version}.tar.gz",
        "r_cache_path": "tests/_r_cache.json",
        "notes": (
            "NNS-python consumes accepted NNS-core snapshots for native code and "
            "verifies public Python behavior against live or cached R NNS."
        ),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Vendored NNS-core {args.core_commit} into {dest}")
    print(f"Recorded R source {args.r_repo}@{args.r_commit} version {args.r_version}")


if __name__ == "__main__":
    main()
