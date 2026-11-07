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

# Only the inputs of the nnscore CMake target are vendored. NNS-core's tests,
# fidelity harness, R reference sources, docs and CI stay upstream: the Python
# build forces NNSCORE_BUILD_TESTS off, and each vendored top-level entry must
# match the upstream git object verbatim (scripts/verify_release_provenance.py).
VENDORED_ENTRIES = ("CMakeLists.txt", "include", "src")


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
    # The R fields describe the R snapshot (scripts/sync_r_nns_snapshot.py);
    # pass them only when the R snapshot moves together with the core.
    parser.add_argument("--r-repo")
    parser.add_argument("--r-commit")
    parser.add_argument("--r-version")
    parser.add_argument("--r-src-tree-hash")
    parser.add_argument("--manifest", type=Path, default=Path("sync/nns_source.json"))
    args = parser.parse_args()

    core = args.core_checkout.resolve()
    if not (core / "CMakeLists.txt").is_file():
        raise SystemExit(f"{core} does not look like NNS-core: missing CMakeLists.txt")

    dest = Path("extern/NNS-core")
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for entry in VENDORED_ENTRIES:
        source = core / entry
        if source.is_dir():
            shutil.copytree(source, dest / entry, ignore=ignore)
        elif source.is_file():
            shutil.copy2(source, dest / entry)
        else:
            raise SystemExit(f"{core} does not look like NNS-core: missing {entry}")

    if args.manifest.exists():
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    else:
        manifest = {}

    manifest.update(
        {
            "core_repo": "OVVO-Financial/NNS-core",
            "core_commit": args.core_commit,
            "vendored_core_path": "extern/NNS-core",
        }
    )
    r_fields = {
        "r_repo": args.r_repo,
        "r_commit": args.r_commit,
        "r_version": args.r_version,
        "r_src_tree_hash": args.r_src_tree_hash,
    }
    manifest.update({key: value for key, value in r_fields.items() if value is not None})
    manifest.setdefault("python_repo", "OVVO-Financial/NNS-python")
    manifest.setdefault("python_commit", None)
    manifest.setdefault("vendored_r_path", "tools/NNS")
    manifest.setdefault("r_cache_path", "tests/_r_cache")
    manifest.setdefault(
        "notes",
        (
            "NNS-python consumes accepted NNS-core snapshots for native code and "
            "verifies public Python behavior against live or cached R NNS."
        ),
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Vendored NNS-core {args.core_commit} into {dest}")
    if args.r_commit is not None:
        print(f"Recorded R source {args.r_repo}@{args.r_commit} version {args.r_version}")


if __name__ == "__main__":
    main()
