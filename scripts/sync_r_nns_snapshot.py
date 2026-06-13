from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from pathlib import Path

EXCLUDES = {
    ".git",
    ".Rproj.user",
    ".Rhistory",
    ".RData",
    "__pycache__",
}


def ignore(_directory: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        if name in EXCLUDES:
            ignored.add(name)
        if name.endswith(("~", ".pyc", ".pyo")):
            ignored.add(name)
    return ignored


def description_version(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Version:"):
            return line.split(":", 1)[1].strip()
    raise SystemExit(f"Version field not found in {path}")


def make_tarball(source_dir: Path, tarball: Path) -> None:
    if tarball.exists():
        tarball.unlink()
    with tarfile.open(tarball, "w:gz") as tf:
        tf.add(source_dir, arcname="NNS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r-checkout", type=Path, required=True)
    parser.add_argument("--r-repo", required=True)
    parser.add_argument("--r-commit", required=True)
    parser.add_argument("--r-version", required=True)
    parser.add_argument("--r-src-tree-hash", required=True)
    parser.add_argument("--manifest", type=Path, default=Path("sync/nns_source.json"))
    args = parser.parse_args()

    r_checkout = args.r_checkout.resolve()
    desc = r_checkout / "DESCRIPTION"
    if not desc.is_file():
        raise SystemExit(f"{r_checkout} does not look like R NNS: missing DESCRIPTION")

    found_version = description_version(desc)
    if found_version != args.r_version:
        raise SystemExit(
            f"DESCRIPTION version {found_version} does not match expected {args.r_version}"
        )

    tools = Path("tools")
    dest = tools / "NNS"
    tarball = tools / f"NNS_{args.r_version}.tar.gz"

    if dest.exists():
        shutil.rmtree(dest)
    tools.mkdir(parents=True, exist_ok=True)
    shutil.copytree(r_checkout, dest, ignore=ignore)

    if not (dest / "DESCRIPTION").is_file():
        raise SystemExit("copied R NNS snapshot is invalid: missing tools/NNS/DESCRIPTION")

    make_tarball(dest, tarball)

    for old in tools.glob("NNS_*.tar.gz"):
        if old != tarball:
            old.unlink()

    if args.manifest.exists():
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    else:
        manifest = {}

    manifest.update(
        {
            "r_repo": args.r_repo,
            "r_commit": args.r_commit,
            "r_version": args.r_version,
            "r_src_tree_hash": args.r_src_tree_hash,
            "python_repo": "OVVO-Financial/NNS-python",
            "python_commit": None,
            "vendored_r_path": "tools/NNS",
            "vendored_r_tarball": str(tarball),
            "r_cache_path": "tests/_r_cache.json",
        }
    )
    manifest.setdefault("core_repo", "OVVO-Financial/NNS-core")
    manifest.setdefault("core_commit", None)
    manifest.setdefault("vendored_core_path", "extern/NNS-core")
    manifest.setdefault(
        "notes",
        (
            "NNS-python consumes accepted NNS-core snapshots for native code and "
            "verifies public Python behavior against live or cached R NNS."
        ),
    )

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Vendored R NNS {args.r_version} from {args.r_commit} into {dest}")
    print(f"Wrote tarball {tarball}")


if __name__ == "__main__":
    main()
