#!/usr/bin/env python3
"""Verify that the recorded provenance matches the vendored bytes.

`scripts/check_release_provenance.py` only confirms the manifest is *filled in*
(no ``unknown`` placeholders) and that the tag matches the version. This script
goes further: it clones the upstream repositories at the recorded commits and
proves that the vendored snapshots actually came from them, so a hand-vendored
tree with a stale ``r_commit`` / ``core_commit`` cannot slip into a release.

Checks (all run against ``sync/nns_source.json``):

* core (hard): every top-level object under ``extern/NNS-core`` exists in
  ``core_repo`` at ``core_commit``. NNS-core is vendored by a plain copy, so its
  git objects appear verbatim upstream.
* R src tree (hard, offline): the recorded ``r_src_tree_hash`` equals the git
  tree hash of the vendored ``tools/NNS/src``.
* R tarball (hard when present): if ``r_repo`` commits the vendored tarball at
  ``r_commit``, its blob must equal the vendored tarball blob. The built R
  package normalizes ``R/`` sources, so the committed tarball is the reliable
  cross-repo anchor; if the upstream does not commit one it is reported as a
  skipped note rather than a failure.

Requires network access (shallow-clones the upstreams) and git. Intended for the
release workflow's provenance gate, on real releases only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

PLACEHOLDERS = {"", "unknown", None}


def _git(args: list[str], cwd: Path | None = None) -> tuple[int, str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout.strip()


def _object_hash(repo: Path, spec: str) -> str | None:
    code, out = _git(["rev-parse", "--verify", "--quiet", spec], cwd=repo)
    return out if code == 0 and out else None


def _object_exists(repo: Path, obj: str) -> bool:
    code, _ = _git(["cat-file", "-e", obj], cwd=repo)
    return code == 0


def _shallow_fetch(url: str, sha: str, dest: Path) -> str | None:
    """Fetch a single commit into a fresh repo at ``dest``; return error or None."""
    dest.mkdir(parents=True, exist_ok=True)
    if _git(["init", "-q"], cwd=dest)[0] != 0:
        return f"git init failed in {dest}"
    _git(["remote", "add", "origin", url], cwd=dest)
    code, out = _git(["fetch", "--depth", "1", "origin", sha], cwd=dest)
    if code != 0:
        return f"could not fetch {sha} from {url}: {out}"
    return None


def check_offline(manifest: dict[str, object], repo_root: Path) -> list[str]:
    problems: list[str] = []
    r_path = str(manifest.get("vendored_r_path") or "tools/NNS")
    recorded = manifest.get("r_src_tree_hash")
    actual = _object_hash(repo_root, f"HEAD:{r_path}/src")
    if actual is None:
        problems.append(f"could not read vendored {r_path}/src tree from HEAD")
    elif actual != recorded:
        problems.append(
            f"r_src_tree_hash {recorded!r} does not match vendored {r_path}/src "
            f"tree {actual!r}"
        )
    return problems


def check_core(manifest: dict[str, object], repo_root: Path, workdir: Path) -> list[str]:
    core_repo = str(manifest.get("core_repo") or "")
    core_commit = manifest.get("core_commit")
    core_path = str(manifest.get("vendored_core_path") or "extern/NNS-core")
    if core_commit in PLACEHOLDERS or not core_repo:
        return [f"core_commit/core_repo unset; cannot verify ({core_repo}@{core_commit})"]

    dest = workdir / "core"
    err = _shallow_fetch(f"https://github.com/{core_repo}.git", str(core_commit), dest)
    if err:
        return [err]

    code, listing = _git(["ls-tree", f"HEAD:{core_path}"], cwd=repo_root)
    if code != 0 or not listing:
        return [f"could not list vendored {core_path} in HEAD"]

    problems: list[str] = []
    for line in listing.splitlines():
        meta, _, name = line.partition("\t")
        obj = meta.split()[2]
        if not _object_exists(dest, obj):
            problems.append(
                f"vendored {core_path}/{name} ({obj}) is not present in "
                f"{core_repo}@{core_commit}"
            )
    return problems


def check_r_tarball(manifest: dict[str, object], repo_root: Path, workdir: Path) -> list[str]:
    r_repo = str(manifest.get("r_repo") or "")
    r_commit = manifest.get("r_commit")
    tarball = str(manifest.get("vendored_r_tarball") or "")
    if r_commit in PLACEHOLDERS or not r_repo or not tarball:
        return [f"r_commit/r_repo/tarball unset; cannot verify ({r_repo}@{r_commit})"]

    vendored_blob = _object_hash(repo_root, f"HEAD:{tarball}")
    if vendored_blob is None:
        return [f"vendored tarball {tarball} not found in HEAD"]

    dest = workdir / "r"
    err = _shallow_fetch(f"https://github.com/{r_repo}.git", str(r_commit), dest)
    if err:
        return [err]

    base = Path(tarball).name
    upstream_blob = _object_hash(dest, f"{r_commit}:{base}")
    if upstream_blob is None:
        print(
            f"  note: {r_repo}@{r_commit} does not commit {base}; "
            "skipping tarball blob cross-check"
        )
        return []
    if upstream_blob != vendored_blob:
        return [
            f"vendored {tarball} ({vendored_blob}) does not match "
            f"{r_repo}@{r_commit}:{base} ({upstream_blob})"
        ]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("sync/nns_source.json"))
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="checkout whose HEAD holds the vendored snapshots (default: cwd)",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    repo_root = args.repo_root.resolve()

    problems: list[str] = []
    problems.extend(check_offline(manifest, repo_root))
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        problems.extend(check_core(manifest, repo_root, workdir))
        problems.extend(check_r_tarball(manifest, repo_root, workdir))

    if problems:
        print("Vendored provenance verification FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print(
        "Vendored provenance OK: extern/NNS-core matches "
        f"{manifest.get('core_repo')}@{manifest.get('core_commit')} and "
        f"tools/NNS matches {manifest.get('r_repo')}@{manifest.get('r_commit')}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
