#!/usr/bin/env python3
"""Regenerate committed R parity cache entries with the installed R/NNS package.

CI should not run this script. It intentionally clears cache-only/offline toggles
and invokes pytest so tests/_r.py can refresh tests/_r_cache.json as needed.

Usage::

    python scripts/regenerate_r_cache.py [--fresh] [-- PYTEST_ARGS...]

By default, existing cache entries are reused and only cache misses call R.
With ``--fresh``, the script detects the installed R NNS version, updates the
cache-version marker in ``tests/_r.py``, moves the old cache aside, and rebuilds
every entry from live R calls. No source-code edit is needed when NNS advances
to a new version.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CACHE_PATH = _REPO_ROOT / "tests" / "_r_cache.json"
_CACHE_BACKUP_PATH = _CACHE_PATH.with_suffix(".json.bak")
_R_HELPER_PATH = _REPO_ROOT / "tests" / "_r.py"
_SCHEMA_VERSION = 1
_VERSION_ASSIGNMENT = re.compile(
    r'^_NNS_VERSION\s*=\s*["\'][^"\']+["\']\s*$',
    flags=re.MULTILINE,
)

_OFFLINE_TOGGLES = (
    "PYNNS_R_CACHE_ONLY",
    "NNS_R_CACHE_ONLY",
    "PYNNS_OFFLINE",
    "NNS_OFFLINE",
    "CI",
)


def _validate_cache(expected_version: str | None = None) -> int:
    if not _CACHE_PATH.exists():
        print(f"ERROR: R cache validation failed: {_CACHE_PATH} does not exist.", file=sys.stderr)
        return 1
    if _CACHE_PATH.stat().st_size == 0:
        print(f"ERROR: R cache validation failed: {_CACHE_PATH} is empty.", file=sys.stderr)
        return 1

    try:
        cache: Any = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(
            f"ERROR: R cache validation failed: {_CACHE_PATH} is not valid JSON: {exc}.",
            file=sys.stderr,
        )
        return 1

    if not isinstance(cache, dict):
        print(
            f"ERROR: R cache validation failed: {_CACHE_PATH} top-level value is not an object.",
            file=sys.stderr,
        )
        return 1

    cache_version = cache.get("nns_version")
    if not isinstance(cache_version, str) or not cache_version.strip():
        print(
            "ERROR: R cache validation failed: nns_version must be a non-empty string.",
            file=sys.stderr,
        )
        return 1
    if expected_version is not None and cache_version != expected_version:
        print(
            "ERROR: R cache validation failed: "
            f"expected nns_version {expected_version!r}, got {cache_version!r}.",
            file=sys.stderr,
        )
        return 1
    if cache.get("schema_version") != _SCHEMA_VERSION:
        print(
            "ERROR: R cache validation failed: "
            f"expected schema_version {_SCHEMA_VERSION!r}, got {cache.get('schema_version')!r}.",
            file=sys.stderr,
        )
        return 1

    entries = cache.get("entries")
    if not isinstance(entries, dict):
        print(
            f"ERROR: R cache validation failed: {_CACHE_PATH} entries value is not an object.",
            file=sys.stderr,
        )
        return 1
    if not entries:
        print(
            f"ERROR: R cache validation failed: {_CACHE_PATH} entries object is empty.",
            file=sys.stderr,
        )
        return 1

    print(f"OK: {_CACHE_PATH} contains {len(entries)} entries for NNS {cache_version}.")
    return 0


def _running_in_ci() -> bool:
    return os.environ.get("CI", "").lower() in {"1", "true", "yes"} or bool(
        os.environ.get("GITHUB_ACTIONS")
    )


def _live_r_nns_version() -> str | None:
    """Return the installed R NNS version, or ``None`` after printing an error."""

    rscript = shutil.which("Rscript")
    if rscript is None:
        print(
            "ERROR: --fresh requires Rscript on PATH. Install R and the desired "
            "NNS package first (scripts/install_local_r_nns.py can install a source or binary).",
            file=sys.stderr,
        )
        return None

    probe = subprocess.run(
        [
            rscript,
            "-e",
            "suppressPackageStartupMessages(library(NNS)); "
            "cat(as.character(packageVersion('NNS')))",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        print("ERROR: --fresh could not load R NNS:\n" + probe.stderr, file=sys.stderr)
        return None

    version = probe.stdout.strip()
    if not version:
        print("ERROR: installed R NNS reported an empty version.", file=sys.stderr)
        return None

    print(f"OK: live R NNS {version} detected for fresh regeneration.")
    return version


def _set_cache_version(version: str) -> int:
    """Update tests/_r.py so cache metadata follows the installed R package."""

    try:
        source = _R_HELPER_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"ERROR: could not read {_R_HELPER_PATH}: {exc}", file=sys.stderr)
        return 1

    replacement = f'_NNS_VERSION = {version!r}'
    updated, count = _VERSION_ASSIGNMENT.subn(replacement, source, count=1)
    if count != 1:
        print(
            f"ERROR: could not find exactly one _NNS_VERSION assignment in {_R_HELPER_PATH}.",
            file=sys.stderr,
        )
        return 1

    if updated != source:
        _R_HELPER_PATH.write_text(updated, encoding="utf-8")
        print(f"Updated {_R_HELPER_PATH} cache marker to NNS {version}.")
    else:
        print(f"Cache marker already targets NNS {version}.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Detect the installed R NNS version, update the cache marker, move the old "
            "cache to tests/_r_cache.json.bak, and regenerate every entry from live R."
        ),
    )
    parser.add_argument(
        "pytest_args",
        nargs="*",
        help="Arguments passed to pytest after an optional '--' separator.",
    )
    parsed = parser.parse_args()

    expected_version: str | None = None
    if parsed.fresh:
        if _running_in_ci():
            print(
                "ERROR: --fresh must not run in CI; it deletes the committed cache "
                "and requires a local R NNS install.",
                file=sys.stderr,
            )
            return 1

        expected_version = _live_r_nns_version()
        if expected_version is None:
            return 1
        if _set_cache_version(expected_version):
            return 1

        if _CACHE_PATH.exists():
            _CACHE_PATH.replace(_CACHE_BACKUP_PATH)
            print(f"Moved existing cache to {_CACHE_BACKUP_PATH}; starting from empty cache.")
        else:
            print("No existing cache found; starting from empty cache.")

    env = os.environ.copy()
    for name in _OFFLINE_TOGGLES:
        env.pop(name, None)

    args = parsed.pytest_args
    if args[:1] == ["--"]:
        args = args[1:]
    if not args:
        args = ["tests/parity"]

    pytest_status = subprocess.call([sys.executable, "-m", "pytest", "-q", *args], env=env)
    validation_status = _validate_cache(expected_version)
    return pytest_status if pytest_status else validation_status


if __name__ == "__main__":
    raise SystemExit(main())
