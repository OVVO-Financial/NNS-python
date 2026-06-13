#!/usr/bin/env python3
"""Regenerate committed R parity cache entries with a local R/NNS install.

CI should not run this script. It intentionally clears cache-only/offline toggles
and invokes pytest so tests/_r.py can refresh tests/_r_cache.json as needed.

Usage::

    python scripts/regenerate_r_cache.py [--fresh] [-- PYTEST_ARGS...]

By default, existing cache entries are reused and only cache misses call R.
With ``--fresh``, the existing ``tests/_r_cache.json`` is moved aside to
``tests/_r_cache.json.bak`` and regeneration starts from an empty cache, so
every entry is produced by a live R call. ``--fresh`` refuses to run in CI and
requires a working local R NNS install reporting ``packageVersion("NNS")`` of
13.0 (see ``scripts/install_local_r_nns.py``).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

_CACHE_PATH = Path(__file__).resolve().parents[1] / "tests" / "_r_cache.json"
_CACHE_BACKUP_PATH = _CACHE_PATH.with_suffix(".json.bak")
_NNS_VERSION = "13.0"
_SCHEMA_VERSION = 1

_OFFLINE_TOGGLES = (
    "PYNNS_R_CACHE_ONLY",
    "NNS_R_CACHE_ONLY",
    "PYNNS_OFFLINE",
    "NNS_OFFLINE",
    "CI",
)


def _validate_cache() -> int:
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
    if cache.get("nns_version") != _NNS_VERSION:
        print(
            "ERROR: R cache validation failed: "
            f"expected nns_version {_NNS_VERSION!r}, got {cache.get('nns_version')!r}.",
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

    print(f"OK: {_CACHE_PATH} contains {len(entries)} entries for NNS {_NNS_VERSION}.")
    return 0


def _running_in_ci() -> bool:
    return os.environ.get("CI", "").lower() in {"1", "true", "yes"} or bool(
        os.environ.get("GITHUB_ACTIONS")
    )


def _verify_live_r_nns() -> int:
    """Confirm a local R NNS install reports the expected version before a fresh run."""

    rscript = shutil.which("Rscript")
    if rscript is None:
        print(
            "ERROR: --fresh requires Rscript on PATH; "
            "run scripts/install_local_r_nns.py first.",
            file=sys.stderr,
        )
        return 1
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
        print(
            "ERROR: --fresh could not load R NNS:\n" + probe.stderr,
            file=sys.stderr,
        )
        return 1
    version = probe.stdout.strip()
    if version != _NNS_VERSION:
        print(
            f"ERROR: --fresh requires R NNS {_NNS_VERSION!r}; "
            f"installed version is {version!r}.",
            file=sys.stderr,
        )
        return 1
    print(f"OK: live R NNS {version} detected for fresh regeneration.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Move the existing cache to tests/_r_cache.json.bak and regenerate every "
            "entry from a live R call. Refuses to run in CI."
        ),
    )
    parser.add_argument(
        "pytest_args",
        nargs="*",
        help="Arguments passed to pytest after an optional '--' separator.",
    )
    parsed = parser.parse_args()

    if parsed.fresh:
        if _running_in_ci():
            print(
                "ERROR: --fresh must not run in CI; it deletes the committed cache "
                "and requires a local R NNS install.",
                file=sys.stderr,
            )
            return 1
        verify_status = _verify_live_r_nns()
        if verify_status:
            return verify_status
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
    validation_status = _validate_cache()
    return pytest_status if pytest_status else validation_status


if __name__ == "__main__":
    raise SystemExit(main())
