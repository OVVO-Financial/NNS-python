#!/usr/bin/env python3
"""Regenerate committed R parity cache entries with a local R/NNS install.

CI should not run this script. It intentionally clears cache-only/offline toggles
and invokes pytest so tests/_r.py can refresh tests/_r_cache.json as needed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

_CACHE_PATH = Path(__file__).resolve().parents[1] / "tests" / "_r_cache.json"
_NNS_VERSION = "13.0"
_SCHEMA_VERSION = 1


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

    return 0


def main() -> int:
    env = os.environ.copy()
    for name in ("PYNNS_R_CACHE_ONLY", "PYNNS_OFFLINE", "CI"):
        env.pop(name, None)

    args = sys.argv[1:]
    if args[:1] == ["--"]:
        args = args[1:]
    if not args:
        args = ["tests/parity"]

    pytest_status = subprocess.call([sys.executable, "-m", "pytest", "-q", *args], env=env)
    validation_status = _validate_cache()
    return pytest_status if pytest_status else validation_status


if __name__ == "__main__":
    raise SystemExit(main())
