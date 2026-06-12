#!/usr/bin/env python3
"""Regenerate committed R parity cache entries with a local R/NNS install.

CI should not run this script. It intentionally clears cache-only/offline toggles
and invokes pytest so tests/_r.py can refresh tests/_r_cache.json as needed.
"""

from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    env = os.environ.copy()
    for name in ("PYNNS_R_CACHE_ONLY", "PYNNS_OFFLINE", "CI"):
        env.pop(name, None)

    args = sys.argv[1:]
    if args[:1] == ["--"]:
        args = args[1:]
    if not args:
        args = ["tests/parity"]

    return subprocess.call([sys.executable, "-m", "pytest", "-q", *args], env=env)


if __name__ == "__main__":
    raise SystemExit(main())
