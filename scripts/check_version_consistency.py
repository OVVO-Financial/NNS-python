#!/usr/bin/env python3
"""Fail if the project version is not identical everywhere it is declared.

The distribution version (``pyproject [project].version``) is the single source
of truth. This asserts that ``nns.__version__`` and the README "Package at a
glance" current-version row match it, so a release can never ship a stale
hard-coded version string -- the README is also the PyPI project description, so
a mismatch there is exactly what shows the wrong version on the package page.

Run as a release gate (``.github/workflows/release.yml``) and as a unit test.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def pyproject_version(root: Path) -> str:
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def init_version(root: Path) -> str | None:
    text = (root / "src" / "nns" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return match.group(1) if match else None


def readme_version(root: Path) -> str | None:
    text = (root / "README.md").read_text(encoding="utf-8")
    match = re.search(r"\|\s*Current version\s*\|\s*`([^`]+)`\s*\|", text)
    return match.group(1) if match else None


def check(root: Path) -> tuple[str, list[str]]:
    version = pyproject_version(root)
    problems: list[str] = []

    init = init_version(root)
    if init is None:
        problems.append("could not find __version__ in src/nns/__init__.py")
    elif init != version:
        problems.append(f"nns.__version__ {init!r} != pyproject version {version!r}")

    readme = readme_version(root)
    if readme is None:
        problems.append("could not find the 'Current version' row in README.md")
    elif readme != version:
        problems.append(
            f"README 'Current version' {readme!r} != pyproject version {version!r}"
        )

    return version, problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_ROOT)
    args = parser.parse_args()

    version, problems = check(args.root.resolve())
    if problems:
        print("Version consistency check FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        print("Bump the version everywhere (pyproject, nns.__version__, README).")
        return 1

    print(f"Version consistency OK: {version} in pyproject, nns.__version__, and README.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
