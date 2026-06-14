#!/usr/bin/env python3
"""Install R NNS 13.0 from the vendored package source in this repository.

This installs NNS from the local source under ``tools/`` and never from CRAN.
It prefers the extracted package directory ``tools/NNS`` and falls back to the
vendored tarball ``tools/NNS_13.0.tar.gz``. After installation it verifies that
the loaded package reports version ``13.0``.

Usage::

    python scripts/install_local_r_nns.py

Requires ``R`` and ``Rscript`` on PATH. CI must not depend on this script; it is
a developer helper for regenerating the committed parity cache with a local,
non-CRAN R NNS install.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TOOLS_DIR = _REPO_ROOT / "tools"
_SOURCE_DIR = _TOOLS_DIR / "NNS"
_SOURCE_TARBALL = _TOOLS_DIR / "NNS_13.0.tar.gz"
_EXPECTED_VERSION = "13.0"

_VERSION_SCRIPT = (
    "suppressPackageStartupMessages(library(NNS)); "
    "cat(as.character(packageVersion('NNS')))"
)


def _resolve_source(override: Path | None = None) -> Path:
    """Return the NNS source path to install.

    With no override, prefer the vendored extracted directory and fall back to
    the vendored tarball. With an override (for example an upstream checkout at a
    recorded R commit), install that path directly after validating it is a
    package source directory or tarball.
    """

    if override is not None:
        if override.is_dir() and (override / "DESCRIPTION").is_file():
            return override
        if override.is_file():
            return override
        raise SystemExit(
            f"ERROR: --source {override} is not an R package source. Expected a "
            "directory containing DESCRIPTION or a package tarball."
        )

    if (_SOURCE_DIR / "DESCRIPTION").is_file():
        return _SOURCE_DIR
    if _SOURCE_TARBALL.is_file():
        return _SOURCE_TARBALL
    raise SystemExit(
        "ERROR: no vendored NNS source found. Expected "
        f"{_SOURCE_DIR}/DESCRIPTION or {_SOURCE_TARBALL}."
    )


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if path is None:
        raise SystemExit(
            f"ERROR: {tool!r} is not on PATH. Install R before running this helper; "
            "this script installs NNS from local source, not from CRAN."
        )
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help=(
            "Install from this R package source (directory with DESCRIPTION or a "
            "tarball) instead of the vendored tools/NNS. Use an upstream checkout "
            "to install live R NNS at a recorded commit."
        ),
    )
    parser.add_argument(
        "--expected-version",
        default=_EXPECTED_VERSION,
        help=(
            "Package version the install must report after loading. Defaults to "
            f"{_EXPECTED_VERSION!r}. Pass the recorded upstream version when "
            "installing from a non-vendored source."
        ),
    )
    args = parser.parse_args()
    expected_version = args.expected_version or _EXPECTED_VERSION

    r_bin = _require("R")
    rscript_bin = _require("Rscript")
    source = _resolve_source(args.source)

    print(f"Installing R NNS from local source: {source} (not CRAN)")
    install = subprocess.run(
        [r_bin, "CMD", "INSTALL", str(source)],
        check=False,
    )
    if install.returncode != 0:
        print("ERROR: R CMD INSTALL failed.", file=sys.stderr)
        return install.returncode

    probe = subprocess.run(
        [rscript_bin, "-e", _VERSION_SCRIPT],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        print(
            "ERROR: failed to load NNS after install:\n" + probe.stderr,
            file=sys.stderr,
        )
        return probe.returncode

    installed_version = probe.stdout.strip()
    print(f"Installed NNS version: {installed_version}")
    if installed_version != expected_version:
        print(
            "ERROR: installed NNS version "
            f"{installed_version!r} does not match expected {expected_version!r}.",
            file=sys.stderr,
        )
        return 1

    print(f"OK: R NNS {expected_version} installed from local source.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
