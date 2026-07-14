#!/usr/bin/env python3
"""Install an R NNS package from a local source directory, tarball, or binary ZIP.

The helper never downloads NNS from CRAN. With no ``--source`` argument it
prefers ``tools/NNS`` and otherwise uses a vendored NNS archive under ``tools``.
A Windows binary can be supplied directly, for example::

    python scripts/install_local_r_nns.py --source C:/Users/me/Documents/NNS_13.1.zip

After installation the helper loads NNS and reports the actual installed
version. Use ``--expected-version`` only when an exact version must be enforced.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TOOLS_DIR = _REPO_ROOT / "tools"
_SOURCE_DIR = _TOOLS_DIR / "NNS"

_VERSION_SCRIPT = (
    "suppressPackageStartupMessages(library(NNS)); "
    "cat(as.character(packageVersion('NNS')))"
)


def _resolve_source(override: Path | None = None) -> Path:
    """Return a local R package source directory or package archive."""

    if override is not None:
        source = override.expanduser().resolve()
        if source.is_dir() and (source / "DESCRIPTION").is_file():
            return source
        if source.is_file() and (
            source.suffix.lower() == ".zip"
            or source.name.lower().endswith((".tar.gz", ".tgz"))
        ):
            return source
        raise SystemExit(
            f"ERROR: --source {source} is not an R package source. Expected a directory "
            "containing DESCRIPTION, a source .tar.gz/.tgz, or a Windows binary .zip."
        )

    if (_SOURCE_DIR / "DESCRIPTION").is_file():
        return _SOURCE_DIR

    archives = sorted(
        [
            *(_TOOLS_DIR.glob("NNS_*.tar.gz")),
            *(_TOOLS_DIR.glob("NNS_*.tgz")),
            *(_TOOLS_DIR.glob("NNS_*.zip")),
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if archives:
        return archives[0]

    raise SystemExit(
        "ERROR: no vendored NNS source found. Expected tools/NNS/DESCRIPTION or an "
        "NNS_*.tar.gz, NNS_*.tgz, or NNS_*.zip archive under tools/."
    )


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if path is None:
        raise SystemExit(
            f"ERROR: {tool!r} is not on PATH. Install R before running this helper."
        )
    return path


def _install(source: Path, r_bin: str, rscript_bin: str) -> int:
    if source.suffix.lower() == ".zip":
        if os.name != "nt":
            print(
                "ERROR: an R Windows binary ZIP can only be installed on Windows.",
                file=sys.stderr,
            )
            return 1
        expression = (
            f"install.packages({json.dumps(str(source))}, repos = NULL, type = 'win.binary')"
        )
        install = subprocess.run([rscript_bin, "-e", expression], check=False)
    else:
        install = subprocess.run([r_bin, "CMD", "INSTALL", str(source)], check=False)

    if install.returncode != 0:
        print("ERROR: local R package installation failed.", file=sys.stderr)
    return install.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help=(
            "Install from this local R package source: a directory with DESCRIPTION, "
            "a source tarball, or a Windows binary ZIP."
        ),
    )
    parser.add_argument(
        "--expected-version",
        default=None,
        help="Optionally require the installed NNS package to report this exact version.",
    )
    args = parser.parse_args()

    r_bin = _require("R")
    rscript_bin = _require("Rscript")
    source = _resolve_source(args.source)

    print(f"Installing R NNS from local package: {source} (not CRAN)")
    install_status = _install(source, r_bin, rscript_bin)
    if install_status != 0:
        return install_status

    probe = subprocess.run(
        [rscript_bin, "-e", _VERSION_SCRIPT],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        print("ERROR: failed to load NNS after install:\n" + probe.stderr, file=sys.stderr)
        return probe.returncode

    installed_version = probe.stdout.strip()
    if not installed_version:
        print("ERROR: installed NNS reported an empty version.", file=sys.stderr)
        return 1

    print(f"Installed NNS version: {installed_version}")
    if args.expected_version is not None and installed_version != args.expected_version:
        print(
            "ERROR: installed NNS version "
            f"{installed_version!r} does not match expected {args.expected_version!r}.",
            file=sys.stderr,
        )
        return 1

    print(f"OK: R NNS {installed_version} installed from local package.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
