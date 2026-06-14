from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPORT_DEFAULT = Path("sync/last_live_r_parity_report.md")
R_CACHE_PATH = Path("tests/_r_cache.json")
# Backup suffix is gitignored (see .gitignore) so it never lands in a PR.
R_CACHE_BACKUP = R_CACHE_PATH.with_suffix(".json.bak")

# Toggles that force tests/_r.py into offline/cache-only mode. Clearing them lets
# a parity test actually shell out to live R on a cache miss.
OFFLINE_TOGGLES = (
    "CI",
    "NNS_R_CACHE_ONLY",
    "PYNNS_R_CACHE_ONLY",
    "NNS_OFFLINE",
    "PYNNS_OFFLINE",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_report(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)


def run(
    cmd: list[str],
    env_extra: dict[str, str] | None = None,
    clear_offline: bool = False,
) -> int:
    env = os.environ.copy()
    if clear_offline:
        for name in OFFLINE_TOGGLES:
            env.pop(name, None)
    if env_extra:
        env.update(env_extra)
    print("+ " + " ".join(cmd))
    completed = subprocess.run(cmd, env=env)
    return completed.returncode


def run_live_subset(out: Path, header: list[str], parity_tests: list[str]) -> None:
    """Recompute the mapped parity subset from live R and compare against Python.

    R NNS must already be installed (the workflow installs it from the upstream
    checkout at the recorded commit). The committed cache is moved aside so every
    mapped ``nns()`` call shells out to live R, the mapped tests run with the
    offline toggles cleared, then the committed cache is restored unchanged. A
    test failure means public Python behavior diverged from live R.
    """

    if shutil.which("Rscript") is None:
        lines = [*header,
            "## Result: live R unavailable",
            "",
            "`Rscript` is not on PATH, so live R parity could not be verified. "
            "Install R NNS before running with `--live`.",
        ]
        write_report(out, lines)
        raise SystemExit(4)

    existing_tests = [t for t in parity_tests if Path(t).exists()]
    missing_tests = [t for t in parity_tests if not Path(t).exists()]
    if not existing_tests:
        lines = [*header,
            "## Result: no mapped parity tests present",
            "",
            "No mapped parity test paths exist on disk; manual review is "
            "recommended.",
        ]
        if missing_tests:
            lines.extend(["", "Missing mapped tests:", ""])
            lines.extend(f"- `{t}`" for t in missing_tests)
        write_report(out, lines)
        return

    cmd = [sys.executable, "-m", "pytest", "-q", "-n", "0", *existing_tests]
    moved = False
    try:
        if R_CACHE_PATH.exists():
            shutil.move(str(R_CACHE_PATH), str(R_CACHE_BACKUP))
            moved = True
        # With the cache absent and offline toggles cleared, each mapped nns()
        # call recomputes against the freshly installed live R and the test
        # asserts the Python implementation matches it.
        code = run(cmd, clear_offline=True)
    finally:
        if R_CACHE_PATH.exists():
            R_CACHE_PATH.unlink()
        if moved:
            shutil.move(str(R_CACHE_BACKUP), str(R_CACHE_PATH))

    if code != 0:
        lines = [*header,
            "## Result: live R parity diverged",
            "",
            "Mapped parity tests recomputed every R value from the freshly "
            "installed live R NNS and the Python implementation did not match. "
            "Public Python behavior differs from live R at the recorded commit.",
            "",
            f"Failing command: `{' '.join(cmd)}`",
            f"Exit status: `{code}`",
        ]
        write_report(out, lines)
        raise SystemExit(code)

    lines = [*header,
        "## Result: live R parity passed (recomputed from live R)",
        "",
        "Every mapped parity value was recomputed from the freshly installed "
        "live R NNS and matched the Python implementation.",
        "",
        "Tests run:",
        "",
    ]
    lines.extend(f"- `{t}`" for t in existing_tests)
    if missing_tests:
        lines.extend(["", "Skipped missing mapped tests:", ""])
        lines.extend(f"- `{t}`" for t in missing_tests)
    write_report(out, lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=Path("sync/last_r_api_plan.json"))
    parser.add_argument("--r-checkout", type=Path, required=False)
    parser.add_argument("--fresh-cache", action="store_true")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "Recompute the mapped parity subset from already-installed live R "
            "(toggles cleared, committed cache moved aside) and compare against "
            "Python. A failure means Python diverged from live R."
        ),
    )
    parser.add_argument("--out", type=Path, default=REPORT_DEFAULT)
    args = parser.parse_args()

    plan = load_json(args.plan)
    parity_tests: list[str] = list(plan.get("parity_tests", []))
    requires_fresh_cache = bool(plan.get("requires_fresh_cache"))
    has_unmapped = bool(plan.get("has_unmapped_r_files"))

    header = [
        "# Live R parity report",
        "",
        f"- Plan: `{args.plan}`",
        f"- R checkout: `{args.r_checkout}`",
        f"- Fresh cache requested: `{args.fresh_cache}`",
        f"- Skip install: `{args.skip_install}`",
        f"- Live R recompute: `{args.live}`",
        "",
    ]

    # 2. Unmapped R files require manual review.
    if has_unmapped:
        lines = [*header,
            "## Result: manual review required",
            "",
            "The plan reports unmapped R files. A human must extend "
            "`sync/r_api_map.json` before automated parity can run:",
            "",
        ]
        lines.extend(f"- `{file}`" for file in plan.get("unmapped_r_files", []))
        write_report(args.out, lines)
        raise SystemExit(2)

    # Live recompute mode: the true fidelity test. It recomputes the mapped
    # subset from already-installed live R regardless of cache state, so it also
    # covers version bumps without a separate fresh-cache pass.
    if args.live:
        run_live_subset(args.out, header, parity_tests)
        return

    # 3. Fresh cache required but not requested.
    if requires_fresh_cache and not args.fresh_cache:
        lines = [*header,
            "## Result: fresh cache required",
            "",
            "The plan reports `requires_fresh_cache=true` (for example a "
            "`DESCRIPTION` version change). Re-run this workflow with "
            "`--fresh-cache` / `fresh_cache=true` to regenerate the parity cache "
            "from empty against live R.",
        ]
        write_report(args.out, lines)
        raise SystemExit(3)

    # 8. Fresh cache path: full regeneration from empty against live R.
    if args.fresh_cache:
        steps = [
            [sys.executable, "scripts/install_local_r_nns.py"],
            [
                sys.executable,
                "scripts/regenerate_r_cache.py",
                "--fresh",
                "--",
                "-n",
                "0",
                "tests/parity",
            ],
        ]
        for cmd in steps:
            code = run(cmd)
            if code != 0:
                lines = [*header,
                    "## Result: fresh cache regeneration failed",
                    "",
                    f"Failing command: `{' '.join(cmd)}`",
                    f"Exit status: `{code}`",
                ]
                write_report(args.out, lines)
                raise SystemExit(code)
        replay = [sys.executable, "-m", "pytest", "-q", "-n", "0", "tests/parity"]
        code = run(replay, env_extra={"NNS_R_CACHE_ONLY": "1"})
        if code != 0:
            lines = [*header,
                "## Result: parity replay failed after fresh regeneration",
                "",
                f"Failing command: `{' '.join(replay)}`",
                f"Exit status: `{code}`",
            ]
            write_report(args.out, lines)
            raise SystemExit(code)
        lines = [*header,
            "## Result: fresh cache regenerated and parity replay passed",
            "",
            "Cache regenerated from empty against live R; "
            "`NNS_R_CACHE_ONLY=1 pytest tests/parity` passed.",
        ]
        write_report(args.out, lines)
        return

    # 4. Install the checked-out R package unless skipped.
    if not args.skip_install:
        code = run([sys.executable, "scripts/install_local_r_nns.py"])
        if code != 0:
            lines = [*header,
                "## Result: R install failed",
                "",
                "`scripts/install_local_r_nns.py` exited nonzero; live R parity "
                "could not be run.",
                f"Exit status: `{code}`",
            ]
            write_report(args.out, lines)
            raise SystemExit(code)

    # 5/6. Run mapped parity tests that exist.
    existing_tests = [t for t in parity_tests if Path(t).exists()]
    missing_tests = [t for t in parity_tests if not Path(t).exists()]

    if not existing_tests:
        lines = [*header,
            "## Result: no mapped parity tests present",
            "",
            "No mapped parity test paths exist on disk; manual review is "
            "recommended.",
        ]
        if missing_tests:
            lines.extend(["", "Missing mapped tests:", ""])
            lines.extend(f"- `{t}`" for t in missing_tests)
        write_report(args.out, lines)
        return

    cmd = [sys.executable, "-m", "pytest", "-q", "-n", "0", *existing_tests]
    code = run(cmd)
    if code != 0:
        lines = [*header,
            "## Result: mapped live R parity tests failed",
            "",
            f"Failing command: `{' '.join(cmd)}`",
            f"Exit status: `{code}`",
        ]
        write_report(args.out, lines)
        raise SystemExit(code)

    lines = [*header,
        "## Result: mapped live R parity tests passed",
        "",
        "Tests run:",
        "",
    ]
    lines.extend(f"- `{t}`" for t in existing_tests)
    if missing_tests:
        lines.extend(["", "Skipped missing mapped tests:", ""])
        lines.extend(f"- `{t}`" for t in missing_tests)
    write_report(args.out, lines)


if __name__ == "__main__":
    main()
