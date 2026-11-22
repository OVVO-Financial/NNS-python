"""Run every live R sweep and fail on any difference from R.

Each sweep builds its cases, evaluates them in one Rscript batch against the
installed R NNS (``NNS_LIVE_R_LIBS`` picks the library) and compares the port's
results exactly (``RTOL = ATOL = 0``). ``--mode fallback`` sets
``NNS_DISABLE_NATIVE=1`` so the NumPy implementation is checked as well as the
native extension; ``--mode both`` runs each sweep both ways.

    python scripts/live_parity/run.py --mode both --jobs 4 --summary live-parity.md

The exit status is 1 when any case differs, a sweep fails to run or a sweep
reports no result.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent

SWEEPS: dict[str, str] = {
    "fz_pm": "Partial moments, VaR, co-moments, `PM.matrix`, `NNS.moments`",
    "fz_ct": "Central tendencies, `NNS.CDF`, `NNS.dep`, `NNS.copula`, `NNS.caus`, `NNS.norm`, `NNS.rescale`",
    "fz_dist": "`NNS.distance`, `NNS.distance.bulk`",
    "fz_part": "`NNS.part`",
    "fz_reg": "`NNS.reg` (univariate, multivariate, classification, dimension reduction, smoothing)",
    "fz_smspl": "R's smoothing spline under `NNS.reg(smooth = TRUE)`",
    "fz_fc": "`NNS.seas`, `NNS.ARMA`, `NNS.ARMA.optim`, `NNS.diff`, `dy.dx`, `dy.d_`",
    "fz_var": "`NNS.VAR`",
    "fz_stack": "`NNS.stack`, `NNS.boost`",
    "stack_ties": "Classification `NNS.stack` built to tie (coarse predictors, pooled folds)",
    "fz_anova": "`NNS.ANOVA`",
    "fz_sd": "Stochastic dominance (`NNS.FSD`/`SSD`/`TSD`, `.uni`, clusters, efficient sets)",
    "fz_sd_edge": "Stochastic dominance edge cases (ties, infinities, tiny scales)",
    "fz_boot": "`NNS.meboot`, `NNS.MC`, `NNS.SS` and ARMA intervals under `set.seed`",
    "fz_pval": "Permutation p-values (`NNS.seas`, `NNS.dep`, `NNS.caus`)",
    "dep_perm": "`NNS.dep` permutation matches",
    "big_mreg": "Multivariate `NNS.reg` predictions on 900-1,500 rows",
    "edge_new": "Compiled paths: integer-degree VaR, `NNS.copula`, `NNS.dep` matrices, class predictions, mixed `dy.d_`",
    "fz_inputs": "Input validation against R 14.2's input layer (accepted inputs and error messages)",
}

RESULT = re.compile(r"^== (?P<name>.+?): (?P<ok>\d+)/(?P<total>\d+) cases match", re.MULTILINE)


@dataclass
class Outcome:
    sweep: str
    mode: str
    matched: int
    total: int
    seconds: float
    error: str | None
    report: str

    @property
    def passed(self) -> bool:
        return self.error is None and self.total > 0 and self.matched == self.total


def run_sweep(sweep: str, mode: str, timeout: float) -> Outcome:
    env = dict(os.environ, RTOL="0", ATOL="0", MPLBACKEND="Agg")
    env.pop("NNS_DISABLE_NATIVE", None)
    if mode == "fallback":
        env["NNS_DISABLE_NATIVE"] = "1"
    start = time.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, str(HERE / f"{sweep}.py")],
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return Outcome(sweep, mode, 0, 0, time.monotonic() - start, "timed out", "")
    seconds = time.monotonic() - start
    found = list(RESULT.finditer(proc.stdout))
    matched = sum(int(m["ok"]) for m in found)
    total = sum(int(m["total"]) for m in found)
    error = None
    if proc.returncode != 0:
        error = f"exit status {proc.returncode}: {proc.stderr.strip()[-2000:]}"
    elif not found:
        error = "no result line"
    report = proc.stdout[found[0].start() :] if found else proc.stdout[-4000:]
    return Outcome(sweep, mode, matched, total, seconds, error, report)


def summary_table(outcomes: list[Outcome], modes: list[str]) -> str:
    by_key = {(o.sweep, o.mode): o for o in outcomes}
    header = "| Sweep | Area | " + " | ".join(modes) + " |"
    lines = [header, "| --- | --- | " + " | ".join("---:" for _ in modes) + " |"]
    for sweep, area in SWEEPS.items():
        cells = []
        for mode in modes:
            o = by_key.get((sweep, mode))
            if o is None:
                cells.append("-")
            elif o.error is not None:
                cells.append(f"**error** ({o.error.splitlines()[0][:60]})")
            else:
                mark = "" if o.passed else " **MISMATCH**"
                cells.append(f"{o.matched:,}/{o.total:,}{mark}")
        if any(c != "-" for c in cells):
            lines.append(f"| `{sweep}` | {area} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mode", choices=("native", "fallback", "both"), default="both")
    parser.add_argument("--only", nargs="+", choices=sorted(SWEEPS), help="run these sweeps")
    parser.add_argument("--jobs", type=int, default=2, help="sweeps run at once")
    parser.add_argument("--timeout", type=float, default=3600.0, help="seconds per sweep")
    parser.add_argument("--summary", type=Path, help="write a Markdown summary here")
    args = parser.parse_args()

    modes = ["native", "fallback"] if args.mode == "both" else [args.mode]
    sweeps = args.only or list(SWEEPS)
    jobs = [(s, m) for m in modes for s in sweeps]
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        outcomes = list(pool.map(lambda job: run_sweep(*job, args.timeout), jobs))

    for o in outcomes:
        status = "ok" if o.passed else "FAIL"
        print(
            f"[{status}] {o.sweep} ({o.mode}, {o.seconds:.0f}s): {o.matched}/{o.total}", flush=True
        )
        if not o.passed:
            print(o.error or o.report, flush=True)

    failed = [o for o in outcomes if not o.passed]
    cases = sum(o.total for o in outcomes)
    verdict = (
        f"All {cases:,} live R comparisons match exactly."
        if not failed
        else f"{len(failed)} of {len(outcomes)} sweep runs differ from R or failed."
    )
    text = "### Live R sweeps (exact comparison)\n\n" + summary_table(outcomes, modes)
    text += f"\n{verdict}\n"
    print(text)
    if args.summary is not None:
        args.summary.write_text(text, encoding="utf-8")
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as handle:
            handle.write(text)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
