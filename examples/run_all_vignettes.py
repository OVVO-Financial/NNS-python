"""Run every NNS Python vignette end to end and print its output.

This is a single, IDLE-friendly driver for the vignette example scripts in
``examples/vignettes/``. Open it in IDLE and press **F5** (or run
``python examples/run_all_vignettes.py`` from a terminal) to execute all
vignettes in the documented order and print each one's output, so you can
compare it against the R NNS vignettes PDF.

Each vignette also self-checks with assertions, so this driver reports PASS/FAIL
per vignette and a final summary, and exits non-zero if any vignette fails.

Requires the package to be importable (``pip install -e .`` from the repo root).
"""

from __future__ import annotations

import importlib.util
import os
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VIGNETTE_DIR = REPO_ROOT / "examples" / "vignettes"

# Ordered to match the R NNS vignettes PDF.
# (number, title, example-script stem)
VIGNETTES = [
    ("00", "Overview", "overview"),
    ("01", "Partial moments", "partial_moments"),
    ("02", "Descriptive & distributional tools",
     "descriptive_distributional_tools"),
    ("03", "Dependence & nonlinear association",
     "dependence_nonlinear_association"),
    ("04", "Normalization & rescaling",
     "normalization_rescaling"),
    ("05", "Hypothesis, ANOVA & stochastic superiority",
     "hypothesis_anova_stochastic_superiority"),
    ("06", "Regression, boosting, stacking & causality",
     "regression_boosting_stacking_causality"),
    ("07", "Time series forecasting",
     "time_series_forecasting"),
    ("08", "Simulation, bootstrap & risk-neutral",
     "simulation_bootstrap_riskneutral"),
    ("09", "Portfolio & stochastic dominance",
     "portfolio_stochastic_dominance"),
]


def _load_vignette_main(stem):
    """Import an example script by path and return its ``main`` callable."""
    path = VIGNETTE_DIR / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(f"nns_vignette_{stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load vignette script {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def run() -> int:
    # Match the cwd the test suite uses so any relative paths resolve.
    os.chdir(REPO_ROOT)

    results = []
    for number, title, stem in VIGNETTES:
        banner = f"  Vignette {number}: {title}  "
        rule = "=" * max(len(banner), 60)
        print("\n" + rule)
        print(banner)
        print(f"  script: examples/vignettes/{stem}.py")
        print(rule)

        start = time.perf_counter()
        try:
            _load_vignette_main(stem)()
            ok = True
        except Exception:
            ok = False
            print(traceback.format_exc())
        elapsed = time.perf_counter() - start
        results.append((number, title, ok, elapsed))

    print("\n" + "=" * 60)
    print("  Vignette verification summary")
    print("=" * 60)
    passed = 0
    for number, title, ok, elapsed in results:
        status = "PASS" if ok else "FAIL"
        passed += int(ok)
        print(f"  [{status}] {number}  {title}  ({elapsed:.2f}s)")
    print("-" * 60)
    print(f"  {passed}/{len(results)} vignettes passed")
    if passed != len(results):
        print("  Some vignettes FAILED — see tracebacks above.")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(run())
