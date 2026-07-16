"""Run the nine canonical NNS Python vignettes in R curriculum order.

The R package is the source of truth. The numbered Python entry points under
``examples/vignettes`` follow the same 01-09 topic sequence and self-check with
assertions. This driver reports PASS/FAIL and exits non-zero on failure.
"""
from __future__ import annotations

import importlib.util
import os
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VIGNETTE_DIR = REPO_ROOT / "examples" / "vignettes"

VIGNETTES = [
    ("01", "Overview", "01_overview"),
    ("02", "Partial Moments", "02_partial_moments"),
    ("03", "Correlation and Dependence", "03_correlation_and_dependence"),
    ("04", "Normalization and Rescaling", "04_normalization_and_rescaling"),
    ("05", "Sampling and Simulation", "05_sampling_and_simulation"),
    ("06", "Comparing Distributions", "06_comparing_distributions"),
    ("07", "Clustering and Regression", "07_clustering_and_regression"),
    ("08", "Classification", "08_classification"),
    ("09", "Forecasting", "09_forecasting"),
]


def _load_vignette_main(stem: str):
    path = VIGNETTE_DIR / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(f"nns_vignette_{stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load vignette script {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def run() -> int:
    os.chdir(REPO_ROOT)
    results: list[tuple[str, str, bool, float]] = []

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
        results.append((number, title, ok, time.perf_counter() - start))

    print("\n" + "=" * 60)
    print("  Canonical vignette verification summary")
    print("=" * 60)
    passed = 0
    for number, title, ok, elapsed in results:
        status = "PASS" if ok else "FAIL"
        passed += int(ok)
        print(f"  [{status}] {number}  {title}  ({elapsed:.2f}s)")
    print("-" * 60)
    print(f"  {passed}/{len(results)} vignettes passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(run())
