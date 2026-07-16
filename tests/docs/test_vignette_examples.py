from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = ROOT / "examples" / "vignettes"
CANONICAL = sorted(EXAMPLE_DIR.glob("[0-9][0-9]_*.py"))


@pytest.mark.parametrize("script", CANONICAL, ids=lambda p: p.stem)
def test_canonical_vignette_executes_and_generates_figures(script: Path) -> None:
    output = EXAMPLE_DIR / "output" / script.stem
    if output.exists():
        for figure in output.glob("*.png"):
            figure.unlink()

    env = os.environ.copy()
    env.update(
        {
            "NNS_VIGNETTE_FAST": "1",
            "MPLBACKEND": "Agg",
            "PYTHONPATH": str(ROOT),
        }
    )
    completed = subprocess.run(
        [sys.executable, str(script)],
        check=True,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )

    assert "Saved figure:" in completed.stdout
    figures = sorted(output.glob("*.png"))
    assert figures, f"{script.name} produced no figures"
    assert all(path.stat().st_size > 0 for path in figures)
