from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "vignettes"


@pytest.mark.parametrize("script", sorted(EXAMPLE_DIR.glob("*.py")), ids=lambda p: p.stem)
def test_vignette_example(script: Path) -> None:
    subprocess.run(
        [sys.executable, str(script)],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
    )
