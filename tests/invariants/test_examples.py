from __future__ import annotations

import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "docs" / "examples"


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.py")))
def test_documented_examples_run(path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    runpy.run_path(str(path), run_name="__main__")
    captured = capsys.readouterr()
    assert captured.out
