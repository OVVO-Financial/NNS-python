from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = ROOT / "examples" / "vignettes"
DATA_DIR = ROOT / "examples" / "data"
EXPECTED = {
    "01": ("NNSvignette_01_Overview.Rmd", "01_overview.py"),
    "02": ("NNSvignette_02_Partial_Moments.Rmd", "02_partial_moments.py"),
    "03": ("NNSvignette_03_Correlation_and_Dependence.Rmd", "03_correlation_and_dependence.py"),
    "04": ("NNSvignette_04_Normalization_and_Rescaling.Rmd", "04_normalization_and_rescaling.py"),
    "05": ("NNSvignette_05_Sampling.Rmd", "05_sampling_and_simulation.py"),
    "06": ("NNSvignette_06_Comparing_Distributions.Rmd", "06_comparing_distributions.py"),
    "07": ("NNSvignette_07_Clustering_and_Regression.Rmd", "07_clustering_and_regression.py"),
    "08": ("NNSvignette_08_Classification.Rmd", "08_classification.py"),
    "09": ("NNSvignette_09_Forecasting.Rmd", "09_forecasting.py"),
}


def test_canonical_vignette_set_is_complete_and_ordered() -> None:
    numbered = sorted(path.name for path in EXAMPLE_DIR.glob("[0-9][0-9]_*.py"))
    assert numbered == [python for _, python in EXPECTED.values()]


def test_each_python_vignette_is_instructional_not_a_wrapper() -> None:
    for r_source, python_name in EXPECTED.values():
        text = (EXAMPLE_DIR / python_name).read_text(encoding="utf-8")
        assert r_source in text
        assert "# %% [markdown]" in text
        assert "save_figure(" in text
        assert "def main()" in text
        assert len(text.splitlines()) >= 90
        assert "from overview import main" not in text
        assert "from time_series_forecasting import main" not in text


def test_manifest_matches_the_canonical_pairs() -> None:
    text = (EXAMPLE_DIR / "manifest.yml").read_text(encoding="utf-8")
    pairs = re.findall(r"\n\s+r: (\S+)\n\s+python: (\S+)", text)
    assert pairs == list(EXPECTED.values())
    for criterion in (
        "section_order",
        "same_data",
        "figures",
        "returned_structures",
        "explicit_gaps",
    ):
        assert criterion in text


def test_exact_r_datasets_are_committed_locally() -> None:
    expected_rows = {"iris.csv": 151, "mtcars.csv": 33, "AirPassengers.csv": 145}
    for filename, line_count in expected_rows.items():
        path = DATA_DIR / filename
        assert path.exists()
        assert len(path.read_text(encoding="utf-8").splitlines()) == line_count


def test_parity_ledger_covers_all_nine_vignettes() -> None:
    text = (EXAMPLE_DIR / "PARITY.md").read_text(encoding="utf-8")
    for vignette_id, (_, python_name) in EXPECTED.items():
        assert f"## {vignette_id}" in text
        assert python_name in text
