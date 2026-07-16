"""Shared, non-vignette support for the canonical Python ports.

The numbered files in ``examples/vignettes`` are intentionally readable as
Jupytext-style instructional scripts.  This module keeps repetitive concerns
(dataset loading, output folders, compact result rendering, and figure saving)
out of the statistical narrative.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
from pprint import pformat
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXAMPLES_DIR = Path(__file__).resolve().parent
DATA_DIR = EXAMPLES_DIR / "data"
OUTPUT_ROOT = EXAMPLES_DIR / "vignettes" / "output"


def fast_mode() -> bool:
    """Return whether CI requested reduced repetitions for expensive sections."""
    return os.environ.get("NNS_VIGNETTE_FAST", "0") == "1"


def output_dir(script_file: str) -> Path:
    path = OUTPUT_ROOT / Path(script_file).stem
    path.mkdir(parents=True, exist_ok=True)
    return path


def section(title: str) -> None:
    line = "=" * len(title)
    print(f"\n{title}\n{line}")


def subsection(title: str) -> None:
    print(f"\n--- {title} ---")


def gap(message: str) -> None:
    """Make a genuine R/Python API difference impossible to overlook."""
    print(f"\nPYTHON API GAP: {message}")


def note(message: str) -> None:
    print(f"\nNOTE: {message}")


def show(label: str, value: Any, *, precision: int = 6) -> None:
    """Display important numerical structures without dumping huge arrays."""
    print(f"\n{label}:")
    if isinstance(value, np.ndarray):
        print(np.array2string(value, precision=precision, threshold=40, edgeitems=5))
        return
    if isinstance(value, dict):
        compact = {key: _compact(item) for key, item in value.items()}
        print(pformat(compact, width=100, sort_dicts=False))
        return
    print(value)


def _compact(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.size <= 20:
            return np.round(value, 6).tolist()
        return {
            "shape": list(value.shape),
            "head": np.round(value.reshape(-1)[:5], 6).tolist(),
            "tail": np.round(value.reshape(-1)[-5:], 6).tolist(),
        }
    if isinstance(value, dict):
        return {key: _compact(item) for key, item in value.items()}
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def table(headers: Sequence[str], rows: Iterable[Sequence[Any]], *, precision: int = 5) -> None:
    rendered = []
    for row in rows:
        rendered.append(
            [f"{item:.{precision}f}" if isinstance(item, (float, np.floating)) else str(item) for item in row]
        )
    widths = [len(str(header)) for header in headers]
    for row in rendered:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row, strict=True)]
    print("  ".join(str(header).ljust(width) for header, width in zip(headers, widths, strict=True)))
    print("  ".join("-" * width for width in widths))
    for row in rendered:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)))


def save_figure(fig: plt.Figure, directory: Path, filename: str) -> Path:
    path = directory / filename
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {path}")
    return path


def load_iris() -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows = _read_csv("iris.csv")
    feature_names = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]
    x = np.asarray([[float(row[name]) for name in feature_names] for row in rows], dtype=float)
    levels = ["setosa", "versicolor", "virginica"]
    y = np.asarray([levels.index(row["Species"]) + 1 for row in rows], dtype=float)
    return x, y, levels


def load_mtcars() -> dict[str, np.ndarray]:
    rows = _read_csv("mtcars.csv")
    numeric = ["mpg", "cyl", "disp", "hp", "drat", "wt", "qsec", "vs", "am", "gear", "carb"]
    return {name: np.asarray([float(row[name]) for row in rows], dtype=float) for name in numeric}


def load_air_passengers() -> np.ndarray:
    rows = _read_csv("AirPassengers.csv")
    return np.asarray([float(row["value"]) for row in rows], dtype=float)


def _read_csv(filename: str) -> list[dict[str, str]]:
    with (DATA_DIR / filename).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(np.asarray(values, dtype=float))
    probs = np.arange(1, ordered.size + 1, dtype=float) / ordered.size
    return ordered, probs


def partition_scatter(ax: plt.Axes, result: dict[str, Any], title: str) -> None:
    dt = result["dt"]
    labels = np.asarray(dt["quadrant"], dtype=str)
    unique, codes = np.unique(labels, return_inverse=True)
    ax.scatter(dt["x"], dt["y"], c=codes, cmap="tab20", s=14, alpha=0.7)
    rp = result["regression.points"]
    ax.scatter(rp["x"], rp["y"], marker="x", s=55, linewidths=1.6)
    ax.set_title(f"{title}\n{len(unique)} terminal paths; order={result['order']}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def regression_scatter(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    result: dict[str, Any],
    title: str,
) -> None:
    fitted = result.get("Fitted.xy")
    ax.scatter(x, y, s=12, alpha=0.45, label="observed")
    if isinstance(fitted, dict):
        order = np.argsort(np.asarray(fitted["x"], dtype=float))
        ax.plot(np.asarray(fitted["x"])[order], np.asarray(fitted["y.hat"])[order], label="NNS fit")
        if "conf.int.neg" in fitted and "conf.int.pos" in fitted:
            ax.fill_between(
                np.asarray(fitted["x"])[order],
                np.asarray(fitted["conf.int.neg"])[order],
                np.asarray(fitted["conf.int.pos"])[order],
                alpha=0.2,
                label="confidence interval",
            )
    rp = result.get("regression.points")
    if isinstance(rp, dict):
        ax.scatter(rp["x"], rp["y"], marker="x", s=45, label="regression points")
    ax.set_title(title)
    ax.legend()


def figure_grid(nrows: int, ncols: int, *, width: float = 5.0, height: float = 4.0):
    return plt.subplots(nrows, ncols, figsize=(width * ncols, height * nrows), squeeze=False)
