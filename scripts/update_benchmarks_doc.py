from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_TESTS = ROOT / "tests" / "benchmarks" / "test_lpm.py"
R_BASELINE = ROOT / "tests" / "benchmarks" / "_r_baseline.json"
BENCHMARK_DOC = ROOT / "docs" / "benchmarks.md"
REALISTIC_SD_R_PLACEHOLDERS = {
    ("sd_efficient_set", 252, 50, 1): 0.0023,
    ("sd_efficient_set", 252, 50, 2): 0.0022,
    ("nns_sd_cluster", 252, 50, 1): 0.0026,
    ("nns_sd_cluster", 252, 50, 2): 0.0073,
    ("sd_efficient_set", 252, 100, 1): 0.0052,
    ("sd_efficient_set", 252, 100, 2): 0.0046,
    ("nns_sd_cluster", 252, 100, 1): 0.0059,
    ("nns_sd_cluster", 252, 100, 2): 0.0155,
    ("sd_efficient_set", 252, 250, 2): 0.0146,
    ("nns_sd_cluster", 252, 250, 2): 0.0579,
    ("sd_efficient_set", 1257, 100, 2): 0.0199,
    ("sd_efficient_set", 252, 478, 2): 0.039,
    ("nns_sd_cluster", 252, 478, 2): 0.185,
    ("sd_efficient_set", 1257, 250, 2): 0.068,
    ("nns_sd_cluster", 1257, 250, 2): 0.186,
    ("sd_efficient_set", 1257, 478, 2): 0.178,
    ("nns_sd_cluster", 1257, 478, 2): 0.618,
    ("rolling_sd_efficient_set_252d_monthly", 252, 100, 2): 0.28,
    ("rolling_sd_efficient_set_252d_monthly", 252, 478, 2): 2.078,
    ("rolling_sd_cluster_252d_monthly", 252, 100, 2): 0.7997,
    ("rolling_sd_cluster_252d_monthly", 252, 478, 2): 9.384,
    ("rolling_sd_cluster_756d_quarterly", 756, 478, 2): 4.2,
    ("rolling_sd_efficient_set_252d_quarterly", 252, 478, 1): 1.149,
    ("rolling_sd_cluster_252d_quarterly", 252, 478, 1): 1.161,
    ("rolling_sd_efficient_set_degree1_vs_degree2_252d_quarterly", 252, 478, 0): 1.847,
    ("mag7_market_downside_stress", 1257, 9, 1): 0.0417,
    ("pm_matrix_degree1_mean", 252, 478, 1): 0.2683,
    ("pm_matrix_degree1_mean", 1257, 478, 1): 1.385,
    ("pm_matrix_degree2_zero", 252, 478, 2): 0.271,
    ("market_relative_daily_dispersion", 1257, 478, 2): 0.0297,
    ("market_relative_rolling_dispersion_63d", 1257, 478, 2): 0.0287,
    ("market_relative_rolling_dispersion_252d", 1257, 478, 2): 0.0287,
}
LABEL_OVERRIDES = {
    **{
        f"test_dy_d_scalar_wrt1_100x2[{eval_points}]": (
            f"`dy_d`, scalar wrt=1, eval_points={eval_points}, N=2, T_obs=100"
        )
        for eval_points in ("mean", "median", "last", "obs", "apd")
    },
    **{
        f"test_nns_var_80x3_h3_tau2[{method}]": (
            f"`nns_var`, dim_red_method={method}, N=3, T_obs=80, h=3, tau=2"
        )
        for method in ("cor", "NNS.dep", "NNS.caus", "all")
    },
}


@dataclass(frozen=True)
class BenchmarkRow:
    name: str
    label: str
    python_seconds: float
    r_seconds: float


@dataclass(frozen=True)
class RealisticSDRow:
    function_name: str
    rows: int
    columns: int
    degree: int
    python_seconds: float
    r_seconds: float | None
    r_source: str


@dataclass(frozen=True)
class PythonOnlyRow:
    label: str
    python_seconds: float
    extra_info: dict[str, Any]
    r_seconds: float | None
    r_source: str


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update docs/benchmarks.md from pytest-benchmark JSON and R baselines."
    )
    parser.add_argument("benchmark_json", type=Path)
    parser.add_argument("--output", type=Path, default=BENCHMARK_DOC)
    parser.add_argument(
        "--realistic-sd-r-csv",
        type=Path,
        default=None,
        help="CSV emitted by scripts/benchmark_realistic_sd_r.R.",
    )
    args = parser.parse_args()

    benchmark_payload = _read_json(args.benchmark_json)
    r_baseline_payload = _read_json(R_BASELINE)
    r_baseline = r_baseline_payload["entries"]
    r_version = str(r_baseline_payload["nns_version"])
    realistic_r = _read_realistic_sd_r_csv(args.realistic_sd_r_csv)
    key_by_test = _r_baseline_keys_by_test()

    rows: list[BenchmarkRow] = []
    realistic_rows: list[RealisticSDRow] = []
    python_only_rows: list[PythonOnlyRow] = []
    for benchmark in benchmark_payload["benchmarks"]:
        name = str(benchmark["name"])
        python_seconds = float(benchmark["stats"]["mean"])
        realistic_case = _realistic_sd_case_from_benchmark_name(name)
        if realistic_case is not None:
            r_seconds = realistic_r.get(realistic_case)
            r_source = "measured"
            if r_seconds is None:
                r_seconds = REALISTIC_SD_R_PLACEHOLDERS.get(realistic_case)
                r_source = "placeholder"
            realistic_rows.append(
                RealisticSDRow(
                    function_name=realistic_case[0],
                    rows=realistic_case[1],
                    columns=realistic_case[2],
                    degree=realistic_case[3],
                    python_seconds=python_seconds,
                    r_seconds=r_seconds,
                    r_source=r_source,
                )
            )
            continue
        python_only_label = _realistic_python_only_label(name)
        if python_only_label is not None:
            workflow_case = _realistic_workflow_case_from_benchmark_name(name)
            r_seconds = realistic_r.get(workflow_case) if workflow_case is not None else None
            r_source = "measured"
            if r_seconds is None and workflow_case is not None:
                r_seconds = REALISTIC_SD_R_PLACEHOLDERS.get(workflow_case)
                r_source = "placeholder"
            python_only_rows.append(
                PythonOnlyRow(
                    label=python_only_label,
                    python_seconds=python_seconds,
                    extra_info=_as_extra_info(benchmark.get("extra_info", {})),
                    r_seconds=r_seconds,
                    r_source=r_source if r_seconds is not None else "none",
                )
            )
            continue
        r_key = _r_baseline_key(name, key_by_test)
        r_seconds = float(r_baseline[r_key])
        label = LABEL_OVERRIDES.get(name, _fallback_label(name))
        rows.append(
            BenchmarkRow(
                name=name,
                label=label,
                python_seconds=python_seconds,
                r_seconds=r_seconds,
            )
        )

    args.output.write_text(
        _render(rows, realistic_rows, python_only_rows, r_version),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}.")
    return payload


def _read_realistic_sd_r_csv(
    path: Path | None,
) -> dict[tuple[str, int, int, int], float]:
    if path is None or not path.exists():
        return {}
    rows: dict[tuple[str, int, int, int], float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            function_name = row["function_name"]
            rows_count = int(row["rows"])
            columns = int(row["columns"])
            degree = int(row["degree"])
            rows[(function_name, rows_count, columns, degree)] = float(row["mean_seconds"])
    return rows


def _as_extra_info(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _r_baseline_keys_by_test() -> dict[str, str]:
    tree = ast.parse(BENCHMARK_TESTS.read_text(encoding="utf-8"))
    keys: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
            continue
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Subscript)
                and isinstance(child.value, ast.Name)
                and child.value.id == "r_baseline"
            ):
                key = _literal_subscript(child.slice)
                if key is not None:
                    keys[node.name] = key
                    break
    return keys


def _literal_subscript(node: ast.expr) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _r_baseline_key(name: str, key_by_test: dict[str, str]) -> str:
    base_name, param = _split_benchmark_name(name)
    if base_name == "test_pm_matrix_scale":
        if param is None:
            raise KeyError(f"Missing parameter for {name}.")
        return f"pm_matrix_{param}x500_seconds"
    if base_name == "test_dy_d_scalar_wrt1_100x2":
        if param is None:
            raise KeyError(f"Missing parameter for {name}.")
        return f"dy_d_scalar_{param}_100x2_seconds"
    if base_name == "test_nns_var_80x3_h3_tau2":
        if param is None:
            raise KeyError(f"Missing parameter for {name}.")
        return f"nns_var_80x3_h3_tau2_{param.lower().replace('.', '_')}_seconds"
    if base_name in key_by_test:
        return key_by_test[base_name]
    raise KeyError(f"No R baseline key mapping found for {name}.")


def _split_benchmark_name(name: str) -> tuple[str, str | None]:
    match = re.fullmatch(r"(?P<base>.+)\[(?P<param>.+)]", name)
    if match:
        return match.group("base"), match.group("param")
    return name, None


def _benchmark_names_from_tests() -> list[str]:
    tree = ast.parse(BENCHMARK_TESTS.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
            continue
        if node.name == "test_pm_matrix_scale":
            names.extend([f"{node.name}[{value}]" for value in (10, 50, 100)])
        elif node.name == "test_dy_d_scalar_wrt1_100x2":
            names.extend(
                [f"{node.name}[{value}]" for value in ("mean", "median", "last", "obs", "apd")]
            )
        elif node.name == "test_nns_var_80x3_h3_tau2":
            names.extend(
                [f"{node.name}[{value}]" for value in ("cor", "NNS.dep", "NNS.caus", "all")]
            )
        else:
            names.append(node.name)
    return names


def _fallback_label(name: str) -> str:
    base_name, param = _split_benchmark_name(name)
    label = base_name.removeprefix("test_").replace("_", " ")
    if param is not None:
        label = f"{label}, {param}"
    return f"`{label}`"


def _render(
    rows: list[BenchmarkRow],
    realistic_rows: list[RealisticSDRow],
    python_only_rows: list[PythonOnlyRow],
    r_version: str,
) -> str:
    lines = [
        "# Benchmarks",
        "",
        "Run with:",
        "",
        "```bash",
        "mkdir -p docs/benchmark_reports",
        "uv run pytest -n0 -m benchmark --benchmark-enable \\",
        "  --benchmark-json=docs/benchmark_reports/benchmark_latest.json tests/benchmarks/",
        "Rscript scripts/benchmark_realistic_sd_r.R \\",
        "  --repeats=3 --max-repeats=1 \\",
        "  --output=docs/benchmark_reports/realistic_sd_r_latest.csv",
        "uv run python scripts/update_benchmarks_doc.py "
        "docs/benchmark_reports/benchmark_latest.json \\",
        "  --realistic-sd-r-csv=docs/benchmark_reports/realistic_sd_r_latest.csv",
        "```",
        "",
        "## Results",
        "",
        f"R baselines use installed R NNS {r_version}.",
        "",
        "`Python speed vs R` is computed as `R baseline / Python mean`. Values above `1.00x` "
        "mean Python is faster; values below `1.00x` mean Python is slower.",
        "",
        "| Benchmark | Python mean | R baseline | Python speed vs R |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.label,
                    _format_ms(row.python_seconds),
                    _format_ms(row.r_seconds),
                    _format_speed_ratio(row.python_seconds, row.r_seconds),
                ]
            )
            + " |"
        )
    if realistic_rows:
        lines.extend(_render_realistic_sd(realistic_rows, python_only_rows))
    return "\n".join(lines) + "\n"


def _render_realistic_sd(
    realistic_rows: list[RealisticSDRow],
    python_only_rows: list[PythonOnlyRow],
) -> list[str]:
    total_return_columns = _fixture_return_column_count()
    constituent_columns = _fixture_constituent_column_count()
    sanity = _fixture_benchmark_column_sanity()
    sorted_rows = sorted(
        realistic_rows,
        key=lambda row: (row.rows, row.columns, row.degree, row.function_name),
    )
    lines = [
        "",
        "## Realistic Finance SD North Stars",
        "",
        "These benchmarks use the static daily-return fixture at",
        "`tests/fixtures/finance/sp500_daily_returns_2019_2023.csv`. That finance",
        "fixture is local-only and not tracked in git; the latest recorded run used 1257",
        f"daily return rows and {total_return_columns} clean return columns after dropping",
        "tickers with missing or non-finite returns. Constituent-universe benchmarks exclude",
        f"`SPY` and `GSPC`, leaving {constituent_columns} columns. Market-relative workflows",
        "prefer `GSPC` and fall back to `SPY`; tradable-proxy examples use `SPY`.",
        "",
        "Benchmark-column sanity metadata:",
        "",
        f"- SPY/GSPC correlation: {sanity.get('spy_gspc_correlation', float('nan')):.6f}",
        "- Mean absolute daily return difference: "
        f"{sanity.get('mean_abs_daily_return_difference', float('nan')):.6f}",
        "- Max absolute daily return difference: "
        f"{sanity.get('max_abs_daily_return_difference', float('nan')):.6f}",
        "",
        "Python timings come from `pytest-benchmark`. R timings come from",
        "`scripts/benchmark_realistic_sd_r.R` when `--realistic-sd-r-csv` is supplied to",
        "the updater. Rows marked `manual placeholder` use the last manually recorded R",
        "baseline so Python/R comparisons remain visible when R has not been rerun.",
        "",
        "Run only the realistic Python benchmarks with:",
        "",
        "```bash",
        "PYNNS_OFFLINE=1 uv run pytest -q -n0 -m benchmark --benchmark-enable \\",
        "  --benchmark-json=docs/benchmark_reports/realistic_sd_python_latest.json \\",
        "  tests/benchmarks/test_stochastic_dominance_realistic.py \\",
        "  tests/benchmarks/test_finance_sd_rolling.py \\",
        "  tests/benchmarks/test_finance_partial_moment_workflows.py",
        "```",
        "",
        "Run matching R baselines with:",
        "",
        "```bash",
        "Rscript scripts/benchmark_realistic_sd_r.R \\",
        "  --repeats=3 --max-repeats=1 \\",
        "  --output=docs/benchmark_reports/realistic_sd_r_latest.csv",
        "```",
        "",
        "`Python/R slowdown` is computed as `Python mean / R mean`. Values above `1.00x`",
        "mean Python is slower than R.",
        "",
        "| Realistic benchmark | Python mean | R mean | R source | Python/R slowdown |",
        "| --- | ---: | ---: | --- | ---: |",
    ]
    for row in sorted_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _realistic_label(row),
                    _format_seconds(row.python_seconds),
                    _format_seconds(row.r_seconds) if row.r_seconds is not None else "n/a",
                    _format_r_source(row.r_source),
                    _format_slowdown(row.python_seconds, row.r_seconds),
                ]
            )
            + " |"
        )
    if python_only_rows:
        lines.extend(
            [
                "",
                "Additional realistic finance workflow benchmarks:",
                "",
                "| Benchmark | Python mean | R mean | R source | Python/R slowdown | "
                "Summary metadata |",
                "| --- | ---: | ---: | --- | ---: | --- |",
            ]
        )
        for workflow_row in sorted(python_only_rows, key=lambda item: item.label):
            r_text = (
                _format_seconds(workflow_row.r_seconds)
                if workflow_row.r_seconds is not None
                else "n/a"
            )
            lines.append(
                f"| {workflow_row.label} | {_format_seconds(workflow_row.python_seconds)} | "
                f"{r_text} | "
                f"{_format_r_source(workflow_row.r_source)} | "
                f"{_format_slowdown(workflow_row.python_seconds, workflow_row.r_seconds)} | "
                f"{_format_extra_info(workflow_row.extra_info)} |"
            )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- Large degree-1 discrete SD uses an exact order-statistic dominance",
            "  matrix: one empirical sample FSD-dominates another iff every sorted",
            "  order statistic is at least as large, with at least one strict",
            "  improvement.",
            "- Guarded prefix-pair evaluation skips curve work for min/mean/identical",
            "  impossible pairs, and the standalone efficient-set path only checks",
            "  already-kept candidates for degree 2/3 and degree-1 continuous cases.",
            "- The implementation deliberately follows R's C++ SD algorithmic structure:",
            "  sorted columns, prefix sums, pair-threshold dominance checks, exact guards, and",
            "  no tolerance-based shortcuts.",
            "- Full-fixture NNS Python runs are feasible for research iteration, but R's C++ SD",
            "  core remains materially faster on the largest cluster cases.",
        ]
    )
    return lines


def _realistic_label(row: RealisticSDRow) -> str:
    return (
        f"`{row.function_name}`, degree={row.degree}, "
        f"N={row.columns}, T_obs={row.rows}"
    )


def _format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    if seconds < 1.0:
        return f"{seconds * 1000.0:.3f} ms"
    return f"{seconds:.3f} s"


def _format_slowdown(python_seconds: float, r_seconds: float | None) -> str:
    if r_seconds is None:
        return "n/a"
    return f"{python_seconds / r_seconds:.2f}x"


def _format_r_source(source: str) -> str:
    if source == "measured":
        return "measured"
    if source == "placeholder":
        return "manual placeholder"
    return "n/a"


def _realistic_sd_case_from_benchmark_name(
    name: str,
) -> tuple[str, int, int, int] | None:
    base_name, param = _split_benchmark_name(name)
    if base_name == "test_sd_efficient_set_sp500_daily_returns":
        if param is None:
            return None
        degree, column_count = _parse_degree_column_param(param)
        return ("sd_efficient_set", 252, column_count, degree)
    if base_name == "test_nns_sd_cluster_sp500_daily_returns":
        if param is None:
            return None
        degree, column_count = _parse_degree_column_param(param)
        return ("nns_sd_cluster", 252, column_count, degree)
    if base_name == "test_sd_efficient_set_sp500_daily_returns_252x250_degree2":
        return ("sd_efficient_set", 252, 250, 2)
    if base_name == "test_nns_sd_cluster_sp500_daily_returns_252x250_degree2":
        return ("nns_sd_cluster", 252, 250, 2)
    if base_name == "test_sd_efficient_set_sp500_daily_returns_1257x100_degree2":
        return ("sd_efficient_set", 1257, 100, 2)
    if base_name == "test_sd_efficient_set_sp500_daily_returns_full_fixture_degree2":
        if param is None:
            return None
        rows, columns = _parse_rows_columns_param(param)
        return ("sd_efficient_set", rows, columns, 2)
    if base_name == "test_nns_sd_cluster_sp500_daily_returns_full_fixture_degree2":
        if param is None:
            return None
        rows, columns = _parse_rows_columns_param(param)
        return ("nns_sd_cluster", rows, columns, 2)
    return None


def _parse_degree_column_param(param: str) -> tuple[int, int]:
    degree_text, column_text = param.split("-", maxsplit=1)
    return int(degree_text.removeprefix("degree")), int(column_text.removeprefix("n"))


def _parse_rows_columns_param(param: str) -> tuple[int, int]:
    rows_text, columns_text = param.split("x", maxsplit=1)
    columns = _fixture_constituent_column_count() if columns_text == "max" else int(columns_text)
    return int(rows_text), columns


def _fixture_constituent_column_count() -> int:
    fixture = ROOT / "tests" / "fixtures" / "finance" / "sp500_daily_returns_2019_2023.csv"
    header = fixture.read_text(encoding="utf-8").splitlines()[0].split(",")
    return len([symbol for symbol in header[1:] if symbol not in {"SPY", "GSPC"}])


def _fixture_return_column_count() -> int:
    fixture = ROOT / "tests" / "fixtures" / "finance" / "sp500_daily_returns_2019_2023.csv"
    header = fixture.read_text(encoding="utf-8").splitlines()[0].split(",")
    return len(header) - 1


def _fixture_benchmark_column_sanity() -> dict[str, float]:
    metadata_path = (
        ROOT
        / "tests"
        / "fixtures"
        / "finance"
        / "sp500_daily_returns_2019_2023_metadata.json"
    )
    payload = _read_json(metadata_path)
    sanity = payload.get("benchmark_column_sanity", {})
    if not isinstance(sanity, dict):
        return {}
    return {str(key): float(value) for key, value in sanity.items()}


def _realistic_python_only_label(name: str) -> str | None:
    base_name, param = _split_benchmark_name(name)
    labels = {
        "test_magnificent_seven_downside_stress_components": (
            "Magnificent Seven downside stress components with SPY"
        ),
        "test_lower_upper_constituent_dispersion_ratio": (
            "Lower/upper constituent dispersion ratio, N=100, T_obs=252"
        ),
        "test_rolling_sd_efficient_set_252d_monthly_degree2": (
            "Rolling SD efficient set, 252-day monthly, degree=2"
        ),
        "test_rolling_sd_cluster_252d_monthly_degree2": (
            "Rolling SD cluster, 252-day monthly, degree=2"
        ),
        "test_rolling_sd_cluster_756d_quarterly_degree2": (
            "Rolling SD cluster, 756-day quarterly, degree=2"
        ),
        "test_rolling_sd_efficient_set_252d_quarterly_degree1": (
            "Rolling SD efficient set, 252-day quarterly, degree=1"
        ),
        "test_rolling_sd_cluster_252d_quarterly_degree1": (
            "Rolling SD cluster, 252-day quarterly, degree=1"
        ),
        "test_rolling_sd_efficient_set_252d_quarterly_degree1_vs_degree2": (
            "Rolling SD efficient set, 252-day quarterly, degree 1 vs 2"
        ),
        "test_mag7_market_downside_stress_components": (
            "Magnificent Seven market-downside stress components"
        ),
        "test_partial_moment_covariance_matrix_workflow": (
            "Partial-moment covariance workflow"
        ),
        "test_market_relative_daily_dispersion_full_fixture": (
            "Market-relative daily dispersion, full fixture"
        ),
        "test_market_relative_rolling_dispersion_signal": (
            "Market-relative rolling dispersion signal"
        ),
    }
    label = labels.get(base_name)
    if label is None:
        return None
    if param is not None:
        label = f"{label}, {param}"
    return label


def _realistic_workflow_case_from_benchmark_name(
    name: str,
) -> tuple[str, int, int, int] | None:
    base_name, param = _split_benchmark_name(name)
    max_columns = _fixture_constituent_column_count()
    if base_name == "test_rolling_sd_efficient_set_252d_monthly_degree2":
        if param is None:
            return None
        columns = max_columns if param == "nmax" else int(param.removeprefix("n"))
        return ("rolling_sd_efficient_set_252d_monthly", 252, columns, 2)
    if base_name == "test_rolling_sd_cluster_252d_monthly_degree2":
        if param is None:
            return None
        columns = max_columns if param == "nmax" else int(param.removeprefix("n"))
        return ("rolling_sd_cluster_252d_monthly", 252, columns, 2)
    if base_name == "test_rolling_sd_cluster_756d_quarterly_degree2":
        return ("rolling_sd_cluster_756d_quarterly", 756, max_columns, 2)
    if base_name == "test_rolling_sd_efficient_set_252d_quarterly_degree1":
        return ("rolling_sd_efficient_set_252d_quarterly", 252, max_columns, 1)
    if base_name == "test_rolling_sd_cluster_252d_quarterly_degree1":
        return ("rolling_sd_cluster_252d_quarterly", 252, max_columns, 1)
    if base_name == "test_rolling_sd_efficient_set_252d_quarterly_degree1_vs_degree2":
        return ("rolling_sd_efficient_set_degree1_vs_degree2_252d_quarterly", 252, max_columns, 0)
    if base_name == "test_mag7_market_downside_stress_components":
        return ("mag7_market_downside_stress", 1257, 9, 1)
    if base_name == "test_partial_moment_covariance_matrix_workflow":
        if param is None:
            return None
        rows_text, degree_text, target_text = param.split("-", maxsplit=2)
        rows = int(rows_text.removesuffix("d"))
        degree = int(degree_text.removeprefix("degree"))
        return (f"pm_matrix_{degree_text}_{target_text}", rows, max_columns, degree)
    if base_name == "test_market_relative_daily_dispersion_full_fixture":
        return ("market_relative_daily_dispersion", 1257, max_columns, 2)
    if base_name == "test_market_relative_rolling_dispersion_signal":
        if param is None:
            return None
        window = int(param.removesuffix("d"))
        return (f"market_relative_rolling_dispersion_{window}d", 1257, max_columns, 2)
    return None


def _format_extra_info(extra_info: dict[str, Any]) -> str:
    labels = {
        "window_count": "windows",
        "average_efficient_set_size": "avg set",
        "average_cluster_count": "avg clusters",
        "average_turnover": "avg turnover",
        "average_degree1_set_size": "avg d1 set",
        "average_degree2_set_size": "avg d2 set",
        "downside_observation_count": "downside obs",
        "stress_regression_r2": "stress R2",
        "rows": "rows",
        "columns": "cols",
        "covariance_shape": "matrix N",
        "signal_length": "signal len",
        "finite_count": "finite",
        "next_day_market_correlation": "next-day corr",
        "spy_gspc_correlation": "SPY/GSPC corr",
        "mean_abs_daily_return_difference": "mean abs diff",
        "max_abs_daily_return_difference": "max abs diff",
    }
    parts = []
    for key, label in labels.items():
        if key not in extra_info:
            continue
        value = extra_info[key]
        if isinstance(value, int):
            formatted = str(value)
        elif isinstance(value, float):
            formatted = f"{value:.4g}"
        else:
            formatted = str(value)
        parts.append(f"{label}: {formatted}")
    return "; ".join(parts) if parts else "n/a"


def _format_ms(seconds: float) -> str:
    return f"{seconds * 1000.0:.3f} ms"


def _format_speed_ratio(python_seconds: float, r_seconds: float) -> str:
    return f"{r_seconds / python_seconds:.2f}x"


if __name__ == "__main__":
    main()
