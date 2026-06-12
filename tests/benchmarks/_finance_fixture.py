from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

FIXTURE = Path(__file__).parents[1] / "fixtures" / "finance" / "sp500_daily_returns_2019_2023.csv"
METADATA = FIXTURE.with_name(f"{FIXTURE.stem}_metadata.json")
MAX_COLUMN_COUNT = "max"

if not FIXTURE.exists() or not METADATA.exists():
    pytest.skip(
        "finance benchmark fixture is local-only; place "
        "sp500_daily_returns_2019_2023.csv and metadata under tests/fixtures/finance "
        "to run these benchmarks.",
        allow_module_level=True,
    )


def load_constituent_returns(
    *,
    row_count: int | None = None,
    column_count: int | str = MAX_COLUMN_COUNT,
) -> NDArray[np.float64]:
    symbols = constituent_symbols()
    resolved_count = len(symbols) if column_count == MAX_COLUMN_COUNT else int(column_count)
    if resolved_count > len(symbols):
        raise AssertionError(
            f"{FIXTURE} has {len(symbols)} constituent columns, "
            f"expected at least {resolved_count}.",
        )
    return load_symbol_returns(tuple(symbols[:resolved_count]), row_count=row_count)


def load_symbol_returns(
    symbols: tuple[str, ...],
    *,
    row_count: int | None = None,
) -> NDArray[np.float64]:
    header = fixture_header()
    missing = [symbol for symbol in symbols if symbol not in header]
    if missing:
        raise AssertionError(f"{FIXTURE} is missing required symbols: {missing}.")
    usecols = [header.index(symbol) for symbol in symbols]
    return np.loadtxt(
        FIXTURE,
        delimiter=",",
        skiprows=1,
        max_rows=row_count,
        usecols=usecols,
        dtype=np.float64,
    )


def load_dates() -> NDArray[np.str_]:
    return np.loadtxt(
        FIXTURE,
        delimiter=",",
        skiprows=1,
        usecols=0,
        dtype=np.str_,
    )


def constituent_symbols() -> tuple[str, ...]:
    raw_excluded = benchmark_columns()["excluded_from_constituents"]
    if not isinstance(raw_excluded, list):
        raise TypeError(f"Expected excluded_from_constituents list in {METADATA}.")
    excluded = {str(symbol) for symbol in raw_excluded}
    return tuple(symbol for symbol in fixture_header()[1:] if symbol not in excluded)


def market_symbol() -> str:
    columns = benchmark_columns()
    market = columns.get("market_index")
    if market and market in fixture_header():
        return str(market)
    proxy = columns.get("tradable_proxy", "SPY")
    if proxy not in fixture_header():
        raise AssertionError(f"{FIXTURE} is missing market proxy {proxy!r}.")
    return str(proxy)


def tradable_proxy_symbol() -> str:
    proxy = benchmark_columns().get("tradable_proxy", "SPY")
    if proxy not in fixture_header():
        raise AssertionError(f"{FIXTURE} is missing tradable proxy {proxy!r}.")
    return str(proxy)


def benchmark_column_sanity() -> dict[str, float]:
    return {
        str(key): float(value)
        for key, value in fixture_metadata().get("benchmark_column_sanity", {}).items()
    }


@lru_cache(maxsize=1)
def fixture_header() -> tuple[str, ...]:
    with FIXTURE.open(encoding="utf-8") as file:
        return tuple(file.readline().rstrip("\n").split(","))


@lru_cache(maxsize=1)
def fixture_metadata() -> dict[str, Any]:
    with METADATA.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected metadata object in {METADATA}.")
    return payload


def benchmark_columns() -> dict[str, object]:
    columns = fixture_metadata().get("benchmark_columns", {})
    if not isinstance(columns, dict):
        raise TypeError(f"Expected benchmark_columns object in {METADATA}.")
    if "excluded_from_constituents" not in columns:
        columns["excluded_from_constituents"] = [
            symbol for symbol in ("SPY", "GSPC") if symbol in fixture_header()
        ]
    return columns
