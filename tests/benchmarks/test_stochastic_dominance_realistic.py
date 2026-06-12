from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from nns import co_lpm, nns_sd_cluster, pm_matrix, sd_efficient_set

_FIXTURE = Path(__file__).parents[1] / "fixtures" / "finance" / "sp500_daily_returns_2019_2023.csv"
_BENCHMARK_ROWS = 252
_FULL_HISTORY_ROWS = 1257
_DISPERSION_COLUMNS = 100
_MAGNIFICENT_SEVEN = ("AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA")
_MAX_COLUMN_COUNT = "max"

if not _FIXTURE.exists():
    pytest.skip(
        "finance benchmark fixture is local-only; place "
        "sp500_daily_returns_2019_2023.csv under tests/fixtures/finance "
        "to run these benchmarks.",
        allow_module_level=True,
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("column_count", [50, 100], ids=["n50", "n100"])
@pytest.mark.parametrize("degree", [1, 2], ids=["degree1", "degree2"])
def test_sd_efficient_set_sp500_daily_returns(
    benchmark: Any,
    column_count: int,
    degree: int,
) -> None:
    returns = _load_daily_returns(row_count=_BENCHMARK_ROWS, column_count=column_count)

    result = benchmark(sd_efficient_set, returns, degree)

    assert all(0 <= index < returns.shape[1] for index in result)


@pytest.mark.benchmark
@pytest.mark.parametrize("column_count", [50, 100], ids=["n50", "n100"])
@pytest.mark.parametrize("degree", [1, 2], ids=["degree1", "degree2"])
def test_nns_sd_cluster_sp500_daily_returns(
    benchmark: Any,
    column_count: int,
    degree: int,
) -> None:
    returns = _load_daily_returns(row_count=_BENCHMARK_ROWS, column_count=column_count)

    result = benchmark(nns_sd_cluster, returns, degree=degree, min_cluster=1)

    clusters = result["Clusters"]
    assert isinstance(clusters, dict)
    members = [name for cluster in clusters.values() for name in cluster]
    assert len(members) == returns.shape[1]
    assert len(set(members)) == returns.shape[1]


@pytest.mark.benchmark
def test_sd_efficient_set_sp500_daily_returns_252x250_degree2(benchmark: Any) -> None:
    returns = _load_daily_returns(row_count=_BENCHMARK_ROWS, column_count=250)

    result = benchmark(sd_efficient_set, returns, 2)

    assert all(0 <= index < returns.shape[1] for index in result)


@pytest.mark.benchmark
def test_nns_sd_cluster_sp500_daily_returns_252x250_degree2(benchmark: Any) -> None:
    returns = _load_daily_returns(row_count=_BENCHMARK_ROWS, column_count=250)

    result = benchmark(nns_sd_cluster, returns, degree=2, min_cluster=1)

    clusters = result["Clusters"]
    assert isinstance(clusters, dict)
    members = [name for cluster in clusters.values() for name in cluster]
    assert len(members) == returns.shape[1]
    assert len(set(members)) == returns.shape[1]


@pytest.mark.benchmark
def test_sd_efficient_set_sp500_daily_returns_1257x100_degree2(benchmark: Any) -> None:
    returns = _load_daily_returns(row_count=_FULL_HISTORY_ROWS, column_count=100)

    result = benchmark(sd_efficient_set, returns, 2)

    assert all(0 <= index < returns.shape[1] for index in result)


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("row_count", "column_count", "rounds"),
    [
        (_BENCHMARK_ROWS, _MAX_COLUMN_COUNT, 3),
        (_FULL_HISTORY_ROWS, 250, 3),
        (_FULL_HISTORY_ROWS, _MAX_COLUMN_COUNT, 1),
    ],
    ids=["252xmax", "1257x250", "1257xmax"],
)
def test_sd_efficient_set_sp500_daily_returns_full_fixture_degree2(
    benchmark: Any,
    row_count: int,
    column_count: int | str,
    rounds: int,
) -> None:
    returns = _load_daily_returns(row_count=row_count, column_count=column_count)

    result = benchmark.pedantic(sd_efficient_set, args=(returns, 2), rounds=rounds, iterations=1)

    assert all(0 <= index < returns.shape[1] for index in result)


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("row_count", "column_count", "rounds"),
    [
        (_BENCHMARK_ROWS, _MAX_COLUMN_COUNT, 3),
        (_FULL_HISTORY_ROWS, 250, 3),
        (_FULL_HISTORY_ROWS, _MAX_COLUMN_COUNT, 1),
    ],
    ids=["252xmax", "1257x250", "1257xmax"],
)
def test_nns_sd_cluster_sp500_daily_returns_full_fixture_degree2(
    benchmark: Any,
    row_count: int,
    column_count: int | str,
    rounds: int,
) -> None:
    returns = _load_daily_returns(row_count=row_count, column_count=column_count)

    result = benchmark.pedantic(
        nns_sd_cluster,
        args=(returns,),
        kwargs={"degree": 2, "min_cluster": 1},
        rounds=rounds,
        iterations=1,
    )

    clusters = result["Clusters"]
    assert isinstance(clusters, dict)
    members = [name for cluster in clusters.values() for name in cluster]
    assert len(members) == returns.shape[1]
    assert len(set(members)) == returns.shape[1]


@pytest.mark.benchmark
def test_magnificent_seven_downside_stress_components(benchmark: Any) -> None:
    returns = _load_symbol_returns((*_MAGNIFICENT_SEVEN, "SPY"))

    result = benchmark(_magnificent_seven_downside_stress_components, returns)

    assert result["observation_count"] >= 20
    assert result["co_lpm_degree0"].shape == (len(_MAGNIFICENT_SEVEN),)
    assert result["co_lpm_degree1"].shape == (len(_MAGNIFICENT_SEVEN),)
    assert result["pm_covariance"].shape == (len(_MAGNIFICENT_SEVEN), len(_MAGNIFICENT_SEVEN))


@pytest.mark.benchmark
def test_lower_upper_constituent_dispersion_ratio(benchmark: Any) -> None:
    returns = _load_daily_returns(
        row_count=_BENCHMARK_ROWS,
        column_count=_DISPERSION_COLUMNS,
    )

    result = benchmark(_rolling_lower_upper_dispersion_ratio, returns)

    assert result.shape == (_BENCHMARK_ROWS - 63 + 1,)
    assert np.all(np.isfinite(result))


def _magnificent_seven_downside_stress_components(
    returns: NDArray[np.float64],
) -> dict[str, NDArray[np.float64] | int]:
    assets = returns[:, :-1]
    index_proxy = returns[:, -1]
    equal_weight_proxy = np.mean(assets, axis=1)
    downside_mask = (index_proxy < 0.0) & (equal_weight_proxy < 0.0)
    stress_assets = assets[downside_mask, :]
    stress_index = index_proxy[downside_mask]

    co_lpm_degree0 = np.asarray(
        [
            co_lpm(0.0, stress_assets[:, index], stress_index, 0.0, 0.0)
            for index in range(assets.shape[1])
        ],
        dtype=np.float64,
    )
    co_lpm_degree1 = np.asarray(
        [
            co_lpm(1.0, stress_assets[:, index], stress_index, 0.0, 0.0)
            for index in range(assets.shape[1])
        ],
        dtype=np.float64,
    )
    matrix = pm_matrix(
        1.0,
        1.0,
        np.zeros(assets.shape[1], dtype=np.float64),
        stress_assets,
        True,
        norm=True,
    )
    return {
        "observation_count": int(stress_assets.shape[0]),
        "co_lpm_degree0": co_lpm_degree0,
        "co_lpm_degree1": co_lpm_degree1,
        "pm_covariance": matrix["cov.matrix"],
    }


def _rolling_lower_upper_dispersion_ratio(
    returns: NDArray[np.float64],
    window: int = 63,
) -> NDArray[np.float64]:
    cross_section_target = np.mean(returns, axis=1, keepdims=True)
    lower = np.mean(np.maximum(0.0, cross_section_target - returns) ** 2, axis=1)
    upper = np.mean(np.maximum(0.0, returns - cross_section_target) ** 2, axis=1)
    ratio = np.divide(lower, upper, out=np.zeros_like(lower), where=upper > 0.0)
    kernel = np.full(window, 1.0 / window, dtype=np.float64)
    return np.convolve(ratio, kernel, mode="valid")


def _load_daily_returns(*, row_count: int, column_count: int | str) -> NDArray[np.float64]:
    available_columns = _fixture_column_count()
    resolved_column_count = (
        available_columns if column_count == _MAX_COLUMN_COUNT else int(column_count)
    )
    if available_columns < resolved_column_count:
        raise AssertionError(
            f"{_FIXTURE} has {available_columns} return columns, "
            f"expected at least {resolved_column_count}.",
        )

    header = _fixture_header()
    symbols = _constituent_symbols()
    usecols = [header.index(symbol) for symbol in symbols[:resolved_column_count]]
    data = np.loadtxt(
        _FIXTURE,
        delimiter=",",
        skiprows=1,
        max_rows=row_count,
        usecols=usecols,
        dtype=np.float64,
    )
    if data.shape != (row_count, resolved_column_count):
        expected_shape = (row_count, resolved_column_count)
        raise AssertionError(f"{_FIXTURE} has shape {data.shape}, expected {expected_shape}.")
    return data


def _load_symbol_returns(symbols: tuple[str, ...]) -> NDArray[np.float64]:
    header = _fixture_header()
    missing = [symbol for symbol in symbols if symbol not in header]
    if missing:
        raise AssertionError(f"{_FIXTURE} is missing required symbols: {missing}.")
    usecols = [header.index(symbol) for symbol in symbols]
    return np.loadtxt(
        _FIXTURE,
        delimiter=",",
        skiprows=1,
        usecols=usecols,
        dtype=np.float64,
    )


@lru_cache(maxsize=1)
def _fixture_header() -> tuple[str, ...]:
    with _FIXTURE.open(encoding="utf-8") as file:
        return tuple(file.readline().rstrip("\n").split(","))


def _fixture_column_count() -> int:
    return len(_constituent_symbols())


def _constituent_symbols() -> tuple[str, ...]:
    return tuple(symbol for symbol in _fixture_header()[1:] if symbol not in {"SPY", "GSPC"})
