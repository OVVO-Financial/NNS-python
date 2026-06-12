from __future__ import annotations

from itertools import pairwise
from typing import Any

import numpy as np
import pytest
from _finance_fixture import MAX_COLUMN_COUNT, load_constituent_returns, load_dates
from numpy.typing import NDArray

from nns import nns_sd_cluster, sd_efficient_set


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("column_count", "rounds"),
    [(100, 3), (MAX_COLUMN_COUNT, 1)],
    ids=["n100", "nmax"],
)
def test_rolling_sd_efficient_set_252d_monthly_degree2(
    benchmark: Any,
    column_count: int | str,
    rounds: int,
) -> None:
    returns = load_constituent_returns(column_count=column_count)
    dates = load_dates()

    result = benchmark.pedantic(
        _rolling_sd_efficient_set_summary,
        args=(returns, dates),
        kwargs={"lookback": 252, "frequency": "monthly", "degree": 2},
        rounds=rounds,
        iterations=1,
    )

    _record_summary(benchmark, result)
    assert result["window_count"] > 0
    assert result["average_efficient_set_size"] > 0.0
    assert 0.0 <= result["average_turnover"] <= 1.0


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("column_count", "rounds"),
    [(100, 3), (MAX_COLUMN_COUNT, 1)],
    ids=["n100", "nmax"],
)
def test_rolling_sd_cluster_252d_monthly_degree2(
    benchmark: Any,
    column_count: int | str,
    rounds: int,
) -> None:
    returns = load_constituent_returns(column_count=column_count)
    dates = load_dates()

    result = benchmark.pedantic(
        _rolling_sd_cluster_summary,
        args=(returns, dates),
        kwargs={"lookback": 252, "frequency": "monthly", "degree": 2},
        rounds=rounds,
        iterations=1,
    )

    _record_summary(benchmark, result)
    assert result["window_count"] > 0
    assert result["average_cluster_count"] > 0.0
    assert result["average_efficient_set_size"] > 0.0


@pytest.mark.benchmark
def test_rolling_sd_cluster_756d_quarterly_degree2(benchmark: Any) -> None:
    returns = load_constituent_returns(column_count=MAX_COLUMN_COUNT)
    dates = load_dates()

    result = benchmark.pedantic(
        _rolling_sd_cluster_summary,
        args=(returns, dates),
        kwargs={"lookback": 756, "frequency": "quarterly", "degree": 2},
        rounds=1,
        iterations=1,
    )

    _record_summary(benchmark, result)
    assert result["window_count"] > 0
    assert result["average_cluster_count"] > 0.0


@pytest.mark.benchmark
def test_rolling_sd_efficient_set_252d_quarterly_degree1(benchmark: Any) -> None:
    returns = load_constituent_returns(column_count=MAX_COLUMN_COUNT)
    dates = load_dates()

    result = benchmark.pedantic(
        _rolling_sd_efficient_set_summary,
        args=(returns, dates),
        kwargs={"lookback": 252, "frequency": "quarterly", "degree": 1},
        rounds=1,
        iterations=1,
    )

    _record_summary(benchmark, result)
    assert result["window_count"] > 0
    assert result["average_efficient_set_size"] > 0.0


@pytest.mark.benchmark
def test_rolling_sd_cluster_252d_quarterly_degree1(benchmark: Any) -> None:
    returns = load_constituent_returns(column_count=MAX_COLUMN_COUNT)
    dates = load_dates()

    result = benchmark.pedantic(
        _rolling_sd_cluster_summary,
        args=(returns, dates),
        kwargs={"lookback": 252, "frequency": "quarterly", "degree": 1},
        rounds=1,
        iterations=1,
    )

    _record_summary(benchmark, result)
    assert result["window_count"] > 0
    assert result["average_cluster_count"] > 0.0


@pytest.mark.benchmark
def test_rolling_sd_efficient_set_252d_quarterly_degree1_vs_degree2(
    benchmark: Any,
) -> None:
    returns = load_constituent_returns(column_count=MAX_COLUMN_COUNT)
    dates = load_dates()

    result = benchmark.pedantic(
        _rolling_sd_degree_comparison_summary,
        args=(returns, dates),
        kwargs={"lookback": 252, "frequency": "quarterly"},
        rounds=1,
        iterations=1,
    )

    _record_summary(benchmark, result)
    assert result["window_count"] > 0
    assert result["average_degree1_set_size"] > 0.0
    assert result["average_degree2_set_size"] > 0.0


def _rolling_sd_efficient_set_summary(
    returns: NDArray[np.float64],
    dates: NDArray[np.str_],
    *,
    lookback: int,
    frequency: str,
    degree: int,
) -> dict[str, float | int]:
    windows = _rolling_windows(dates, lookback, frequency)
    efficient_sets: list[set[int]] = []
    sizes: list[int] = []
    for start, stop in windows:
        indices = sd_efficient_set(returns[start:stop, :], degree)
        efficient_sets.append(set(indices))
        sizes.append(len(indices))
    return {
        "window_count": len(windows),
        "average_efficient_set_size": float(np.mean(sizes)),
        "average_turnover": _average_turnover(efficient_sets),
    }


def _rolling_sd_cluster_summary(
    returns: NDArray[np.float64],
    dates: NDArray[np.str_],
    *,
    lookback: int,
    frequency: str,
    degree: int,
) -> dict[str, float | int]:
    windows = _rolling_windows(dates, lookback, frequency)
    cluster_counts: list[int] = []
    first_cluster_sizes: list[int] = []
    for start, stop in windows:
        result = nns_sd_cluster(returns[start:stop, :], degree=degree, min_cluster=1)
        clusters = result["Clusters"]
        assert isinstance(clusters, dict)
        cluster_counts.append(len(clusters))
        first_cluster = clusters.get("Cluster_1", [])
        assert isinstance(first_cluster, list)
        first_cluster_sizes.append(len(first_cluster))
    return {
        "window_count": len(windows),
        "average_cluster_count": float(np.mean(cluster_counts)),
        "average_efficient_set_size": float(np.mean(first_cluster_sizes)),
    }


def _rolling_sd_degree_comparison_summary(
    returns: NDArray[np.float64],
    dates: NDArray[np.str_],
    *,
    lookback: int,
    frequency: str,
) -> dict[str, float | int]:
    windows = _rolling_windows(dates, lookback, frequency)
    degree1_sizes: list[int] = []
    degree2_sizes: list[int] = []
    for start, stop in windows:
        window = returns[start:stop, :]
        degree1_sizes.append(len(sd_efficient_set(window, 1)))
        degree2_sizes.append(len(sd_efficient_set(window, 2)))
    return {
        "window_count": len(windows),
        "average_degree1_set_size": float(np.mean(degree1_sizes)),
        "average_degree2_set_size": float(np.mean(degree2_sizes)),
    }


def _rolling_windows(
    dates: NDArray[np.str_],
    lookback: int,
    frequency: str,
) -> list[tuple[int, int]]:
    stops = _period_end_positions(dates, frequency)
    windows = [(stop - lookback, stop) for stop in stops if stop >= lookback]
    if not windows:
        raise AssertionError(f"No {frequency} windows with lookback={lookback}.")
    return windows


def _period_end_positions(dates: NDArray[np.str_], frequency: str) -> list[int]:
    positions: list[int] = []
    for index, value in enumerate(dates):
        current = str(value)
        next_value = str(dates[index + 1]) if index + 1 < len(dates) else None
        if next_value is None:
            positions.append(index + 1)
            continue
        if frequency == "monthly" and current[:7] != next_value[:7]:
            positions.append(index + 1)
        elif frequency == "quarterly" and _quarter_key(current) != _quarter_key(next_value):
            positions.append(index + 1)
    return positions


def _quarter_key(date_value: str) -> tuple[str, int]:
    month = int(date_value[5:7])
    return date_value[:4], (month - 1) // 3


def _average_turnover(efficient_sets: list[set[int]]) -> float:
    if len(efficient_sets) < 2:
        return 0.0
    turnovers = []
    for previous, current in pairwise(efficient_sets):
        union = previous | current
        turnovers.append(0.0 if not union else 1.0 - len(previous & current) / len(union))
    return float(np.mean(turnovers))


def _record_summary(benchmark: Any, summary: dict[str, float | int]) -> None:
    benchmark.extra_info.update({
        key: float(value) if isinstance(value, float) else int(value)
        for key, value in summary.items()
    })
