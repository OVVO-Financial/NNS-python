from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pytest
from _finance_fixture import (
    MAX_COLUMN_COUNT,
    benchmark_column_sanity,
    load_constituent_returns,
    load_symbol_returns,
    market_symbol,
    tradable_proxy_symbol,
)
from numpy.typing import NDArray

from pynns import co_lpm, nns_reg, pm_matrix

_BENCHMARK_ROWS = 252
_FULL_HISTORY_ROWS = 1257
_MAGNIFICENT_SEVEN = ("AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA")


@pytest.mark.benchmark
def test_mag7_market_downside_stress_components(benchmark: Any) -> None:
    symbols = (*_MAGNIFICENT_SEVEN, market_symbol(), tradable_proxy_symbol())
    returns = load_symbol_returns(symbols)

    result = benchmark(_mag7_market_downside_stress_components, returns)

    _record_summary(benchmark, result)
    assert result["downside_observation_count"] >= 20
    assert result["co_lpm_degree1"].shape == (len(_MAGNIFICENT_SEVEN),)
    assert result["co_lpm_degree2"].shape == (len(_MAGNIFICENT_SEVEN),)
    assert result["pm_covariance"].shape == (len(_MAGNIFICENT_SEVEN), len(_MAGNIFICENT_SEVEN))
    assert result["stress_estimates"].shape == (2,)
    assert np.all(np.isfinite(result["stress_estimates"]))


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("row_count", "degree", "target_kind", "rounds"),
    [
        (_BENCHMARK_ROWS, 1, "mean", 3),
        (_FULL_HISTORY_ROWS, 1, "mean", 1),
        (_BENCHMARK_ROWS, 2, "zero", 3),
    ],
    ids=["252d-degree1-mean", "1257d-degree1-mean", "252d-degree2-zero"],
)
def test_partial_moment_covariance_matrix_workflow(
    benchmark: Any,
    row_count: int,
    degree: int,
    target_kind: str,
    rounds: int,
) -> None:
    returns = load_constituent_returns(row_count=row_count, column_count=MAX_COLUMN_COUNT)
    target = "mean" if target_kind == "mean" else np.zeros(returns.shape[1], dtype=np.float64)

    result = benchmark.pedantic(
        _partial_moment_covariance_workflow,
        args=(returns,),
        kwargs={"degree": degree, "target": target},
        rounds=rounds,
        iterations=1,
    )

    _record_summary(benchmark, result)
    assert result["rows"] == row_count
    assert result["columns"] == returns.shape[1]
    assert result["covariance_shape"] == returns.shape[1]
    if degree == 1 and target_kind == "mean":
        np.testing.assert_allclose(
            result["covariance_trace"],
            float(np.trace(np.cov(returns, rowvar=False))),
            rtol=1e-10,
            atol=1e-12,
        )


@pytest.mark.benchmark
def test_market_relative_daily_dispersion_full_fixture(benchmark: Any) -> None:
    constituents = load_constituent_returns(column_count=MAX_COLUMN_COUNT)
    market = load_symbol_returns((market_symbol(),)).reshape(-1)

    result = benchmark(_market_relative_daily_dispersion_ratio, constituents, market)

    _record_summary(benchmark, result)
    assert result["signal_length"] == constituents.shape[0]
    assert result["finite_count"] == constituents.shape[0]
    assert np.isfinite(result["next_day_market_correlation"])


@pytest.mark.benchmark
@pytest.mark.parametrize("window", [63, 252], ids=["63d", "252d"])
def test_market_relative_rolling_dispersion_signal(
    benchmark: Any,
    window: int,
) -> None:
    constituents = load_constituent_returns(column_count=MAX_COLUMN_COUNT)
    market = load_symbol_returns((market_symbol(),)).reshape(-1)

    result = benchmark(
        _market_relative_rolling_dispersion_signal,
        constituents,
        market,
        window,
    )

    _record_summary(benchmark, result)
    assert result["signal_length"] == constituents.shape[0] - window + 1
    assert result["finite_count"] == result["signal_length"]
    assert np.isfinite(result["next_day_market_correlation"])


def _mag7_market_downside_stress_components(
    returns: NDArray[np.float64],
) -> dict[str, NDArray[np.float64] | float | int]:
    assets = returns[:, : len(_MAGNIFICENT_SEVEN)]
    market = returns[:, len(_MAGNIFICENT_SEVEN)]
    downside_mask = market <= -0.01
    stress_assets = assets[downside_mask, :]
    stress_market = market[downside_mask]

    co_lpm_degree1 = np.asarray(
        [
            co_lpm(1.0, stress_assets[:, index], stress_market, 0.0, 0.0)
            for index in range(assets.shape[1])
        ],
        dtype=np.float64,
    )
    co_lpm_degree2 = np.asarray(
        [
            co_lpm(2.0, stress_assets[:, index], stress_market, 0.0, 0.0)
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
    stress_points = np.asarray(
        [[-0.05] * assets.shape[1], [-0.10] * assets.shape[1]],
        dtype=np.float64,
    )
    regression = nns_reg(
        stress_assets,
        stress_market,
        dim_red_method="cor",
        order=2,
        point_est=stress_points,
    )
    sanity = benchmark_column_sanity()
    return {
        "downside_observation_count": int(stress_assets.shape[0]),
        "co_lpm_degree1": co_lpm_degree1,
        "co_lpm_degree2": co_lpm_degree2,
        "pm_covariance": matrix["cov.matrix"],
        "stress_estimates": np.asarray(regression["Point.est"], dtype=np.float64),
        "stress_regression_r2": float(regression["R2"]),
        **sanity,
    }


def _partial_moment_covariance_workflow(
    returns: NDArray[np.float64],
    *,
    degree: int,
    target: Literal["mean"] | NDArray[np.float64],
) -> dict[str, float | int]:
    matrix = pm_matrix(float(degree), float(degree), target, returns, True)
    cov = matrix["cov.matrix"]
    return {
        "rows": returns.shape[0],
        "columns": returns.shape[1],
        "covariance_shape": cov.shape[0],
        "covariance_trace": float(np.trace(cov)),
        "clpm_trace": float(np.trace(matrix["clpm"])),
        "cupm_trace": float(np.trace(matrix["cupm"])),
    }


def _market_relative_daily_dispersion_ratio(
    constituents: NDArray[np.float64],
    market: NDArray[np.float64],
) -> dict[str, float | int]:
    ratio = _market_relative_ratio(constituents, market)
    return _dispersion_summary(ratio, market)


def _market_relative_rolling_dispersion_signal(
    constituents: NDArray[np.float64],
    market: NDArray[np.float64],
    window: int,
) -> dict[str, float | int]:
    ratio = _market_relative_ratio(constituents, market)
    kernel = np.full(window, 1.0 / window, dtype=np.float64)
    signal = np.convolve(ratio, kernel, mode="valid")
    return _dispersion_summary(signal, market[window - 1 :])


def _market_relative_ratio(
    constituents: NDArray[np.float64],
    market: NDArray[np.float64],
) -> NDArray[np.float64]:
    target = market[:, np.newaxis]
    lower = np.sqrt(np.mean(np.maximum(0.0, target - constituents) ** 2, axis=1))
    upper = np.sqrt(np.mean(np.maximum(0.0, constituents - target) ** 2, axis=1))
    ratio: NDArray[np.float64] = np.divide(
        upper,
        lower,
        out=np.zeros_like(upper),
        where=lower > 0.0,
    )
    return ratio


def _dispersion_summary(
    signal: NDArray[np.float64],
    market: NDArray[np.float64],
) -> dict[str, float | int]:
    finite = np.isfinite(signal)
    if signal.size > 1:
        correlation = float(np.corrcoef(signal[:-1], market[1:])[0, 1])
    else:
        correlation = 0.0
    return {
        "signal_length": signal.size,
        "finite_count": int(np.count_nonzero(finite)),
        "signal_min": float(np.min(signal)),
        "signal_max": float(np.max(signal)),
        "next_day_market_correlation": correlation,
        **benchmark_column_sanity(),
    }


def _record_summary(
    benchmark: Any,
    summary: dict[str, NDArray[np.float64] | float | int],
) -> None:
    for key, value in summary.items():
        if isinstance(value, np.ndarray):
            continue
        benchmark.extra_info[key] = float(value) if isinstance(value, float) else int(value)
