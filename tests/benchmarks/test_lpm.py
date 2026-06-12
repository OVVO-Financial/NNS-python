from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pynns import (
    dy_d,
    dy_dx,
    lpm,
    nns_anova,
    nns_arma,
    nns_arma_optim,
    nns_boost,
    nns_causation,
    nns_cdf,
    nns_copula,
    nns_dep,
    nns_diff,
    nns_distance,
    nns_distance_bulk,
    nns_m_reg,
    nns_mc,
    nns_meboot,
    nns_mode,
    nns_norm,
    nns_part,
    nns_reg,
    nns_sd_cluster,
    nns_seas,
    nns_ss,
    nns_stack,
    nns_var,
    pm_matrix,
    sd_efficient_set,
)


@pytest.mark.benchmark
def test_lpm_small(benchmark: Any, r_baseline: dict[str, object]) -> None:
    x = np.linspace(-3.0, 3.0, 1000)

    result = benchmark(lpm, 1, 0, x)

    assert result == pytest.approx(0.7507507507507507)
    assert isinstance(r_baseline["lpm_small_seconds"], float)


@pytest.mark.benchmark
@pytest.mark.parametrize("n_variables", [10, 50, 100])
def test_pm_matrix_scale(
    benchmark: Any,
    r_baseline: dict[str, object],
    n_variables: int,
) -> None:
    row = np.arange(1, 501, dtype=np.float64)[:, np.newaxis]
    col = np.arange(1, n_variables + 1, dtype=np.float64)[np.newaxis, :]
    variable = np.sin(row * col / 11.0) + np.cos((row + 1.0) / (col + 2.0))

    result = benchmark(pm_matrix, 1, 1, "mean", variable, True)

    assert set(result) == {"cupm", "dupm", "dlpm", "clpm", "cov.matrix"}
    assert isinstance(r_baseline[f"pm_matrix_{n_variables}x500_seconds"], float)


@pytest.mark.benchmark
def test_sd_efficient_set_degree_2_scale(
    benchmark: Any,
    r_baseline: dict[str, object],
) -> None:
    row = np.arange(1, 253, dtype=np.float64)[:, np.newaxis]
    col = np.arange(1, 51, dtype=np.float64)[np.newaxis, :]
    returns = np.sin(row * col / 17.0) + np.cos((row + 3.0) / (col + 5.0))

    result = benchmark(sd_efficient_set, returns, 2)

    assert all(0 <= index < 50 for index in result)
    assert isinstance(r_baseline["sd_efficient_set_50x252_degree2_seconds"], float)


@pytest.mark.benchmark
def test_nns_sd_cluster_252x50_degree2(
    benchmark: Any,
    r_baseline: dict[str, object],
) -> None:
    row = np.arange(252, dtype=np.float64)
    data = np.column_stack([np.sin(row / (index + 2)) + 0.01 * index for index in range(50)])

    result = benchmark(nns_sd_cluster, data, degree=2, min_cluster=1)

    assert isinstance(result["Clusters"], dict)
    assert isinstance(r_baseline["nns_sd_cluster_252x50_degree2_seconds"], float)


@pytest.mark.benchmark
def test_nns_sd_cluster_252x50_degree2_dendrogram(
    benchmark: Any,
    r_baseline: dict[str, object],
) -> None:
    row = np.arange(252, dtype=np.float64)
    data = np.column_stack([np.sin(row / (index + 2)) + 0.01 * index for index in range(50)])

    result = benchmark(nns_sd_cluster, data, degree=2, min_cluster=1, dendrogram=True)

    assert isinstance(result["Clusters"], dict)
    assert isinstance(result["Dendrogram"], dict)
    assert isinstance(r_baseline["nns_sd_cluster_252x50_degree2_dendrogram_seconds"], float)


@pytest.mark.benchmark
def test_nns_cdf_1000_degree0(benchmark: Any, r_baseline: dict[str, object]) -> None:
    x = np.linspace(-3.0, 3.0, 1000) + 0.1 * np.sin(np.arange(1000, dtype=np.float64))

    result = benchmark(nns_cdf, x, degree=0.0, type="CDF")

    assert set(result) == {"Function", "target.value"}
    assert isinstance(r_baseline["nns_cdf_1000_degree0_seconds"], float)


@pytest.mark.benchmark
def test_nns_cdf_1000_degree2(benchmark: Any, r_baseline: dict[str, object]) -> None:
    x = np.linspace(-3.0, 3.0, 1000) + 0.1 * np.sin(np.arange(1000, dtype=np.float64))

    result = benchmark(nns_cdf, x, degree=2.0, type="CDF")

    assert set(result) == {"Function", "target.value"}
    assert isinstance(r_baseline["nns_cdf_1000_degree2_seconds"], float)


@pytest.mark.benchmark
def test_nns_cdf_500x3_degree1(benchmark: Any, r_baseline: dict[str, object]) -> None:
    row = np.arange(1, 501, dtype=np.float64)[:, np.newaxis]
    col = np.arange(1, 4, dtype=np.float64)[np.newaxis, :]
    variable = np.sin(row * col / 11.0) + np.cos((row + 1.0) / (col + 2.0))

    result = benchmark(nns_cdf, variable, degree=1.0, type="CDF")

    assert set(result) == {"Function", "target.value"}
    assert isinstance(r_baseline["nns_cdf_500x3_degree1_seconds"], float)


@pytest.mark.benchmark
def test_nns_dep_1000(benchmark: Any, r_baseline: dict[str, object]) -> None:
    x = np.linspace(-3.0, 3.0, 1000)
    y = np.sin(x) + 0.05 * np.cos(7.0 * x)

    result = benchmark(nns_dep, x, y)

    assert set(result) == {"Correlation", "Dependence"}
    assert isinstance(r_baseline["nns_dep_1000_seconds"], float)


@pytest.mark.benchmark
def test_nns_dep_asym_1000(benchmark: Any, r_baseline: dict[str, object]) -> None:
    x = np.linspace(-3.0, 3.0, 1000)
    y = np.sin(x) + 0.05 * np.cos(7.0 * x)

    result = benchmark(nns_dep, x, y, True)

    assert set(result) == {"Correlation", "Dependence"}
    assert isinstance(r_baseline["nns_dep_asym_1000_seconds"], float)


@pytest.mark.benchmark
def test_nns_copula_1000(benchmark: Any, r_baseline: dict[str, object]) -> None:
    x = np.linspace(-3.0, 3.0, 1000)
    y = np.sin(x) + 0.05 * np.cos(7.0 * x)

    result = benchmark(nns_copula, x, y)

    assert 0.0 <= result <= 1.0
    assert isinstance(r_baseline["nns_copula_1000_seconds"], float)


@pytest.mark.benchmark
def test_nns_causation_1000(benchmark: Any, r_baseline: dict[str, object]) -> None:
    x = np.linspace(-3.0, 3.0, 1000)
    y = np.sin(x) + 0.05 * np.cos(7.0 * x)

    result = benchmark(nns_causation, x, y)

    assert "Causation.x.given.y" in result
    assert "Causation.y.given.x" in result
    assert isinstance(r_baseline["nns_causation_1000_seconds"], float)


@pytest.mark.benchmark
def test_nns_norm_1000x3(benchmark: Any, r_baseline: dict[str, object]) -> None:
    x = np.linspace(-2.0, 2.0, 1000)
    variable = np.column_stack((x + 3.0, x**2 + 1.0, np.sin(x) + 2.0))

    result = benchmark(nns_norm, variable)

    assert result.shape == variable.shape
    assert isinstance(r_baseline["nns_norm_1000x3_seconds"], float)


@pytest.mark.benchmark
def test_nns_distance_1000x3(benchmark: Any, r_baseline: dict[str, object]) -> None:
    row = np.arange(1, 1001, dtype=np.float64)
    features = np.column_stack((np.sin(row / 3.0) + 1.5, np.cos(row / 5.0) + 2.0, row / 1000.0))
    rpm = np.column_stack((features, np.sin(row / 7.0)))

    result = benchmark(nns_distance, rpm, np.array([1.25, 2.75, 0.4]), 20)

    assert np.isfinite(result)
    assert isinstance(r_baseline["nns_distance_1000x3_seconds"], float)


@pytest.mark.benchmark
def test_nns_distance_bulk_1000x3_100(
    benchmark: Any,
    r_baseline: dict[str, object],
) -> None:
    row = np.arange(1, 1001, dtype=np.float64)
    features = np.column_stack((np.sin(row / 3.0) + 1.5, np.cos(row / 5.0) + 2.0, row / 1000.0))
    rpm = np.column_stack((features, np.sin(row / 7.0)))
    test_row = np.arange(1, 101, dtype=np.float64)
    x_test = np.column_stack(
        (np.sin(test_row / 4.0) + 1.5, np.cos(test_row / 6.0) + 2.0, test_row / 100.0)
    )

    result = benchmark(nns_distance_bulk, rpm, x_test, 20)

    assert result.shape == (100,)
    assert isinstance(r_baseline["nns_distance_bulk_1000x3_100_seconds"], float)


@pytest.mark.benchmark
def test_nns_distance_class_500x3(benchmark: Any, r_baseline: dict[str, object]) -> None:
    row = np.arange(1, 501, dtype=np.float64)
    features = np.column_stack((np.sin(row / 3.0) + 1.5, np.cos(row / 5.0) + 2.0, row / 500.0))
    rpm = np.column_stack((features, (row % 3.0) + 1.0))

    result = benchmark(nns_distance, rpm, np.array([1.25, 2.75, 0.4]), 5, "class")

    assert result in {1.0, 2.0, 3.0}
    assert isinstance(r_baseline["nns_distance_class_500x3_seconds"], float)


@pytest.mark.benchmark
def test_nns_distance_bulk_class_500x3_50(
    benchmark: Any,
    r_baseline: dict[str, object],
) -> None:
    row = np.arange(1, 501, dtype=np.float64)
    features = np.column_stack((np.sin(row / 3.0) + 1.5, np.cos(row / 5.0) + 2.0, row / 500.0))
    rpm = np.column_stack((features, (row % 3.0) + 1.0))
    test_row = np.arange(1, 51, dtype=np.float64)
    x_test = np.column_stack(
        (np.sin(test_row / 4.0) + 1.5, np.cos(test_row / 6.0) + 2.0, test_row / 50.0)
    )

    result = benchmark(nns_distance_bulk, rpm, x_test, 5, "class")

    assert result.shape == (50,)
    assert isinstance(r_baseline["nns_distance_bulk_class_500x3_50_seconds"], float)


@pytest.mark.benchmark
def test_nns_diff_sin(benchmark: Any, r_baseline: dict[str, object]) -> None:
    result = benchmark(nns_diff, np.sin, 1.0)

    assert result["DERIVATIVE"] == pytest.approx(np.cos(1.0), abs=1e-6)
    assert isinstance(r_baseline["nns_diff_sin_seconds"], float)


@pytest.mark.benchmark
def test_dy_dx_numeric_eval_points(benchmark: Any, r_baseline: dict[str, object]) -> None:
    x = np.linspace(-2.0, 2.0, 100)
    y = x + np.sin(x)

    result = benchmark(dy_dx, x, y, np.array([-1.0, 0.0, 1.0]))

    assert isinstance(result, dict)
    assert result["eval.point"].shape == (3,)
    assert isinstance(r_baseline["dy_dx_numeric_100_seconds"], float)


@pytest.mark.benchmark
@pytest.mark.parametrize("eval_points", ["mean", "median", "last", "obs", "apd"])
def test_dy_d_scalar_wrt1_100x2(
    benchmark: Any,
    r_baseline: dict[str, object],
    eval_points: str,
) -> None:
    x1 = np.linspace(-1.5, 1.5, 100)
    x2 = np.cos(np.linspace(0.0, 2.0, 100))
    x = np.column_stack((x1, x2))
    y = x[:, 0] ** 2 + 0.5 * x[:, 1] + np.sin(x[:, 0] * x[:, 1])

    result = benchmark(dy_d, x, y, wrt=1, eval_points=eval_points)

    assert set(result) == {"First", "Second"}
    assert isinstance(r_baseline[f"dy_d_scalar_{eval_points}_100x2_seconds"], float)


@pytest.mark.benchmark
def test_nns_anova_100x2(benchmark: Any, r_baseline: dict[str, object]) -> None:
    idx = np.arange(100, dtype=np.float64)
    x = np.linspace(-2.0, 2.0, 100) + 0.1 * np.sin(idx / 3.0)
    y = x + 0.25 + 0.05 * np.cos(idx / 5.0)

    result = benchmark(nns_anova, x, y, confidence_interval=None)

    assert isinstance(result, dict)
    assert 0.0 <= result["Certainty"] <= 1.0
    assert isinstance(r_baseline["nns_anova_100x2_seconds"], float)


@pytest.mark.benchmark
def test_nns_part_500(benchmark: Any, r_baseline: dict[str, object]) -> None:
    x = np.linspace(-3.0, 3.0, 500)
    y = np.sin(x) + 0.05 * np.cos(7.0 * x)

    result = benchmark(nns_part, x, y)

    assert result["order"] >= 0
    assert isinstance(r_baseline["nns_part_500_seconds"], float)


@pytest.mark.benchmark
def test_nns_reg_500(benchmark: Any, r_baseline: dict[str, object]) -> None:
    x = np.linspace(-3.0, 3.0, 500)
    y = np.sin(x) + 0.05 * np.cos(7.0 * x)

    result = benchmark(nns_reg, x, y)

    assert "Fitted.xy" in result
    assert isinstance(r_baseline["nns_reg_500_seconds"], float)


@pytest.mark.benchmark
def test_nns_reg_200_confidence_interval(benchmark: Any, r_baseline: dict[str, object]) -> None:
    x = np.linspace(-3.0, 3.0, 200)
    y = np.sin(x) + 0.05 * np.cos(7.0 * x)
    point_est = np.linspace(-3.0, 3.0, 20)

    result = benchmark(nns_reg, x, y, point_est=point_est, confidence_interval=0.95)

    assert result["pred.int"] is not None
    assert isinstance(r_baseline["nns_reg_200_ci_seconds"], float)


@pytest.mark.benchmark
def test_nns_reg_200_smooth(benchmark: Any, r_baseline: dict[str, object]) -> None:
    x = np.linspace(-3.0, 3.0, 200)
    y = np.sin(x) + 0.05 * np.cos(7.0 * x)
    point_est = np.linspace(-3.0, 3.0, 20)

    result = benchmark(
        nns_reg,
        x,
        y,
        order=2,
        point_est=point_est,
        smooth=True,
        confidence_interval=0.95,
    )

    assert result["pred.int"] is not None
    assert isinstance(r_baseline["nns_reg_200_smooth_seconds"], float)


@pytest.mark.benchmark
def test_nns_reg_factor_predictor_200(
    benchmark: Any,
    r_baseline: dict[str, object],
) -> None:
    levels = ["a", "b", "c"]
    x = np.asarray([levels[index % len(levels)] for index in range(200)])
    y = np.sin(np.arange(200, dtype=np.float64) / 11.0) + (np.arange(200) % 3)
    point_est = np.asarray(["a", "c", "b", "a"])

    result = benchmark(
        nns_reg,
        x,
        y,
        factor_2_dummy=True,
        factor_levels=levels,
        point_est=point_est,
    )

    assert result["Point.est"].shape == (4,)
    assert isinstance(r_baseline["nns_reg_factor_predictor_200_seconds"], float)


@pytest.mark.benchmark
def test_nns_reg_factor_predictor_dimred_120(
    benchmark: Any,
    r_baseline: dict[str, object],
) -> None:
    levels = ["a", "b", "c"]
    factor = np.asarray([levels[index % len(levels)] for index in range(120)], dtype=object)
    numeric = np.linspace(-2.0, 2.0, 120, dtype=np.float64).astype(object)
    x = np.column_stack((factor, numeric))
    y = np.sin(np.arange(120, dtype=np.float64) / 9.0) + (np.arange(120) % 3)

    result = benchmark(
        nns_reg,
        x,
        y,
        factor_2_dummy=True,
        factor_levels=[levels, None],
        dim_red_method="cor",
    )

    assert result["equation"]["Coefficient"].shape == (5,)
    assert isinstance(r_baseline["nns_reg_factor_dimred_120x2_seconds"], float)


@pytest.mark.benchmark
def test_nns_reg_class_200(benchmark: Any, r_baseline: dict[str, object]) -> None:
    x = np.linspace(-3.0, 3.0, 200)
    y = (np.arange(200, dtype=np.float64) % 3.0) + 1.0
    point_est = np.linspace(-3.0, 3.0, 20)

    result = benchmark(nns_reg, x, y, point_est=point_est, type="class")

    assert result["Prediction.Accuracy"] is not None
    assert isinstance(r_baseline["nns_reg_class_200_seconds"], float)


@pytest.mark.benchmark
def test_nns_reg_class_200_confidence_interval(
    benchmark: Any,
    r_baseline: dict[str, object],
) -> None:
    x = np.linspace(-3.0, 3.0, 200)
    y = (np.arange(200, dtype=np.float64) % 3.0) + 1.0
    point_est = np.linspace(-3.0, 3.0, 20)

    result = benchmark(
        nns_reg,
        x,
        y,
        point_est=point_est,
        type="class",
        confidence_interval=0.95,
    )

    assert result["pred.int"] is not None
    assert isinstance(r_baseline["nns_reg_class_200_ci_seconds"], float)


@pytest.mark.benchmark
def test_nns_reg_dimred_200x3(benchmark: Any, r_baseline: dict[str, object]) -> None:
    x = np.linspace(-3.0, 3.0, 200)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = x + np.sin(x) + 0.25 * np.cos(x)

    result = benchmark(nns_reg, variable, y, dim_red_method="cor")

    assert result["x.star"]["x"].shape == (200,)
    assert isinstance(r_baseline["nns_reg_dimred_200x3_seconds"], float)


@pytest.mark.benchmark
def test_nns_m_reg_200x3(benchmark: Any, r_baseline: dict[str, object]) -> None:
    x = np.linspace(-3.0, 3.0, 200)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = x + np.sin(x) + 0.25 * np.cos(x)

    result = benchmark(nns_m_reg, variable, y)

    assert "Fitted.xy" in result
    assert isinstance(r_baseline["nns_m_reg_200x3_seconds"], float)


@pytest.mark.benchmark
def test_nns_m_reg_200x3_confidence_interval(
    benchmark: Any,
    r_baseline: dict[str, object],
) -> None:
    x = np.linspace(-3.0, 3.0, 200)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = x + np.sin(x) + 0.25 * np.cos(x)

    result = benchmark(nns_m_reg, variable, y, point_est=variable[:20], confidence_interval=0.95)

    assert result["pred.int"] is not None
    assert isinstance(r_baseline["nns_m_reg_200x3_ci_seconds"], float)


@pytest.mark.benchmark
def test_nns_m_reg_class_200x3(benchmark: Any, r_baseline: dict[str, object]) -> None:
    x = np.linspace(-3.0, 3.0, 200)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = (np.arange(200, dtype=np.float64) % 3.0) + 1.0

    result = benchmark(nns_m_reg, variable, y, point_est=variable[:20], type="class")

    assert "Fitted.xy" in result
    assert isinstance(r_baseline["nns_m_reg_class_200x3_seconds"], float)


@pytest.mark.benchmark
def test_nns_m_reg_class_200x3_confidence_interval(
    benchmark: Any,
    r_baseline: dict[str, object],
) -> None:
    x = np.linspace(-3.0, 3.0, 200)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = (np.arange(200, dtype=np.float64) % 3.0) + 1.0

    result = benchmark(
        nns_m_reg,
        variable,
        y,
        point_est=variable[:20],
        type="class",
        confidence_interval=0.95,
    )

    assert result["pred.int"] is not None
    assert isinstance(r_baseline["nns_m_reg_class_200x3_ci_seconds"], float)


@pytest.mark.benchmark
def test_nns_stack_100x3(benchmark: Any, r_baseline: dict[str, object]) -> None:
    x = np.linspace(-2.0, 2.0, 100)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = x + np.sin(x) + 0.25 * np.cos(x)

    result = benchmark(
        nns_stack,
        variable,
        y,
        variable[:20],
        cv_size=0.25,
        folds=2,
        method=(1, 2),
        dim_red_method="cor",
    )

    assert result["stack"].shape == (20,)
    assert isinstance(r_baseline["nns_stack_100x3_seconds"], float)


@pytest.mark.benchmark
def test_nns_stack_factor_predictor_60_method1(
    benchmark: Any,
    r_baseline: dict[str, object],
) -> None:
    levels = ["a", "b", "c"]
    x = np.asarray([levels[index % len(levels)] for index in range(60)])
    y = np.sin(np.arange(60, dtype=np.float64) / 7.0) + (np.arange(60) % 3)
    point_est = np.asarray(["a", "c", "b", "a", "b"])

    result = benchmark(
        nns_stack,
        x,
        y,
        point_est,
        factor_levels=levels,
        cv_size=0.25,
        folds=1,
        method=1,
    )

    assert result["stack"].shape == (5,)
    assert isinstance(r_baseline["nns_stack_factor_predictor_60_method1_seconds"], float)


@pytest.mark.benchmark
def test_nns_stack_mixed_factor_predictor_60_method2(
    benchmark: Any,
    r_baseline: dict[str, object],
) -> None:
    levels = ["a", "b", "c"]
    factor = np.asarray([levels[index % len(levels)] for index in range(60)], dtype=object)
    numeric = np.linspace(-1.0, 1.0, 60)
    variable = np.column_stack((factor, numeric.astype(object)))
    y = np.sin(np.arange(60, dtype=np.float64) / 7.0) + (np.arange(60) % 3)
    point_est = np.column_stack(
        (
            np.asarray(["a", "c", "b", "a", "b"], dtype=object),
            np.linspace(-0.75, 0.75, 5).astype(object),
        )
    )

    result = benchmark(
        nns_stack,
        variable,
        y,
        point_est,
        factor_levels=(levels, None),
        cv_size=0.25,
        folds=1,
        method=2,
    )

    assert result["stack"].shape == (5,)
    assert isinstance(r_baseline["nns_stack_mixed_factor_predictor_60_method2_seconds"], float)


@pytest.mark.benchmark
def test_nns_stack_mixed_factor_predictor_100x3_method12(
    benchmark: Any,
    r_baseline: dict[str, object],
) -> None:
    levels = ["a", "b", "c"]
    factor = np.asarray([levels[index % len(levels)] for index in range(100)], dtype=object)
    numeric = np.linspace(-1.0, 1.0, 100)
    variable = np.column_stack((factor, numeric.astype(object)))
    y = numeric + np.where(factor == "a", 0.0, np.where(factor == "b", 0.5, 1.0))
    point_est = np.column_stack(
        (
            np.asarray(
                [
                    "a",
                    "c",
                    "b",
                    "a",
                    "b",
                    "c",
                    "a",
                    "c",
                    "b",
                    "a",
                    "c",
                    "b",
                    "a",
                    "b",
                    "c",
                    "a",
                    "c",
                    "b",
                    "a",
                    "c",
                ],
                dtype=object,
            ),
            np.linspace(-0.8, 0.8, 20).astype(object),
        )
    )

    result = benchmark(
        nns_stack,
        variable,
        y,
        point_est,
        factor_levels=(levels, None),
        cv_size=0.25,
        folds=1,
        method=(1, 2),
        dim_red_method="cor",
    )

    assert result["stack"].shape == (20,)
    assert isinstance(r_baseline["nns_stack_mixed_factor_predictor_100x3_method12_seconds"], float)


@pytest.mark.benchmark
def test_nns_stack_100x3_pred_int(benchmark: Any, r_baseline: dict[str, object]) -> None:
    x = np.linspace(-2.0, 2.0, 100)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = x + np.sin(x) + 0.25 * np.cos(x)

    result = benchmark(
        nns_stack,
        variable,
        y,
        variable[:20],
        cv_size=0.25,
        folds=1,
        method=(1, 2),
        dim_red_method="cor",
        pred_int=0.95,
    )

    assert result["pred.int"] is not None
    assert isinstance(r_baseline["nns_stack_100x3_pred_int_seconds"], float)


@pytest.mark.benchmark
def test_nns_stack_100x3_ts_test(benchmark: Any, r_baseline: dict[str, object]) -> None:
    x = np.linspace(-2.0, 2.0, 100)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = x + np.sin(x) + 0.25 * np.cos(x)

    result = benchmark(
        nns_stack,
        variable,
        y,
        variable[:20],
        cv_size=0.25,
        folds=1,
        method=(1, 2),
        dim_red_method="cor",
        ts_test=20,
    )

    assert result["stack"].shape == (20,)
    assert isinstance(r_baseline["nns_stack_100x3_ts_test_seconds"], float)


@pytest.mark.benchmark
def test_nns_stack_class_100x3(benchmark: Any, r_baseline: dict[str, object]) -> None:
    x = np.linspace(-2.0, 2.0, 100)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = np.where(x < -0.5, 1.0, np.where(x > 0.75, 3.0, 2.0))

    result = benchmark(
        nns_stack,
        variable,
        y,
        variable[:20],
        cv_size=0.25,
        folds=1,
        method=(1, 2),
        dim_red_method="cor",
        type="class",
    )

    assert result["stack"].shape == (20,)
    assert np.all(np.isin(result["stack"], np.unique(y)))
    assert isinstance(r_baseline["nns_stack_class_100x3_seconds"], float)


@pytest.mark.benchmark
def test_nns_stack_class_100x3_pred_int(benchmark: Any, r_baseline: dict[str, object]) -> None:
    x = np.linspace(-2.0, 2.0, 100)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = np.where(x < -0.5, 1.0, np.where(x > 0.75, 3.0, 2.0))

    result = benchmark(
        nns_stack,
        variable,
        y,
        variable[:20],
        cv_size=0.25,
        folds=1,
        method=(1, 2),
        dim_red_method="cor",
        type="class",
        pred_int=0.95,
    )

    assert result["pred.int"] is not None
    assert isinstance(r_baseline["nns_stack_class_100x3_pred_int_seconds"], float)


@pytest.mark.benchmark
def test_nns_stack_class_balance_150x3(benchmark: Any, r_baseline: dict[str, object]) -> None:
    x = np.linspace(-2.0, 2.0, 150)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = np.where(x < -0.75, 1.0, np.where(x > 1.0, 3.0, 2.0))

    result = benchmark(
        nns_stack,
        variable,
        y,
        variable[:20],
        cv_size=0.25,
        folds=1,
        method=(1, 2),
        dim_red_method="cor",
        type="class",
        balance=True,
        random_seed=42,
    )

    assert result["stack"].shape == (20,)
    assert np.all(np.isin(result["stack"], np.unique(y)))
    assert isinstance(r_baseline["nns_stack_class_balance_150x3_seconds"], float)


@pytest.mark.benchmark
def test_nns_boost_50x3(benchmark: Any, r_baseline: dict[str, object]) -> None:
    x = np.linspace(-2.0, 2.0, 50)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = x + np.sin(x) + 0.25 * np.cos(x)

    result = benchmark(
        nns_boost,
        variable,
        y,
        variable[:10],
        learner_trials=10,
        cv_size=0.25,
        feature_importance=False,
    )

    assert result["results"].shape == (10,)
    assert isinstance(r_baseline["nns_boost_50x3_seconds"], float)


@pytest.mark.benchmark
def test_nns_boost_50x3_pred_int(benchmark: Any, r_baseline: dict[str, object]) -> None:
    x = np.linspace(-2.0, 2.0, 50)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = x + np.sin(x) + 0.25 * np.cos(x)

    result = benchmark(
        nns_boost,
        variable,
        y,
        variable[:10],
        learner_trials=10,
        cv_size=0.25,
        depth=2,
        pred_int=0.95,
        feature_importance=False,
    )

    assert result["results"].shape == (10,)
    assert result["pred.int"] is not None
    assert isinstance(r_baseline["nns_boost_50x3_pred_int_seconds"], float)


@pytest.mark.benchmark
def test_nns_boost_50x3_ts_test(benchmark: Any, r_baseline: dict[str, object]) -> None:
    x = np.linspace(-2.0, 2.0, 50)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = x + np.sin(x) + 0.25 * np.cos(x)

    result = benchmark(
        nns_boost,
        variable,
        y,
        variable[:10],
        learner_trials=10,
        cv_size=0.25,
        ts_test=8,
        feature_importance=False,
    )

    assert result["results"].shape == (10,)
    assert isinstance(r_baseline["nns_boost_50x3_ts_test_seconds"], float)


@pytest.mark.benchmark
def test_nns_boost_stochastic_64x11(benchmark: Any, r_baseline: dict[str, object]) -> None:
    x = np.linspace(-2.0, 2.0, 64)
    variable = np.column_stack([np.sin((idx + 1) * x) for idx in range(11)])
    y = x + np.sin(x)

    result = benchmark(
        nns_boost,
        variable,
        y,
        variable[:3],
        learner_trials=4,
        epochs=4,
        cv_size=0.25,
        random_seed=4,
        feature_importance=False,
    )

    assert result["results"].shape == (3,)
    assert isinstance(r_baseline["nns_boost_stochastic_64x11_seconds"], float)


@pytest.mark.benchmark
def test_nns_boost_stochastic_ts_test_64x11(
    benchmark: Any,
    r_baseline: dict[str, object],
) -> None:
    x = np.linspace(-2.0, 2.0, 64)
    variable = np.column_stack([np.sin((idx + 1) * x) for idx in range(11)])
    y = x + np.sin(x)

    result = benchmark(
        nns_boost,
        variable,
        y,
        variable[:3],
        learner_trials=4,
        epochs=4,
        cv_size=0.25,
        ts_test=5,
        random_seed=5,
        feature_importance=False,
    )

    assert result["results"].shape == (3,)
    assert isinstance(r_baseline["nns_boost_stochastic_ts_test_64x11_seconds"], float)


@pytest.mark.benchmark
def test_nns_boost_factor_predictor_50x2(
    benchmark: Any,
    r_baseline: dict[str, object],
) -> None:
    x = np.linspace(-2.0, 2.0, 50)
    labels = np.where(x < -0.5, "low", np.where(x > 0.75, "high", "mid"))
    variable = np.column_stack((labels, x))
    y = x + np.where(labels == "low", 1.0, np.where(labels == "mid", 2.0, 3.0)) * 0.25

    result = benchmark(
        nns_boost,
        variable,
        y,
        variable[:10],
        learner_trials=10,
        cv_size=0.25,
        factor_levels=(["low", "mid", "high"], None),
        feature_importance=False,
    )

    assert result["results"].shape == (10,)
    assert isinstance(r_baseline["nns_boost_factor_predictor_50x2_seconds"], float)


@pytest.mark.benchmark
def test_nns_boost_multi_factor_predictor_50x3(
    benchmark: Any,
    r_baseline: dict[str, object],
) -> None:
    x = np.linspace(-2.0, 2.0, 50)
    first = np.where(x < -0.5, "low", np.where(x > 0.75, "high", "mid"))
    second = np.where(np.sin(x) > 0.0, "up", "down")
    variable = np.column_stack((first, x.astype(object), second))
    y = (
        x
        + np.where(first == "low", 1.0, np.where(first == "mid", 2.0, 3.0)) * 0.25
        + np.where(second == "up", 0.1, -0.1)
    )

    result = benchmark(
        nns_boost,
        variable,
        y,
        variable[:10],
        learner_trials=10,
        cv_size=0.25,
        factor_levels=(["low", "mid", "high"], None, ["down", "up"]),
        feature_importance=False,
        random_seed=1,
    )

    assert result["results"].shape == (10,)
    assert isinstance(r_baseline["nns_boost_multi_factor_predictor_50x3_seconds"], float)


@pytest.mark.benchmark
def test_nns_boost_class_50x3(benchmark: Any, r_baseline: dict[str, object]) -> None:
    x = np.linspace(-2.0, 2.0, 50)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = np.where(x < -0.5, 1.0, np.where(x > 0.75, 3.0, 2.0))

    result = benchmark(
        nns_boost,
        variable,
        y,
        variable[:10],
        learner_trials=10,
        cv_size=0.25,
        depth=2,
        type="class",
        feature_importance=False,
    )

    assert result["results"].shape == (10,)
    assert np.all(np.isin(result["results"], np.unique(y)))
    assert isinstance(r_baseline["nns_boost_class_50x3_seconds"], float)


@pytest.mark.benchmark
def test_nns_boost_class_50x3_pred_int(benchmark: Any, r_baseline: dict[str, object]) -> None:
    x = np.linspace(-2.0, 2.0, 50)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = np.where(x < -0.5, 1.0, np.where(x > 0.75, 3.0, 2.0))

    result = benchmark(
        nns_boost,
        variable,
        y,
        variable[:10],
        learner_trials=10,
        cv_size=0.25,
        depth=2,
        type="class",
        pred_int=0.95,
        feature_importance=False,
    )

    assert result["pred.int"] is not None
    assert isinstance(r_baseline["nns_boost_class_50x3_pred_int_seconds"], float)


@pytest.mark.benchmark
def test_nns_boost_class_balance_80x3(benchmark: Any, r_baseline: dict[str, object]) -> None:
    x = np.linspace(-2.0, 2.0, 80)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = np.where(x < -0.75, 1.0, np.where(x > 1.0, 3.0, 2.0))

    result = benchmark(
        nns_boost,
        variable,
        y,
        variable[:10],
        learner_trials=10,
        cv_size=0.25,
        depth=2,
        type="class",
        balance=True,
        random_seed=42,
        feature_importance=False,
    )

    assert result["results"].shape == (10,)
    assert np.all(np.isin(result["results"], np.unique(y)))
    assert isinstance(r_baseline["nns_boost_class_balance_80x3_seconds"], float)


@pytest.mark.benchmark
def test_nns_mode_continuous_1000(benchmark: Any, r_baseline: dict[str, object]) -> None:
    x = np.concatenate((np.linspace(-3.0, 3.0, 500), np.linspace(1.0, 2.0, 500)))

    result = benchmark(nns_mode, x)

    assert np.isfinite(result)
    assert isinstance(r_baseline["nns_mode_continuous_1000_seconds"], float)


@pytest.mark.benchmark
def test_nns_seas_1000(benchmark: Any, r_baseline: dict[str, object]) -> None:
    t = np.arange(1, 1001, dtype=np.float64)
    variable = np.sin(2.0 * np.pi * t / 12.0) + 0.05 * np.cos(t / 3.0)

    result = benchmark(nns_seas, variable)

    assert result["best.period"] == int(result["periods"][0])
    assert isinstance(r_baseline["nns_seas_1000_seconds"], float)


@pytest.mark.benchmark
def test_nns_seas_5000(benchmark: Any, r_baseline: dict[str, object]) -> None:
    t = np.arange(1, 5001, dtype=np.float64)
    variable = np.sin(2.0 * np.pi * t / 12.0) + 0.05 * np.cos(t / 3.0)

    result = benchmark(nns_seas, variable)

    assert result["best.period"] == int(result["periods"][0])
    assert isinstance(r_baseline["nns_seas_5000_seconds"], float)


@pytest.mark.benchmark
def test_nns_arma_500_auto_nonlin(benchmark: Any, r_baseline: dict[str, object]) -> None:
    t = np.arange(1, 501, dtype=np.float64)
    variable = np.sin(2.0 * np.pi * t / 12.0) + 0.05 * np.cos(t / 3.0) + 2.0

    result = benchmark(nns_arma, variable, h=12, seasonal_factor=True, method="nonlin")

    assert result.shape == (12,)
    assert isinstance(r_baseline["nns_arma_500_auto_nonlin_seconds"], float)


@pytest.mark.benchmark
def test_nns_arma_500_explicit12_nonlin(
    benchmark: Any,
    r_baseline: dict[str, object],
) -> None:
    t = np.arange(1, 501, dtype=np.float64)
    variable = np.sin(2.0 * np.pi * t / 12.0) + 0.05 * np.cos(t / 3.0) + 2.0

    result = benchmark(nns_arma, variable, h=12, seasonal_factor=12, method="nonlin")

    assert result.shape == (12,)
    assert isinstance(r_baseline["nns_arma_500_explicit12_nonlin_seconds"], float)


@pytest.mark.benchmark
def test_nns_arma_200_explicit4_lin_predint(
    benchmark: Any,
    r_baseline: dict[str, object],
) -> None:
    t = np.arange(1, 201, dtype=np.float64)
    variable = np.sin(2.0 * np.pi * t / 12.0) + 0.05 * np.cos(t / 3.0) + 2.0

    result = benchmark(
        nns_arma,
        variable,
        5,
        None,
        [3, 4],
        method="lin",
        pred_int=0.95,
        random_seed=123,
    )

    assert isinstance(result, dict)
    assert result["Estimates"].shape == (5,)
    assert isinstance(r_baseline["nns_arma_200_explicit4_lin_predint_seconds"], float)


@pytest.mark.benchmark
def test_nns_arma_200_auto_nonlin_predint(
    benchmark: Any,
    r_baseline: dict[str, object],
) -> None:
    t = np.arange(1, 201, dtype=np.float64)
    variable = np.sin(2.0 * np.pi * t / 12.0) + 0.05 * np.cos(t / 3.0) + 2.0

    result = benchmark(
        nns_arma,
        variable,
        5,
        None,
        True,
        method="nonlin",
        pred_int=0.95,
        random_seed=123,
    )

    assert isinstance(result, dict)
    assert result["Estimates"].shape == (5,)
    assert isinstance(r_baseline["nns_arma_200_auto_nonlin_predint_seconds"], float)


@pytest.mark.benchmark
def test_nns_arma_optim_80_small(
    benchmark: Any,
    r_baseline: dict[str, object],
) -> None:
    t = np.arange(1, 81, dtype=np.float64)
    variable = np.sin(2.0 * np.pi * t / 12.0) + 0.05 * np.cos(t / 3.0) + 2.0

    result = benchmark(
        nns_arma_optim,
        variable,
        5,
        None,
        [3, 4, 5, 6, 7, 8, 9, 10],
        lin_only=True,
        print_trace=False,
    )

    assert result["results"].shape == (5,)
    assert isinstance(r_baseline["nns_arma_optim_80_small_seconds"], float)


@pytest.mark.benchmark
@pytest.mark.parametrize("dim_red_method", ["cor", "NNS.dep", "NNS.caus", "all"])
def test_nns_var_80x3_h3_tau2(
    benchmark: Any,
    r_baseline: dict[str, object],
    dim_red_method: str,
) -> None:
    t = np.arange(1, 81, dtype=np.float64)
    variables = np.column_stack(
        (
            np.sin(t / 5.0) + 0.01 * t,
            np.cos(t / 7.0) + 0.02 * t,
            np.sin(t / 11.0) + np.cos(t / 13.0),
        )
    )

    result = benchmark(nns_var, variables, h=3, tau=2, dim_red_method=dim_red_method)

    assert result["ensemble"].shape == (3, 3)
    key_method = dim_red_method.lower().replace(".", "_")
    assert isinstance(r_baseline[f"nns_var_80x3_h3_tau2_{key_method}_seconds"], float)


@pytest.mark.benchmark
def test_nns_meboot_500_reps100(benchmark: Any, r_baseline: dict[str, object]) -> None:
    t = np.arange(1, 501, dtype=np.float64)
    x = 0.01 * t + np.sin(t / 11.0) + 0.2 * np.cos(t / 5.0)

    result = benchmark(nns_meboot, x, 100, 0.0, random_seed=123)

    assert result["replicates"].shape == (500, 100)
    assert isinstance(r_baseline["nns_meboot_500_reps100_seconds"], float)


@pytest.mark.benchmark
def test_nns_meboot_1000_reps100(benchmark: Any, r_baseline: dict[str, object]) -> None:
    t = np.arange(1, 1001, dtype=np.float64)
    x = 0.01 * t + np.sin(t / 11.0) + 0.2 * np.cos(t / 5.0)

    result = benchmark(nns_meboot, x, 100, 0.0, random_seed=123)

    assert result["replicates"].shape == (1000, 100)
    assert isinstance(r_baseline["nns_meboot_1000_reps100_seconds"], float)


@pytest.mark.benchmark
def test_nns_mc_500_reps30_by02(benchmark: Any, r_baseline: dict[str, object]) -> None:
    t = np.arange(1, 501, dtype=np.float64)
    x = 0.01 * t + np.sin(t / 11.0) + 0.2 * np.cos(t / 5.0)

    result = benchmark(
        nns_mc,
        x,
        30,
        -1.0,
        1.0,
        0.2,
        1.0,
        random_seed=123,
    )

    assert result["ensemble"].shape == (500,)
    assert isinstance(r_baseline["nns_mc_500_reps30_by02_seconds"], float)


@pytest.mark.benchmark
def test_nns_mc_500_reps30_by01(benchmark: Any, r_baseline: dict[str, object]) -> None:
    t = np.arange(1, 501, dtype=np.float64)
    x = 0.01 * t + np.sin(t / 11.0) + 0.2 * np.cos(t / 5.0)

    result = benchmark(
        nns_mc,
        x,
        30,
        -1.0,
        1.0,
        0.1,
        1.0,
        random_seed=123,
    )

    assert result["ensemble"].shape == (500,)
    assert isinstance(r_baseline["nns_mc_500_reps30_by01_seconds"], float)


@pytest.mark.benchmark
def test_nns_ss_1000(benchmark: Any, r_baseline: dict[str, object]) -> None:
    x = np.linspace(-2.0, 3.0, 1000) + 0.2 * np.sin(np.arange(1000, dtype=np.float64))
    y = np.linspace(-1.5, 2.5, 1000) + 0.3 * np.cos(np.arange(1000, dtype=np.float64))

    result = benchmark(nns_ss, x, y)

    assert set(result) == {"p_gt", "p_tie", "p_star"}
    assert isinstance(r_baseline["nns_ss_1000_seconds"], float)


@pytest.mark.benchmark
def test_nns_ss_200_ci_reps100(benchmark: Any, r_baseline: dict[str, object]) -> None:
    x = np.linspace(-2.0, 3.0, 200) + 0.2 * np.sin(np.arange(200, dtype=np.float64))
    y = np.linspace(-1.5, 2.5, 200) + 0.3 * np.cos(np.arange(200, dtype=np.float64))

    result = benchmark(
        nns_ss,
        x,
        y,
        confidence_interval=True,
        reps=100,
        rho=1.0,
        random_seed=123,
    )

    assert result["boot_vals"].shape == (100,)
    assert isinstance(r_baseline["nns_ss_200_ci_reps100_seconds"], float)
