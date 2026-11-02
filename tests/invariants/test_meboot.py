from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

import nns.meboot as meboot_module
from nns import nns_meboot

pytestmark = pytest.mark.stochastic


def test_nns_meboot_replicate_and_ensemble_shapes() -> None:
    x = np.linspace(-2.0, 3.0, 25) + 0.1 * np.sin(np.arange(25, dtype=np.float64))

    result = nns_meboot(x, reps=7, rho=0.0, random_seed=11)

    assert result["replicates"].shape == (25, 7)
    assert result["ensemble"].shape == (25,)
    assert np.all(np.isfinite(result["replicates"]))
    assert np.all(np.isfinite(result["ensemble"]))


def test_nns_meboot_random_seed_is_reproducible() -> None:
    x = np.linspace(-2.0, 3.0, 25) + 0.1 * np.sin(np.arange(25, dtype=np.float64))

    first = nns_meboot(x, reps=5, rho=0.0, random_seed=22)
    second = nns_meboot(x, reps=5, rho=0.0, random_seed=22)
    third = nns_meboot(x, reps=5, rho=0.0, random_seed=23)

    np.testing.assert_array_equal(first["replicates"], second["replicates"])
    assert not np.array_equal(first["replicates"], third["replicates"])


def test_nns_meboot_xmin_xmax_clipping_is_respected() -> None:
    x = np.linspace(-2.0, 3.0, 30) + 0.15 * np.sin(np.arange(30, dtype=np.float64))

    result = nns_meboot(
        x,
        reps=8,
        rho=0.0,
        xmin=-1.0,
        xmax=2.0,
        random_seed=33,
    )

    assert np.min(result["replicates"]) >= -1.0
    assert np.max(result["replicates"]) <= 2.0


@pytest.mark.parametrize(
    ("force_clt", "expected_calls"),
    [(True, 1), (False, 2)],
)
def test_nns_meboot_bounds_disable_expand_sd_only_with_force_clt(
    force_clt: bool, expected_calls: int
) -> None:
    # R (NNS_meboot.R:158-159) turns expand.sd off for xmin/xmax only inside
    # if (isTRUE(force.clt)); otherwise the level-scale expansion still runs. The
    # residual-scale expansion always runs, so the level step is the second call.
    x = np.linspace(1.0, 4.0, 30) + 0.2 * np.sin(np.arange(30, dtype=np.float64))

    with mock.patch.object(
        meboot_module, "_meboot_expand_sd", wraps=meboot_module._meboot_expand_sd
    ) as spy:
        nns_meboot(
            x, reps=6, rho=0.0, xmin=0.0, force_clt=force_clt, expand_sd=True, random_seed=5
        )

    assert spy.call_count == expected_calls


def test_nns_meboot_vector_rho_returns_one_result_per_rho() -> None:
    x = np.linspace(-2.0, 3.0, 25) + 0.1 * np.sin(np.arange(25, dtype=np.float64))

    result = nns_meboot(x, reps=3, rho=[-1.0, 0.0, 1.0], random_seed=44)

    assert isinstance(result, list)
    assert len(result) == 3
    assert all(item["replicates"].shape == (25, 3) for item in result)


def test_nns_meboot_target_drift_scale_changes_ensemble_trend() -> None:
    x = np.linspace(1.0, 5.0, 30) + 0.1 * np.sin(np.arange(30, dtype=np.float64))

    flat = nns_meboot(x, reps=20, rho=0.0, drift=False, random_seed=55)
    scaled = nns_meboot(x, reps=20, rho=0.0, target_drift_scale=0.5, random_seed=55)

    flat_slope = np.polyfit(np.arange(1, 31, dtype=np.float64), flat["ensemble"], 1)[0]
    scaled_slope = np.polyfit(np.arange(1, 31, dtype=np.float64), scaled["ensemble"], 1)[0]

    assert abs(scaled_slope) > abs(flat_slope)


def test_nns_meboot_rho_targeting_moves_spearman_direction() -> None:
    x = np.linspace(-2.0, 4.0, 40) + 0.3 * np.sin(np.arange(40, dtype=np.float64))

    positive = nns_meboot(x, reps=20, rho=1.0, random_seed=66, force_clt=False, expand_sd=False)
    negative = nns_meboot(x, reps=20, rho=-1.0, random_seed=66, force_clt=False, expand_sd=False)

    pos_corr = _spearman(positive["ensemble"], x)
    neg_corr = _spearman(negative["ensemble"], x)

    assert pos_corr > neg_corr


def test_nns_meboot_rejects_empty() -> None:
    with pytest.raises(ValueError):
        nns_meboot(np.array([], dtype=np.float64), rho=0.0)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x_rank = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(np.float64)
    y_rank = np.argsort(np.argsort(y, kind="stable"), kind="stable").astype(np.float64)
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _slope(values: np.ndarray) -> float:
    t = np.arange(1, values.size + 1, dtype=np.float64)
    return float(np.polyfit(t, values, 1)[0])


@pytest.mark.parametrize("argument", ["target_drift", "target_drift_scale"])
def test_nns_meboot_vectorises_drift_arguments_like_r(argument: str) -> None:
    # R's documented example: NNS.meboot(x, rho = 0, target_drift = c(1, 7)) returns
    # one result per drift value (Vectorize over rho, target_drift, target_drift_scale).
    t = np.arange(1, 61, dtype=np.float64)
    x = 100.0 + 2.0 * t + 5.0 * np.sin(t)
    targets = [0.5, 6.0]

    results = nns_meboot(x, reps=20, rho=0.0, random_seed=4, **{argument: targets})
    first = nns_meboot(x, reps=20, rho=0.0, random_seed=4, **{argument: targets[0]})

    assert isinstance(results, list)
    assert len(results) == 2
    assert isinstance(first, dict)
    # Results are produced in order from one generator, so the first matches a scalar call.
    np.testing.assert_array_equal(results[0]["replicates"], first["replicates"])
    assert _slope(results[0]["ensemble"]) < _slope(results[1]["ensemble"])


def test_nns_meboot_recycles_vectorised_arguments_elementwise() -> None:
    x = np.linspace(1.0, 4.0, 30) + 0.2 * np.sin(np.arange(30, dtype=np.float64))

    paired = nns_meboot(x, reps=4, rho=[0.0, 0.5], target_drift=[0.1, 0.2], random_seed=1)
    assert isinstance(paired, list)
    assert len(paired) == 2

    with pytest.warns(RuntimeWarning, match="not a multiple"):
        recycled = nns_meboot(
            x, reps=4, rho=[0.0, 0.5, 1.0], target_drift=[0.1, 0.2], random_seed=1
        )
    assert isinstance(recycled, list)
    assert len(recycled) == 3
