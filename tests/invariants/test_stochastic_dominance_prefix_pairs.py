from __future__ import annotations

import numpy as np
import pytest

from nns import nns_sd_cluster, sd_efficient_set
from nns import stochastic_dominance as sd


@pytest.mark.parametrize(
    ("degree", "discrete"),
    [
        (1, True),
        (1, False),
        (2, True),
        (3, True),
    ],
)
@pytest.mark.parametrize(
    "returns",
    [
        np.asarray(
            [
                [0.0, 0.0, -1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0, -1.0, 2.0],
                [2.0, 2.0, 1.0, 0.0, 0.0],
                [3.0, 3.0, 2.0, 2.0, 2.0],
                [4.0, 4.0, 3.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        np.asarray(
            [
                [0.0, 0.2, -1.0, 0.7],
                [0.0, 0.3, 2.0, -0.2],
                [1.0, 0.5, -0.5, 1.4],
                [1.0, 0.8, 2.5, -0.1],
                [2.0, 1.1, 0.0, 0.2],
                [2.0, 1.3, 1.2, 1.9],
            ],
            dtype=np.float64,
        ),
    ],
)
def test_prefix_pair_dominance_matrix_matches_global_grid(
    degree: int,
    discrete: bool,
    returns: np.ndarray,
) -> None:
    global_precomputed = sd._precompute_sd_table(returns, degree, discrete=discrete)
    prefix_precomputed = sd._prefix_sd_precompute(returns, degree, discrete=discrete)

    expected = sd._dominance_matrix_from_precomputed(global_precomputed, degree)
    actual = sd._dominance_matrix_from_prefix_pairs(
        prefix_precomputed,
        degree,
        discrete=discrete,
    )

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_prefix_pair_dominance_matrix_matches_random_global_grid(degree: int) -> None:
    rng = np.random.default_rng(20260517 + degree)
    returns = rng.normal(size=(11, 9))
    returns[:, 1] = returns[:, 0]
    returns[:, 2] = returns[:, 0] + 0.25
    returns[:, 3] = np.round(returns[:, 3], 1)

    global_precomputed = sd._precompute_sd_table(returns, degree, discrete=True)
    prefix_precomputed = sd._prefix_sd_precompute(returns, degree, discrete=True)

    expected = sd._dominance_matrix_from_precomputed(global_precomputed, degree)
    actual = sd._dominance_matrix_from_prefix_pairs(
        prefix_precomputed,
        degree,
        discrete=True,
    )

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "returns",
    [
        np.asarray(
            [
                [0.0, 0.0, -1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0, -1.0, 2.0],
                [2.0, 2.0, 1.0, 0.0, 0.0],
                [3.0, 3.0, 2.0, 2.0, 2.0],
                [4.0, 4.0, 3.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        np.asarray(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        np.asarray(
            [
                [-2.0, -1.0, -2.0, 1.0, -1.0],
                [0.0, 0.0, -2.0, -1.0, 1.0],
                [2.0, 1.0, 2.0, 0.0, -1.0],
                [2.0, 3.0, 2.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        ),
        np.asarray(
            [
                [-0.2, -0.1, 0.4, -0.4, 0.0, 0.0],
                [0.1, 0.2, -0.3, 0.5, 0.0, 0.1],
                [0.4, 0.5, 0.2, -0.2, 0.0, 0.2],
                [0.7, 0.8, -0.1, 0.6, 0.0, 0.3],
                [1.0, 1.1, 0.6, -0.6, 0.0, 0.4],
            ],
            dtype=np.float64,
        ),
    ],
)
def test_order_stat_dominance_matrix_matches_prefix_and_global(returns: np.ndarray) -> None:
    global_precomputed = sd._precompute_sd_table(returns, 1, discrete=True)
    prefix_precomputed = sd._prefix_sd_precompute(returns, 1, discrete=True)
    order_stat_precomputed = sd._order_stat_sd_precompute(returns)

    global_expected = sd._dominance_matrix_from_precomputed(global_precomputed, 1)
    prefix_expected = sd._dominance_matrix_from_prefix_pairs(
        prefix_precomputed,
        1,
        discrete=True,
    )
    actual = sd._dominance_matrix_from_order_stats(order_stat_precomputed)

    np.testing.assert_array_equal(actual, global_expected)
    np.testing.assert_array_equal(actual, prefix_expected)


def test_order_stat_dominance_matrix_matches_random_prefix_matrix() -> None:
    rng = np.random.default_rng(7519)
    returns = rng.normal(size=(17, 26))
    returns[:, 1] = returns[:, 0]
    returns[:, 2] = returns[:, 0] + 0.5
    returns[:, 3] = np.round(returns[:, 3], 1)
    returns[:, 4] = np.linspace(-1.0, 1.0, returns.shape[0])
    returns[:, 5] = returns[:, 4][::-1]

    prefix_precomputed = sd._prefix_sd_precompute(returns, 1, discrete=True)
    order_stat_precomputed = sd._order_stat_sd_precompute(returns)

    expected = sd._dominance_matrix_from_prefix_pairs(
        prefix_precomputed,
        1,
        discrete=True,
    )
    actual = sd._dominance_matrix_from_order_stats(order_stat_precomputed)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("degree", "discrete"),
    [
        (1, True),
        (1, False),
        (2, True),
        (3, True),
    ],
)
@pytest.mark.parametrize(
    "returns",
    [
        np.asarray(
            [
                [0.0, 0.0, 0.0, 1.0, -1.0, 2.0],
                [0.0, 0.0, 1.0, 0.0, 2.0, -1.0],
                [1.0, 1.0, 0.0, 2.0, -1.0, 2.0],
                [1.0, 1.0, 1.0, 0.0, 2.0, -1.0],
                [2.0, 2.0, 0.0, 1.0, -1.0, 2.0],
                [2.0, 2.0, 1.0, 2.0, 2.0, -1.0],
            ],
            dtype=np.float64,
        ),
        np.asarray(
            [
                [0.00, 0.10, -0.20, 0.15, 0.00, 0.25],
                [0.05, 0.15, 0.40, -0.30, 0.05, -0.10],
                [0.10, 0.20, -0.10, 0.35, 0.10, 0.05],
                [0.15, 0.25, 0.30, -0.15, 0.15, 0.30],
                [0.20, 0.30, 0.00, 0.25, 0.20, -0.05],
                [0.25, 0.35, 0.20, -0.05, 0.25, 0.20],
            ],
            dtype=np.float64,
        ),
    ],
)
def test_prefix_pair_one_direction_evaluator_matches_matrix(
    degree: int,
    discrete: bool,
    returns: np.ndarray,
) -> None:
    prefix_precomputed = sd._prefix_sd_precompute(returns, degree, discrete=discrete)
    dominance_matrix = sd._dominance_matrix_from_prefix_pairs(
        prefix_precomputed,
        degree,
        discrete=discrete,
    )

    for source_index in range(returns.shape[1]):
        for target_index in range(returns.shape[1]):
            actual = sd._dominates_from_prefix_pair(
                prefix_precomputed,
                source_index,
                target_index,
                degree,
                discrete=discrete,
            )
            assert actual == bool(dominance_matrix[source_index, target_index])


@pytest.mark.parametrize(
    ("degree", "discrete"),
    [
        (1, True),
        (1, False),
        (2, True),
        (3, True),
    ],
)
def test_prefix_pair_one_direction_evaluator_matches_random_matrix(
    degree: int,
    discrete: bool,
) -> None:
    rng = np.random.default_rng(9100 + degree + int(discrete))
    returns = rng.normal(size=(13, 11))
    returns[:, 1] = returns[:, 0]
    returns[:, 2] = returns[:, 0] + 0.4
    returns[:, 3] = np.round(returns[:, 3], 1)
    returns[:, 4] = np.linspace(-1.0, 1.0, returns.shape[0])
    returns[:, 5] = returns[:, 4][::-1]
    prefix_precomputed = sd._prefix_sd_precompute(returns, degree, discrete=discrete)
    dominance_matrix = sd._dominance_matrix_from_prefix_pairs(
        prefix_precomputed,
        degree,
        discrete=discrete,
    )

    for source_index in range(returns.shape[1]):
        for target_index in range(returns.shape[1]):
            actual = sd._dominates_from_prefix_pair(
                prefix_precomputed,
                source_index,
                target_index,
                degree,
                discrete=discrete,
            )
            assert actual == bool(dominance_matrix[source_index, target_index])


def test_sd_efficient_set_prefix_path_matches_lazy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    returns = _large_fixture()

    monkeypatch.setattr(sd, "_SD_PREFIX_PAIR_MATRIX_MIN_COLUMNS", returns.shape[1] + 1)
    lazy = sd_efficient_set(returns, 2)

    monkeypatch.setattr(sd, "_SD_PREFIX_PAIR_MATRIX_MIN_COLUMNS", 1)
    prefix = sd_efficient_set(returns, 2)

    assert prefix == lazy


def test_sd_efficient_set_order_stat_path_matches_lazy_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    returns = _large_fixture()

    monkeypatch.setattr(sd, "_SD_PREFIX_PAIR_MATRIX_MIN_COLUMNS", returns.shape[1] + 1)
    lazy = sd_efficient_set(returns, 1)

    monkeypatch.setattr(sd, "_SD_PREFIX_PAIR_MATRIX_MIN_COLUMNS", 1)
    order_stat = sd_efficient_set(returns, 1)

    assert order_stat == lazy


def test_order_stat_active_subset_matches_prefix_matrix() -> None:
    returns = _large_fixture()
    active = [0, 1, 2, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    prefix_precomputed = sd._prefix_sd_precompute(returns, 1, discrete=True)
    order_stat_precomputed = sd._order_stat_sd_precompute(returns)
    dominance_matrix = sd._dominance_matrix_from_order_stats(order_stat_precomputed)
    prefix_matrix = sd._dominance_matrix_from_prefix_pairs(
        prefix_precomputed,
        1,
        discrete=True,
    )

    expected = sd._sd_efficient_active_indices_from_matrix(
        returns,
        active,
        1,
        prefix_matrix,
    )
    actual = sd._sd_efficient_active_indices_from_matrix(
        returns,
        active,
        1,
        dominance_matrix,
    )

    assert actual == expected


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_kept_only_prefix_efficient_set_matches_matrix_path_on_active_subset(degree: int) -> None:
    returns = _large_fixture()
    active = [0, 1, 2, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    discrete = True
    prefix_precomputed = sd._prefix_sd_precompute(returns, degree, discrete=discrete)
    dominance_matrix = sd._dominance_matrix_from_prefix_pairs(
        prefix_precomputed,
        degree,
        discrete=discrete,
    )

    expected = sd._sd_efficient_active_indices_from_matrix(
        returns,
        active,
        degree,
        dominance_matrix,
    )
    actual = sd._sd_efficient_active_indices_from_prefix_kept(
        prefix_precomputed,
        active,
        degree,
        discrete=discrete,
    )

    assert actual == expected


def test_kept_only_prefix_efficient_set_matches_matrix_path_for_repeated_returns() -> None:
    returns = np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0, -1.0, 2.0],
            [0.0, 0.0, 1.0, 0.0, 2.0, -1.0],
            [1.0, 1.0, 0.0, 2.0, -1.0, 2.0],
            [1.0, 1.0, 1.0, 0.0, 2.0, -1.0],
            [2.0, 2.0, 0.0, 1.0, -1.0, 2.0],
            [2.0, 2.0, 1.0, 2.0, 2.0, -1.0],
        ],
        dtype=np.float64,
    )
    active = list(range(returns.shape[1]))
    prefix_precomputed = sd._prefix_sd_precompute(returns, 2, discrete=True)
    dominance_matrix = sd._dominance_matrix_from_prefix_pairs(
        prefix_precomputed,
        2,
        discrete=True,
    )

    expected = sd._sd_efficient_active_indices_from_matrix(
        returns,
        active,
        2,
        dominance_matrix,
    )
    actual = sd._sd_efficient_active_indices_from_prefix_kept(
        prefix_precomputed,
        active,
        2,
        discrete=True,
    )

    assert actual == expected


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_kept_only_prefix_efficient_set_matches_matrix_path_for_random_fixture(
    degree: int,
) -> None:
    rng = np.random.default_rng(1701 + degree)
    returns = rng.normal(size=(18, 24))
    returns[:, 1] = returns[:, 0]
    returns[:, 2] = np.round(returns[:, 2], 1)
    returns[:, 3] = returns[:, 0] + 0.3
    active = list(range(returns.shape[1]))
    prefix_precomputed = sd._prefix_sd_precompute(returns, degree, discrete=True)
    dominance_matrix = sd._dominance_matrix_from_prefix_pairs(
        prefix_precomputed,
        degree,
        discrete=True,
    )

    expected = sd._sd_efficient_active_indices_from_matrix(
        returns,
        active,
        degree,
        dominance_matrix,
    )
    actual = sd._sd_efficient_active_indices_from_prefix_kept(
        prefix_precomputed,
        active,
        degree,
        discrete=True,
    )

    assert actual == expected


@pytest.mark.parametrize("dendrogram", [False, True])
def test_nns_sd_cluster_prefix_path_matches_lazy_path(
    monkeypatch: pytest.MonkeyPatch,
    dendrogram: bool,
) -> None:
    returns = _large_fixture()

    monkeypatch.setattr(sd, "_SD_CLUSTER_DOMINANCE_MATRIX_MIN_COLUMNS", returns.shape[1] + 1)
    lazy = nns_sd_cluster(returns, degree=2, min_cluster=1, dendrogram=dendrogram)

    monkeypatch.setattr(sd, "_SD_CLUSTER_DOMINANCE_MATRIX_MIN_COLUMNS", 1)
    prefix = nns_sd_cluster(returns, degree=2, min_cluster=1, dendrogram=dendrogram)

    if not dendrogram:
        assert prefix == lazy
        return

    assert prefix["Clusters"] == lazy["Clusters"]
    prefix_dendrogram = prefix["Dendrogram"]
    lazy_dendrogram = lazy["Dendrogram"]
    assert isinstance(prefix_dendrogram, dict)
    assert isinstance(lazy_dendrogram, dict)
    for key in ("merge", "height", "order", "labels"):
        np.testing.assert_array_equal(prefix_dendrogram[key], lazy_dendrogram[key])
    assert prefix_dendrogram["method"] == lazy_dendrogram["method"]
    assert prefix_dendrogram["dist.method"] == lazy_dendrogram["dist.method"]


@pytest.mark.parametrize("dendrogram", [False, True])
def test_nns_sd_cluster_order_stat_path_matches_lazy_path(
    monkeypatch: pytest.MonkeyPatch,
    dendrogram: bool,
) -> None:
    returns = _large_fixture()

    monkeypatch.setattr(sd, "_SD_CLUSTER_DOMINANCE_MATRIX_MIN_COLUMNS", returns.shape[1] + 1)
    lazy = nns_sd_cluster(returns, degree=1, min_cluster=1, dendrogram=dendrogram)

    monkeypatch.setattr(sd, "_SD_CLUSTER_DOMINANCE_MATRIX_MIN_COLUMNS", 1)
    order_stat = nns_sd_cluster(returns, degree=1, min_cluster=1, dendrogram=dendrogram)

    if not dendrogram:
        assert order_stat == lazy
        return

    assert order_stat["Clusters"] == lazy["Clusters"]
    order_stat_dendrogram = order_stat["Dendrogram"]
    lazy_dendrogram = lazy["Dendrogram"]
    assert isinstance(order_stat_dendrogram, dict)
    assert isinstance(lazy_dendrogram, dict)
    for key in ("merge", "height", "order", "labels"):
        np.testing.assert_array_equal(order_stat_dendrogram[key], lazy_dendrogram[key])
    assert order_stat_dendrogram["method"] == lazy_dendrogram["method"]
    assert order_stat_dendrogram["dist.method"] == lazy_dendrogram["dist.method"]


def test_degree1_continuous_large_path_stays_on_prefix_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    returns = _large_fixture()

    monkeypatch.setattr(sd, "_SD_PREFIX_PAIR_MATRIX_MIN_COLUMNS", returns.shape[1] + 1)
    lazy = sd_efficient_set(returns, 1, type="continuous")

    monkeypatch.setattr(sd, "_SD_PREFIX_PAIR_MATRIX_MIN_COLUMNS", 1)
    prefix = sd_efficient_set(returns, 1, type="continuous")

    assert prefix == lazy


@pytest.mark.parametrize("degree", [2, 3])
def test_degree2_and_degree3_large_paths_stay_on_prefix_behavior(
    monkeypatch: pytest.MonkeyPatch,
    degree: int,
) -> None:
    returns = _large_fixture()

    monkeypatch.setattr(sd, "_SD_PREFIX_PAIR_MATRIX_MIN_COLUMNS", returns.shape[1] + 1)
    lazy = sd_efficient_set(returns, degree)

    monkeypatch.setattr(sd, "_SD_PREFIX_PAIR_MATRIX_MIN_COLUMNS", 1)
    prefix = sd_efficient_set(returns, degree)

    assert prefix == lazy


def _large_fixture() -> np.ndarray:
    rng = np.random.default_rng(17)
    returns = rng.normal(size=(16, 80))
    returns[:, 1] = returns[:, 0]
    returns[:, 2] = returns[:, 0] + 0.2
    returns[:, 3] = np.linspace(-1.0, 1.0, returns.shape[0])
    returns[:, 4] = returns[:, 3][::-1]
    returns[:, 5:10] = np.round(returns[:, 5:10], 1)
    return returns
