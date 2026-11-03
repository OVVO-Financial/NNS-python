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
def test_prefix_pair_evaluator_matches_pairwise_tests(
    degree: int,
    discrete: bool,
    returns: np.ndarray,
) -> None:
    prefix_precomputed = sd._prefix_sd_precompute(returns, degree, discrete=discrete)

    expected = _pairwise_dominance_matrix(returns, degree, discrete=discrete)
    actual = _prefix_dominance_matrix(prefix_precomputed, degree, discrete=discrete)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_prefix_pair_evaluator_matches_random_pairwise_tests(degree: int) -> None:
    rng = np.random.default_rng(20260517 + degree)
    returns = rng.normal(size=(11, 9))
    returns[:, 1] = returns[:, 0]
    returns[:, 2] = returns[:, 0] + 0.25
    returns[:, 3] = np.round(returns[:, 3], 1)

    prefix_precomputed = sd._prefix_sd_precompute(returns, degree, discrete=True)

    expected = _pairwise_dominance_matrix(returns, degree, discrete=True)
    actual = _prefix_dominance_matrix(prefix_precomputed, degree, discrete=True)

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
def test_order_stat_dominance_matrix_matches_prefix_and_pairwise(returns: np.ndarray) -> None:
    prefix_precomputed = sd._prefix_sd_precompute(returns, 1, discrete=True)
    order_stat_precomputed = sd._order_stat_sd_precompute(returns)

    pairwise_expected = _pairwise_dominance_matrix(returns, 1, discrete=True)
    prefix_expected = _prefix_dominance_matrix(prefix_precomputed, 1, discrete=True)
    actual = sd._dominance_matrix_from_order_stats(order_stat_precomputed)

    np.testing.assert_array_equal(actual, pairwise_expected)
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

    expected = _prefix_dominance_matrix(prefix_precomputed, 1, discrete=True)
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
    dominance_matrix = _pairwise_dominance_matrix(returns, degree, discrete=discrete)

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
    dominance_matrix = _pairwise_dominance_matrix(returns, degree, discrete=discrete)

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
    ("degree", "type_"),
    [(1, "discrete"), (1, "continuous"), (2, "discrete"), (3, "discrete")],
)
def test_sd_efficient_set_matches_pairwise_oracle(degree: int, type_: str) -> None:
    returns = _large_fixture()
    discrete = type_ != "continuous"
    active = list(range(returns.shape[1]))
    expected = sd._sd_efficient_active_indices_from_matrix(
        returns,
        active,
        degree,
        _pairwise_dominance_matrix(returns, degree, discrete=discrete),
    )

    assert sd_efficient_set(returns, degree, type=type_) == expected
    assert nns_sd_cluster(returns, degree=degree, type=type_)["Clusters"]["Cluster_1"] == [
        f"X_{index + 1}" for index in expected
    ]


def test_order_stat_active_subset_matches_prefix_matrix() -> None:
    returns = _large_fixture()
    active = [0, 1, 2, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    prefix_precomputed = sd._prefix_sd_precompute(returns, 1, discrete=True)
    order_stat_precomputed = sd._order_stat_sd_precompute(returns)
    dominance_matrix = sd._dominance_matrix_from_order_stats(order_stat_precomputed)
    prefix_matrix = _prefix_dominance_matrix(prefix_precomputed, 1, discrete=True)

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
def test_lazy_prefix_efficient_set_matches_matrix_path_on_active_subset(degree: int) -> None:
    returns = _large_fixture()
    active = [0, 1, 2, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    discrete = True
    prefix_precomputed = sd._prefix_sd_precompute(returns, degree, discrete=discrete)
    dominance_matrix = _pairwise_dominance_matrix(returns, degree, discrete=discrete)

    expected = sd._sd_efficient_active_indices_from_matrix(
        returns,
        active,
        degree,
        dominance_matrix,
    )
    actual = sd._sd_efficient_active_indices_from_prefix(
        prefix_precomputed,
        active,
        degree,
        discrete=discrete,
    )

    assert actual == expected


def test_lazy_prefix_efficient_set_matches_matrix_path_for_repeated_returns() -> None:
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
    dominance_matrix = _pairwise_dominance_matrix(returns, 2, discrete=True)

    expected = sd._sd_efficient_active_indices_from_matrix(
        returns,
        active,
        2,
        dominance_matrix,
    )
    actual = sd._sd_efficient_active_indices_from_prefix(
        prefix_precomputed,
        active,
        2,
        discrete=True,
    )

    assert actual == expected


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_lazy_prefix_efficient_set_matches_matrix_path_for_random_fixture(
    degree: int,
) -> None:
    rng = np.random.default_rng(1701 + degree)
    returns = rng.normal(size=(18, 24))
    returns[:, 1] = returns[:, 0]
    returns[:, 2] = np.round(returns[:, 2], 1)
    returns[:, 3] = returns[:, 0] + 0.3
    active = list(range(returns.shape[1]))
    prefix_precomputed = sd._prefix_sd_precompute(returns, degree, discrete=True)
    dominance_matrix = _pairwise_dominance_matrix(returns, degree, discrete=True)

    expected = sd._sd_efficient_active_indices_from_matrix(
        returns,
        active,
        degree,
        dominance_matrix,
    )
    actual = sd._sd_efficient_active_indices_from_prefix(
        prefix_precomputed,
        active,
        degree,
        discrete=True,
    )

    assert actual == expected


@pytest.mark.parametrize(
    ("degree", "type_"),
    [(1, "discrete"), (1, "continuous"), (2, "discrete"), (3, "discrete")],
)
@pytest.mark.parametrize("junk_columns", [0, 1, 71, 72, 120])
def test_sd_efficient_set_ignores_unrelated_dominated_columns(
    degree: int,
    type_: str,
    junk_columns: int,
) -> None:
    # A column's verdict against another must not depend on which other columns
    # are present: neither on extra knots they add nor on the column count.
    x = np.array([-0.85, -1.52, 0.68])
    y = np.array([-0.21, -0.33, -1.67])
    z = np.array([-3.0, 0.16, 0.2])
    junk = [np.array([-10.0, -9.0, -8.0]) - index for index in range(junk_columns)]
    base = sd_efficient_set(np.column_stack([x, y]), degree, type=type_)
    padded = np.column_stack([x, y, z, *junk])

    assert [index for index in sd_efficient_set(padded, degree, type=type_) if index < 2] == base
    clusters = nns_sd_cluster(padded, degree=degree, type=type_)["Clusters"]
    assert [name for name in clusters["Cluster_1"] if name in {"X_1", "X_2"}] == [
        f"X_{index + 1}" for index in base
    ]


def _pairwise_dominance_matrix(
    returns: np.ndarray,
    degree: int,
    *,
    discrete: bool,
) -> np.ndarray:
    """Oracle: the bivariate test on each pair's own merged grid, as fsd/ssd/tsd use."""
    columns = returns.shape[1]
    out = np.zeros((columns, columns), dtype=np.bool_)
    for source in range(columns):
        for target in range(columns):
            if source != target:
                out[source, target] = sd._dominates_uni(
                    returns[:, source],
                    returns[:, target],
                    degree,
                    discrete=discrete,
                )
    return out


def _prefix_dominance_matrix(
    precomputed: sd._SDPrefixPrecomputed,
    degree: int,
    *,
    discrete: bool,
) -> np.ndarray:
    columns = precomputed.sorted_values.shape[1]
    out = np.zeros((columns, columns), dtype=np.bool_)
    for source in range(columns):
        for target in range(columns):
            if source != target:
                out[source, target] = sd._dominates_from_prefix_pair(
                    precomputed,
                    source,
                    target,
                    degree,
                    discrete=discrete,
                )
    return out


def _large_fixture() -> np.ndarray:
    rng = np.random.default_rng(17)
    returns = rng.normal(size=(16, 80))
    returns[:, 1] = returns[:, 0]
    returns[:, 2] = returns[:, 0] + 0.2
    returns[:, 3] = np.linspace(-1.0, 1.0, returns.shape[0])
    returns[:, 4] = returns[:, 3][::-1]
    returns[:, 5:10] = np.round(returns[:, 5:10], 1)
    return returns


def test_identical_sorted_columns_matches_dense_comparison() -> None:
    rng = np.random.default_rng(42)
    returns = np.round(rng.normal(size=(9, 12)), 1)
    returns[:, 1] = returns[::-1, 0]
    returns[:, 2] = returns[:, 0]
    returns[:, 3] = 0.0
    returns[:, 4] = -0.0
    returns[:, 5] = returns[:, 6]
    returns[0, 7] = returns[0, 7] + 1e-15
    sorted_values = np.sort(returns, axis=0)

    expected = np.all(sorted_values.T[:, None, :] == sorted_values.T[None, :, :], axis=2)
    actual = sd._identical_sorted_columns(sorted_values)

    np.testing.assert_array_equal(actual, expected)
    assert actual[0, 1] and actual[0, 2] and actual[3, 4] and actual[5, 6]
