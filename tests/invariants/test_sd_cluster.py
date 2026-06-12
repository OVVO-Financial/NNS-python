from __future__ import annotations

import numpy as np
import pytest

from pynns import nns_sd_cluster


def test_nns_sd_cluster_covers_columns_once_and_is_deterministic() -> None:
    data = np.column_stack(
        [
            np.linspace(1.0, 5.0, 8),
            np.linspace(0.0, 4.0, 8),
            np.sin(np.arange(8, dtype=np.float64)),
            np.cos(np.arange(8, dtype=np.float64)),
        ]
    )

    first = nns_sd_cluster(data, degree=1, min_cluster=1, names=["A", "B", "C", "D"])
    second = nns_sd_cluster(data, degree=1, min_cluster=1, names=["A", "B", "C", "D"])

    assert first == second
    expected_keys = [f"Cluster_{index}" for index in range(1, len(first["Clusters"]) + 1)]
    assert list(first["Clusters"]) == expected_keys
    members = [name for cluster in first["Clusters"].values() for name in cluster]
    assert sorted(members) == ["A", "B", "C", "D"]
    assert len(members) == len(set(members))


def test_nns_sd_cluster_min_cluster_above_columns_is_empty() -> None:
    data = np.arange(12, dtype=np.float64).reshape(4, 3)

    assert nns_sd_cluster(data, min_cluster=3) == {"Clusters": {}}
    assert nns_sd_cluster(data, min_cluster=4) == {"Clusters": {}}


def test_nns_sd_cluster_validates_name_count() -> None:
    with pytest.raises(ValueError, match="names length"):
        nns_sd_cluster(np.ones((4, 3)), names=["A", "B"])


def test_nns_sd_cluster_rejects_1d_input_like_r_error_path() -> None:
    with pytest.raises(ValueError, match="2D"):
        nns_sd_cluster(np.arange(5, dtype=np.float64))


def test_nns_sd_cluster_duplicate_columns_can_share_cluster() -> None:
    data = np.column_stack(
        [
            np.arange(1, 6, dtype=np.float64),
            np.arange(0, 5, dtype=np.float64),
            np.arange(1, 6, dtype=np.float64),
        ]
    )

    result = nns_sd_cluster(data, degree=1, min_cluster=1, names=["A", "B", "C"])

    assert result["Clusters"]["Cluster_1"] == ["A", "C"]


def test_nns_sd_cluster_dendrogram_hclust_fields_are_consistent() -> None:
    data = np.column_stack(
        [
            np.arange(1, 6, dtype=np.float64),
            np.arange(0, 5, dtype=np.float64),
            np.sin(np.arange(1, 6, dtype=np.float64)),
            np.arange(1, 6, dtype=np.float64),
        ]
    )

    result = nns_sd_cluster(data, degree=1, min_cluster=1, dendrogram=True)

    assert set(result) == {"Clusters", "Dendrogram"}
    dendrogram = result["Dendrogram"]
    assert isinstance(dendrogram, dict)
    labels = dendrogram["labels"]
    assert len(labels) == data.shape[1]
    assert dendrogram["merge"].shape == (data.shape[1] - 1, 2)
    assert dendrogram["height"].shape == (data.shape[1] - 1,)
    assert dendrogram["order"].shape == (data.shape[1],)
    assert dendrogram["method"] == "complete"
    assert dendrogram["dist.method"] is None
