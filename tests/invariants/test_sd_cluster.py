from __future__ import annotations

import numpy as np
import pytest

from nns import nns_sd_cluster


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


def test_nns_sd_cluster_variables_left_when_peeling_stops_at_once_form_the_final_cluster() -> None:
    # R NNS.SD.cluster returned no clusters here.
    data = np.arange(12, dtype=np.float64).reshape(4, 3)
    every = {"Cluster_1": ["X_1", "X_2", "X_3"]}

    assert nns_sd_cluster(data, min_cluster=3) == {"Clusters": every}
    assert nns_sd_cluster(data, min_cluster=4) == {"Clusters": every}
    pair = np.column_stack([np.arange(1.0, 6.0), np.arange(2.0, 7.0)])
    assert nns_sd_cluster(pair, min_cluster=2, names=["A", "B"]) == {
        "Clusters": {"Cluster_1": ["A", "B"]}
    }
    assert nns_sd_cluster(pair[:, :1], names=["A"]) == {"Clusters": {"Cluster_1": ["A"]}}
    with pytest.warns(UserWarning, match="Not enough variables"):
        single = nns_sd_cluster(pair[:, :1], dendrogram=True, names=["A"])
    assert single == {"Clusters": {"Cluster_1": ["A"]}, "Order": None}
    both = nns_sd_cluster(pair, min_cluster=2, dendrogram=True, names=["A", "B"])
    assert both["Clusters"] == {"Cluster_1": ["A", "B"]}
    assert both["Dendrogram"]["labels"].tolist() == ["A", "B"]


def test_nns_sd_cluster_dendrogram_keeps_hclust_leaf_order() -> None:
    # R NNS.SD.cluster overwrote hclust's order with the original column positions,
    # which is not a leaf order of the tree. R main values.
    rows = np.arange(1, 10)
    data = np.column_stack([((j * 7 + rows * 5) % 11) - j % 4 for j in range(1, 13)])
    names = [f"V{j}" for j in range(1, 13)]
    result = nns_sd_cluster(data.astype(np.float64), min_cluster=2, dendrogram=True, names=names)
    dendrogram = result["Dendrogram"]
    assert dendrogram["order"].tolist() == [1, 2, 3, 4, 9, 10, 11, 12, 5, 6, 7, 8]
    leaves = dendrogram["labels"][dendrogram["order"] - 1].tolist()
    assert leaves == "V4 V8 V12 V1 V7 V10 V3 V11 V5 V9 V2 V6".split()
    # The order lists each leaf once, left to right through the merges.
    assert sorted(dendrogram["order"].tolist()) == list(range(1, 13))


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
