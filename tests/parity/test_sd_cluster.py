from __future__ import annotations

import numpy as np
import pytest
from _r import RValue, nns, nns_sd_cluster_dendrogram

from nns import nns_sd_cluster


@pytest.mark.parity
@pytest.mark.parametrize(
    ("degree", "min_cluster", "expected"),
    [
        (
            1,
            1,
            {"Cluster_1": ["A", "D"], "Cluster_2": ["B"], "Cluster_3": ["C"]},
        ),
        (2, 2, {"Cluster_1": ["A", "D"], "Cluster_2": ["B", "C"]}),
    ],
)
def test_nns_sd_cluster_known_matrix_matches_installed_r_probe(
    degree: int,
    min_cluster: int,
    expected: dict[str, list[str]],
) -> None:
    data = _known_matrix()

    actual = nns_sd_cluster(
        data,
        degree=degree,
        min_cluster=min_cluster,
        names=["A", "B", "C", "D"],
    )

    assert actual == {"Clusters": expected}


@pytest.mark.parity
def test_nns_sd_cluster_unnamed_matrix_matches_r() -> None:
    data = _known_matrix()
    expected = _normalize_clusters(nns("NNS.SD.cluster", data.tolist(), 1, "discrete", 1, False))

    actual = nns_sd_cluster(data, degree=1, min_cluster=1)

    assert actual == expected


@pytest.mark.parity
def test_nns_sd_cluster_constant_columns_match_installed_r_probe() -> None:
    data = np.column_stack([np.ones(5), np.ones(5), np.arange(1, 6, dtype=np.float64)])

    actual = nns_sd_cluster(data, degree=1, min_cluster=1, names=["A", "B", "C"])

    assert actual == {"Clusters": {"Cluster_1": ["C"], "Cluster_2": ["A", "B"]}}


@pytest.mark.parity
@pytest.mark.parametrize("min_cluster", [4, 5])
def test_nns_sd_cluster_min_cluster_at_or_above_columns_matches_r(min_cluster: int) -> None:
    data = _known_matrix()
    expected = _normalize_clusters(
        nns("NNS.SD.cluster", data.tolist(), 1, "discrete", min_cluster, False)
    )

    actual = nns_sd_cluster(data, degree=1, min_cluster=min_cluster)

    # Nothing is peeled, and the variables left form the final cluster.
    assert actual == expected == {"Clusters": {"Cluster_1": ["X_1", "X_2", "X_3", "X_4"]}}


@pytest.mark.parity
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_nns_sd_cluster_random_matrix_matches_r(degree: int) -> None:
    row = np.arange(1, 9, dtype=np.float64)
    data = np.column_stack(
        [
            0.2 * row,
            np.sin(row),
            np.cos(row) + 0.1 * row,
            np.where(row % 2 == 0, 1.0, -1.0),
            row[::-1] / 3.0,
        ]
    )
    expected = _normalize_clusters(
        nns("NNS.SD.cluster", data.tolist(), degree, "discrete", 1, False)
    )

    actual = nns_sd_cluster(data, degree=degree, min_cluster=1)

    assert actual == expected


@pytest.mark.parity
@pytest.mark.parametrize("min_cluster", [0, 1, 2])
def test_nns_sd_cluster_shifted_copies_match_r(min_cluster: int) -> None:
    # Each peel takes one column; with min_cluster=0 the last peel empties the
    # pool and R adds no empty trailing cluster.
    base = np.sin(np.arange(1, 21, dtype=np.float64))
    data = np.column_stack([base + shift for shift in (3.0, 2.0, 1.0, 0.0, -1.0)])
    expected = _normalize_clusters(
        nns("NNS.SD.cluster", data.tolist(), 1, "discrete", min_cluster, False)
    )

    actual = nns_sd_cluster(data, degree=1, min_cluster=min_cluster)

    assert actual == expected
    assert all(actual["Clusters"].values())


@pytest.mark.parity
def test_nns_sd_cluster_continuous_type_matches_r() -> None:
    data = _known_matrix()
    expected = _normalize_clusters(nns("NNS.SD.cluster", data.tolist(), 1, "continuous", 1, False))

    actual = nns_sd_cluster(data, degree=1, type="continuous", min_cluster=1)

    assert actual == expected


@pytest.mark.parity
def test_nns_sd_cluster_invalid_degree_raises() -> None:
    with pytest.raises(ValueError, match="degree must be 1, 2, or 3"):
        nns_sd_cluster(_known_matrix(), degree=4)


@pytest.mark.parity
def test_nns_sd_cluster_missing_values_raise() -> None:
    data = _known_matrix()
    data[0, 0] = np.nan

    with pytest.raises(ValueError, match="You have some missing values, please address"):
        nns_sd_cluster(data)


@pytest.mark.parity
def test_nns_sd_cluster_dendrogram_matches_r_hclust_shape() -> None:
    data = _known_matrix()
    expected = _normalize_dendrogram(nns_sd_cluster_dendrogram(data.tolist(), 1, "discrete", 1))

    actual = nns_sd_cluster(data, degree=1, min_cluster=1, dendrogram=True)

    assert actual["Clusters"] == expected["Clusters"]
    assert isinstance(actual["Dendrogram"], dict)
    assert isinstance(expected["Dendrogram"], dict)
    for key in ("merge", "height", "order", "labels"):
        np.testing.assert_array_equal(actual["Dendrogram"][key], expected["Dendrogram"][key])
    assert actual["Dendrogram"]["method"] == expected["Dendrogram"]["method"] == "complete"
    assert actual["Dendrogram"]["dist.method"] is None


@pytest.mark.parity
def test_nns_sd_cluster_dendrogram_too_few_variables_matches_r() -> None:
    data = _known_matrix()[:, :1]
    expected = _normalize_dendrogram(nns_sd_cluster_dendrogram(data.tolist(), 1, "discrete", 1))

    with pytest.warns(UserWarning, match="Not enough variables"):
        actual = nns_sd_cluster(data, degree=1, min_cluster=1, dendrogram=True)

    assert actual == expected == {"Clusters": {"Cluster_1": ["X_1"]}, "Order": None}


def _known_matrix() -> np.ndarray:
    return np.asarray(
        [
            [2.0, 1.0, 0.0, 2.0],
            [3.0, 2.0, 4.0, 3.0],
            [4.0, 3.0, 0.0, 4.0],
            [5.0, 4.0, 4.0, 5.0],
            [6.0, 5.0, 0.0, 6.0],
        ],
        dtype=np.float64,
    )


def _normalize_clusters(value: RValue) -> dict[str, dict[str, list[str]]]:
    assert isinstance(value, dict)
    clusters = value["Clusters"]
    if clusters == []:
        return {"Clusters": {}}
    assert isinstance(clusters, dict)
    normalized: dict[str, list[str]] = {}
    for key, item in clusters.items():
        if isinstance(item, str):
            normalized[key] = [item]
        elif isinstance(item, list):
            normalized[key] = ["" if element is None else str(element) for element in item]
        else:
            normalized[key] = [str(element) for element in np.asarray(item).tolist()]
    return {"Clusters": normalized}


def _normalize_dendrogram(value: RValue) -> dict[str, object]:
    assert isinstance(value, dict)
    clusters_value = _normalize_clusters(value)["Clusters"]
    if "Order" in value:
        return {"Clusters": clusters_value, "Order": None}
    dendrogram = value["Dendrogram"]
    assert isinstance(dendrogram, dict)
    return {
        "Clusters": clusters_value,
        "Dendrogram": {
            "merge": np.asarray(dendrogram["merge"], dtype=np.int64),
            "height": np.asarray(dendrogram["height"], dtype=np.float64),
            "order": np.asarray(dendrogram["order"], dtype=np.int64),
            "labels": np.asarray(dendrogram["labels"], dtype=str),
            "method": str(dendrogram["method"]),
            "dist.method": None,
        },
    }


def test_nns_sd_cluster_uses_dataframe_column_names() -> None:
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(_known_matrix(), columns=["A", "B", "C", "D"])

    actual = nns_sd_cluster(frame, degree=1, min_cluster=1)

    assert actual == nns_sd_cluster(_known_matrix(), degree=1, names=["A", "B", "C", "D"])
    assert actual == {"Clusters": {"Cluster_1": ["A", "D"], "Cluster_2": ["B"], "Cluster_3": ["C"]}}
    renamed = nns_sd_cluster(frame, degree=1, names=["w", "x", "y", "z"])
    assert renamed["Clusters"]["Cluster_1"] == ["w", "z"]
