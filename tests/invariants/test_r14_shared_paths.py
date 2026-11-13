"""Regression guards for the R14 rules shared by compound estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nns._reg_engine import _encode_predictors, _mreg_default_nbest, _mreg_partition_matrix


@pytest.mark.parametrize("order", [None, 1, 2, "max"])
def test_binary_coordinates_retain_both_terminal_atoms(order: int | str | None) -> None:
    x = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    partition = _mreg_partition_matrix(x, np.arange(4.0), order, "off", False)
    np.testing.assert_array_equal(partition["ids"], ["1.1", "1.2", "2.1", "2.2"])
    np.testing.assert_array_equal(partition["rhs"], [[0.0, 0.0], [1.0, 1.0]])


def test_default_neighbor_count_averages_response_orientations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nns.copula as copula

    seen = []

    def directional(data: np.ndarray) -> float:
        seen.append(data[:, -1].copy())
        return 0.1 if data[0, -1] > 0 else 0.9

    monkeypatch.setattr(copula, "nns_copula", directional)
    x = np.arange(200.0).reshape(100, 2)
    y = np.arange(1.0, 101.0)
    assert _mreg_default_nbest(x, y, 100) == 5
    np.testing.assert_array_equal(seen[0], y)
    np.testing.assert_array_equal(seen[1], -y)
    assert _mreg_default_nbest(x, -y, 100) == 5


@pytest.mark.parametrize("ordered", [False, True])
def test_predictor_levels_are_canonical_only_when_unordered(ordered: bool) -> None:
    x = pd.Series(
        pd.Categorical(["b", "a", "c", "a"], categories=["c", "b", "a"], ordered=ordered),
        name="flag",
    )
    encoded = _encode_predictors(x, pd.Series(["a", "c"]), True)
    levels = ["c", "b", "a"] if ordered else ["a", "b", "c"]
    assert encoded["metadata"][0]["levels"] == levels
    expected = np.array([[float(v == level) for level in levels] for v in x])
    np.testing.assert_array_equal(encoded["x"], expected)
    np.testing.assert_array_equal(
        encoded["point_est"], [[float(v == level) for level in levels] for v in ["a", "c"]]
    )


def test_pipeline_validation_forecasts_do_not_see_heldout_observations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nns.var as var

    t = np.arange(1.0, 41.0)
    values = np.column_stack((t + np.sin(t), 2 * t + np.cos(t)))
    captured = []

    def capture(interpolated: np.ndarray, univariate: np.ndarray, **kwargs: object) -> dict:
        train, test, actual = kwargs["validation_data"]
        captured.append((train.copy(), test.copy(), actual.copy()))
        return {
            "multivariate": univariate.copy(),
            "relevant_variables": np.full((0, values.shape[1]), None, dtype=object),
        }

    monkeypatch.setattr(var, "_var_multivariate_stack_stage", capture)
    var.nns_var(values, 2, tau=1)
    changed = values.copy()
    changed[-8:] += 1000
    var.nns_var(changed, 2, tau=1)
    np.testing.assert_array_equal(captured[0][0], captured[1][0])
    np.testing.assert_array_equal(captured[0][1], captured[1][1])
    np.testing.assert_array_equal(captured[0][2], values[-8:])
    np.testing.assert_array_equal(captured[1][2], changed[-8:])
    assert not np.array_equal(captured[0][2], captured[1][2])


def test_native_r14_overlays_have_separate_source_provenance() -> None:
    import hashlib
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / "sync/nns_source.json").read_text())
    assert manifest["r_version"] == "14.0"
    assert len(manifest["r14_native_overlays"]) == 4
    for overlay in manifest["r14_native_overlays"]:
        # Each overlay names the R commit its source was ported from and the
        # pinned oracle it was verified against bit for bit.
        assert overlay["verified_r_commit"] == manifest["r_commit"]
        assert len(overlay["r_source_commit"]) == 40
        assert not overlay["path"].startswith("extern/")
        actual = hashlib.sha256((root / overlay["path"]).read_bytes()).hexdigest()
        assert actual == overlay["sha256"], overlay["path"]
        assert len(overlay["r_source_sha256"]) == 64


def test_public_dependence_requires_versioned_native_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nns.dependence as dependence

    requested = []

    def lookup(name: str):
        requested.append(name)
        if name == "nns_dep_pair_r14":
            return lambda x, y, asym: {"Correlation": 0.25, "Dependence": 0.5}
        raise AssertionError(f"Unversioned native entry point requested: {name}")

    monkeypatch.setattr(dependence, "native_fn", lookup)
    result = dependence.nns_dep(np.arange(5.0), np.arange(5.0))
    assert requested == ["nns_dep_pair_r14"]
    assert result == {"Correlation": 0.25, "Dependence": 0.5}
