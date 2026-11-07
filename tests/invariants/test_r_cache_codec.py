"""R cache fidelity checks that run without Rscript or the native extension."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from _r import _NNS_VERSION, _decode, _encode

pytestmark = pytest.mark.invariant


@pytest.mark.parametrize(
    "raw",
    [
        {"result": [[1, 2], 2]},
        {"result": [[1, 2, 3], [4]]},
        {"result": [{"coefficients": [1, 2]}, {"converged": 1}]},
        {"result": [1.0, ["non-numeric", "labels"]]},
        {"result": [[1, 2], None, [3]]},
    ],
)
def test_heterogeneous_r_lists_survive_cache_roundtrip(raw: dict[str, object]) -> None:
    """The R oracle may return mixed-shape lists, not just numeric matrices."""
    decoded = _decode(raw)
    assert isinstance(decoded, dict)
    assert isinstance(decoded["result"], list)
    assert _encode(decoded) == raw
    json.dumps(_encode(decoded), allow_nan=False)


def test_rectangular_numeric_r_matrix_stays_ndarray() -> None:
    decoded = _decode([[1, 2], [3, 4]])
    assert isinstance(decoded, np.ndarray)
    assert decoded.shape == (2, 2)
    np.testing.assert_array_equal(decoded, [[1, 2], [3, 4]])
    assert _encode(decoded) == [[1.0, 2.0], [3.0, 4.0]]


def test_numeric_special_values_remain_numeric() -> None:
    decoded = _decode([1.0, "NA", "Inf", "-Inf"])
    assert isinstance(decoded, np.ndarray)
    assert float(decoded[0]) == 1.0
    assert np.isnan(decoded[1])
    assert np.isposinf(decoded[2])
    assert np.isneginf(decoded[3])


def test_r_cache_shards_match_pinned_r_manifest() -> None:
    repo = Path(__file__).resolve().parents[2]
    manifest = json.loads((repo / "sync" / "nns_source.json").read_text(encoding="utf-8"))
    assert manifest["r_version"] == _NNS_VERSION

    shards = sorted((repo / "tests" / "_r_cache").glob("*.json"))
    assert shards, "The committed R parity cache must not be empty."
    total = 0
    for path in shards:
        shard = json.loads(path.read_text(encoding="utf-8"))
        assert shard["schema_version"] == 2, path.name
        assert shard["nns_version"] == _NNS_VERSION, path.name
        label = shard["label"]
        assert path.name == label.replace("/", "_").replace("\\", "_") + ".json"
        entries = shard["entries"]
        assert isinstance(entries, dict) and entries, path.name
        assert all(re.fullmatch(r"[0-9a-f]{64}", key) for key in entries), path.name
        total += len(entries)
    assert total > 0
