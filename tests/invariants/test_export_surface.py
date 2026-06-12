from __future__ import annotations

import pytest

import nns


def test_removed_r_nowcast_is_not_public() -> None:
    assert "nns_nowcast" not in nns.__all__
    with pytest.raises(AttributeError):
        nns.__getattr__("nns_nowcast")
