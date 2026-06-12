from __future__ import annotations

import pytest

import pynns


def test_removed_r_nowcast_is_not_public() -> None:
    assert "nns_nowcast" not in pynns.__all__
    with pytest.raises(AttributeError):
        pynns.__getattr__("nns_nowcast")
