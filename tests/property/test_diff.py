from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from pynns import nns_diff


@given(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False))
def test_nns_diff_identity_property(point: float) -> None:
    result = nns_diff(lambda x: x, point)

    assert result["DERIVATIVE"] == 1.0
    assert np.isfinite(result["Value of f(x) at point"])
