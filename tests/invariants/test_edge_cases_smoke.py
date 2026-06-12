from __future__ import annotations

import warnings

import numpy as np
from conftest import EdgeCase


def test_edge_case_battery_applies_to_numpy_mean(edge_case: EdgeCase) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = np.mean(edge_case.values)

    assert np.isscalar(result)
