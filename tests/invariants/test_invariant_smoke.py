import numpy as np
import pytest


@pytest.mark.invariant
def test_numpy_mean_smoke() -> None:
    values = np.array([1.0, 2.0, 3.0])

    assert values.mean() == 2.0
