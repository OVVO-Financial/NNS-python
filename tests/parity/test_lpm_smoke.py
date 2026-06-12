import numpy as np
import pytest
from _r import nns
from _tolerances import EXACT


@pytest.mark.parity
def test_lpm_smoke() -> None:
    result = nns("LPM", 1, 0, [-2, -1, 0, 1, 2])

    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, np.array(0.6), atol=EXACT)
