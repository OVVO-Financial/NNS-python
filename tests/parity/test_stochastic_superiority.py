from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from _r import nns
from _tolerances import EXACT

from pynns import nns_ss


@pytest.mark.parity
@pytest.mark.parametrize(
    ("x", "y"),
    [
        ([2.0, 3.0, 4.0], [1.0, 2.0, 3.0]),
        ([1.0, 2.0, 3.0], [2.0, 3.0, 4.0]),
        ([1.0, 4.0], [2.0, 3.0]),
        ([1.0, 2.0, 2.0], [1.0, 2.0, 2.0]),
        ([1.0, 2.0], [1.0, 2.0, 3.0, 4.0]),
    ],
)
def test_nns_ss_deterministic_matches_r(x: list[float], y: list[float]) -> None:
    expected = cast(dict[str, np.ndarray], nns("NNS.SS", x, y, False))
    actual = nns_ss(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))

    assert actual["p_gt"] == pytest.approx(float(expected["p_gt"]), abs=EXACT)
    assert actual["p_tie"] == pytest.approx(float(expected["p_tie"]), abs=EXACT)
    assert actual["p_star"] == pytest.approx(float(expected["p_star"]), abs=EXACT)


@pytest.mark.parity
def test_nns_ss_nan_omission_matches_installed_r_probe() -> None:
    actual = nns_ss(np.array([np.nan, 2.0, 3.0]), np.array([1.0, np.nan, 3.0]))

    assert actual["p_gt"] == pytest.approx(0.5, abs=EXACT)
    assert actual["p_tie"] == pytest.approx(0.25, abs=EXACT)
    assert actual["p_star"] == pytest.approx(0.625, abs=EXACT)


@pytest.mark.parity
def test_nns_ss_infinity_matches_installed_r_probe() -> None:
    actual = nns_ss(np.array([1.0, np.inf, 3.0]), np.array([1.0, 2.0, np.inf]))

    assert actual["p_gt"] == pytest.approx(0.4444444444444444, abs=EXACT)
    assert actual["p_tie"] == pytest.approx(0.2222222222222222, abs=EXACT)
    assert actual["p_star"] == pytest.approx(0.5555555555555556, abs=EXACT)


@pytest.mark.parity
def test_nns_ss_probe_values_match_installed_r() -> None:
    actual = nns_ss(np.array([2.0, 3.0, 4.0]), np.array([1.0, 2.0, 3.0]))

    assert actual["p_gt"] == pytest.approx(0.666666666666667)
    assert actual["p_tie"] == pytest.approx(0.222222222222222)
    assert actual["p_star"] == pytest.approx(0.777777777777778)


@pytest.mark.parity
def test_nns_ss_empty_after_nan_raises() -> None:
    with pytest.raises(ValueError, match="at least one non-missing"):
        nns_ss(np.array([np.nan]), np.array([1.0, 2.0]))


@pytest.mark.parity
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"confidence_interval": True, "reps": 1}, "reps"),
        ({"confidence_interval": True, "reps": 3, "ci": 1.0}, "ci"),
    ],
)
def test_nns_ss_invalid_ci_arguments_raise(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        nns_ss(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), **kwargs)
