from __future__ import annotations

import numpy as np
import pytest
from _r import nns
from _tolerances import EXACT

from nns import nns_norm

SIZES = [50, 200, 1000]


@pytest.mark.parity
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("linear", [False, True])
def test_nns_norm_matches_r_for_correlation_scale_path(
    size: int,
    linear: bool,
) -> None:
    x = _small_matrix(size)

    expected = nns("NNS.norm", x.tolist(), linear, None)
    actual = nns_norm(x, linear=linear)

    np.testing.assert_allclose(actual, _matrix(expected), atol=EXACT)


@pytest.mark.parity
@pytest.mark.parametrize("linear", [False, True])
def test_nns_norm_matches_r_for_dependence_scale_path(linear: bool) -> None:
    x = _wide_matrix(50)

    expected = nns("NNS.norm", x.tolist(), linear, None)
    actual = nns_norm(x, linear=linear)

    np.testing.assert_allclose(actual, _matrix(expected), atol=EXACT)


def _matrix(value: object) -> np.ndarray:
    assert isinstance(value, np.ndarray)
    return value.astype(np.float64)


def _small_matrix(size: int) -> np.ndarray:
    row = np.linspace(-2.0, 2.0, size)
    return np.column_stack((row + 3.0, row**2 + 1.0, np.sin(row) + 2.0))


def _wide_matrix(size: int) -> np.ndarray:
    row = np.arange(1, size + 1, dtype=np.float64)[:, np.newaxis]
    col = np.arange(1, 11, dtype=np.float64)[np.newaxis, :]
    return np.sin(row * col / 13.0) + np.cos((row + 3.0) / (col + 5.0)) + 3.0


@pytest.mark.parity
@pytest.mark.parametrize("linear", [False, True])
def test_nns_norm_accepts_dataframe_like_r(linear: bool) -> None:
    pd = pytest.importorskip("pandas")
    x = _small_matrix(60)
    frame = pd.DataFrame(x, columns=["a", "b", "c"], index=pd.RangeIndex(10, 70))

    expected = nns("NNS.norm", {name: frame[name].tolist() for name in frame}, linear, None)
    actual = nns_norm(frame, linear=linear)

    assert isinstance(actual, pd.DataFrame)
    assert list(actual.columns) == ["a Normalized", "b Normalized", "c Normalized"]
    assert actual.index.equals(frame.index)
    np.testing.assert_allclose(actual.to_numpy(), _matrix(expected), atol=EXACT)
    np.testing.assert_array_equal(actual.to_numpy(), nns_norm(x, linear=linear))
