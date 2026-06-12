from __future__ import annotations

import numpy as np

from pynns import nns_norm


def test_nns_norm_shape_matches_input() -> None:
    x = np.arange(1, 13, dtype=np.float64).reshape(4, 3)

    assert nns_norm(x).shape == x.shape


def test_linear_nns_norm_equalizes_column_means() -> None:
    x = np.column_stack(
        (
            np.linspace(1.0, 3.0, 50),
            np.linspace(2.0, 8.0, 50),
            np.linspace(10.0, 20.0, 50),
        )
    )

    result = nns_norm(x, linear=True)

    np.testing.assert_allclose(np.mean(result, axis=0), np.mean(result))


def test_nonlinear_nns_norm_preserves_shape_for_wide_matrix() -> None:
    row = np.arange(1, 51, dtype=np.float64)[:, np.newaxis]
    col = np.arange(1, 11, dtype=np.float64)[np.newaxis, :]
    x = np.sin(row * col / 13.0) + np.cos((row + 3.0) / (col + 5.0)) + 3.0

    result = nns_norm(x)

    assert result.shape == x.shape
    assert np.all(np.isfinite(result))
