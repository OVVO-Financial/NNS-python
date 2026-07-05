from __future__ import annotations

import numpy as np

from nns import nns_norm


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


def test_equal_length_sequence_matches_matrix_path() -> None:
    x = np.linspace(1.0, 3.0, 50)
    y = np.linspace(2.0, 8.0, 50)
    z = np.linspace(10.0, 20.0, 50)
    matrix = np.column_stack((x, y, z))

    for linear in (False, True):
        from_sequence = nns_norm([x, y, z], linear=linear)
        from_matrix = nns_norm(matrix, linear=linear)
        assert isinstance(from_sequence, np.ndarray)
        np.testing.assert_allclose(from_sequence, from_matrix)


def test_unequal_length_sequence_forces_linear_scaling() -> None:
    vec1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    vec2 = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    vec3 = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])

    result = nns_norm([vec1, vec2, vec3])

    assert isinstance(result, list)
    assert [item.size for item in result] == [7, 6, 9]

    # Linear scaling equalizes every scaled mean at the grand mean of means,
    # and the linear flag is forced regardless of its passed value (as in R).
    grand_mean = np.mean([vec1.mean(), vec2.mean(), vec3.mean()])
    for item in result:
        np.testing.assert_allclose(item.mean(), grand_mean)
    forced = nns_norm([vec1, vec2, vec3], linear=True)
    for got, expected in zip(result, forced, strict=True):
        np.testing.assert_allclose(got, expected)


def test_zero_sum_length_differences_detected_as_unequal() -> None:
    # Lengths (5, 6, 5) have pairwise diffs summing to zero; they must still
    # take the unequal-length path rather than being treated as a matrix.
    vec1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    vec2 = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    vec3 = np.array([0.5, 0.6, 0.7, 0.8, 0.9])

    result = nns_norm([vec1, vec2, vec3])

    assert isinstance(result, list)
    assert [item.size for item in result] == [5, 6, 5]


def test_sequence_input_validation() -> None:
    import pytest

    with pytest.raises(ValueError, match="non-empty"):
        nns_norm([])
    with pytest.raises(ValueError, match=r"x\[1\] must be a 1D"):
        nns_norm([np.ones(3), np.ones((3, 2))])
    with pytest.raises(ValueError, match=r"x\[0\] must contain only finite"):
        nns_norm([np.array([1.0, np.nan]), np.ones(3)])
