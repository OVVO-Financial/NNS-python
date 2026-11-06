"""Invariants of R14 book-ch14 directional association."""
from __future__ import annotations

import numpy as np
import pytest

from nns import causal_matrix, nns_causation
from nns._dep_r14 import dep_levels_r14


def test_nns_causation_identical_self_case() -> None:
    x = np.linspace(-2.0, 2.0, 200)
    result = nns_causation(x, x)
    assert result["C(x--->y)"] == pytest.approx(0.5)
    assert result["C(y--->x)"] == pytest.approx(0.5)


def test_nns_causation_is_directional_for_asymmetric_pair() -> None:
    x = np.linspace(-2.0, 2.0, 200)
    y = x**2 + 0.1 * np.sin(x)
    forward = nns_causation(x, y)
    reverse = nns_causation(y, x)
    assert forward["C(x--->y)"] == pytest.approx(reverse["C(y--->x)"])
    assert forward["C(y--->x)"] == pytest.approx(reverse["C(x--->y)"])
    assert forward["C(x--->y)"] != pytest.approx(0.5)


def test_nns_causation_values_are_bounded_like_r_conventions() -> None:
    x = np.linspace(-2.0, 2.0, 200)
    y = np.sin(x)
    result = nns_causation(x, y)
    assert set(result) == {"C(x--->y)", "C(y--->x)"}
    assert all(0.0 <= value <= 1.0 for value in result.values())
    assert sum(result.values()) == pytest.approx(1.0, abs=1e-15)


def test_causal_matrix_offdiagonal_indices_are_complementary() -> None:
    x = np.linspace(-2.0, 2.0, 100)
    variable = np.column_stack((x, x**2, np.sin(x)))
    result = causal_matrix(variable)
    np.testing.assert_array_equal(np.diag(result), 0.0)
    complement = result + result.T
    np.testing.assert_allclose(complement, 1.0 - np.eye(3), atol=1e-15)


def test_causal_matrix_entry_means_row_causes_column() -> None:
    x = np.linspace(-2.0, 2.0, 100)
    y = x * x
    pair = nns_causation(x, y)
    result = causal_matrix(np.column_stack((x, y)))
    assert result[0, 1] == pytest.approx(pair["C(x--->y)"])
    assert result[1, 0] == pytest.approx(pair["C(y--->x)"])
    assert result[0, 1] + result[1, 0] == pytest.approx(1.0)


def test_causal_matrix_accepts_cross_sectional_tau_for_two_observations() -> None:
    variable = np.array([[1.0, 2.0], [2.0, 1.0]])
    np.testing.assert_allclose(
        causal_matrix(variable, tau="cs"), np.array([[0.0, 0.5], [0.5, 0.0]])
    )


def test_nns_causation_ts_tau_no_longer_raises() -> None:
    x = np.linspace(-2.0, 2.0, 100)
    result = nns_causation(x, np.sin(x), tau="ts")
    assert set(result) == {"C(x--->y)", "C(y--->x)"}
    assert sum(result.values()) == pytest.approx(1.0, abs=1e-15)


def test_causal_matrix_ts_tau_no_longer_raises() -> None:
    t = np.arange(1, 61, dtype=np.float64)
    variable = np.column_stack(
        (
            np.sin(2.0 * np.pi * t / 7.0),
            np.cos(2.0 * np.pi * t / 7.0),
            np.sin(2.0 * np.pi * t / 5.0),
        )
    )
    result = causal_matrix(variable, tau="ts")
    assert result.shape == (3, 3)
    np.testing.assert_allclose(result + result.T, 1.0 - np.eye(3), atol=1e-15)


@pytest.mark.parametrize("n", [5, 40, 600])
def test_causation_rho_is_unfloored_mean_of_partition_levels(n: int) -> None:
    from nns.causation import _asym_dep, _asym_dep_python

    rng = np.random.default_rng(n)
    x = rng.normal(size=n)
    for y in (x**2 + rng.normal(size=n), rng.normal(size=n), np.round(x), -x):
        levels = dep_levels_r14(x, y)
        expected = float(np.mean(levels))
        assert _asym_dep(x, y) == pytest.approx(expected, abs=1e-15)
        assert _asym_dep_python(x, y) == pytest.approx(expected, abs=1e-15)
        assert expected <= max(levels) + 1e-12
