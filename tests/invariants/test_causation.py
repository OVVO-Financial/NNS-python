from __future__ import annotations

import numpy as np
import pytest

from nns import causal_matrix, nns_causation


def test_nns_causation_identical_self_case() -> None:
    x = np.linspace(-2.0, 2.0, 200)

    result = nns_causation(x, x)

    assert result["Causation.x.given.y"] == pytest.approx(result["Causation.y.given.x"])
    assert abs(next(value for key, value in result.items() if key.startswith("C("))) <= 100.0


def test_nns_causation_is_directional_for_asymmetric_pair() -> None:
    x = np.linspace(-2.0, 2.0, 200)
    y = x**2 + 0.1 * np.sin(x)

    forward = nns_causation(x, y)
    reverse = nns_causation(y, x)

    assert forward != reverse


def test_nns_causation_values_are_bounded_like_r_conventions() -> None:
    x = np.linspace(-2.0, 2.0, 200)
    y = np.sin(x)

    result = nns_causation(x, y)
    directional = list(result.values())[:2]
    net = next(value for key, value in result.items() if key.startswith("C("))

    assert all(0.0 <= value <= 1.0 for value in directional)
    assert abs(net) <= 100.0


def test_causal_matrix_is_antisymmetric() -> None:
    x = np.linspace(-2.0, 2.0, 100)
    variable = np.column_stack((x, x**2, np.sin(x)))

    result = causal_matrix(variable)

    np.testing.assert_allclose(np.diag(result), 0.0)
    np.testing.assert_allclose(result, -result.T)


def test_nns_causation_ts_tau_no_longer_raises() -> None:
    x = np.linspace(-2.0, 2.0, 100)

    result = nns_causation(x, np.sin(x), tau="ts")

    assert set(result) in (
        {"Causation.x.given.y", "Causation.y.given.x", "C(x--->y)"},
        {"Causation.x.given.y", "Causation.y.given.x", "C(y--->x)"},
    )


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
    np.testing.assert_allclose(np.diag(result), 0.0)


@pytest.mark.parametrize("n", [5, 40, 600])
def test_causation_native_asym_dep_matches_python_partition(n: int) -> None:
    # _asym_dep uses the native nns_dep(asym=True) kernel. The pure-Python X-only
    # partition computes the same cells and cell copulas; only the order in which
    # the cell terms are added differs (R's std::unordered_map order natively,
    # first-seen order in Python), which moves the total by a few ulp at most.
    from nns.causation import _asym_dep, _asym_dep_python

    rng = np.random.default_rng(n)
    x = rng.normal(size=n)
    for y in (x**2 + rng.normal(size=n), rng.normal(size=n), np.round(x), -x):
        assert _asym_dep(x, y) == pytest.approx(_asym_dep_python(x, y), rel=1e-14, abs=1e-15)
        assert _asym_dep(y, x) == pytest.approx(_asym_dep_python(y, x), rel=1e-14, abs=1e-15)
    ints = rng.integers(0, 4, size=n).astype(float)
    assert _asym_dep(ints, x) == pytest.approx(_asym_dep_python(ints, x), rel=1e-14, abs=1e-15)
