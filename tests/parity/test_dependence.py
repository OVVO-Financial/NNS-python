from __future__ import annotations

import numpy as np
import pytest
from _r import nns
from _tolerances import EXACT

from nns import nns_cor, nns_dep

SIZES = [50, 200, 1000]
RELATIONSHIPS = ["linear", "independent", "quadratic", "sin", "noise"]


@pytest.mark.parity
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("relationship", RELATIONSHIPS)
def test_nns_dep_matches_r(
    rng: np.random.Generator,
    size: int,
    relationship: str,
) -> None:
    x, y = _relationship(relationship, size, rng)

    expected = nns("NNS.dep", x.tolist(), y.tolist(), False, False, False)
    assert isinstance(expected, dict)
    expected_correlation = _scalar(expected["Correlation"])
    expected_dependence = _scalar(expected["Dependence"])
    actual = nns_dep(x, y)

    np.testing.assert_allclose(actual["Correlation"], expected_correlation, atol=EXACT)
    np.testing.assert_allclose(actual["Dependence"], expected_dependence, atol=EXACT)
    np.testing.assert_allclose(nns_cor(x, y), expected_correlation, atol=EXACT)


@pytest.mark.parity
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("relationship", RELATIONSHIPS)
def test_nns_dep_asym_matches_r(
    rng: np.random.Generator,
    size: int,
    relationship: str,
) -> None:
    x, y = _relationship(relationship, size, rng)

    expected = nns("NNS.dep", x.tolist(), y.tolist(), True, False, False)
    assert isinstance(expected, dict)
    expected_correlation = _scalar(expected["Correlation"])
    expected_dependence = _scalar(expected["Dependence"])
    actual = nns_dep(x, y, asym=True)

    np.testing.assert_allclose(actual["Correlation"], expected_correlation, atol=EXACT)
    np.testing.assert_allclose(actual["Dependence"], expected_dependence, atol=EXACT)


@pytest.mark.parity
def test_nns_dep_identical_pair_matches_r() -> None:
    x = np.linspace(-2.0, 2.0, 200)

    expected = nns("NNS.dep", x.tolist(), x.tolist(), False, False, False)
    assert isinstance(expected, dict)
    expected_correlation = _scalar(expected["Correlation"])
    expected_dependence = _scalar(expected["Dependence"])
    actual = nns_dep(x, x)

    np.testing.assert_allclose(actual["Correlation"], expected_correlation, atol=EXACT)
    np.testing.assert_allclose(actual["Dependence"], expected_dependence, atol=EXACT)


@pytest.mark.parity
def test_nns_dep_asym_identical_pair_matches_r() -> None:
    x = np.linspace(-2.0, 2.0, 200)

    expected = nns("NNS.dep", x.tolist(), x.tolist(), True, False, False)
    assert isinstance(expected, dict)
    expected_correlation = _scalar(expected["Correlation"])
    expected_dependence = _scalar(expected["Dependence"])
    actual = nns_dep(x, x, asym=True)

    np.testing.assert_allclose(actual["Correlation"], expected_correlation, atol=EXACT)
    np.testing.assert_allclose(actual["Dependence"], expected_dependence, atol=EXACT)


def _scalar(value: object) -> float:
    assert isinstance(value, np.ndarray)
    return float(value)


def _relationship(
    relationship: str,
    size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    x = rng.normal(size=size)
    if relationship == "linear":
        noise = rng.normal(size=size)
        return x, 0.7 * x + np.sqrt(1.0 - 0.7**2) * noise
    if relationship == "independent":
        return x, rng.normal(size=size)
    if relationship == "quadratic":
        return x, x * x + 0.1 * rng.normal(size=size)
    if relationship == "sin":
        return x, np.sin(x) + 0.05 * rng.normal(size=size)
    if relationship == "cubic":
        return x, x**3
    return rng.normal(size=size), rng.normal(size=size)


@pytest.mark.parity
@pytest.mark.parametrize("asym", [False, True])
def test_nns_dep_matrix_matches_r(asym: bool) -> None:
    # R NNS.dep(X) returns Correlation / Dependence matrices; with asym = TRUE
    # entry [i, j] is NNS.dep(X[, i], X[, j], asym = TRUE) (row -> column).
    rng = np.random.default_rng(2024)
    x = rng.normal(size=150)
    variable = np.column_stack(
        (x, x**2 + 0.2 * rng.normal(size=150), np.sin(2 * x), rng.normal(size=150))
    )

    expected = nns("NNS.dep", variable.tolist(), None, asym, False, False)
    assert isinstance(expected, dict)
    actual = nns_dep(variable, asym=asym)

    for key in ("Correlation", "Dependence"):
        expected_matrix = np.asarray(expected[key], dtype=np.float64)
        assert actual[key].shape == (4, 4)
        np.testing.assert_allclose(actual[key], expected_matrix, atol=EXACT)
    if not asym:
        np.testing.assert_array_equal(actual["Dependence"], actual["Dependence"].T)
    else:
        assert actual["Dependence"][0, 1] == nns_dep(x, variable[:, 1], asym=True)["Dependence"]


def test_nns_dep_matrix_keeps_dataframe_labels_and_validates() -> None:
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(rng.normal(size=(60, 3)), columns=["spy", "tlt", "gld"])

    result = nns_dep(frame)

    for key in ("Correlation", "Dependence"):
        assert list(result[key].index) == ["spy", "tlt", "gld"]
        assert list(result[key].columns) == ["spy", "tlt", "gld"]
        np.testing.assert_array_equal(np.diag(result[key].to_numpy()), 1.0)
    pair = nns_dep(frame["spy"].to_numpy(), frame["gld"].to_numpy())
    assert result["Dependence"].loc["spy", "gld"] == pair["Dependence"]
    with pytest.raises(ValueError):
        nns_dep(frame.to_numpy()[:, 0])
    with pytest.raises(ValueError):
        nns_dep(frame.to_numpy()[:, :1])
