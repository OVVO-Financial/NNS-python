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


def _r_seq(start: float, stop: float, by: float) -> np.ndarray:
    """R ``seq(start, stop, by)``: start + (0:n) * by, clamped at ``stop``."""
    count = int(np.floor((stop - start) / by + 1e-10))
    return np.minimum(start + np.arange(count + 1, dtype=np.float64) * by, stop)


def _tie_sensitive_case(name: str) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(9)
    if name == "sin_grid_4pi":
        x = _r_seq(0.0, 4.0 * np.pi, np.pi / 100.0)
        return x, np.sin(x)
    if name == "sin_grid_2pi":
        x = _r_seq(0.0, 2.0 * np.pi, np.pi / 100.0)
        return x, np.sin(x)
    if name == "rounded":
        x = np.round(rng.normal(size=300), 1)
        return x, np.round(x**2 + rng.normal(size=300), 1)
    if name == "three_valued":
        x = np.round(rng.normal(5.0, 1.0, size=216), 1)
        return x, np.digitize(x, [4.5, 5.5]).astype(np.float64)
    x = rng.integers(0, 6, size=400).astype(np.float64) / 10.0
    return x, (x * 10.0 + rng.integers(0, 3, size=400)) / 10.0


@pytest.mark.parity
@pytest.mark.parametrize("asym", [False, True])
@pytest.mark.parametrize(
    "case", ["sin_grid_4pi", "sin_grid_2pi", "rounded", "three_valued", "tenths"]
)
def test_nns_dep_tie_sensitive_cells_match_r(case: str, asym: bool) -> None:
    # Points that sit exactly on a cell mean: R takes the cell means as a
    # sequential double sum / n, and a pairwise (np.mean) mean that differs in
    # the last ulp moved such points between quadrants (DEP-09; sin_grid_4pi
    # gave Dependence 0.92402 vs R 0.92466).
    x, y = _tie_sensitive_case(case)

    expected = nns("NNS.dep", x.tolist(), y.tolist(), asym, False, False)
    assert isinstance(expected, dict)
    actual = nns_dep(x, y, asym=asym)

    np.testing.assert_allclose(actual["Correlation"], _scalar(expected["Correlation"]), atol=EXACT)
    np.testing.assert_allclose(actual["Dependence"], _scalar(expected["Dependence"]), atol=EXACT)


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


_NORMAL_300 = np.round(np.random.default_rng(1).standard_normal(300), 8)


@pytest.mark.parity
@pytest.mark.parametrize("asym", [False, True])
@pytest.mark.parametrize(
    "x, y",
    [
        (_NORMAL_300, -_NORMAL_300),
        (np.arange(1.0, 101.0), -np.arange(1.0, 101.0)),
        (np.array([1.0, 2.0]), np.array([2.0, 1.0])),
    ],
    ids=["neg_linear", "int_seq_rev", "len2"],
)
def test_nns_dep_fallback_keeps_the_sign_of_fully_discordant_cells(
    monkeypatch: pytest.MonkeyPatch, x: np.ndarray, y: np.ndarray, asym: bool
) -> None:
    # R fix 1f1490e: a partition cell whose points are all discordant scores -1, not
    # +1, so NNS.dep(x, -x) has Correlation -1. The NumPy path mirrors it; the native
    # nns_dep_pair kernel still has the old early return until it is rebuilt.
    import nns.dependence as dependence

    monkeypatch.setattr(dependence, "native_fn", lambda name: None)
    expected = nns("NNS.dep", x.tolist(), y.tolist(), asym, False, False)
    assert isinstance(expected, dict)
    actual = nns_dep(x, y, asym=asym)

    np.testing.assert_allclose(actual["Correlation"], _scalar(expected["Correlation"]), atol=EXACT)
    np.testing.assert_allclose(actual["Dependence"], _scalar(expected["Dependence"]), atol=EXACT)
    assert actual["Correlation"] < 0.0
