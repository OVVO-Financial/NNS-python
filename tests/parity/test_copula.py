from __future__ import annotations

import numpy as np
import pytest
from _r import nns
from _tolerances import EXACT

from nns import nns_copula

SIZES = [50, 200, 1000]
RELATIONSHIPS = ["correlated_normal", "independent", "anti_monotonic"]


@pytest.mark.parity
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("relationship", RELATIONSHIPS)
def test_nns_copula_matches_r(
    rng: np.random.Generator,
    size: int,
    relationship: str,
) -> None:
    x, y = _relationship(relationship, size, rng)

    expected = nns("NNS.copula", np.column_stack((x, y)).tolist(), None, True, False, False)
    actual = nns_copula(x, y)

    np.testing.assert_allclose(actual, _scalar(expected), atol=EXACT)


@pytest.mark.parity
def test_nns_copula_target_matches_r() -> None:
    x = np.linspace(-2.0, 2.0, 200)
    y = np.sin(x)

    expected = nns(
        "NNS.copula",
        np.column_stack((x, y)).tolist(),
        [0.25, -0.1],
        True,
        False,
        False,
    )
    actual = nns_copula(x, y, target_x=0.25, target_y=-0.1)

    np.testing.assert_allclose(actual, _scalar(expected), atol=EXACT)


def _scalar(value: object) -> float:
    assert isinstance(value, np.ndarray)
    return float(value)


def _relationship(
    relationship: str,
    size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    x = rng.normal(size=size)
    if relationship == "correlated_normal":
        noise = rng.normal(size=size)
        return x, 0.7 * x + np.sqrt(1.0 - 0.7**2) * noise
    if relationship == "independent":
        return x, rng.normal(size=size)
    if relationship == "monotonic_nonlinear":
        return x, x**3
    return x, -x



@pytest.mark.parity
def test_nns_copula_default_target_uses_r_col_means() -> None:
    # np.mean(x) is 0.4999999999999999 here, R colMeans(x) is 0.5, and four
    # observations sit exactly on 0.5. The last ulp decides whether they count as
    # lower (all points concordant, R's shortcut returns 1) or upper (0.882).
    x = np.array(
        [0.1, 0.2, 0.9, 0.6, 0.3, 0.5, 0.9, 0.5, 0.1, 0.6, 0.8, 0.8, 0.7, 0.2, 0.5, 0.5, 0.7, 0.1]
    )
    y = np.where(x <= 0.5, x, x + 10.0)
    m = np.column_stack((x, y))
    assert float(np.mean(x)) != 0.5

    expected = _scalar(nns("NNS.copula", m.tolist(), None, True, False, False))

    np.testing.assert_allclose(nns_copula(m), expected, atol=EXACT)
    np.testing.assert_allclose(nns_copula(x, y), expected, atol=EXACT)


def _integer_matrices() -> list[np.ndarray]:
    rng = np.random.default_rng(1)
    draws = [rng.integers(1, 6, (10, 3)).astype(float) for _ in range(30)]
    # Draws 6, 14 and 24 have a degree-0 pair sum of exactly 1 (an interior value
    # for three variables), so the bivariate shortcut used to return 1 for them.
    return [draws[6], draws[14], draws[24], draws[3]]


@pytest.mark.parity
@pytest.mark.parametrize("native", [True, False])
@pytest.mark.parametrize("continuous", [True, False])
@pytest.mark.parametrize("case", range(4))
def test_nns_copula_perfect_dependence_shortcut_is_dimension_generic(
    monkeypatch: pytest.MonkeyPatch, case: int, continuous: bool, native: bool
) -> None:
    # CORE-10, R fix "NNS.copula: dimension-generic perfect-dependence early
    # return": the shortcut fires only when every pair is fully concordant.
    m = _integer_matrices()[case]
    expected = _scalar(nns("NNS.copula", m.tolist(), None, continuous, False, False))
    if not native:
        import nns.copula as copula

        monkeypatch.setattr(copula, "native_fn", lambda name: None)
    actual = nns_copula(m, continuous=continuous)
    np.testing.assert_allclose(actual, expected, atol=EXACT)
    if case < 3:
        assert actual < 1.0


@pytest.mark.parity
@pytest.mark.parametrize("native", [True, False])
def test_nns_copula_degree0_ties_partition_the_sample(
    monkeypatch: pytest.MonkeyPatch, native: bool
) -> None:
    # CORE-09: columns of -1/0/1 with means exactly 0, so a third of each column
    # ties the target; DPM_nD at degree 0 counts them as lower, not discordant.
    rng = np.random.default_rng(2)
    base = np.repeat([-1.0, 0.0, 1.0], 40)
    m = np.column_stack([rng.permutation(base) for _ in range(3)])
    cases = [m[:, :2], m]
    expected = [
        _scalar(nns("NNS.copula", c.tolist(), None, True, False, False)) for c in cases
    ]
    if not native:
        import nns.copula as copula

        monkeypatch.setattr(copula, "native_fn", lambda name: None)
    for c, want in zip(cases, expected, strict=True):
        np.testing.assert_allclose(nns_copula(c), want, atol=EXACT)
