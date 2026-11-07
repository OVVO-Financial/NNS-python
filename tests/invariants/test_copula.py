from __future__ import annotations

import numpy as np
import pytest

from nns import nns_copula


def test_nns_copula_is_bounded() -> None:
    x = np.linspace(-2.0, 2.0, 200)
    y = np.sin(x)

    result = nns_copula(x, y)

    assert result >= 0.0
    assert result <= 1.0


def test_nns_copula_is_symmetric() -> None:
    x = np.linspace(-2.0, 2.0, 200)
    y = x**3

    assert nns_copula(x, y) == pytest.approx(nns_copula(y, x), abs=1e-12)


def test_nns_copula_identical_pair_is_unit() -> None:
    x = np.linspace(-2.0, 2.0, 200)

    assert nns_copula(x, x) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("x", "y", "kwargs"),
    [
        (np.array([]), np.array([]), {}),
        (np.array([0.1, np.nan, 0.3, 0.4]), np.array([1.0, 2.0, 3.0, 4.0]), {}),
        (np.array([0.1, 0.2, 0.3, np.inf]), np.array([1.0, 2.0, 3.0, 4.0]), {}),
        (np.array([0.1, 0.2, 0.3, 0.4]), np.array([1.0, -np.inf, 3.0, 4.0]), {}),
        (np.array([0.1, 0.2, 0.3, 0.4]), np.array([1.0, 2.0, 3.0, 4.0]), {"target_x": np.nan}),
        (np.array([0.1, 0.2, 0.3, 0.4]), np.array([1.0, 2.0, 3.0, 4.0]), {"target_y": np.inf}),
    ],
)
def test_nns_copula_rejects_empty_and_non_finite_bivariate_input(
    x: np.ndarray, y: np.ndarray, kwargs: dict[str, float]
) -> None:
    # R NNS.copula stops on missing / non-finite data. The bivariate form used to
    # segfault on empty input and return 1.0 for a NaN.
    with pytest.raises(ValueError):
        nns_copula(x, y, **kwargs)


def test_nns_copula_rejects_empty_matrix() -> None:
    with pytest.raises(ValueError):
        nns_copula(np.empty((0, 2)))


def _grid(*ranges: range) -> np.ndarray:
    # R's expand.grid: the first column varies fastest.
    mesh = np.meshgrid(*ranges, indexing="ij")
    return np.column_stack([m.ravel(order="F") for m in mesh]).astype(np.float64)


@pytest.mark.parametrize("native", [True, False])
def test_nns_copula_is_zero_when_every_margin_is_one_sided(
    monkeypatch: pytest.MonkeyPatch, native: bool
) -> None:
    # R NNS.copula: each pair is then trivially concordant or discordant; these returned 1.
    if not native:
        import nns.copula as copula

        monkeypatch.setattr(copula, "native_fn", lambda name: None)
    g = _grid(range(1, 11), range(1, 11))
    x = np.arange(1.0, 11.0)
    b = np.column_stack([np.tile([0.0, 1.0], 50), np.tile([1.0, 0.0], 50)])
    assert nns_copula(g, target=[0, 0]) == 0.0
    assert nns_copula(g, target=[0, 0], continuous=False) == 0.0
    assert nns_copula(g, target=[10, 10]) == 0.0
    assert nns_copula(g, target=[0, 11]) == 0.0
    assert nns_copula(_grid(range(1, 5), range(1, 5), range(1, 5)), target=[0, 0, 0]) == 0.0
    assert nns_copula(np.column_stack([x, x]), target=[0, 0]) == 0.0
    assert nns_copula(b, target=[1, 1]) == 0.0
    assert nns_copula(b, target=[1, 1], continuous=False) == 0.0
    # One margin inside its data keeps the full measure (R main 0.52704627669472992).
    assert nns_copula(np.column_stack([x, x]), target=[0, 5]) == pytest.approx(
        0.52704627669472992, abs=1e-15
    )


@pytest.mark.parametrize("native", [True, False])
@pytest.mark.parametrize("continuous", [True, False])
def test_nns_copula_is_one_for_perfect_dependence_inside_the_data(
    monkeypatch: pytest.MonkeyPatch, native: bool, continuous: bool
) -> None:
    if not native:
        import nns.copula as copula

        monkeypatch.setattr(copula, "native_fn", lambda name: None)
    x = np.arange(1.0, 11.0)
    assert nns_copula(np.column_stack([x, x]), continuous=continuous) == 1.0
    assert nns_copula(np.column_stack([x, -x]), continuous=continuous) == 1.0
    assert nns_copula(np.column_stack([x, 2 * x + 1]), target=[3, 7], continuous=continuous) == 1.0
    assert nns_copula(np.column_stack([x, -x]), target=[3.5, -3.5], continuous=continuous) == 1.0
    cube = np.column_stack([x, x, -x])
    assert nns_copula(cube, target=[3.5, 3.5, -3.5], continuous=continuous) == 1.0
