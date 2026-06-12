from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from _r import RValue, nns

from pynns import fsd, fsd_uni, sd_efficient_set, ssd, ssd_uni, tsd, tsd_uni

SIZES = [50, 200, 1000]


@pytest.mark.parity
@pytest.mark.parametrize(
    ("r_name", "function", "x_dominates", "y_dominates", "none"),
    [
        ("NNS.FSD", fsd, "X FSD Y", "Y FSD X", "NO FSD EXISTS"),
        ("NNS.SSD", ssd, "X SSD Y", "Y SSD X", "NO SSD EXISTS"),
        ("NNS.TSD", tsd, "X TSD Y", "Y TSD X", "NO TSD EXISTS"),
    ],
)
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("case", ["shifted", "stretched", "crossing", "random"])
def test_sd_functions_match_r(
    rng: np.random.Generator,
    r_name: str,
    function: SDPairFunction,
    x_dominates: str,
    y_dominates: str,
    none: str,
    size: int,
    case: str,
) -> None:
    x, y = _pair(case, size, rng)
    expected = _sd_result_from_r(r_name, x, y, x_dominates, y_dominates, none)

    assert function(x, y) == expected


@pytest.mark.parity
@pytest.mark.parametrize(
    ("r_name", "function"),
    [
        ("NNS.FSD.uni", fsd_uni),
        ("NNS.SSD.uni", ssd_uni),
        ("NNS.TSD.uni", tsd_uni),
    ],
)
@pytest.mark.parametrize("case", ["dominance", "reverse", "crossing", "identical"])
def test_sd_uni_wrappers_match_r(
    r_name: str,
    function: Callable[..., int],
    case: str,
) -> None:
    x, y = _uni_pair(case)
    if r_name == "NNS.FSD.uni":
        r_value = nns(r_name, x.tolist(), y.tolist(), "discrete")
        expected_value = int(np.asarray(r_value).reshape(-1)[0])
        actual = function(x, y, "discrete")
    else:
        bidirectional = _sd_result_from_r(
            r_name.removesuffix(".uni"),
            x,
            y,
            f"X {r_name[4:7]} Y",
            f"Y {r_name[4:7]} X",
            f"NO {r_name[4:7]} EXISTS",
        )
        expected_value = 1 if bidirectional == 1 else 0
        actual = function(x, y)

    assert actual == expected_value


@pytest.mark.parity
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("case", ["constructed", "random"])
def test_sd_efficient_set_matches_r(
    rng: np.random.Generator,
    degree: int,
    size: int,
    case: str,
) -> None:
    if case == "constructed":
        base = np.linspace(-1.0, 1.0, size)
        returns = np.column_stack(
            [
                base + 0.05,
                base,
                np.sin(np.linspace(0.0, 4.0, size)),
                np.cos(np.linspace(0.0, 4.0, size)) * 0.4,
            ]
        )
    else:
        returns = rng.normal(size=(size, 6))

    expected = nns(
        "NNS.SD.efficient.set",
        returns.tolist(),
        degree,
        "discrete",
        False,
    )
    assert _strings(expected) == [f"X_{index + 1}" for index in sd_efficient_set(returns, degree)]


@pytest.mark.parity
def test_sd_efficient_set_continuous_fsd_matches_r() -> None:
    row = np.arange(1, 9, dtype=np.float64)
    returns = np.column_stack(
        [
            0.2 * row,
            np.sin(row),
            np.cos(row) + 0.1 * row,
            np.where(row % 2 == 0, 1.0, -1.0),
        ]
    )

    expected = nns("NNS.SD.efficient.set", returns.tolist(), 1, "continuous", False)

    assert _strings(expected) == [
        f"X_{index + 1}" for index in sd_efficient_set(returns, 1, type="continuous")
    ]


SDPairFunction = Callable[[np.ndarray, np.ndarray], int]


def _pair(
    case: str,
    size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if case == "shifted":
        y = rng.normal(size=size)
        return y + 0.25, y
    if case == "stretched":
        base = rng.normal(size=size)
        return base + 0.05, base * 1.4 - 0.05
    if case == "crossing":
        half = size // 2
        x = np.concatenate((np.full(half, -0.2), np.full(size - half, 1.0)))
        y = np.concatenate((np.full(half, 0.0), np.full(size - half, 0.7)))
        return x, y
    return rng.normal(size=size), rng.normal(size=size)


def _uni_pair(case: str) -> tuple[np.ndarray, np.ndarray]:
    base = np.array([-1.0, -0.25, 0.5, 1.0, 2.0], dtype=np.float64)
    if case == "dominance":
        return base + 0.5, base
    if case == "reverse":
        return base, base + 0.5
    if case == "crossing":
        return np.array([-1.0, 0.0, 3.0, 3.5]), np.array([-0.5, 1.0, 1.5, 2.0])
    return base, base.copy()


def _sd_result_from_r(
    r_name: str,
    x: np.ndarray,
    y: np.ndarray,
    x_dominates: str,
    y_dominates: str,
    none: str,
) -> int:
    if r_name == "NNS.FSD":
        result = nns(r_name, x.tolist(), y.tolist(), "discrete", False)
    else:
        result = nns(r_name, x.tolist(), y.tolist(), False)
    assert isinstance(result, str)
    return {x_dominates: 1, y_dominates: -1, none: 0}[result]


def _strings(value: RValue) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, np.ndarray):
        return [str(item) for item in value.tolist()]
    raise TypeError("Expected an R character vector.")
