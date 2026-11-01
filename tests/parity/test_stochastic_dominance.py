from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from _r import RValue, nns

from nns import fsd, fsd_uni, sd_efficient_set, ssd, ssd_uni, tsd, tsd_uni

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
@pytest.mark.parametrize("size", [8, 200, 30_000])
@pytest.mark.parametrize("case", ["shifted", "stretched", "crossing", "random"])
def test_fsd_uni_continuous_matches_r(size: int, case: str) -> None:
    # n = 30_000 used to build a dense (2n x n) matrix and run out of memory.
    x, y = _pair(case, size, np.random.default_rng(size))
    for first, second in ((x, y), (y, x)):
        expected = nns("NNS.FSD.uni", first.tolist(), second.tolist(), "continuous")
        assert fsd_uni(first, second, "continuous") == int(np.asarray(expected).reshape(-1)[0])


_UNI_EDGE_CASES = [
    ([5.0, 5.0, 0.0, 4.0, 0.0, 2.0], [4.0] * 6),  # point mass at a threshold
    ([0.0, 0.0], [0.0, 3.0]),  # constant sample: 0/0 continuous ratio
    ([0.0, 0.0, 3.0, 3.0], [1.0, 1.0, 3.0, 0.0]),  # curves coincide at the knots
    ([1.0, 2.0], [1.0, 1.0, 2.0, 2.0]),  # same distribution, unequal lengths
    ([1.0, 2.0, 5.0], [0.0, 1.0, 1.0, 2.0, 3.0]),  # unequal lengths
]


@pytest.mark.parity
@pytest.mark.parametrize(("x", "y"), _UNI_EDGE_CASES)
def test_sd_uni_edge_cases_match_patched_r(x: list[float], y: list[float]) -> None:
    # R fixes 019fc1a (each sample walked with its own length), ec87e4a (no dominance
    # when the curves coincide) and 05f6d3d (continuous FSD 0/0 ratio = the degree-0
    # ratio, as LPM.ratio): the patched R .uni routines now agree with Python's and
    # with the bidirectional tests.
    a = np.asarray(x)
    b = np.asarray(y)
    for first, second in ((a, b), (b, a)):
        for fsd_type in ("continuous", "discrete"):
            expected = nns("NNS.FSD.uni", first.tolist(), second.tolist(), fsd_type)
            actual = fsd_uni(first, second, fsd_type)
            assert actual == int(np.asarray(expected).reshape(-1)[0])
        for order, function in (("SSD", ssd_uni), ("TSD", tsd_uni)):
            if first.size != second.size:
                expected = nns(f"NNS.{order}.uni", first.tolist(), second.tolist())
                expected_value = int(np.asarray(expected).reshape(-1)[0])
            else:
                # jsonlite turns two equal-length vectors into a matrix, so check
                # equal lengths against the bidirectional test instead.
                bidirectional = _sd_result_from_r(
                    f"NNS.{order}", first, second, f"X {order} Y", f"Y {order} X",
                    f"NO {order} EXISTS",
                )
                expected_value = 1 if bidirectional == 1 else 0
            assert function(first, second) == expected_value
    assert fsd(a, b, "continuous") == _sd_result_from_r_typed(a, b, "continuous")


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


@pytest.mark.parity
@pytest.mark.parametrize("junk_columns", [0, 71, 72])
def test_sd_efficient_set_tsd_with_extra_knots_matches_r(junk_columns: int) -> None:
    # z's knot at 0.16 lies where LPM2_x - LPM2_y peaks between the (x, y) knots;
    # R tests each pair on its own merged grid, so y stays dominated by x.
    x = [-0.85, -1.52, 0.68]
    y = [-0.21, -0.33, -1.67]
    z = [-3.0, 0.16, 0.2]
    junk = [[-10.0 - index, -9.0 - index, -8.0 - index] for index in range(junk_columns)]
    returns = np.column_stack([x, y, z, *junk])

    expected = nns("NNS.SD.efficient.set", returns.tolist(), 3, "discrete", False)

    assert _strings(expected) == [f"X_{index + 1}" for index in sd_efficient_set(returns, 3)]


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


def _sd_result_from_r_typed(x: np.ndarray, y: np.ndarray, type_: str) -> int:
    result = nns("NNS.FSD", x.tolist(), y.tolist(), type_, False)
    assert isinstance(result, str)
    return {"X FSD Y": 1, "Y FSD X": -1, "NO FSD EXISTS": 0}[result]


def _strings(value: RValue) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, np.ndarray):
        return [str(item) for item in value.tolist()]
    raise TypeError("Expected an R character vector.")


@pytest.mark.parity
@pytest.mark.parametrize(("degree", "r_degree"), [(1.5, 1.5), (2.9, 2.9), ("2", 2)])
def test_sd_efficient_set_truncates_degree_like_r(degree: object, r_degree: float) -> None:
    # R passes as.integer(degree) to its C++ routine, so 1.5 is FSD (not TSD).
    rng = np.random.default_rng(15)
    returns = rng.normal(size=(40, 8)) + np.linspace(-0.3, 0.3, 8)

    expected = nns("NNS.SD.efficient.set", returns.tolist(), r_degree, "discrete", False)

    actual = sd_efficient_set(returns, degree)  # type: ignore[arg-type]
    assert _strings(expected) == [f"X_{index + 1}" for index in actual]
    assert actual == sd_efficient_set(returns, int(float(degree)))  # type: ignore[arg-type]


@pytest.mark.parametrize("degree", [0.5, 4, "abc", None, float("nan")])
def test_sd_efficient_set_rejects_degrees_outside_one_to_three(degree: object) -> None:
    with pytest.raises(ValueError, match="degree must be 1, 2, or 3"):
        sd_efficient_set(np.zeros((3, 2)), degree)  # type: ignore[arg-type]


@pytest.mark.parity
@pytest.mark.parametrize("case", ["shifted", "stretched", "crossing", "random"])
@pytest.mark.parametrize("sizes", [(40, 40), (37, 61)])
def test_fsd_continuous_type_matches_r(case: str, sizes: tuple[int, int]) -> None:
    rng = np.random.default_rng(sizes[0] + sizes[1])
    x, y = _pair(case, max(sizes), rng)
    x = x[: sizes[0]]
    expected = _sd_result_from_r_typed(x, y, "continuous")

    assert fsd(x, y, type="continuous") == expected
    assert fsd(x, y, "discrete") == fsd(x, y)


@pytest.mark.parity
def test_fsd_unknown_type_warns_and_runs_r_branch() -> None:
    # R's NNS.FSD runs its continuous (degree-1) branch for anything but
    # "discrete"; NNS.FSD.uni and the efficient set run the discrete one.
    x = np.array([0.15, 1.55, -0.86, -0.8, 0.25, 0.33])
    y = np.array([-0.15, 1.17, 0.26, -0.04, -2.44, 1.19]) - 0.2
    expected = _sd_result_from_r_typed(x, y, "contin")

    with pytest.warns(UserWarning, match="type needs to be either"):
        assert fsd(x, y, type="contin") == expected == fsd(x, y, type="continuous")
    with pytest.warns(UserWarning, match="type needs to be either"):
        assert fsd_uni(x, y, type="contin") == fsd_uni(x, y)
    with pytest.warns(UserWarning, match="type needs to be either"):
        assert sd_efficient_set(np.column_stack([x, y]), 1, type="contin") == sd_efficient_set(
            np.column_stack([x, y]), 1
        )


def test_fsd_accepts_single_column_objects_like_r() -> None:
    x = np.array([2.0, 3.0, 4.0, 5.0])
    y = np.array([1.0, 2.0, 3.0])

    assert fsd(x[:, None], y[:, None]) == fsd(x, y) == 1
    assert fsd_uni(x[:, None], y) == 1
    with pytest.raises(ValueError, match="x must be a single-column object"):
        fsd(np.column_stack([x, x]), y)
