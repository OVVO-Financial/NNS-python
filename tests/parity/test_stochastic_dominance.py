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
    # LPM2_x - LPM2_y peaks at +0.039 at t = 0.16, between the (x, y) knots, and z
    # adds a knot there. R fix d4fc624 checks that interior maximum for every pair,
    # so x no longer TSD-dominates y, whatever other columns are present.
    x = [-0.85, -1.52, 0.68]
    y = [-0.21, -0.33, -1.67]
    z = [-3.0, 0.16, 0.2]
    junk = [[-10.0 - index, -9.0 - index, -8.0 - index] for index in range(junk_columns)]
    returns = np.column_stack([x, y, z, *junk])

    expected = nns("NNS.SD.efficient.set", returns.tolist(), 3, "discrete", False)

    assert _strings(expected) == [f"X_{index + 1}" for index in sd_efficient_set(returns, 3)]


# Pairs where LPM2_x - LPM2_y (TSD) or the continuous FSD ratio difference peaks
# above 0 strictly between the merged sample points while it is <= 0 at every point:
# 13.3 reported X dominates Y (work/SDA/tsd_knots_counterex.json, fsdc_knots_verify.R).
_INTERIOR_TSD_PAIRS = [
    ([-0.85, -1.52, 0.68], [-0.21, -0.33, -1.67]),
    ([-0.15, -0.58, -0.47, 2.42, 0.42], [-0.99, 1.19, 1.35, 0.03, -0.41]),
    ([1.37, -0.67, -0.63, 0.91], [-0.21, 0.94, -0.11, -0.89]),
    ([-1.28, -0.99, -0.27, 0.29, 2.58], [1.08, 0.92, -1.31, -1.51, 0.43]),
    ([-0.16, -1.02, 2.53], [0.37, -1.14, 0.45]),
    ([1.07, -1.42, -0.41, 0.85, -0.19], [-0.22, -0.05, 0.78, -1.52, 0.25]),
    ([-0.67, 1.56, -0.78, 1.04], [1.42, -0.04, -0.32, -0.98]),
    ([0.57, 0.61, -0.49, 0.4, -1.23], [-0.24, -0.17, -1.28, 0.74, 0.33]),
]
_INTERIOR_FSD_CONTINUOUS_PAIR = (
    [0.15, 1.55, -0.86, -0.8, 0.25, 0.33],
    [-0.15, 1.17, 0.26, -0.04, -2.44, 1.19],
)


@pytest.mark.parity
@pytest.mark.parametrize(("x", "y"), _INTERIOR_TSD_PAIRS)
def test_tsd_checks_the_curves_between_sample_points(x: list[float], y: list[float]) -> None:
    # R fix d4fc624: the pairwise test also evaluates the vertex of the piecewise
    # quadratic LPM2 difference on each interval between knots.
    a = np.asarray(x)
    b = np.asarray(y)
    grid = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), 20_001)

    def lpm2(values: np.ndarray) -> np.ndarray:
        return np.mean(np.maximum(grid[:, None] - values[None, :], 0.0) ** 2, axis=1)

    assert np.max(lpm2(a) - lpm2(b)) > 1e-3

    assert tsd(a, b) == _sd_result_from_r("NNS.TSD", a, b, "X TSD Y", "Y TSD X", "NO TSD EXISTS")
    assert tsd_uni(a, b) == 0
    assert sd_efficient_set(np.column_stack([a, b]), 3) == [0, 1]
    expected = nns("NNS.SD.efficient.set", np.column_stack([a, b]).tolist(), 3, "discrete", False)
    assert _strings(expected) == ["X_1", "X_2"]


@pytest.mark.parity
def test_continuous_fsd_checks_the_curves_between_sample_points() -> None:
    # R fix d4fc624: the continuous ratio L1/(L1+U1) is linear-fractional between
    # knots; the difference peaks at +0.003 at t = 0.50 although it is <= 0 at them.
    a = np.asarray(_INTERIOR_FSD_CONTINUOUS_PAIR[0])
    b = np.asarray(_INTERIOR_FSD_CONTINUOUS_PAIR[1])

    assert fsd(a, b, type="continuous") == _sd_result_from_r_typed(a, b, "continuous") == 0
    for first, second in ((a, b), (b, a)):
        expected = nns("NNS.FSD.uni", first.tolist(), second.tolist(), "continuous")
        assert fsd_uni(first, second, "continuous") == int(np.asarray(expected).reshape(-1)[0])
    assert fsd_uni(a, b, "continuous") == 0
    returns = np.column_stack([a, b])
    expected_set = nns("NNS.SD.efficient.set", returns.tolist(), 1, "continuous", False)
    assert _strings(expected_set) == [
        f"X_{index + 1}" for index in sd_efficient_set(returns, 1, type="continuous")
    ]


def _mean_preserving_spread(shift: float) -> tuple[np.ndarray, np.ndarray]:
    from scipy.stats import norm  # type: ignore[import-untyped]

    base = norm.ppf((np.arange(1, 101) - 0.5) / 100.0)
    return base + shift, 2.0 * base + shift


@pytest.mark.parity
@pytest.mark.parametrize(
    ("x", "y"),
    [
        # x = qnorm(ppoints(100)), y = 2x: an exact mean-preserving spread, so X SSD Y
        # and X TSD Y; 13.3 said NO SSD (LPM1 difference +8.9e-16, mean -3.7e-18).
        _mean_preserving_spread(0.0),
        _mean_preserving_spread(0.37),
        # 0.1 + 0.7 = 0.7999999999999999: 13.3 said NO SSD, the same data x10 X SSD Y.
        (np.array([0.1, 0.7]), np.array([0.0, 0.8])),
        (np.array([1.0, 7.0]), np.array([0.0, 8.0])),
    ],
)
def test_sd_rounding_noise_does_not_decide_dominance(x: np.ndarray, y: np.ndarray) -> None:
    # R fix 2ca5aca: differences up to 64 * eps * S^d are ties, and the bidirectional
    # tests take their verdict from the pairwise (.uni) test in both directions.
    assert ssd(x, y) == 1 == ssd_uni(x, y)
    assert tsd(x, y) == 1 == tsd_uni(x, y)
    assert ssd_uni(y, x) == tsd_uni(y, x) == 0
    assert ssd(x, y) == _sd_result_from_r("NNS.SSD", x, y, "X SSD Y", "Y SSD X", "NO SSD EXISTS")
    assert tsd(x, y) == _sd_result_from_r("NNS.TSD", x, y, "X TSD Y", "Y TSD X", "NO TSD EXISTS")
    for degree in (2, 3):
        returns = np.column_stack([y, x])
        expected = nns("NNS.SD.efficient.set", returns.tolist(), degree, "discrete", False)
        actual = sd_efficient_set(returns, degree)
        assert _strings(expected) == [f"X_{index + 1}" for index in actual]


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
