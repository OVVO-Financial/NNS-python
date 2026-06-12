from collections.abc import Sequence
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

class FastLmResult(TypedDict):
    coef: Sequence[float]
    fitted_values: Sequence[float]
    residuals: Sequence[float]
    df_residual: int

class FastLmMultResult(TypedDict):
    coefficients: Sequence[float]
    fitted_values: Sequence[float]
    residuals: Sequence[float]
    r_squared: float

PMMatrixResult = TypedDict(
    "PMMatrixResult",
    {
        "cupm": Sequence[float],
        "dupm": Sequence[float],
        "dlpm": Sequence[float],
        "clpm": Sequence[float],
        "cov.matrix": Sequence[float],
        "dim": int,
    },
)

class DummyMatrixResult(TypedDict):
    data: Sequence[float]
    names: Sequence[str]
    nrow: int
    ncol: int

class TimeSeriesVectorsResult(TypedDict):
    series: Sequence[Sequence[float]]
    index: Sequence[Sequence[int]]

class ForecastVectorsResult(TypedDict):
    series: Sequence[Sequence[float]]
    index: Sequence[Sequence[int]]
    forecast_values: Sequence[Sequence[float]]
    forecast_index: Sequence[Sequence[int]]

class StochSupResult(TypedDict):
    p_gt: float
    p_tie: float
    p_star: float

def lpm(
    degree: float,
    target: float | NDArray[np.float64],
    x: Sequence[float] | NDArray[np.float64],
) -> float | Sequence[float]: ...
def upm(
    degree: float,
    target: float | NDArray[np.float64],
    x: Sequence[float] | NDArray[np.float64],
) -> float | Sequence[float]: ...
def lpm_v(
    degree: float, target: NDArray[np.float64], x: NDArray[np.float64]
) -> Sequence[float]: ...
def upm_v(
    degree: float, target: NDArray[np.float64], x: NDArray[np.float64]
) -> Sequence[float]: ...
def lpm_ratio_v(
    degree: float, target: NDArray[np.float64], x: NDArray[np.float64]
) -> Sequence[float]: ...
def upm_ratio_v(
    degree: float, target: NDArray[np.float64], x: NDArray[np.float64]
) -> Sequence[float]: ...
def co_lpm(
    degree_x: float,
    degree_y: float,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    target_x: float,
    target_y: float,
) -> float: ...
def co_upm(
    degree_x: float,
    degree_y: float,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    target_x: float,
    target_y: float,
) -> float: ...
def d_lpm(
    degree_lpm: float,
    degree_upm: float,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    target_x: float,
    target_y: float,
) -> float: ...
def d_upm(
    degree_lpm: float,
    degree_upm: float,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    target_x: float,
    target_y: float,
) -> float: ...
def co_lpm_v(
    degree_x: float,
    degree_y: float,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    target_x: NDArray[np.float64],
    target_y: NDArray[np.float64],
) -> Sequence[float]: ...
def co_upm_v(
    degree_x: float,
    degree_y: float,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    target_x: NDArray[np.float64],
    target_y: NDArray[np.float64],
) -> Sequence[float]: ...
def d_lpm_v(
    degree_lpm: float,
    degree_upm: float,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    target_x: NDArray[np.float64],
    target_y: NDArray[np.float64],
) -> Sequence[float]: ...
def d_upm_v(
    degree_lpm: float,
    degree_upm: float,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    target_x: NDArray[np.float64],
    target_y: NDArray[np.float64],
) -> Sequence[float]: ...
def clpm_nd(
    data: NDArray[np.float64],
    n: int,
    d: int,
    target: NDArray[np.float64],
    degree: float,
    norm: bool,
) -> float: ...
def cupm_nd(
    data: NDArray[np.float64],
    n: int,
    d: int,
    target: NDArray[np.float64],
    degree: float,
    norm: bool,
) -> float: ...
def dpm_nd(
    data: NDArray[np.float64],
    n: int,
    d: int,
    target: NDArray[np.float64],
    degree: float,
    norm: bool,
) -> float: ...
def clpm_nd_batch(
    data: NDArray[np.float64],
    n: int,
    d: int,
    targets: NDArray[np.float64],
    n_targets: int,
    degree: float,
    norm: bool,
) -> Sequence[float]: ...
def pm_matrix(
    degree_lpm: float,
    degree_upm: float,
    target: NDArray[np.float64],
    variable: NDArray[np.float64],
    n: int,
    d: int,
    pop_adj: bool,
    norm: bool,
) -> PMMatrixResult: ...
def fast_lm(x: NDArray[np.float64], y: NDArray[np.float64]) -> FastLmResult: ...
def fast_lm_mult(
    x: NDArray[np.float64], y: NDArray[np.float64], n: int, p: int
) -> FastLmMultResult: ...
def is_discrete(x: NDArray[np.float64]) -> bool: ...
def vec_sd(x: NDArray[np.float64]) -> float: ...
def col_sd(x: NDArray[np.float64], n: int, p: int) -> Sequence[float]: ...
def factor_2_dummy(
    codes: Sequence[int], levels: Sequence[str]
) -> DummyMatrixResult: ...
def factor_2_dummy_fr(
    codes: Sequence[int], levels: Sequence[str]
) -> DummyMatrixResult: ...
def generate_vectors(
    x: NDArray[np.float64], lags: NDArray[np.int32]
) -> TimeSeriesVectorsResult: ...
def generate_lin_vectors(
    x: NDArray[np.float64], l: int, h: int
) -> ForecastVectorsResult: ...
def gravity(x: NDArray[np.float64], discrete: bool) -> float: ...
def mode(x: NDArray[np.float64], discrete: bool, multi: bool) -> Sequence[float]: ...
def stochastic_superiority(
    x: NDArray[np.float64], y: NDArray[np.float64]
) -> StochSupResult: ...
