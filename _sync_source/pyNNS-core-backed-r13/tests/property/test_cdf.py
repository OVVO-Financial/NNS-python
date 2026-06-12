from __future__ import annotations

from typing import cast

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pynns import nns_cdf

finite_floats = st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False)


@given(
    x=arrays(np.float64, st.integers(3, 80), elements=finite_floats),
    degree=st.sampled_from([0.0, 1.0, 2.0, 3.0]),
    type_name=st.sampled_from(["cdf", "survival", "cumulative hazard"]),
)
def test_nns_cdf_univariate_shape_and_range_properties(
    x: np.ndarray,
    degree: float,
    type_name: str,
) -> None:
    result = nns_cdf(x, degree=degree, type=type_name)
    function = cast(dict[str, np.ndarray], result["Function"])
    values = next(value for key, value in function.items() if key != "x")

    assert function["x"].shape == values.shape
    assert np.asarray(result["target.value"]).size == 0
    if degree == 0.0 and type_name in {"cdf", "survival"}:
        assert np.all(values >= 0.0)
        assert np.all(values <= 1.0)


@given(
    rows=st.integers(5, 30),
    cols=st.integers(2, 4),
    degree=st.sampled_from([0.0, 1.0, 2.0, 3.0]),
    type_name=st.sampled_from(["cdf", "survival", "cumulative hazard"]),
)
def test_nns_cdf_multivariate_shape_properties(
    rows: int,
    cols: int,
    degree: float,
    type_name: str,
) -> None:
    values = np.linspace(-2.0, 2.0, rows * cols, dtype=np.float64).reshape(rows, cols)
    values = values + np.arange(cols, dtype=np.float64)

    result = nns_cdf(values, degree=degree, type=type_name)
    function = cast(dict[str, np.ndarray], result["Function"])

    assert function["CDF"].shape == (rows,)
    assert list(function) == [*(f"V{index + 1}" for index in range(cols)), "CDF"]
