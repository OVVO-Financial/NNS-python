from __future__ import annotations

from typing import cast

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pynns import nns_reg
from pynns.part import NoiseReduction
from pynns.regression import Order

finite_arrays = arrays(
    dtype=np.float64,
    shape=st.integers(min_value=12, max_value=100),
    elements=st.floats(
        min_value=-1e4,
        max_value=1e4,
        allow_nan=False,
        allow_infinity=False,
        width=64,
    ),
)

finite_matrices = arrays(
    dtype=np.float64,
    shape=st.tuples(st.integers(min_value=12, max_value=60), st.integers(min_value=2, max_value=4)),
    elements=st.floats(
        min_value=-1e3,
        max_value=1e3,
        allow_nan=False,
        allow_infinity=False,
        width=64,
    ),
)


@given(
    finite_arrays,
    finite_arrays,
    st.sampled_from([None, 1, 2, 3, "max"]),
    st.sampled_from(["off", "mean", "median", "mode", "mode_class"]),
)
def test_nns_reg_univariate_shape_invariants_hold(
    x: np.ndarray,
    y: np.ndarray,
    order: int | str | None,
    noise: str,
) -> None:
    size = min(x.size, y.size)
    x = x[:size]
    y = y[:size]
    assume(np.unique(x).size > 1)
    assume(np.unique(y).size > 1)

    result = nns_reg(x, y, order=cast(Order, order), noise_reduction=cast(NoiseReduction, noise))

    assert np.isnan(result["R2"]) or -1e-12 <= result["R2"] <= 1.0 + 1e-12
    assert result["SE"] >= 0.0
    assert result["Fitted.xy"]["x"].shape == (size,)
    assert result["Fitted.xy"]["y"].shape == (size,)
    assert result["Fitted.xy"]["y.hat"].shape == (size,)
    assert result["Fitted.xy"]["gradient"].shape == (size,)
    assert result["derivative"]["Coefficient"].size == result["derivative"]["X.Lower.Range"].size
    assert result["derivative"]["Coefficient"].size == result["derivative"]["X.Upper.Range"].size
    assert result["regression.points"]["x"].size == result["regression.points"]["y"].size


@given(
    finite_matrices,
    st.sampled_from(["cor", "NNS.dep", "equal", [1.0, 0.5, 0.25, 0.125]]),
)
def test_nns_reg_dim_red_shape_invariants_hold(
    x: np.ndarray,
    method: str | list[float],
) -> None:
    y = 0.5 * x[:, 0] - 0.25 * x[:, 1]
    assume(np.unique(y).size > 1)
    assume(all(np.unique(x[:, col]).size > 1 for col in range(x.shape[1])))
    if isinstance(method, list):
        method = method[: x.shape[1]]
        assume(len(method) == x.shape[1])

    result = nns_reg(x, y, dim_red_method=method)

    assert np.isnan(result["R2"]) or -1e-12 <= result["R2"] <= 1.0 + 1e-12
    assert result["x.star"]["x"].shape == (x.shape[0],)
    assert result["equation"]["Variable"].shape == (x.shape[1] + 1,)
    assert result["equation"]["Coefficient"].shape == (x.shape[1] + 1,)
    assert result["Fitted.xy"]["x"].shape == (x.shape[0],)


@given(
    finite_arrays,
    finite_arrays,
    st.sampled_from([0.8, 0.95]),
)
def test_nns_reg_confidence_interval_shape_invariants_hold(
    x: np.ndarray,
    y: np.ndarray,
    confidence_interval: float,
) -> None:
    size = min(x.size, y.size)
    x = x[:size]
    y = y[:size]
    assume(np.unique(x).size > 1)
    assume(np.unique(y).size > 1)
    points = np.array([float(np.min(x)), float(np.mean(x)), float(np.max(x))])

    result = nns_reg(x, y, order=1, point_est=points, confidence_interval=confidence_interval)

    assert result["Fitted.xy"]["conf.int.pos"].shape == (size,)
    assert result["Fitted.xy"]["conf.int.neg"].shape == (size,)
    assert result["pred.int"] is not None
    assert set(result["pred.int"]) == {"pred.int.neg", "pred.int.pos"}
    assert result["pred.int"]["pred.int.neg"].shape == result["pred.int"]["pred.int.pos"].shape


@given(
    finite_arrays,
    finite_arrays,
)
@settings(max_examples=5)
def test_nns_reg_smooth_shape_invariants_hold(x: np.ndarray, y: np.ndarray) -> None:
    size = min(x.size, y.size, 40)
    x = x[:size]
    y = y[:size]
    assume(np.unique(x).size > 3)
    assume(np.unique(y).size > 1)
    points = np.array([float(np.min(x)), float(np.mean(x)), float(np.max(x))])

    result = nns_reg(x, y, order=1, point_est=points, smooth=True)

    assert result["Fitted.xy"]["y.hat"].shape == (size,)
    assert result["Point.est"].shape == points.shape
    assert result["regression.points"]["x"].size == result["regression.points"]["y"].size
    assert np.all(np.isfinite(result["Fitted.xy"]["y.hat"]))
    assert np.all(np.isfinite(result["Point.est"]))


@given(
    finite_arrays,
    st.integers(min_value=2, max_value=4),
)
def test_nns_reg_classification_bounds_hold(x: np.ndarray, n_classes: int) -> None:
    assume(np.unique(x).size > 1)
    classes = (np.arange(x.size) % n_classes + 1).astype(np.float64)
    points = np.array([float(np.min(x)), float(np.mean(x)), float(np.max(x))])

    result = nns_reg(x, classes, order=1, type="class", point_est=points)

    assert result["Prediction.Accuracy"] is not None
    assert set(result["Fitted.xy"]["y.hat"]).issubset(set(classes))
    assert set(result["Point.est"]).issubset(set(classes))


@given(
    finite_arrays,
    st.integers(min_value=2, max_value=4),
    st.sampled_from([0.8, 0.95]),
)
def test_nns_reg_class_confidence_interval_shape_invariants_hold(
    x: np.ndarray,
    n_classes: int,
    confidence_interval: float,
) -> None:
    assume(np.unique(x).size > 1)
    classes = (np.arange(x.size) % n_classes + 1).astype(np.float64)
    points = np.array([float(np.min(x)), float(np.mean(x)), float(np.max(x))])

    result = nns_reg(
        x,
        classes,
        order=1,
        type="class",
        point_est=points,
        confidence_interval=confidence_interval,
    )

    assert result["Fitted.xy"]["conf.int.pos"].shape == (x.shape[0],)
    assert result["Fitted.xy"]["conf.int.neg"].shape == (x.shape[0],)
    assert result["pred.int"] is not None
    assert set(result["pred.int"]) == {"pred.int.neg", "pred.int.pos"}
    for values in result["pred.int"].values():
        assert np.all(np.isfinite(values))
        np.testing.assert_allclose(values, np.round(values))
