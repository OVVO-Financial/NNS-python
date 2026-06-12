from __future__ import annotations

from typing import cast

import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pynns import nns_m_reg
from pynns.part import NoiseReduction
from pynns.regression import Order

matrix_arrays = arrays(
    dtype=np.float64,
    shape=st.tuples(st.integers(min_value=12, max_value=60), st.integers(min_value=2, max_value=3)),
    elements=st.floats(
        min_value=-1e3,
        max_value=1e3,
        allow_nan=False,
        allow_infinity=False,
        width=64,
    ),
)


@given(
    matrix_arrays,
    st.sampled_from([None, 1, 2, "max"]),
    st.sampled_from(["off", "mean", "median"]),
)
def test_nns_m_reg_shape_invariants_hold(
    x: np.ndarray,
    order: int | str | None,
    noise: str,
) -> None:
    y = 0.5 * x[:, 0] - 0.25 * x[:, 1]
    assume(np.unique(y).size > 1)
    assume(all(np.unique(x[:, col]).size > 1 for col in range(x.shape[1])))

    result = nns_m_reg(x, y, order=cast(Order, order), noise_reduction=cast(NoiseReduction, noise))

    assert np.isnan(result["R2"]) or -1e-12 <= result["R2"] <= 1.0 + 1e-12
    assert result["Fitted.xy"]["y"].shape == (x.shape[0],)
    assert result["Fitted.xy"]["y.hat"].shape == (x.shape[0],)
    assert result["Fitted.xy"]["NNS.ID"].shape == (x.shape[0],)
    assert result["RPM"]["y.hat"].size <= x.shape[0]


@given(matrix_arrays, st.sampled_from([0.8, 0.95]))
def test_nns_m_reg_confidence_interval_shape_invariants_hold(
    x: np.ndarray,
    confidence_interval: float,
) -> None:
    y = 0.5 * x[:, 0] - 0.25 * x[:, 1]
    assume(np.unique(y).size > 1)
    assume(all(np.unique(x[:, col]).size > 1 for col in range(x.shape[1])))

    result = nns_m_reg(
        x,
        y,
        order=1,
        n_best=1,
        point_est=x[:3],
        confidence_interval=confidence_interval,
    )

    assert result["Fitted.xy"]["conf.int.pos"].shape == (x.shape[0],)
    assert result["Fitted.xy"]["conf.int.neg"].shape == (x.shape[0],)
    assert result["pred.int"] is not None
    assert result["pred.int"]["lower.pred.int"].shape == (3,)
    assert result["pred.int"]["upper.pred.int"].shape == (3,)


@given(matrix_arrays, st.integers(min_value=2, max_value=4))
def test_nns_m_reg_classification_bounds_hold(x: np.ndarray, n_classes: int) -> None:
    assume(all(np.unique(x[:, col]).size > 1 for col in range(x.shape[1])))
    classes = (np.arange(x.shape[0]) % n_classes + 1).astype(np.float64)

    result = nns_m_reg(x, classes, order=1, n_best=1, type="class", point_est=x[:3])

    assert 0.0 <= result["R2"] <= 1.0
    assert set(result["Fitted.xy"]["y.hat"]).issubset(set(classes))
    assert result["Point.est"] is not None
    assert set(result["Point.est"]).issubset(set(classes))


@given(matrix_arrays, st.integers(min_value=2, max_value=4), st.sampled_from([0.8, 0.95]))
def test_nns_m_reg_class_confidence_interval_shape_invariants_hold(
    x: np.ndarray,
    n_classes: int,
    confidence_interval: float,
) -> None:
    assume(all(np.unique(x[:, col]).size > 1 for col in range(x.shape[1])))
    classes = (np.arange(x.shape[0]) % n_classes + 1).astype(np.float64)

    result = nns_m_reg(
        x,
        classes,
        order=1,
        n_best=1,
        type="class",
        point_est=x[:3],
        confidence_interval=confidence_interval,
    )

    assert result["Fitted.xy"]["conf.int.pos"].shape == (x.shape[0],)
    assert result["Fitted.xy"]["conf.int.neg"].shape == (x.shape[0],)
    assert result["pred.int"] is not None
    assert set(result["pred.int"]) == {"lower.pred.int", "upper.pred.int"}
    assert result["pred.int"]["lower.pred.int"].shape == (3,)
    assert result["pred.int"]["upper.pred.int"].shape == (3,)
    assert np.all(np.isfinite(result["pred.int"]["lower.pred.int"]))
    assert np.all(np.isfinite(result["pred.int"]["upper.pred.int"]))
