from __future__ import annotations

import numpy as np
import pytest

from pynns import nns_m_reg


def test_nns_m_reg_shapes_and_bounds() -> None:
    x1 = np.linspace(-2.0, 2.0, 100)
    x = np.column_stack((x1, np.sin(x1), np.cos(x1)))
    y = x1 + np.sin(x1)
    points = np.array([[0.0, 0.0, 1.0], [3.0, 0.0, 1.0]])

    result = nns_m_reg(x, y, order=2, n_best=1, point_est=points)

    assert np.isnan(result["R2"]) or 0.0 <= result["R2"] <= 1.0
    assert result["Fitted.xy"]["y"].shape == y.shape
    assert result["Fitted.xy"]["y.hat"].shape == y.shape
    assert result["Fitted.xy"]["NNS.ID"].shape == y.shape
    assert result["Point.est"].shape == (2,)
    assert result["RPM"]["y.hat"].size <= y.size
    assert all("." in item for item in result["Fitted.xy"]["NNS.ID"].astype(str))


def test_nns_m_reg_point_only_returns_point_est_and_rpm() -> None:
    x1 = np.linspace(-2.0, 2.0, 50)
    x = np.column_stack((x1, np.sin(x1)))
    y = x1 + np.sin(x1)

    result = nns_m_reg(x, y, order=1, n_best=1, point_est=np.array([[0.0, 0.0]]), point_only=True)

    assert set(result) == {"Point.est", "RPM"}
    assert result["Point.est"].shape == (1,)
    assert result["RPM"]["y.hat"].size <= y.size


def test_nns_m_reg_order_max_is_perfect_fit() -> None:
    x1 = np.linspace(-2.0, 2.0, 30)
    x = np.column_stack((x1, np.sin(x1)))
    y = x1 + np.sin(x1)

    result = nns_m_reg(x, y, order="max")

    np.testing.assert_allclose(result["Fitted.xy"]["y.hat"], y, atol=1e-12)
    assert result["R2"] == pytest.approx(1.0)


def test_nns_m_reg_confidence_interval_shapes() -> None:
    x1 = np.linspace(-2.0, 2.0, 50)
    x = np.column_stack((x1, np.sin(x1)))
    y = x1 + np.sin(x1)
    points = np.array([[0.0, 0.0], [1.0, np.sin(1.0)]])

    result = nns_m_reg(x, y, order=1, n_best=1, point_est=points, confidence_interval=0.95)

    assert result["Fitted.xy"]["conf.int.pos"].shape == y.shape
    assert result["Fitted.xy"]["conf.int.neg"].shape == y.shape
    assert result["pred.int"] is not None
    assert set(result["pred.int"]) == {"lower.pred.int", "upper.pred.int"}
    assert result["pred.int"]["lower.pred.int"].shape == (2,)
    assert result["pred.int"]["upper.pred.int"].shape == (2,)


def test_nns_m_reg_classification_outputs_numeric_codes() -> None:
    x1 = np.linspace(-2.0, 2.0, 20)
    x = np.column_stack((x1, np.sin(x1)))
    y = np.where(x1 < 0.0, 1.0, 2.0)

    result = nns_m_reg(x, y, type="class", point_est=x[:3], n_best=1)

    assert 0.0 <= result["R2"] <= 1.0
    assert set(result["Fitted.xy"]["y.hat"]).issubset(set(y))
    assert result["Point.est"] is not None
    assert set(result["Point.est"]).issubset(set(y))


def test_nns_m_reg_class_confidence_interval_keeps_raw_bounds() -> None:
    x1 = np.linspace(-2.0, 2.0, 20)
    x = np.column_stack((x1, np.sin(x1)))
    y = np.where(x1 < 0.0, 1.0, 2.0)

    result = nns_m_reg(x, y, type="class", point_est=x[:3], n_best=1, confidence_interval=0.95)

    assert result["Fitted.xy"]["conf.int.pos"].shape == y.shape
    assert result["Fitted.xy"]["conf.int.neg"].shape == y.shape
    assert result["pred.int"] is not None
    assert set(result["pred.int"]) == {"lower.pred.int", "upper.pred.int"}
    assert not np.allclose(
        result["pred.int"]["lower.pred.int"],
        np.round(result["pred.int"]["lower.pred.int"]),
    )
    assert set(result["Point.est"]).issubset(set(y))


def test_nns_m_reg_direct_factor_dummy_path_stays_rejected() -> None:
    x = np.array(
        [
            ["b", 0.0],
            ["a", 1.0],
            ["b", 2.0],
            ["c", 3.0],
        ],
        dtype=object,
    )
    y = np.array([2.0, 1.0, 3.0, 4.0])

    with pytest.raises(NotImplementedError, match=r"prepare_factor_predictors"):
        nns_m_reg(x, y, factor_2_dummy=True)
