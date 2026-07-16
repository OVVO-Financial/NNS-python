from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pytest

from nns import nns_boost, nns_reg, nns_stack
from nns._reg_engine import _mreg_distances, nns_reg_engine


def _assert_numeric_equal(a: Any, b: Any) -> None:
    if isinstance(a, dict):
        assert isinstance(b, dict)
        assert a.keys() == b.keys()
        for key in a:
            _assert_numeric_equal(a[key], b[key])
    elif isinstance(a, np.ndarray):
        np.testing.assert_allclose(a, b, equal_nan=True)
    elif isinstance(a, (float, int, np.floating, np.integer)) or a is None:
        assert a == pytest.approx(b) if a is not None else b is None


def test_public_signatures_default_dist_none() -> None:
    assert inspect.signature(nns_reg).parameters["dist"].default is None
    assert inspect.signature(nns_stack).parameters["dist"].default is None
    assert inspect.signature(nns_boost).parameters["dist"].default is None


def test_nns_reg_none_nns_and_lowercase_are_equivalent() -> None:
    x = np.array([[0.0, 0.0], [1.0, 3.0], [2.0, 1.0], [4.0, 5.0], [7.0, 2.0]])
    y = np.array([0.0, 1.0, 1.5, 2.2, 3.0])
    point = np.array([[1.5, 1.0], [5.0, 4.0]])
    base = nns_reg(x, y, point_est=point, point_only=True, dist=None)
    explicit = nns_reg(x, y, point_est=point, point_only=True, dist="NNS")
    lower = nns_reg(x, y, point_est=point, point_only=True, dist="nns")
    np.testing.assert_allclose(base["Point.est"], explicit["Point.est"])
    np.testing.assert_allclose(base["Point.est"], lower["Point.est"])
    assert base["dist"] == explicit["dist"] == lower["dist"] == "NNS"


def test_native_distance_formula_differs_from_l1_and_l2() -> None:
    rpm = np.array([[0.0, 0.0], [2.0, 3.0], [4.0, 1.0], [8.0, 6.0]])
    xtest = np.array([[1.0, 4.0]])
    mins = rpm.min(axis=0)
    maxs = rpm.max(axis=0)
    nns = _mreg_distances(rpm, xtest, "NNS", mins, maxs)
    l1 = _mreg_distances(rpm, xtest, "L1", mins, maxs)
    l2 = _mreg_distances(rpm, xtest, "L2", mins, maxs)
    assert not np.allclose(nns, l1)
    assert not np.allclose(nns, l2)
    z = (xtest[:, None, :] - rpm[None, :, :]) / (maxs - mins)
    np.testing.assert_allclose(nns, np.sum(np.abs(z) + z**2, axis=2))


def test_invalid_distance_raises_clear_value_error() -> None:
    with pytest.raises(ValueError, match=r"dist.*NNS.*L1.*L2.*FACTOR"):
        nns_reg_engine([[0.0], [1.0]], [0.0, 1.0], dist="bad")


def test_explicit_l1_l2_factor_still_accepted() -> None:
    x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 1.0], [3.0, 0.0]])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    for dist in ("L1", "L2", "FACTOR"):
        out = nns_reg(x, y, point_est=np.array([[1.5, 0.5]]), point_only=True, dist=dist)
        assert out["dist"] == dist
        assert np.isfinite(out["Point.est"]).all()


def test_nns_stack_none_and_nns_are_equivalent() -> None:
    x = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 5.0]])
    y = np.array([0.0, 1.0, 1.4, 2.2, 2.8, 3.5])
    kwargs = dict(method=(1,), stack=False, folds=2, seed=42, status=False)
    none = nns_stack(x, y, x[:2], dist=None, **kwargs)
    explicit = nns_stack(x, y, x[:2], dist="NNS", **kwargs)
    np.testing.assert_allclose(none["reg"], explicit["reg"])


def test_nns_boost_none_and_nns_are_equivalent_and_l2_propagates() -> None:
    x = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 5.0]])
    y = np.array([0.0, 1.0, 1.4, 2.2, 2.8, 3.5])
    kwargs = dict(learner_trials=2, epochs=1, seed=7, status=False)
    none = nns_boost(x, y, x[:2], dist=None, **kwargs)
    explicit = nns_boost(x, y, x[:2], dist="NNS", **kwargs)
    l2 = nns_boost(x, y, x[:2], dist="L2", **kwargs)
    np.testing.assert_allclose(none["results"], explicit["results"])
    assert np.isfinite(l2["results"]).all()


def test_nns_predict_path_matches_batch_predict_for_native_distance() -> None:
    from nns._reg_engine import _mreg_predict, _mreg_predict_path

    rpm_x = np.array([[0.0, 0.0], [2.0, 3.0], [4.0, 1.0], [8.0, 6.0]])
    rpm_y = np.array([0.0, 1.5, 2.0, 4.0])
    xtest = np.array([[1.0, 4.0], [6.0, 2.0]])
    mins = rpm_x.min(axis=0)
    maxs = rpm_x.max(axis=0)
    path = _mreg_predict_path(xtest, rpm_x, rpm_y, 3, "NNS", mins, maxs)
    for k in range(1, 4):
        batch = _mreg_predict(xtest, rpm_x, rpm_y, k, "NNS", mins, maxs, False)
        np.testing.assert_allclose(path[:, k - 1], batch)


def test_nns_stack_native_scoring_matches_nns_alias_and_differs_from_l2_objective() -> None:
    x = np.array(
        [
            [0.0, 0.0],
            [1.0, 4.0],
            [2.0, 1.0],
            [3.0, 8.0],
            [4.0, 2.0],
            [5.0, 7.0],
            [6.0, 3.0],
            [7.0, 6.0],
        ]
    )
    y = np.array([0.0, 1.0, 1.4, 2.2, 2.8, 3.5, 3.7, 4.1])
    kwargs = dict(method=(1,), stack=False, folds=2, cv_size=0.25, seed=3, status=False)
    native = nns_stack(x, y, x[:3], dist=None, **kwargs)
    alias = nns_stack(x, y, x[:3], dist="NNS", **kwargs)
    l2 = nns_stack(x, y, x[:3], dist="L2", **kwargs)
    assert native["NNS.reg.n.best"] == alias["NNS.reg.n.best"]
    assert native["OBJfn.reg"] == pytest.approx(alias["OBJfn.reg"])
    np.testing.assert_allclose(native["reg"], alias["reg"])
    assert native["OBJfn.reg"] != pytest.approx(l2["OBJfn.reg"])
    assert not np.allclose(native["reg"], l2["reg"])


def test_nns_boost_l2_propagates_to_learner_trials_and_final_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nns.boost as boost_mod

    seen_engine: list[str | None] = []
    seen_stack: list[str | None] = []
    real_engine = boost_mod.nns_reg_engine
    real_stack = boost_mod.nns_stack

    def wrapped_engine(*args: Any, **kwargs: Any) -> Any:
        seen_engine.append(kwargs.get("dist"))
        return real_engine(*args, **kwargs)

    def wrapped_stack(*args: Any, **kwargs: Any) -> Any:
        seen_stack.append(kwargs.get("dist"))
        return real_stack(*args, **kwargs)

    monkeypatch.setattr(boost_mod, "nns_reg_engine", wrapped_engine)
    monkeypatch.setattr(boost_mod, "nns_stack", wrapped_stack)
    x = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 5.0]])
    y = np.array([0.0, 1.0, 1.4, 2.2, 2.8, 3.5])
    nns_boost(x, y, x[:2], dist="L2", learner_trials=2, epochs=1, seed=7, status=False)
    assert seen_engine
    assert seen_stack
    assert set(seen_engine) == {"L2"}
    assert set(seen_stack) == {"L2"}
