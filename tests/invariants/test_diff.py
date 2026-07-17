from __future__ import annotations

import numpy as np
import pytest

from nns import dy_d, dy_dx, nns_diff


def test_nns_diff_constant_derivative_is_zero() -> None:
    result = nns_diff(lambda x: 12.0, 3.0)

    assert result["DERIVATIVE"] == pytest.approx(0.0)


def test_nns_diff_identity_derivative_is_one() -> None:
    result = nns_diff(lambda x: x, -2.0)

    assert result["DERIVATIVE"] == pytest.approx(1.0)


def test_nns_diff_smooth_function_derivative_has_bounded_error() -> None:
    point = 1.25
    result = nns_diff(np.sin, point)

    assert result["DERIVATIVE"] == pytest.approx(np.cos(point), abs=1e-6)


def test_dy_dx_numeric_eval_point_returns_derivative_table() -> None:
    x = np.linspace(-2.0, 2.0, 24)
    y = x + np.sin(x)

    result = dy_dx(x, y, eval_point=np.array([-1.0, 0.0, 1.0]))

    assert isinstance(result, dict)
    assert list(result) == ["eval.point", "first.derivative", "second.derivative"]
    assert all(value.shape == (3,) for value in result.values())
    assert np.all(np.isfinite(result["first.derivative"]))


def test_dy_d_vectorized_wrt_obs_is_implemented() -> None:
    x = np.random.RandomState(0).randn(40, 3)
    y = x[:, 0] + 2.0 * x[:, 1] - x[:, 2]

    result = dy_d(x, y, wrt=np.array([1, 2]), eval_points="obs")

    assert result.keys() == {"First", "Second"}
    assert result["First"].shape == (40, 2)
    assert result["Second"].shape == (40, 2)


def test_dy_d_vectorized_wrt_mixed_three_column_input_falls_back_to_first_second() -> None:
    x = np.random.RandomState(1).randn(40, 3)
    y = x[:, 0] + x[:, 1] + x[:, 2]

    result = dy_d(x, y, wrt=np.array([1, 2]), eval_points="mean", mixed=True)

    assert result.keys() == {"First", "Second"}
    assert result["First"].shape == (1, 2)
    assert result["Second"].shape == (1, 2)


def test_dy_d_vectorized_wrt_mixed_two_column_input_returns_mixed() -> None:
    x = np.random.RandomState(1).randn(40, 2)
    y = x[:, 0] + x[:, 1]

    result = dy_d(x, y, wrt=np.array([1, 2]), eval_points="mean", mixed=True)

    assert result.keys() == {"First", "Second", "Mixed"}
    assert result["First"].shape == (1, 2)
    assert result["Second"].shape == (1, 2)
    assert result["Mixed"].shape == (1, 2)


def test_dy_d_vectorized_wrt_obs_mixed_uses_pointwise_python_shape() -> None:
    x = np.random.RandomState(3).randn(24, 2)
    y = x[:, 0] ** 2 + x[:, 1]

    result = dy_d(x, y, wrt=np.array([1, 2]), eval_points="obs", mixed=True)

    assert result.keys() == {"First", "Second", "Mixed"}
    assert result["First"].shape == (24, 2)
    assert result["Second"].shape == (24, 2)
    assert result["Mixed"].shape == (24, 2)
    assert np.all(np.isfinite(result["Mixed"]))


def test_dy_d_vectorized_wrt_apd_mixed_remains_invalid() -> None:
    x = np.random.RandomState(1).randn(40, 2)
    y = x[:, 0] + x[:, 1]

    with pytest.raises(
        ValueError,
        match="Mixed derivatives require a complete two-predictor evaluation tuple",
    ):
        dy_d(x, y, wrt=np.array([1, 2]), eval_points="apd", mixed=True)


def test_dy_d_vectorized_wrt_mean_is_implemented() -> None:
    x = np.random.RandomState(2).randn(40, 2)
    y = x[:, 0] * 2.0 - x[:, 1]
    result = dy_d(x, y, wrt=np.array([1, 2]), eval_points="mean")

    assert isinstance(result, dict)
    assert result.keys() == {"First", "Second"}
    assert result["First"].shape == (1, 2)
    assert result["Second"].shape == (1, 2)


def test_dy_d_point_modes_preserve_linear_slope_direction() -> None:
    rng = np.random.default_rng(0)
    x = np.column_stack((rng.uniform(-2.0, 2.0, 120), rng.uniform(-1.0, 1.0, 120)))
    positive = 2.0 * x[:, 0] + 0.5 * x[:, 1]
    negative = -2.0 * x[:, 0] + 0.5 * x[:, 1]

    for eval_points in ("mean", "median", "last"):
        assert dy_d(x, positive, wrt=1, eval_points=eval_points)["First"][0] > 0.0
        assert dy_d(x, negative, wrt=1, eval_points=eval_points)["First"][0] < 0.0


def test_dy_d_best_is_exported() -> None:
    import nns

    assert "dy_d_best" in nns.__all__
    assert callable(nns.dy_d_best)


def test_dy_d_best_vectorized_wrt_tuple_returns_first_second() -> None:
    from nns import dy_d_best

    x = np.random.RandomState(0).randn(60, 3)
    y = x[:, 0] + 2.0 * x[:, 1] - x[:, 2]

    result = dy_d_best(x, y, wrt=np.array([1, 2, 3]), eval_points=np.zeros((1, 3)))

    assert result.keys() == {"First", "Second"}
    assert result["First"].shape == (1, 3)
    assert result["Second"].shape == (1, 3)
    assert np.all(np.isfinite(result["First"]))


def test_dy_d_best_single_wrt_returns_one_dimensional_first() -> None:
    from nns import dy_d_best

    x = np.random.RandomState(0).randn(60, 3)
    y = x[:, 0] + 2.0 * x[:, 1] - x[:, 2]

    # A single `wrt` returns 1-D arrays (one value per evaluation point).
    result = dy_d_best(x, y, wrt=1, eval_points=np.array([[0.0, 0.0, 0.0]]))

    assert result["First"].shape == (1,)
    assert np.all(np.isfinite(result["First"]))


def test_dy_d_best_two_column_mixed_returns_mixed() -> None:
    from nns import dy_d_best

    x = np.random.RandomState(1).randn(60, 2)
    y = x[:, 0] * x[:, 1]

    result = dy_d_best(x, y, wrt=np.array([1, 2]), eval_points=np.array([[0.0, 0.0]]), mixed=True)

    assert result.keys() == {"First", "Second", "Mixed"}
    assert result["First"].shape == (1, 2)
    assert result["Mixed"].shape == (1, 2)


def test_dy_d_best_is_deterministic() -> None:
    from nns import dy_d_best

    x = np.random.RandomState(0).randn(60, 3)
    y = x[:, 0] + 2.0 * x[:, 1] - x[:, 2]

    a = dy_d_best(x, y, wrt=np.array([1, 2, 3]), eval_points=np.zeros((1, 3)))
    b = dy_d_best(x, y, wrt=np.array([1, 2, 3]), eval_points=np.zeros((1, 3)))

    assert np.allclose(a["First"], b["First"])
    assert np.allclose(a["Second"], b["Second"])


def test_dy_d_best_matches_pinned_values() -> None:
    # Golden values guard the v0.5.7 finite-difference design against regressions.
    from nns import dy_d_best

    x = np.random.RandomState(0).randn(60, 3)
    y = x[:, 0] + 2.0 * x[:, 1] - x[:, 2]

    result = dy_d_best(x, y, wrt=np.array([1, 2, 3]), eval_points=np.zeros((1, 3)))

    np.testing.assert_allclose(
        np.ravel(result["First"]),
        np.array([0.754963, 0.792164, 0.699243]),
        rtol=0.0,
        atol=1e-4,
    )


def test_stable_topk_matches_full_stable_sort() -> None:
    # Partial neighbor selection must be byte-identical to a full stable argsort,
    # including heavy distance ties (e.g. duplicated 1-D synthetic coordinates).
    from nns._reg_engine import _stable_topk

    rng = np.random.RandomState(0)
    for _ in range(300):
        m = rng.randint(1, 6)
        n = rng.randint(2, 40)
        k = rng.randint(1, n + 1)
        d = rng.randint(0, 4, size=(m, n)).astype(float)  # ties on purpose
        expected = np.argsort(d, axis=1, kind="stable")[:, :k]
        assert np.array_equal(_stable_topk(d, k), expected)


def test_mreg_predict_path_chunking_is_exact() -> None:
    # The chunked/partial-selection path must equal a full-sort reference.
    from nns._reg_engine import _mreg_distances, _mreg_ensemble_weights, _mreg_predict_path
    from nns.central_tendencies import nns_gravity

    def reference(xtest, rpm_x, rpm_yhat, kmax, dist, mins, maxs):
        n = rpm_x.shape[0]
        kmax = min(int(kmax), n)
        m = xtest.shape[0]
        out = np.empty((m, kmax))
        d = _mreg_distances(rpm_x, xtest, dist, mins, maxs)
        idx = np.argsort(d, axis=1, kind="stable")
        ds = np.take_along_axis(d, idx, axis=1)
        ys = rpm_yhat[idx]
        dmin = d.min(axis=1, keepdims=True)
        for i in range(m):
            t = rpm_yhat[d[i] == dmin[i, 0]]
            out[i, 0] = float(nns_gravity(t[np.isfinite(t)]))
        for k in range(2, kmax + 1):
            w = _mreg_ensemble_weights(ds[:, :k])
            out[:, k - 1] = np.sum(ys[:, :k] * w, axis=1)
        return out

    rng = np.random.RandomState(1)
    rpm_x = rng.uniform(-2, 2, size=(300, 3))
    rpm_y = rng.uniform(size=300)
    xt = rng.uniform(-2, 2, size=(120, 3))
    mins, maxs = rpm_x.min(0), rpm_x.max(0)
    for kmax in (1, 5, 50, 300):
        a = reference(xt, rpm_x, rpm_y, kmax, "NNS", mins, maxs)
        b = _mreg_predict_path(xt, rpm_x, rpm_y, kmax, "NNS", mins, maxs)
        np.testing.assert_allclose(a, b, rtol=0, atol=0)
