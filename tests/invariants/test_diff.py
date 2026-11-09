from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from nns import dy_d, dy_dx, nns_diff, nns_gravity, nns_reg


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


def test_dy_dx_without_eval_point_raises_clear_error() -> None:
    x = np.linspace(0.0, 1.0, 30)

    with pytest.raises(ValueError, match="eval_point must be provided"):
        dy_dx(x, x**2)
    # Raised before any input checks or fitting work.
    with pytest.raises(ValueError, match="eval_point must be provided"):
        dy_dx(x, x[:-1])


def test_dy_dx_numeric_small_sample_uses_distinct_bandwidths() -> None:
    # Below n = 16 every R bandwidth repeats and R's dy.dx fails; the port
    # falls back to the distinct bandwidths.
    x = np.linspace(0.0, 1.0, 12)

    result = dy_dx(x, 2.0 * x + 1.0, eval_point=np.array([0.3, 0.6]))

    assert isinstance(result, dict)
    np.testing.assert_allclose(result["first.derivative"], [2.0, 2.0], atol=1e-9)
    np.testing.assert_allclose(result["second.derivative"], [0.0, 0.0], atol=1e-9)


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


def test_dy_d_vectorized_wrt_mixed_three_column_input_is_rejected_like_r14() -> None:
    x = np.random.RandomState(1).randn(40, 3)
    y = x[:, 0] + x[:, 1] + x[:, 2]

    # R 14.0 dy_d_wrt.R rejects mixed derivatives for more than two
    # expanded predictors; silently falling back would conceal a bad request.
    with pytest.raises(
        ValueError,
        match="Mixed derivatives require two predictors and explicit matrix evaluation points",
    ):
        dy_d(x, y, wrt=np.array([1, 2]), eval_points="mean", mixed=True)


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
        match="Mixed derivatives require two predictors and explicit matrix evaluation points",
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


def test_dy_d_best_is_a_deprecated_alias_of_dy_d() -> None:
    import nns

    assert "dy_d_best" in nns.__all__
    x = np.random.RandomState(0).randn(60, 3)
    y = x[:, 0] + 2.0 * x[:, 1] - x[:, 2]
    with pytest.warns(DeprecationWarning, match="dy_d_best is deprecated"):
        alias = nns.dy_d_best(x, y, wrt=1, eval_points=np.zeros((1, 3)))
    direct = nns.dy_d(x, y, wrt=1, eval_points=np.zeros((1, 3)), messages=False)
    np.testing.assert_array_equal(alias["First"], direct["First"])


def test_dy_d_vectorized_wrt_tuple_returns_first_second() -> None:
    from nns import dy_d

    x = np.random.RandomState(0).randn(60, 3)
    y = x[:, 0] + 2.0 * x[:, 1] - x[:, 2]

    result = dy_d(x, y, wrt=np.array([1, 2, 3]), eval_points=np.zeros((1, 3)))

    assert result.keys() == {"First", "Second"}
    assert result["First"].shape == (1, 3)
    assert result["Second"].shape == (1, 3)
    assert np.all(np.isfinite(result["First"]))


def test_dy_d_single_wrt_returns_one_dimensional_first() -> None:
    from nns import dy_d

    x = np.random.RandomState(0).randn(60, 3)
    y = x[:, 0] + 2.0 * x[:, 1] - x[:, 2]

    # A single `wrt` returns 1-D arrays (one value per evaluation point).
    result = dy_d(x, y, wrt=1, eval_points=np.array([[0.0, 0.0, 0.0]]))

    assert result["First"].shape == (1,)
    assert np.all(np.isfinite(result["First"]))


def test_dy_d_two_column_mixed_returns_mixed() -> None:
    from nns import dy_d

    x = np.random.RandomState(1).randn(60, 2)
    y = x[:, 0] * x[:, 1]

    result = dy_d(x, y, wrt=np.array([1, 2]), eval_points=np.array([[0.0, 0.0]]), mixed=True)

    assert result.keys() == {"First", "Second", "Mixed"}
    assert result["First"].shape == (1, 2)
    assert result["Mixed"].shape == (1, 2)


def test_dy_d_is_deterministic() -> None:
    from nns import dy_d

    x = np.random.RandomState(0).randn(60, 3)
    y = x[:, 0] + 2.0 * x[:, 1] - x[:, 2]

    a = dy_d(x, y, wrt=np.array([1, 2, 3]), eval_points=np.zeros((1, 3)))
    b = dy_d(x, y, wrt=np.array([1, 2, 3]), eval_points=np.zeros((1, 3)))

    assert np.allclose(a["First"], b["First"])
    assert np.allclose(a["Second"], b["Second"])


def test_dy_d_matches_pinned_values() -> None:
    # Verified with pinned R14 7b07a837c130ba7220dbe16c584221508757d5ae.
    # The fitted derivative is not the exact slope of the generating equation.
    from nns import dy_d

    x = np.random.RandomState(0).randn(60, 3)
    y = x[:, 0] + 2.0 * x[:, 1] - x[:, 2]

    result = dy_d(x, y, wrt=np.array([1, 2, 3]), eval_points=np.zeros((1, 3)))

    np.testing.assert_allclose(
        np.ravel(result["First"]),
        np.array([0.5552936444351168, 1.4747889156529164, -0.6454516118551835]),
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


def test_mreg_predict_path_chunking_is_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    # The chunked/partial-selection NumPy path must equal a full-sort reference,
    # as does the native kernel (R's NNS_mreg_predict_path_v2_cpp).
    import nns._reg_engine as engine
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
            # R's kernel accumulates the weighted average sequentially.
            out[:, k - 1] = np.cumsum(ys[:, :k] * w, axis=1)[:, -1]
        return out

    rng = np.random.RandomState(1)
    rpm_x = rng.uniform(-2, 2, size=(300, 3))
    rpm_y = rng.uniform(size=300)
    xt = rng.uniform(-2, 2, size=(120, 3))
    mins, maxs = rpm_x.min(0), rpm_x.max(0)
    for kmax in (1, 5, 50, 300):
        a = reference(xt, rpm_x, rpm_y, kmax, "NNS", mins, maxs)
        native = _mreg_predict_path(xt, rpm_x, rpm_y, kmax, "NNS", mins, maxs)
        with monkeypatch.context() as patch:
            patch.setattr(engine, "native_fn", lambda name: None)
            b = _mreg_predict_path(xt, rpm_x, rpm_y, kmax, "NNS", mins, maxs)
        np.testing.assert_allclose(a, b, rtol=0, atol=0)
        np.testing.assert_array_equal(native, b)


def _band_means(x: np.ndarray, y: np.ndarray, p: float) -> list[float]:
    h_s = np.rint(np.exp(np.linspace(np.log(2.0), np.log(np.floor(np.sqrt(x.size))), 5)))
    h_step = nns_gravity(np.diff(np.unique(x))) * h_s
    lo = np.maximum(x.min(), p - h_step)
    hi = np.minimum(x.max(), p + h_step)
    point_est = np.concatenate((lo, np.full(5, p), hi))
    fit = nns_reg(x, y, point_est=point_est, point_only=True, smooth=True, plot=False, dist=None)
    est = np.asarray(fit["Point.est"], dtype=np.float64).reshape(3, -1).T
    run_1, run_2 = hi - p, p - lo
    rise_1, rise_2 = est[:, 2] - est[:, 1], est[:, 1] - est[:, 0]
    return [
        float(np.mean((rise_1 + rise_2) / (run_1 + run_2))),
        float(np.mean((rise_1 / run_1 - rise_2 / run_2) / ((run_1 + run_2) / 2.0))),
    ]


def _dy_dx_at(x: np.ndarray, p: float) -> list[float]:
    d = cast(dict[str, np.ndarray], dy_dx(x, x**2, eval_point=p))
    return [float(np.ravel(d["first.derivative"])[0]), float(np.ravel(d["second.derivative"])[0])]


@pytest.mark.parametrize("n", [10, 12])
def test_dy_dx_works_when_small_samples_repeat_a_bandwidth(n: int) -> None:
    first, second = _dy_dx_at(np.linspace(0.0, 3.0, n), 1.5)
    assert np.isfinite(first) and np.isfinite(second)
    assert first == pytest.approx(3.0, rel=0.05)


@pytest.mark.parametrize("n", [20, 30, 50, 10])
def test_dy_dx_averages_over_all_five_bandwidth_positions(n: int) -> None:
    # R dy.dx dropped repeated bandwidths (n < 49) and failed below n = 16.
    x = np.linspace(0.0, 3.0, n)
    np.testing.assert_allclose(_dy_dx_at(x, 1.5), _band_means(x, x**2, 1.5), rtol=1e-12)


def test_dy_dx_small_sample_matches_r_main() -> None:
    first, second = _dy_dx_at(np.linspace(0.0, 3.0, 10), 1.5)
    assert first == pytest.approx(3.0531359393852653, rel=1e-10)
    assert second == pytest.approx(0.1317797419408123, rel=1e-10)
