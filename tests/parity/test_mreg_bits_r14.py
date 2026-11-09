"""R 14.0 parity, to the last bit, for multivariate regression and classification.

* R's C++ ``NNS_mreg_predict_v2_cpp`` weights the k nearest regression points
  with an eight-component ensemble whose Student-t and normal terms are R's own
  nmath ``dt`` and ``dnorm4``, summed sequentially; the native overlay and the
  NumPy fallback both reproduce it (and its libm ``exp``).
* R's classifier uses the R-level ``.nns_mreg_ensemble_weights`` instead:
  long-double ``sum()``/``var()``, ``dexp(rate = 1 / k)``, ``ranks^(-2)`` and
  ``rowSums`` distances, which differ from the C++ kernel from three
  predictors on.
* ``nns_m_reg`` is R's internal ``NNS.M.reg``, whose ``categorical`` argument
  defaults to FALSE; ``NNS.reg`` sets it when every predictor is categorical.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest
from _r import nns_frame_call, r_column

import nns

pytestmark = pytest.mark.parity

reg_engine = importlib.import_module("nns._reg_engine")


def _num(values: Any) -> dict[str, Any]:
    return r_column([float(v) for v in np.asarray(values, dtype=np.float64).ravel()], "numeric")


def _design(seed: int, n: int, p: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, p))
    y = x[:, 0] * x[:, 1] + rng.normal(size=n) * 0.2
    return x, y


@pytest.mark.parametrize("native", [True, False])
@pytest.mark.parametrize(("dist", "n_best", "r_args"), [
    ("NNS", None, ""), ("L2", "all", ", order = 2, n.best = 'all', dist = 'L2'"),
])
def test_multivariate_fit_matches_r_bitwise(
    monkeypatch: pytest.MonkeyPatch, native: bool, dist: str, n_best: str | None, r_args: str
) -> None:
    x, y = _design(6, 60, 3)
    expected = nns_frame_call(
        "mreg14.bits",
        f"r <- NNS.reg(matrix(v, ncol = 3), y, point.est = matrix(v, ncol = 3)[1:5, ]{r_args}, "
        "plot = FALSE); list(fit = r$Fitted.xy$y.hat, point = r$Point.est)",
        {"v": _num(x.ravel(order="F")), "y": _num(y)},
        exact=True,
    )
    if not native:
        monkeypatch.setattr(reg_engine, "native_fn", lambda name: None)
    order = 2 if n_best == "all" else None
    actual = nns.nns_reg(x, y, point_est=x[:5], order=order, n_best=n_best, dist=dist)
    np.testing.assert_array_equal(actual["Fitted.xy"]["y.hat"], np.asarray(expected["fit"]))
    np.testing.assert_array_equal(actual["Point.est"], np.asarray(expected["point"]))


def test_three_predictor_class_probabilities_match_r_bitwise() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(150, 3))
    y = (x[:, 0] + x[:, 1] > 0).astype(float)
    expected = nns_frame_call(
        "mreg14.class_bits",
        "r <- NNS.reg(matrix(v, ncol = 3), y, point.est = matrix(v, ncol = 3)[1:4, ], "
        "type = 'CLASS', plot = FALSE); list(fit = unname(r$Fitted.probabilities), "
        "point = unname(r$Point.probabilities))",
        {"v": _num(x.ravel(order="F")), "y": _num(y)},
        exact=True,
    )
    actual = nns.nns_reg(x, y, point_est=x[:4], type="CLASS")

    def matrix(probabilities: Any) -> np.ndarray:
        # The port keys each class's column by its label, in R's column order.
        return np.column_stack([np.asarray(v, dtype=np.float64) for v in probabilities.values()])

    np.testing.assert_array_equal(
        matrix(actual["Fitted.probabilities"]), np.asarray(expected["fit"], dtype=np.float64)
    )
    np.testing.assert_array_equal(
        matrix(actual["Point.probabilities"]), np.asarray(expected["point"], dtype=np.float64)
    )


@pytest.mark.parametrize("categorical", [False, True])
def test_nns_m_reg_categorical_follows_r_internal_default(categorical: bool) -> None:
    import pandas as pd  # type: ignore[import-untyped]

    rng = np.random.default_rng(8)
    a = rng.choice(["p", "q", "r"], 40)
    b = rng.choice(["u", "v"], 40)
    y = (pd.Categorical(a).codes * 2.0 + pd.Categorical(b).codes) + rng.normal(size=40) * 0.3
    flag = "TRUE" if categorical else "FALSE"
    expected = nns_frame_call(
        "mreg14.categorical",
        f"d <- data.frame(a = factor(a), b = factor(b)); NNS:::NNS.M.reg(d, y, "
        f"point.est = d[1:5, ], categorical = {flag})$Point.est",
        {"a": r_column(a.tolist(), "character"), "b": r_column(b.tolist(), "character"),
         "y": _num(y)},
        exact=True,
    )
    frame = pd.DataFrame({"a": pd.Categorical(a), "b": pd.Categorical(b)})
    actual = nns.nns_m_reg(frame, y, point_est=frame.iloc[:5], categorical=categorical)
    np.testing.assert_array_equal(actual["Point.est"], np.asarray(expected, dtype=np.float64))
