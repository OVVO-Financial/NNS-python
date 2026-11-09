"""NNS.stack / NNS.boost behaviour that depends on R's exact arithmetic and dispatch.

- Regression stacks pick the blend weight from ``seq(0, 1, by = 0.01)`` by the
  out-of-fold objective, taking the middle of exactly tied weights. When both
  components fit the same coordinate the 101 scores differ only by rounding,
  so the default objective must be R's ``sum((predicted - actual)^2)`` (a
  long-double ``sum()``) and the grid R's ``seq()`` values.
- Nominal cells use ``findInterval(..., rightmost.closed = TRUE)``: only a value
  equal to the last breakpoint joins the final interval, so observations above
  the last cell mean (the breakpoints are means) keep a cell of their own.
- NNS.boost passes ``type = type`` to its learners: a numeric 0/1 response is
  classified by each base NNS.reg even though the boost itself is a regression.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from _r import nns_frame_call, r_column
from _tolerances import COMPOUND

from nns import nns_boost, nns_stack

pytestmark = pytest.mark.parity


def _num(values: Any) -> dict[str, Any]:
    return r_column([float(v) for v in np.asarray(values, dtype=np.float64).ravel()], "numeric")


def _frame(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> dict[str, Any]:
    return {"v": _num(x.ravel(order="F")), "y": _num(y), "t": _num(z.ravel(order="F"))}


def _three_level(seed: int, n: int = 60) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    y = np.digitize(x[:, 0] + rng.normal(size=n) * 0.3, [-0.5, 0.5]).astype(float) + 1
    return x, y, rng.normal(size=(7, 2))


@pytest.mark.parametrize("seed", [101, 102, 103, 104])
def test_regression_stack_blend_weight_ties_follow_r(seed: int) -> None:
    x, y, z = _three_level(seed)
    expected = nns_frame_call(
        "stack14.blend",
        "r <- NNS.stack(matrix(v, ncol = 2), y, IVs.test = matrix(t, ncol = 2), folds = 2, "
        "status = FALSE, ncores = 1); list(w = unname(r$weights), stack = r$stack, "
        "obj = c(r$OBJfn.reg, r$OBJfn.dim.red))",
        _frame(x, y, z),
        exact=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actual = nns_stack(x, y, z, folds=2, ncores=1)
    np.testing.assert_array_equal(
        [actual["weights"]["reg"], actual["weights"]["dim.red"]], expected["w"]
    )
    np.testing.assert_array_equal([actual["OBJfn.reg"], actual["OBJfn.dim.red"]], expected["obj"])
    np.testing.assert_allclose(actual["stack"], expected["stack"], rtol=0, atol=COMPOUND)


@pytest.mark.parametrize("seed", [3, 35, 46, 95])
def test_balanced_class_stack_cells_follow_r(seed: int) -> None:
    rng = np.random.default_rng(200 + seed)
    x = rng.normal(size=(60, 2))
    y = np.digitize(x[:, 0] + rng.normal(size=60) * 0.3, [-0.5, 0.5]).astype(float) + 1
    z = rng.normal(size=(7, 2))
    expected = nns_frame_call(
        "stack14.class.balanced",
        f"r <- NNS.stack(matrix(v, ncol = 2), y, IVs.test = matrix(t, ncol = 2), folds = 2, "
        f"balance = TRUE, type = 'CLASS', seed = {seed}L, status = FALSE, ncores = 1); "
        "list(reg = r$reg, dim = r$dim.red, stack = r$stack, "
        "rp = as.numeric(r$reg.probabilities), sp = as.numeric(r$probabilities))",
        _frame(x, y, z),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actual = nns_stack(x, y, z, folds=2, balance=True, type="CLASS", seed=seed, ncores=1)
    np.testing.assert_array_equal(actual["reg"], expected["reg"])
    np.testing.assert_array_equal(actual["dim.red"], expected["dim"])
    np.testing.assert_array_equal(actual["stack"], expected["stack"])
    reg_probs = np.asarray(actual["reg.probabilities"], dtype=np.float64)
    probs = np.asarray(actual["probabilities"], dtype=np.float64)
    np.testing.assert_allclose(reg_probs.ravel(order="F"), expected["rp"], rtol=0, atol=COMPOUND)
    np.testing.assert_allclose(probs.ravel(order="F"), expected["sp"], rtol=0, atol=COMPOUND)


@pytest.mark.parametrize("seed", [123, 9])
def test_boost_numeric_binary_response_learners_classify_as_in_r(seed: int) -> None:
    rng = np.random.default_rng(300 + seed)
    x = rng.normal(size=(80, 3))
    y = (x[:, 0] - x[:, 1] + rng.normal(size=80) * 0.5 > 0).astype(float)
    z = rng.normal(size=(6, 3))
    expected = nns_frame_call(
        "boost14.binary.numeric",
        f"r <- NNS.boost(matrix(v, ncol = 3), y, IVs.test = matrix(t, ncol = 3), "
        f"learner.trials = 10, seed = {seed}L, status = FALSE, feature.importance = FALSE); "
        "list(results = r$results, weights = unname(r$feature.weights[c('V1', 'V2', 'V3')]), "
        "frequency = unname(r$feature.frequency[c('V1', 'V2', 'V3')]))",
        _frame(x, y, z),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actual = nns_boost(x, y, z, learner_trials=10, seed=seed)
    np.testing.assert_allclose(actual["results"], expected["results"], rtol=0, atol=COMPOUND)
    weights = [actual["feature.weights"].get(k, 0.0) for k in ("V1", "V2", "V3")]
    frequency = [actual["feature.frequency"].get(k, 0.0) for k in ("V1", "V2", "V3")]
    np.testing.assert_allclose(weights, np.nan_to_num(expected["weights"]), rtol=0, atol=COMPOUND)
    np.testing.assert_allclose(frequency, np.nan_to_num(expected["frequency"]), rtol=0, atol=0)
