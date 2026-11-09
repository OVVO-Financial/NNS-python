"""NNS.reg argument and dispatch contracts of R 14.0.

These follow R/Regression.R and R/Classification.R where the port used to
differ: a class response is classified even with ``multivariate.call = TRUE``;
``threshold = NULL``/``NA`` means no screening and ``n.best = NA`` the default,
so values reported by ``NNS.stack`` can be passed back; the dimension-reduction
classifier reports ``n.best = NA`` and integer row ids.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from _r import nns_frame_call, r_column
from _tolerances import COMPOUND

import nns

pytestmark = pytest.mark.parity


def _num(values: Any) -> dict[str, Any]:
    return r_column([float(v) for v in np.asarray(values, dtype=np.float64).ravel()], "numeric")


def _design(seed: int, n: int = 60) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    y = x[:, 0] + x[:, 1] ** 2 + rng.normal(size=n) * 0.3
    classes = (x[:, 0] + x[:, 2] > 0).astype(float)
    return x, y, classes


def test_class_response_with_multivariate_call_is_classified_as_in_r() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=80)
    y = (x + rng.normal(size=80) * 0.5 > 0).astype(float)
    point = np.array([-0.5, 0.25, 1.5])
    expected = nns_frame_call(
        "reg14.mvcall.class",
        "r <- NNS.reg(x, y, point.est = p, multivariate.call = TRUE, plot = FALSE); "
        "list(names = names(r), point = r$Point.est, probs = r$Point.probabilities[, 2])",
        {"x": _num(x), "y": _num(y), "p": _num(point)},
    )
    actual = nns.nns_reg(x, y, point_est=point, multivariate_call=True)
    assert list(actual) == list(expected["names"])
    np.testing.assert_array_equal(actual["Point.est"], expected["point"])
    np.testing.assert_allclose(
        actual["Point.probabilities"]["1"], expected["probs"], rtol=0, atol=COMPOUND
    )


@pytest.mark.parametrize("threshold", [None, float("nan")])
@pytest.mark.parametrize("response", ["numeric", "class"])
def test_threshold_null_or_na_means_no_screening(threshold: float | None, response: str) -> None:
    x, y, classes = _design(4)
    target = y if response == "numeric" else classes
    arg = "NULL" if threshold is None else "NA"
    expected = nns_frame_call(
        "reg14.threshold.na",
        f"NNS.reg(matrix(v, ncol = 3), y, dim.red.method = 'cor', point.est = matrix(p, ncol = 3), "
        f"threshold = {arg}, plot = FALSE)$Point.est",
        {"v": _num(x.ravel(order="F")), "y": _num(target), "p": _num(x[:5].ravel(order="F"))},
    )
    actual = nns.nns_reg(x, target, dim_red_method="cor", point_est=x[:5], threshold=threshold)
    np.testing.assert_allclose(actual["Point.est"], np.asarray(expected), rtol=0, atol=COMPOUND)


def test_n_best_na_selects_the_default_as_in_r() -> None:
    x, y, _ = _design(5)
    expected = nns_frame_call(
        "reg14.nbest.na",
        "NNS.reg(matrix(v, ncol = 3), y, point.est = matrix(p, ncol = 3), n.best = NA, "
        "plot = FALSE)$Point.est",
        {"v": _num(x.ravel(order="F")), "y": _num(y), "p": _num(x[:5].ravel(order="F"))},
    )
    actual = nns.nns_reg(x, y, point_est=x[:5], n_best=float("nan"))
    np.testing.assert_allclose(actual["Point.est"], np.asarray(expected), rtol=0, atol=COMPOUND)


def test_per_class_thresholds_follow_r() -> None:
    x, _, classes = _design(6)
    expected = nns_frame_call(
        "reg14.threshold.class",
        "r <- NNS.reg(matrix(v, ncol = 3), y, dim.red.method = 'cor', "
        "point.est = matrix(p, ncol = 3), threshold = c(0.05, 0.3), plot = FALSE); "
        "list(point = r$Point.est, probs = as.numeric(r$Point.probabilities))",
        {"v": _num(x.ravel(order="F")), "y": _num(classes), "p": _num(x[:5].ravel(order="F"))},
    )
    actual = nns.nns_reg(x, classes, dim_red_method="cor", point_est=x[:5], threshold=[0.05, 0.3])
    np.testing.assert_array_equal(actual["Point.est"], expected["point"])
    probs = np.column_stack([actual["Point.probabilities"][k] for k in ("0", "1")])
    np.testing.assert_allclose(probs.ravel(order="F"), expected["probs"], rtol=0, atol=COMPOUND)


def test_vector_threshold_is_rejected_for_regression_as_in_r() -> None:
    x, y, _ = _design(7)
    with pytest.raises(ValueError, match="single number for regression"):
        nns.nns_reg(x, y, dim_red_method="cor", threshold=[0.1, 0.2])


def test_dimension_reduction_classifier_reports_na_n_best_and_row_ids() -> None:
    x, _, classes = _design(8)
    expected = nns_frame_call(
        "reg14.dimred.class.meta",
        "r <- NNS.reg(matrix(v, ncol = 3), y, dim.red.method = 'cor', plot = FALSE); "
        "list(nbest = r$n.best, ids = r$Fitted.xy$NNS.ID)",
        {"v": _num(x.ravel(order="F")), "y": _num(classes)},
    )
    actual = nns.nns_reg(x, classes, dim_red_method="cor")
    assert np.isnan(np.asarray(expected["nbest"], dtype=np.float64)).all()
    assert np.isnan(actual["n.best"])
    np.testing.assert_array_equal(actual["Fitted.xy"]["NNS.ID"], expected["ids"])
