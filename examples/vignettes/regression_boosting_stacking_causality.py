"""Vignette 06 — Regression, boosting, stacking, and causality.

Python translation of the R NNS "Clustering and Regression" vignette
(``tools/NNS/vignettes/NNSvignette_07_Clustering_and_Regression.Rmd``) plus the
causality example from the Overview vignette.

The boost and stack examples use the deterministic numeric design verified
against live R NNS 13.0 (PR #3): stack returns ``reg``/``dim.red``/``stack``
and boost returns ``results``/``feature.weights``/``feature.frequency``.

Run with::

    python examples/vignettes/regression_boosting_stacking_causality.py
"""

from __future__ import annotations

import numpy as np

from nns import nns_boost, nns_causation, nns_reg, nns_stack


def main() -> None:
    # Nonlinear univariate regression.
    x = np.arange(-5.0, 5.05, 0.05)
    y = x**3
    reg = nns_reg(x, y, point_est=np.array([-2.0, 0.0, 2.0]))
    assert 0.0 <= reg["R2"] <= 1.0
    assert reg["Point.est"].shape == (3,)

    # Deterministic numeric design used for stack and boost.
    xb = np.linspace(-2.0, 2.0, 30)
    variable = np.column_stack((xb, np.sin(xb), np.cos(xb)))
    target = xb + np.sin(xb) + 0.25 * np.cos(xb)
    point = variable[:5]

    stack = nns_stack(variable, target, point, method=(1, 2), cv_size=0.25, folds=1)
    for key in ("reg", "dim.red", "stack"):
        assert np.asarray(stack[key]).shape == (5,)

    boost = nns_boost(
        variable, target, point,
        learner_trials=10, cv_size=0.25, depth=None, feature_importance=False,
    )
    assert set(boost) == {"results", "pred.int", "feature.weights", "feature.frequency"}
    assert np.asarray(boost["results"]).shape == (5,)

    # Causality is directional: the conditional causation of x given y differs
    # from y given x. The net-direction summary key is named C(x--->y) or
    # C(y--->x) depending on which direction dominates, so read the stable
    # directional keys here.
    rng = np.random.default_rng(1)
    driver = np.cumsum(rng.normal(size=200))
    response = np.concatenate(([0.0, 0.0], driver[:-2])) + rng.normal(scale=0.1, size=200)
    caus = nns_causation(driver, response)
    cxy = float(caus["Causation.x.given.y"])
    cyx = float(caus["Causation.y.given.x"])
    net_key = next(k for k in caus if k.startswith("C(") and "--->" in k)

    print("regression R2:", round(reg["R2"], 4))
    print("regression point estimates:", np.round(reg["Point.est"], 4))
    print("stack reg:    ", np.round(np.asarray(stack["reg"]), 6))
    print("stack dim.red:", np.round(np.asarray(stack["dim.red"]), 6))
    print("stack stack:  ", np.round(np.asarray(stack["stack"]), 6))
    print("boost results:", np.round(np.asarray(boost["results"]), 6))
    print("boost feature.weights:", np.asarray(boost["feature.weights"]))
    print("boost feature.frequency:", np.asarray(boost["feature.frequency"]))
    print("causation x|y:", round(cxy, 4), " y|x:", round(cyx, 4))
    print(f"net causation {net_key}:", round(float(caus[net_key]), 4))


if __name__ == "__main__":
    main()
