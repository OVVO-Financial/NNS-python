"""Canonical vignette 07: Clustering and regression.

Source of truth:
    NNS/vignettes/NNSvignette_07_Clustering_and_Regression.Rmd
"""
from __future__ import annotations

import numpy as np

from nns import nns_part, nns_reg


def main() -> None:
    x = np.arange(-5.0, 5.05, 0.05)
    y = x**3

    full = nns_part(x, y, order=4, obs_req=0)
    x_only = nns_part(x, y, type="XONLY", order=4, obs_req=0)
    assert full["order"] == 4 and x_only["order"] == 4

    # X-only partition paths may contain multiple levels (for example q1121),
    # but every branch after the root must be a left/right label: 1 or 2.
    quadrant_paths = np.asarray(x_only["dt"]["quadrant"], dtype=str)
    assert all(set(path.removeprefix("q")) <= {"1", "2"} for path in quadrant_paths)

    points = np.array([-6.0, -2.0, 0.0, 2.0, 6.0])
    univariate = nns_reg(x, y, point_est=points, confidence_interval=None)
    assert np.asarray(univariate["Point.est"]).shape == points.shape

    rng = np.random.default_rng(123)
    design = rng.uniform(-2.0, 2.0, size=(160, 2))
    target = design[:, 0] ** 3 + 3.0 * design[:, 1] - design[:, 1] ** 3 - 3.0 * design[:, 0]
    multivariate = nns_reg(design, target, point_est=design[:10], order="max")
    smoothed = nns_reg(x, y + rng.normal(scale=2.0, size=x.size), point_est=points, smooth=True)

    print("joint partition regression points:", len(full["regression.points"]["x"]))
    print("X-only partition regression points:", len(x_only["regression.points"]["x"]))
    print("univariate R2:", round(float(univariate["R2"]), 4))
    print("univariate point estimates:", np.round(univariate["Point.est"], 4))
    print("multivariate point estimates:", np.round(multivariate["Point.est"], 4))
    print("smoothed point estimates:", np.round(smoothed["Point.est"], 4))


if __name__ == "__main__":
    main()
