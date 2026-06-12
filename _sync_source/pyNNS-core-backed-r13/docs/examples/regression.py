from __future__ import annotations

import numpy as np

from pynns import nns_m_reg, nns_part, nns_reg


def main() -> None:
    x = np.linspace(-3.0, 3.0, 80, dtype=np.float64)
    y = np.sin(x) + 0.2 * x
    points = np.array([-1.5, 0.0, 1.5], dtype=np.float64)

    fit = nns_reg(x, y, point_est=points, confidence_interval=None)
    partition = nns_part(x, y, order=3, obs_req=6)

    features = np.column_stack((x, x**2))
    multi_points = np.array([[-2.0, 4.0], [0.0, 0.0], [2.0, 4.0]], dtype=np.float64)
    multi_fit = nns_m_reg(
        features,
        y,
        point_est=multi_points,
        order=3,
        n_best=2,
        confidence_interval=None,
    )

    fitted = fit["Fitted.xy"]
    assert fitted["x"].shape == x.shape
    assert fitted["y.hat"].shape == y.shape
    assert fit["Point.est"].shape == points.shape
    assert 0.0 <= fit["R2"] <= 1.0
    assert partition["dt"]["quadrant"].shape == x.shape
    assert multi_fit["Point.est"].shape == (multi_points.shape[0],)
    assert 0.0 <= multi_fit["R2"] <= 1.0

    print("univariate R2:", fit["R2"])
    print("univariate point estimates:")
    print(np.column_stack((points, fit["Point.est"])))
    print("partition order:", partition["order"])
    print("first partition labels:", partition["dt"]["quadrant"][:8])
    print("multivariate R2:", multi_fit["R2"])
    print("multivariate point estimates:")
    print(np.column_stack((multi_points, multi_fit["Point.est"])))


if __name__ == "__main__":
    main()
