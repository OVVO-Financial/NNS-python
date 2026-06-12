from __future__ import annotations

import numpy as np

from pynns import nns_m_reg, nns_reg, nns_stack


def main() -> None:
    x = np.linspace(-2.0, 2.0, 72, dtype=np.float64)
    second_feature = np.cos(2.0 * x)
    features = np.column_stack((x, second_feature))
    y = np.where(x < -0.6, 1.0, np.where(x > 0.65, 3.0, 2.0))

    one_dim_points = np.array([-1.0, 0.0, 1.25], dtype=np.float64)
    one_dim = nns_reg(
        x,
        y,
        type="class",
        point_est=one_dim_points,
        confidence_interval=None,
    )

    two_dim_points = np.array(
        [
            [-1.25, np.cos(-2.5)],
            [0.1, np.cos(0.2)],
            [1.2, np.cos(2.4)],
        ],
        dtype=np.float64,
    )
    multi = nns_m_reg(
        features,
        y,
        type="class",
        point_est=two_dim_points,
        confidence_interval=None,
    )

    # Stacking uses a simple cross-validation split to choose between candidate methods.
    stacked = nns_stack(
        features,
        y,
        two_dim_points,
        type="class",
        method=(1, 2),
        folds=1,
        cv_size=0.25,
        random_seed=7,
    )

    one_dim_predictions = np.asarray(one_dim["Point.est"], dtype=np.float64)
    multi_predictions = np.asarray(multi["Point.est"], dtype=np.float64)
    stack_predictions = np.asarray(stacked["stack"], dtype=np.float64)
    classes = set(np.unique(y))

    assert one_dim_predictions.shape == one_dim_points.shape
    assert multi_predictions.shape == (two_dim_points.shape[0],)
    assert stack_predictions.shape == (two_dim_points.shape[0],)
    assert set(one_dim_predictions).issubset(classes)
    assert set(multi_predictions).issubset(classes)
    assert set(stack_predictions).issubset(classes)
    assert 0.0 <= multi["R2"] <= 1.0

    print("1D points:", one_dim_points)
    print("1D class predictions:", one_dim_predictions)
    print("2D points:")
    print(two_dim_points)
    print("multivariate class predictions:", multi_predictions)
    print("stacked class predictions:", stack_predictions)
    print("training accuracy proxy:", multi["R2"])


if __name__ == "__main__":
    main()
