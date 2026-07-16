"""Canonical vignette 08: Classification.

Source of truth:
    NNS/vignettes/NNSvignette_08_Classification.Rmd

Demonstrates the same three public classification paths as R: ``nns_reg`` as
the base learner, ``nns_boost`` as the resampled ensemble, and ``nns_stack``
as the cross-validated regression/dimension-reduction ensemble. Class codes
start at 1, matching the R contract.
"""
from __future__ import annotations

import numpy as np

from nns import nns_boost, nns_reg, nns_stack


def _three_class_data(seed: int = 123) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = np.array([[-3.0, -2.0, 0.0, 1.0], [0.0, 2.5, 3.0, -1.0], [3.0, -1.0, -2.5, 2.0]])
    train = np.vstack([rng.normal(center, 0.45, size=(35, 4)) for center in centers])
    test = np.vstack([rng.normal(center, 0.45, size=(5, 4)) for center in centers])
    y_train = np.repeat(np.arange(1, 4), 35).astype(float)
    y_test = np.repeat(np.arange(1, 4), 5).astype(float)
    return train, y_train, test, y_test


def main() -> None:
    x_train, y_train, x_test, y_test = _three_class_data()

    reg = nns_reg(x_train, y_train, type="CLASS", point_est=x_test)
    boost = nns_boost(
        x_train,
        y_train,
        x_test,
        type="CLASS",
        epochs=5,
        learner_trials=10,
        cv_size=0.25,
        balance=True,
        status=False,
        seed=123,
    )
    stack = nns_stack(
        x_train,
        y_train,
        x_test,
        type="CLASS",
        balance=True,
        folds=1,
        cv_size=0.25,
        status=False,
        seed=123,
    )

    predictions = {
        "reg": np.asarray(reg["Point.est"], dtype=float),
        "boost": np.asarray(boost["results"], dtype=float),
        "stack": np.asarray(stack["stack"], dtype=float),
    }
    for values in predictions.values():
        assert values.shape == y_test.shape
        assert set(np.unique(values)) <= {1.0, 2.0, 3.0}

    for name, values in predictions.items():
        print(f"{name} accuracy:", round(float(np.mean(values == y_test)), 4))
        print(f"{name} predictions:", values.astype(int))
    print("boost feature weights:", boost["feature.weights"])
    print("stack weights:", stack["weights"])


if __name__ == "__main__":
    main()
