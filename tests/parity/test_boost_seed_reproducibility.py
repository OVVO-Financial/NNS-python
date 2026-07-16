from __future__ import annotations

import numpy as np
import pytest

from nns import nns_boost


@pytest.mark.parity
@pytest.mark.parametrize("seed", [0, 1, 4, 42, 1234])
def test_nns_boost_ivs_test_none_is_seed_reproducible(seed: int) -> None:
    """Reusing a seed must reproduce the same delegated NNS.stack estimate."""
    x = np.linspace(-2.0, 2.0, 24)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = x + np.sin(x) + 0.25 * np.cos(x)

    def fitted() -> np.ndarray:
        return np.asarray(
            nns_boost(
                variable,
                y,
                learner_trials=10,
                cv_size=0.25,
                feature_importance=False,
                random_seed=seed,
            )["results"],
            dtype=np.float64,
        )

    np.testing.assert_array_equal(fitted(), fitted())
