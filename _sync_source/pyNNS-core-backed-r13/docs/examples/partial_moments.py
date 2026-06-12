from __future__ import annotations

import numpy as np

from pynns import (
    co_lpm,
    co_upm,
    d_lpm,
    d_upm,
    lpm,
    lpm_ratio,
    nns_moments,
    pm_matrix,
    upm,
    upm_ratio,
)


def main() -> None:
    x = np.array([-2.0, -1.0, 0.5, 3.0, 4.5], dtype=np.float64)
    y = np.array([4.0, 2.5, 1.0, 1.5, 3.0], dtype=np.float64)
    target = float(np.mean(x))
    target_y = float(np.mean(y))

    lower_degree_zero = lpm(0, target, x)
    upper_degree_zero = upm(0, target, x)
    variance_from_partials = lpm(2, target, x) + upm(2, target, x)
    downside_share = lpm_ratio(2, target, x)
    upside_share = upm_ratio(2, target, x)

    # Co-partial moments split joint movement into same-side and opposite-side terms.
    same_lower = co_lpm(1, x, y, target, target_y)
    same_upper = co_upm(1, x, y, target, target_y)
    lower_x_upper_y = d_upm(1, 1, x, y, target, target_y)
    upper_x_lower_y = d_lpm(1, 1, x, y, target, target_y)

    matrix = pm_matrix(
        1,
        1,
        "mean",
        np.column_stack((x, y)),
        pop_adj=True,
        norm=True,
    )

    np.testing.assert_allclose(lower_degree_zero + upper_degree_zero, 1.0)
    np.testing.assert_allclose(variance_from_partials, np.var(x, ddof=0))
    np.testing.assert_allclose(downside_share + upside_share, 1.0)
    assert set(matrix) == {"cupm", "dupm", "dlpm", "clpm", "cov.matrix"}
    assert matrix["cov.matrix"].shape == (2, 2)

    print("target:", target)
    print("P(x <= target):", lower_degree_zero)
    print("P(x > target):", upper_degree_zero)
    print("downside/upside variance shares:", downside_share, upside_share)
    print("population variance from partial moments:", variance_from_partials)
    print("same-side co-moments:", same_lower, same_upper)
    print("opposite-side co-moments:", lower_x_upper_y, upper_x_lower_y)
    print("normalized partial-moment covariance matrix:")
    print(matrix["cov.matrix"])
    print("NNS moments:", nns_moments(x))


if __name__ == "__main__":
    main()
