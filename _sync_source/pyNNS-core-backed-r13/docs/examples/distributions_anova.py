from __future__ import annotations

import numpy as np

from pynns import nns_anova, nns_cdf


def main() -> None:
    control = np.linspace(-1.0, 1.0, 25, dtype=np.float64)
    treatment = control + 0.35
    wider_treatment = 1.2 * control + 0.55

    cdf = nns_cdf(control, degree=0)
    survival = nns_cdf(control, degree=0, type="survival")
    cumulative_hazard = nns_cdf(control, degree=0, type="cumulative hazard", target=0.0)
    comparison = nns_anova(control, treatment, confidence_interval=None)
    robust = nns_anova(
        control,
        treatment,
        robust=True,
        n_boot=64,
        random_seed=11,
        confidence_interval=None,
    )
    pairwise = nns_anova(
        [control, treatment, wider_treatment],
        pairwise=True,
        confidence_interval=None,
    )

    function = cdf["Function"]
    survival_function = survival["Function"]
    assert isinstance(function, dict)
    assert isinstance(survival_function, dict)
    assert set(function) == {"x", "CDF"}
    assert set(survival_function) == {"x", "S(x)"}
    assert 0.0 <= comparison["Certainty"] <= 1.0
    assert 0.0 <= robust["Certainty"] <= 1.0
    np.testing.assert_allclose(function["CDF"] + survival_function["S(x)"], 1.0)
    np.testing.assert_allclose(pairwise, pairwise.T, equal_nan=True)
    np.testing.assert_allclose(np.diag(pairwise), 1.0)

    print("first CDF rows:")
    print(np.column_stack((function["x"][:5], function["CDF"][:5])))
    print("cumulative hazard at target 0:", cumulative_hazard["target.value"])
    print("ANOVA certainty:", comparison["Certainty"])
    print("robust ANOVA certainty:", robust["Certainty"])
    print("pairwise certainty matrix:")
    print(pairwise)


if __name__ == "__main__":
    main()
