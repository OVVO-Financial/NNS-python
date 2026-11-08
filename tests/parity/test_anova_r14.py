"""R 14.0 parity for NNS.ANOVA, down to R's own summary arithmetic.

The certainty compares each group's mean (or median) with the grand statistic
and with partial-moment VaR targets that R averages with ``mean()``, which
accumulates in long double; the effect-size bounds are quantiles of bootstrap
replicates drawn with R's ``sample()`` and summarised with ``colMeans()`` or
``median()``. The port follows that arithmetic, so the results agree with R to
the last bit.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pytest
from _r import nns_frame_call, r_column

import nns

pytestmark = pytest.mark.parity

def _num(values: Any) -> dict[str, Any]:
    return r_column([float(v) for v in np.asarray(values, dtype=np.float64)], "numeric")


def _named(expected: Any) -> dict[str, float]:
    # unlist() appends the quantile labels R's quantile() gives the effect-size
    # bounds ("Effect_Size_LB.2.5%"); the port names the bound alone.
    names = [re.sub(r"\.\d+(\.\d+)?%$", "", name) for name in expected["names"]]
    return dict(zip(names, (float(v) for v in expected["values"]), strict=True))


def _groups(kind: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if kind == "ties":
        return np.round(rng.normal(size=60)), np.round(rng.normal(size=45) + 0.3)
    return rng.exponential(size=60), rng.exponential(size=45) * 1.5


@pytest.mark.parametrize("kind", ["ties", "skew"])
@pytest.mark.parametrize(
    ("means_only", "medians"), [(False, False), (True, False), (False, True), (True, True)]
)
def test_nns_anova_binary_with_effect_sizes_matches_r(
    kind: str, means_only: bool, medians: bool
) -> None:
    control, treatment = _groups(kind, 61)
    flag = {True: "TRUE", False: "FALSE"}
    expected = nns_frame_call(
        "anova14.binary",
        f"set.seed(7); unlist(NNS.ANOVA(a, b, means.only = {flag[means_only]}, "
        f"medians = {flag[medians]}, confidence.interval = 0.95, tails = 'Both', "
        "plot = FALSE))",
        {"a": _num(control), "b": _num(treatment)},
        exact=True,
    )
    actual = nns.nns_anova(
        control,
        treatment,
        means_only=means_only,
        medians=medians,
        confidence_interval=0.95,
        tails="Both",
        random_seed=7,
    )
    assert isinstance(actual, dict)
    want = _named(expected)
    assert set(actual) == set(want)
    for key, value in want.items():
        assert actual[key] == value, key


@pytest.mark.parametrize("medians", [False, True])
def test_nns_anova_multi_group_certainty_matches_r(medians: bool) -> None:
    rng = np.random.default_rng(62)
    groups = [rng.normal(size=size) + 0.2 * i for i, size in enumerate((23, 31, 17, 40))]
    flag = "TRUE" if medians else "FALSE"
    data = {f"g{i}": _num(group) for i, group in enumerate(groups)}
    expected = nns_frame_call(
        "anova14.multi",
        "G <- list(g0, g1, g2, g3); "
        f"c(unname(NNS.ANOVA(G, medians = {flag}, plot = FALSE)['Certainty']), "
        f"as.numeric(NNS.ANOVA(G, medians = {flag}, pairwise = TRUE, plot = FALSE)))",
        data,
        exact=True,
    )
    certainty = nns.nns_anova(groups, medians=medians)
    pairwise = nns.nns_anova(groups, medians=medians, pairwise=True)
    assert isinstance(certainty, dict)
    actual = np.concatenate([[certainty["Certainty"]], np.asarray(pairwise).ravel(order="F")])
    np.testing.assert_array_equal(actual, np.asarray(expected, dtype=np.float64))
