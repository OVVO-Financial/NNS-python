from __future__ import annotations

import numpy as np
import pytest
from _r import RValue, nns_anova_custom

from pynns import nns_anova

ANOVA_PARITY = 3e-5
SIZES = [30, 100, 500]


@pytest.mark.parity
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize(
    ("means_only", "medians"),
    [(False, False), (True, False), (False, True)],
)
def test_nns_anova_binary_matches_r(size: int, means_only: bool, medians: bool) -> None:
    control, treatment = _groups(size)

    expected = _r_anova_binary(control, treatment, means_only=means_only, medians=medians)
    actual = nns_anova(
        control,
        treatment,
        means_only=means_only,
        medians=medians,
        confidence_interval=None,
    )

    assert isinstance(actual, dict)
    assert set(actual) == set(expected)
    for key, value in expected.items():
        np.testing.assert_allclose(actual[key], value, atol=ANOVA_PARITY)


@pytest.mark.parity
def test_nns_anova_binary_unequal_sizes_matches_r() -> None:
    control, treatment = _groups(100)
    treatment = treatment[:73]

    expected = _r_anova_binary(control, treatment)
    actual = nns_anova(control, treatment, confidence_interval=None)

    assert isinstance(actual, dict)
    for key, value in expected.items():
        np.testing.assert_allclose(actual[key], value, atol=ANOVA_PARITY)


@pytest.mark.parity
@pytest.mark.parametrize("size", SIZES)
def test_nns_anova_multi_group_certainty_matches_r(size: int) -> None:
    groups = _multi_groups(size)

    expected = _r_anova_groups(groups, pairwise=False)
    actual = nns_anova(groups, confidence_interval=None)

    assert isinstance(actual, dict)
    assert isinstance(expected, float)
    np.testing.assert_allclose(actual["Certainty"], expected, atol=ANOVA_PARITY)


@pytest.mark.parity
def test_nns_anova_pairwise_matches_r() -> None:
    groups = _multi_groups(100)

    expected = _r_anova_groups(groups, pairwise=True)
    actual = nns_anova(groups, confidence_interval=None, pairwise=True)

    assert isinstance(actual, np.ndarray)
    assert isinstance(expected, np.ndarray)
    np.testing.assert_allclose(actual, expected, atol=ANOVA_PARITY)


@pytest.mark.parity
@pytest.mark.stochastic
def test_nns_anova_robust_structure_matches_r_shape() -> None:
    control, treatment = _groups(30)

    expected = _r_anova_binary(control, treatment, robust=True)
    actual = nns_anova(control, treatment, robust=True, random_seed=123)

    assert isinstance(actual, dict)
    assert set(expected).issuperset(
        {"Control", "Treatment", "Grand_Statistic", "Control_CDF", "Treatment_CDF", "Certainty"}
    )
    assert set(actual) == {
        "Control",
        "Treatment",
        "Grand_Statistic",
        "Control_CDF",
        "Treatment_CDF",
        "Certainty",
        "Effect_Size_LB",
        "Effect_Size_UB",
        "Confidence_Level",
        "Robust Certainty Estimate",
        "Lower Bound Robust Certainty",
        "Upper Bound Robust Certainty",
    }
    assert 0.0 <= actual["Robust Certainty Estimate"] <= 1.0
    assert 0.0 <= actual["Lower Bound Robust Certainty"] <= 1.0
    assert 0.0 <= actual["Upper Bound Robust Certainty"] <= 1.0


def _groups(size: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(size, dtype=np.float64)
    x = np.linspace(-2.0, 2.0, size) + 0.1 * np.sin(idx / 3.0)
    y = x + 0.25 + 0.05 * np.cos(idx / 5.0)
    return x, y


def _multi_groups(size: int) -> list[np.ndarray]:
    x, y = _groups(size)
    z = np.cos(np.linspace(0.0, 3.0, size)) + 0.1 * np.sin(np.arange(size) / 7.0)
    return [x, y, z]


def _r_anova_binary(
    control: np.ndarray,
    treatment: np.ndarray,
    *,
    means_only: bool = False,
    medians: bool = False,
    robust: bool = False,
) -> dict[str, float]:
    result = _r_anova(
        {
            "mode": "binary",
            "control": control.tolist(),
            "treatment": treatment.tolist(),
            "means_only": means_only,
            "medians": medians,
            "robust": robust,
        }
    )
    assert isinstance(result, dict)
    return _scalar_dict(result)


def _r_anova_groups(groups: list[np.ndarray], *, pairwise: bool) -> float | np.ndarray:
    result = _r_anova(
        {
            "mode": "groups",
            "groups": [group.tolist() for group in groups],
            "means_only": False,
            "medians": False,
            "pairwise": pairwise,
        }
    )
    if isinstance(result, dict):
        raise AssertionError(f"Unexpected R ANOVA group result: {result!r}")
    if isinstance(result, np.ndarray) and result.ndim == 0:
        return float(result)
    assert isinstance(result, np.ndarray)
    return result


def _r_anova(payload: dict[str, object]) -> RValue:
    return nns_anova_custom(payload)


def _scalar_dict(value: dict[str, RValue]) -> dict[str, float]:
    return {key: float(np.asarray(item).reshape(-1)[0]) for key, item in value.items()}
