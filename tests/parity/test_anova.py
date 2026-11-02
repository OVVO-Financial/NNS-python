from __future__ import annotations

import numpy as np
import pytest
from _r import RValue, nns_anova_custom

from nns import nns_anova

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
def test_nns_anova_quantile_targets_match_r_exactly() -> None:
    # The 25%/12.5% targets use the exact LPM.VaR/UPM.VaR inversion (nns.var),
    # not an optimiser, so the full Certainty agrees with R to rounding error.
    rng = np.random.default_rng(14)
    groups = [rng.standard_t(3, 400) * 0.01 + shift for shift in (0.0, 0.001, -0.002)]

    binary = nns_anova(groups[0], groups[1], confidence_interval=None)
    expected = _r_anova_binary(groups[0], groups[1])
    assert isinstance(binary, dict)
    for key, value in expected.items():
        np.testing.assert_allclose(binary[key], value, rtol=0.0, atol=1e-12)

    pairwise = nns_anova(groups, confidence_interval=None, pairwise=True)
    np.testing.assert_allclose(
        pairwise, _r_anova_groups(groups, pairwise=True), rtol=0.0, atol=1e-12
    )
    aggregate = nns_anova(groups, confidence_interval=None)
    assert isinstance(aggregate, dict)
    np.testing.assert_allclose(
        aggregate["Certainty"], _r_anova_groups(groups, pairwise=False), rtol=0.0, atol=1e-12
    )


@pytest.mark.parity
@pytest.mark.parametrize("seed", [3, 77])
@pytest.mark.parametrize("confidence_interval", [0.95, 0.9, None])
@pytest.mark.parametrize("medians", [False, True])
def test_nns_anova_robust_matches_seeded_r(
    seed: int,
    confidence_interval: float | None,
    medians: bool,
) -> None:
    # Same resample indices and the same control/treatment pairing as R, so the
    # robust estimate and its bounds match set.seed(seed) in R exactly.
    control, treatment = _groups(30)
    treatment = np.sin(3.0 * treatment[:26]) + 0.1

    expected = _r_anova_seeded(
        control,
        treatment,
        seed=seed,
        ci=confidence_interval,
        robust=True,
        medians=medians,
    )
    actual = nns_anova(
        control,
        treatment,
        robust=True,
        medians=medians,
        confidence_interval=confidence_interval,
        random_seed=seed,
    )

    assert isinstance(actual, dict)
    # R's unlist() turns the effect-size keys into "Effect_Size_LB.2.5%" etc.;
    # NNS Python keeps the plain names used by the non-robust path.
    renamed = {
        key.split(".")[0] if key.startswith("Effect_Size") else key: value
        for key, value in expected.items()
    }
    assert set(actual) == set(renamed)
    if confidence_interval is None:
        assert "Robust Certainty Estimate" in actual and len(actual) == 7
    else:
        level = round(100 * confidence_interval)
        assert {f"Lower {level}% CI", f"Upper {level}% CI"} <= set(actual)
    for key, value in renamed.items():
        np.testing.assert_allclose(actual[key], np.asarray(value), rtol=0.0, atol=1e-12)


@pytest.mark.parity
@pytest.mark.parametrize("seed", [5, 2024])
@pytest.mark.parametrize("medians", [False, True])
@pytest.mark.parametrize("confidence_interval", [0.95, 0.9])
def test_nns_anova_effect_size_bootstrap_matches_seeded_r(
    seed: int,
    medians: bool,
    confidence_interval: float,
) -> None:
    # random_seed=s replays R's set.seed(s) stream, so the bootstrap CI is exact.
    control, treatment = _groups(30)
    treatment = treatment[:23] ** 2

    expected = _r_anova_seeded(
        control, treatment, seed=seed, ci=confidence_interval, medians=medians
    )
    actual = nns_anova(
        control,
        treatment,
        medians=medians,
        confidence_interval=confidence_interval,
        random_seed=seed,
    )

    assert isinstance(actual, dict)
    assert list(actual) == [
        "Control",
        "Treatment",
        "Grand_Statistic",
        "Control_CDF",
        "Treatment_CDF",
        "Certainty",
        "Effect_Size_LB",
        "Effect_Size_UB",
        "Confidence_Level",
    ]
    assert set(actual) == set(expected)
    for key, value in expected.items():
        np.testing.assert_allclose(actual[key], np.asarray(value), rtol=0.0, atol=1e-12)


@pytest.mark.parity
@pytest.mark.parametrize("tails", ["left", "right"])
@pytest.mark.parametrize("robust", [False, True])
def test_nns_anova_one_sided_ci_raises_where_r_fails(tails: str, robust: bool) -> None:
    control, treatment = _groups(30)

    r_result = _r_anova_seeded(
        control, treatment, seed=1, ci=0.95, tails=tails, robust=robust, catch=True
    )
    assert "not found" in str(r_result["error"])
    with pytest.raises(ValueError, match="confidence_interval is not supported"):
        nns_anova(control, treatment, tails=tails, robust=robust, random_seed=1)

    # Without an interval the tail choice is irrelevant to the binary test.
    expected = _r_anova_seeded(control, treatment, seed=1, ci=None, tails=tails)
    actual = nns_anova(control, treatment, tails=tails, confidence_interval=None)
    assert isinstance(actual, dict)
    assert set(actual) == set(expected)
    for key, value in expected.items():
        np.testing.assert_allclose(actual[key], np.asarray(value), rtol=0.0, atol=1e-12)


def _r_anova_seeded(
    control: np.ndarray,
    treatment: np.ndarray,
    *,
    seed: int,
    ci: float | None,
    tails: str = "both",
    robust: bool = False,
    medians: bool = False,
    catch: bool = False,
) -> dict[str, RValue]:
    payload: dict[str, object] = {
        "mode": "binary",
        "control": control.tolist(),
        "treatment": treatment.tolist(),
        "means_only": False,
        "medians": medians,
        "robust": robust,
        "seed": seed,
        "ci": "none" if ci is None else ci,
        "tails": tails,
    }
    if catch:
        payload["catch"] = True
    result = _r_anova(payload)
    assert isinstance(result, dict)
    return result


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
