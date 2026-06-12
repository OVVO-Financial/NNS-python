from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from pynns.core import lpm_ratio, upm_ratio
from pynns.dependence import _gravity

AnovaResult = dict[str, float]
Tail = Literal["both", "left", "right"]


def nns_anova(
    control: NDArray[np.float64] | Sequence[NDArray[np.float64]],
    treatment: NDArray[np.float64] | None = None,
    *,
    means_only: bool = False,
    medians: bool = False,
    confidence_interval: float | None = 0.95,
    tails: Tail | str = "Both",
    pairwise: bool = False,
    robust: bool = False,
    n_boot: int = 1000,
    random_seed: int | None = None,
) -> AnovaResult | NDArray[np.float64]:
    """Partial-moment ANOVA, matching R's non-plotting NNS.ANOVA paths."""
    tail = _tail(tails)
    if treatment is not None:
        control_values = _as_group(control, "control")
        treatment_values = _as_group(treatment, "treatment")
        rng = np.random.default_rng(random_seed)
        if robust:
            return _anova_robust(
                control_values,
                treatment_values,
                means_only=means_only,
                medians=medians,
                confidence_interval=confidence_interval,
                tails=tail,
                n_boot=n_boot,
                rng=rng,
            )
        return _anova_bin(
            control_values,
            treatment_values,
            means_only=means_only,
            medians=medians,
            confidence_interval=confidence_interval,
            tails=tail,
            n_boot=n_boot,
            rng=rng,
        )

    groups = _as_groups(control)
    if len(groups) < 2:
        raise ValueError("supply both control and treatment or at least two control groups.")

    grand = (
        float(np.mean([np.median(group) for group in groups]))
        if medians
        else float(np.mean([np.mean(group) for group in groups]))
    )

    if pairwise:
        out = np.full((len(groups), len(groups)), np.nan, dtype=np.float64)
        np.fill_diagonal(out, 1.0)
        for i in range(len(groups) - 1):
            for j in range(i + 1, len(groups)):
                certainty = _anova_bin(
                    groups[i],
                    groups[j],
                    means_only=means_only,
                    medians=medians,
                    confidence_interval=None,
                    tails=tail,
                )["Certainty"]
                out[i, j] = certainty
                out[j, i] = certainty
        return out

    upper_25 = float(np.mean([_upm_var(0.25, 1, group) for group in groups]))
    lower_25 = float(np.mean([_lpm_var(0.25, 1, group) for group in groups]))
    upper_125 = float(np.mean([_upm_var(0.125, 1, group) for group in groups]))
    lower_125 = float(np.mean([_lpm_var(0.125, 1, group) for group in groups]))

    certainties = [
        _anova_bin(
            groups[i],
            groups[j],
            means_only=means_only,
            medians=medians,
            mean_of_means=grand,
            upper_25_target=upper_25,
            lower_25_target=lower_25,
            upper_125_target=upper_125,
            lower_125_target=lower_125,
            confidence_interval=None,
            tails=tail,
        )["Certainty"]
        for i in range(len(groups) - 1)
        for j in range(i + 1, len(groups))
    ]
    return {"Certainty": float(np.mean(certainties))}


def _anova_bin(
    control: NDArray[np.float64],
    treatment: NDArray[np.float64],
    *,
    means_only: bool,
    medians: bool,
    mean_of_means: float | None = None,
    upper_25_target: float | None = None,
    lower_25_target: float | None = None,
    upper_125_target: float | None = None,
    lower_125_target: float | None = None,
    confidence_interval: float | None = None,
    tails: Tail = "both",
    n_boot: int = 1000,
    rng: np.random.Generator | None = None,
) -> AnovaResult:
    if mean_of_means is None:
        control_stat = float(np.median(control) if medians else np.mean(control))
        treatment_stat = float(np.median(treatment) if medians else np.mean(treatment))
        mean_of_means = (control.size * control_stat + treatment.size * treatment_stat) / (
            control.size + treatment.size
        )
    else:
        control_stat = float(np.median(control) if medians else np.mean(control))
        treatment_stat = float(np.median(treatment) if medians else np.mean(treatment))

    if upper_25_target is None or lower_25_target is None:
        upper_25_target = float(np.mean([_upm_var(0.25, 1, control), _upm_var(0.25, 1, treatment)]))
        lower_25_target = float(np.mean([_lpm_var(0.25, 1, control), _lpm_var(0.25, 1, treatment)]))
        upper_125_target = float(
            np.mean([_upm_var(0.125, 1, control), _upm_var(0.125, 1, treatment)])
        )
        lower_125_target = float(
            np.mean([_lpm_var(0.125, 1, control), _lpm_var(0.125, 1, treatment)])
        )
    assert upper_25_target is not None
    assert lower_25_target is not None
    assert upper_125_target is not None
    assert lower_125_target is not None

    if medians:
        lpm_ratio_1 = float(lpm_ratio(0, mean_of_means, control))
        lpm_ratio_2 = float(lpm_ratio(0, mean_of_means, treatment))
    else:
        lpm_ratio_1 = _lower_area_share(mean_of_means, control)
        lpm_ratio_2 = _lower_area_share(mean_of_means, treatment)

    upper_25_ratio_1 = float(upm_ratio(1, upper_25_target, control))
    upper_25_ratio_2 = float(upm_ratio(1, upper_25_target, treatment))
    lower_25_ratio_1 = float(lpm_ratio(1, lower_25_target, control))
    lower_25_ratio_2 = float(lpm_ratio(1, lower_25_target, treatment))
    upper_125_ratio_1 = float(upm_ratio(1, upper_125_target, control))
    upper_125_ratio_2 = float(upm_ratio(1, upper_125_target, treatment))
    lower_125_ratio_1 = float(lpm_ratio(1, lower_125_target, control))
    lower_125_ratio_2 = float(lpm_ratio(1, lower_125_target, treatment))

    mad_cdf = _r_min(0.5, _r_max(abs(lpm_ratio_1 - 0.5), abs(lpm_ratio_2 - 0.5)))
    upper_25_cdf = _r_min(
        0.25,
        _r_max(abs(upper_25_ratio_1 - 0.25), abs(upper_25_ratio_2 - 0.25)),
    )
    lower_25_cdf = _r_min(
        0.25,
        _r_max(abs(lower_25_ratio_1 - 0.25), abs(lower_25_ratio_2 - 0.25)),
    )
    upper_125_cdf = _r_min(
        0.125,
        _r_max(abs(upper_125_ratio_1 - 0.125), abs(upper_125_ratio_2 - 0.125)),
    )
    lower_125_cdf = _r_min(
        0.125,
        _r_max(abs(lower_125_ratio_1 - 0.125), abs(lower_125_ratio_2 - 0.125)),
    )

    if means_only:
        rho = ((0.5 - mad_cdf) ** 2) / 0.25
    else:
        rho = (
            ((0.5 - mad_cdf) ** 2) / 0.25
            + 0.5 * (((0.25 - upper_25_cdf) ** 2) / (0.25**2))
            + 0.5 * (((0.25 - lower_25_cdf) ** 2) / (0.25**2))
            + 0.25 * (((0.125 - upper_125_cdf) ** 2) / (0.125**2))
            + 0.25 * (((0.125 - lower_125_cdf) ** 2) / (0.125**2))
        ) / 2.5

    pop_adjustment = ((control.size + treatment.size - 2) / (control.size + treatment.size)) ** 2
    result = {
        "Control": control_stat,
        "Treatment": treatment_stat,
        "Grand_Statistic": float(mean_of_means),
        "Control_CDF": lpm_ratio_1,
        "Treatment_CDF": lpm_ratio_2,
        "Certainty": _r_min(1.0, float(rho * pop_adjustment)),
    }
    if confidence_interval is not None:
        result.update(
            _effect_size_bounds(
                control,
                treatment,
                medians=medians,
                confidence_interval=confidence_interval,
                tails=tails,
                n_boot=n_boot,
                rng=np.random.default_rng() if rng is None else rng,
            )
        )
    return result


def _anova_robust(
    control: NDArray[np.float64],
    treatment: NDArray[np.float64],
    *,
    means_only: bool,
    medians: bool,
    confidence_interval: float | None,
    tails: Tail,
    n_boot: int,
    rng: np.random.Generator,
) -> AnovaResult:
    base = _anova_bin(
        control,
        treatment,
        means_only=means_only,
        medians=medians,
        confidence_interval=confidence_interval,
        tails=tails,
        n_boot=n_boot,
        rng=rng,
    )
    sample_size = min(control.size, treatment.size)
    indices = rng.integers(0, sample_size, size=(sample_size, 100))
    certainties = np.array(
        [
            _anova_bin(
                control[indices[:, col]],
                treatment[indices[:, col]],
                means_only=means_only,
                medians=medians,
                confidence_interval=None,
                tails=tails,
            )["Certainty"]
            for col in range(indices.shape[1])
        ],
        dtype=np.float64,
    )

    alpha = _ci_alpha(confidence_interval, tails)
    base["Robust Certainty Estimate"] = _gravity(certainties)
    base["Lower Bound Robust Certainty"] = _lpm_var(alpha, 0, certainties)
    base["Upper Bound Robust Certainty"] = _upm_var(alpha, 0, certainties)
    return base


def _effect_size_bounds(
    control: NDArray[np.float64],
    treatment: NDArray[np.float64],
    *,
    medians: bool,
    confidence_interval: float,
    tails: Tail,
    n_boot: int,
    rng: np.random.Generator,
) -> AnovaResult:
    if not 0.0 <= confidence_interval <= 1.0:
        raise ValueError("confidence_interval must be in [0, 1].")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1.")

    control_boot = rng.choice(control, size=(control.size, n_boot), replace=True)
    treatment_boot = rng.choice(treatment, size=(treatment.size, n_boot), replace=True)
    control_stats = np.median(control_boot, axis=0) if medians else np.mean(control_boot, axis=0)
    treatment_stats = (
        np.median(treatment_boot, axis=0) if medians else np.mean(treatment_boot, axis=0)
    )
    alpha = _ci_alpha(confidence_interval, tails)

    control_upper = treatment_upper = np.inf
    control_lower = treatment_lower = -np.inf
    if tails in {"both", "right"}:
        control_upper = _upm_var(alpha, 0, control_stats)
        treatment_upper = _upm_var(alpha, 0, treatment_stats)
    if tails in {"both", "left"}:
        control_lower = _lpm_var(alpha, 0, control_stats)
        treatment_lower = _lpm_var(alpha, 0, treatment_stats)

    if tails == "both":
        min_effect = treatment_lower - control_upper
        max_effect = treatment_upper - control_lower
    elif tails == "left":
        min_effect = treatment_lower - control_upper
        max_effect = np.inf
    else:
        min_effect = -np.inf
        max_effect = treatment_upper - control_lower
    return {
        "Effect_Size_LB": float(min_effect),
        "Effect_Size_UB": float(max_effect),
        "Confidence_Level": float(confidence_interval),
    }


def _ci_alpha(confidence_interval: float | None, tails: Tail) -> float:
    interval = 0.95 if confidence_interval is None else confidence_interval
    if not 0.0 <= interval <= 1.0:
        raise ValueError("confidence_interval must be in [0, 1].")
    return (1.0 - interval) / 2.0 if tails == "both" else 1.0 - interval


def _lower_area_share(target: float, values: NDArray[np.float64]) -> float:
    lower = float(lpm_ratio(1, target, values))
    upper = float(upm_ratio(1, target, values))
    return lower / (lower + upper)


def _r_min(left: float, right: float) -> float:
    return float(np.minimum(left, right))


def _r_max(left: float, right: float) -> float:
    return float(np.maximum(left, right))


def _lpm_var(percentile: float, degree: int, values: NDArray[np.float64]) -> float:
    percentile = float(np.clip(percentile, 0.0, 1.0))
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    if degree == 0:
        return float(np.quantile(values, percentile))
    if float(np.min(values)) == float(np.max(values)):
        return float(np.min(values))

    def objective(target: float) -> float:
        return abs(float(lpm_ratio(degree, target, values)) - percentile)

    from scipy import optimize  # type: ignore[import-untyped]

    result = optimize.minimize_scalar(
        objective,
        bounds=(float(np.min(values)), float(np.max(values))),
        method="bounded",
        options={"xatol": np.sqrt(np.finfo(float).eps)},
    )
    return float(result.x)


def _upm_var(percentile: float, degree: int, values: NDArray[np.float64]) -> float:
    percentile = float(np.clip(percentile, 0.0, 1.0))
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    if degree == 0:
        return float(np.quantile(values, 1.0 - percentile))
    if float(np.min(values)) == float(np.max(values)):
        return float(np.min(values))

    def objective(target: float) -> float:
        return abs(float(upm_ratio(degree, target, values)) - percentile)

    from scipy import optimize

    result = optimize.minimize_scalar(
        objective,
        bounds=(float(np.min(values)), float(np.max(values))),
        method="bounded",
        options={"xatol": np.sqrt(np.finfo(float).eps)},
    )
    return float(result.x)


def _as_group(value: object, name: str) -> NDArray[np.float64]:
    values = np.asarray(value, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError(f"{name} must contain at least one finite value.")
    return values


def _as_groups(
    control: NDArray[np.float64] | Sequence[NDArray[np.float64]],
) -> list[NDArray[np.float64]]:
    if isinstance(control, Sequence) and not isinstance(control, np.ndarray):
        return [_as_group(group, "control group") for group in control]
    values = np.asarray(control, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("control must be 2D when treatment is omitted.")
    return [_as_group(values[:, col], "control column") for col in range(values.shape[1])]


def _tail(value: str) -> Tail:
    tail = value.lower()
    if tail not in {"left", "right", "both"}:
        raise ValueError("tails must be 'left', 'right', or 'both'.")
    return tail  # type: ignore[return-value]
