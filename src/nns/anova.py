from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from nns._rrng import RRNG
from nns.core import lpm_ratio, upm_ratio
from nns.dependence import _gravity
from nns.var import lpm_var, upm_var

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
    """Partial-moment ANOVA, matching R's non-plotting NNS.ANOVA paths.

    The bootstrap draws (effect-size interval and ``robust`` resamples) come from
    R's Mersenne-Twister stream: ``random_seed=s`` reproduces R's
    ``set.seed(s); NNS.ANOVA(...)`` draw for draw. Without a seed the stream is
    seeded from fresh entropy.
    """
    tail = _tail(tails)
    if treatment is not None:
        control_values = _as_group(control, "control")
        treatment_values = _as_group(treatment, "treatment")
        _check_one_sided_ci(confidence_interval, tail)
        rng = _r_rng(random_seed)
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

    upper_25 = float(np.mean([upm_var(0.25, 1, group) for group in groups]))
    lower_25 = float(np.mean([lpm_var(0.25, 1, group) for group in groups]))
    upper_125 = float(np.mean([upm_var(0.125, 1, group) for group in groups]))
    lower_125 = float(np.mean([lpm_var(0.125, 1, group) for group in groups]))

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
    rng: RRNG | None = None,
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
        upper_25_target = float(np.mean([upm_var(0.25, 1, control), upm_var(0.25, 1, treatment)]))
        lower_25_target = float(np.mean([lpm_var(0.25, 1, control), lpm_var(0.25, 1, treatment)]))
        upper_125_target = float(
            np.mean([upm_var(0.125, 1, control), upm_var(0.125, 1, treatment)])
        )
        lower_125_target = float(
            np.mean([lpm_var(0.125, 1, control), lpm_var(0.125, 1, treatment)])
        )
    assert upper_25_target is not None
    assert lower_25_target is not None
    assert upper_125_target is not None
    assert lower_125_target is not None

    if medians:
        lpm_ratio_1 = float(lpm_ratio(0, mean_of_means, control))
        lpm_ratio_2 = float(lpm_ratio(0, mean_of_means, treatment))
    else:
        lpm_ratio_1 = float(lpm_ratio(1, mean_of_means, control))
        lpm_ratio_2 = float(lpm_ratio(1, mean_of_means, treatment))

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
                rng=_r_rng(None) if rng is None else rng,
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
    rng: RRNG,
) -> AnovaResult:
    # R draws the resample indices first: replicate(100, sample.int(l, replace =
    # TRUE)) with l = min(length(treatment), length(control)). (Resampling only
    # the first l observations of the longer group is shared with R; SDA-02.)
    sample_size = min(control.size, treatment.size)
    draws = (rng.sample_int_replace(sample_size, sample_size * 100) - 1).reshape(100, -1)
    # R's treatment_matrix is cbind(resample 1, resamples 100, 99, ..., 1), so
    # control resample g is paired with treatment resample 1 for g = 1 and with
    # 102 - g otherwise: the two groups are resampled (almost) independently.
    treatment_draws = draws[np.r_[0, np.arange(99, 0, -1)]]
    certainties = np.array(
        [
            _anova_bin(
                control[draws[replicate]],
                treatment[treatment_draws[replicate]],
                means_only=means_only,
                medians=medians,
                confidence_interval=None,
                tails=tails,
            )["Certainty"]
            for replicate in range(draws.shape[0])
        ],
        dtype=np.float64,
    )

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
    base["Robust Certainty Estimate"] = _gravity(certainties)
    if confidence_interval is None:
        return base

    # R labels the bounds by the CI level: "Lower 95% CI" / "Upper 95% CI" for two
    # tails, "Lower 95% bound" or "Upper 95% bound" for one.
    alpha = _ci_alpha(confidence_interval, tails)
    level = round(100.0 * confidence_interval)
    kind = "CI" if tails == "both" else "bound"
    if tails in {"both", "left"}:
        base[f"Lower {level}% {kind}"] = _quantile_var(alpha, upper=False, values=certainties)
    if tails in {"both", "right"}:
        base[f"Upper {level}% {kind}"] = _quantile_var(alpha, upper=True, values=certainties)
    return base


def _effect_size_bounds(
    control: NDArray[np.float64],
    treatment: NDArray[np.float64],
    *,
    medians: bool,
    confidence_interval: float,
    tails: Tail,
    n_boot: int,
    rng: RRNG,
) -> AnovaResult:
    if not 0.0 <= confidence_interval <= 1.0:
        raise ValueError("confidence_interval must be in [0, 1].")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1.")

    # R: matrix(sample(x, n_boot * length(x), replace = TRUE), nrow = length(x)),
    # i.e. column j holds draws j * n .. (j + 1) * n - 1; row j here.
    control_boot = control[_r_boot_indices(rng, control.size, n_boot)]
    treatment_boot = treatment[_r_boot_indices(rng, treatment.size, n_boot)]
    control_stats = np.median(control_boot, axis=1) if medians else np.mean(control_boot, axis=1)
    treatment_stats = (
        np.median(treatment_boot, axis=1) if medians else np.mean(treatment_boot, axis=1)
    )
    _check_one_sided_ci(confidence_interval, tails)
    alpha = _ci_alpha(confidence_interval, tails)

    control_upper = _quantile_var(alpha, upper=True, values=control_stats)
    treatment_upper = _quantile_var(alpha, upper=True, values=treatment_stats)
    control_lower = _quantile_var(alpha, upper=False, values=control_stats)
    treatment_lower = _quantile_var(alpha, upper=False, values=treatment_stats)
    min_effect = treatment_lower - control_upper
    max_effect = treatment_upper - control_lower
    return {
        "Effect_Size_LB": float(min_effect),
        "Effect_Size_UB": float(max_effect),
        "Confidence_Level": float(confidence_interval),
    }


def _r_rng(seed: int | None) -> RRNG:
    if seed is None:
        return RRNG(int(np.random.SeedSequence().generate_state(1)[0]))
    return RRNG(int(seed))


def _r_boot_indices(rng: RRNG, size: int, n_boot: int) -> NDArray[np.int64]:
    return (rng.sample_int_replace(size, size * n_boot) - 1).reshape(n_boot, size)


def _check_one_sided_ci(confidence_interval: float | None, tails: Tail) -> None:
    # R's NNS.ANOVA.bin computes control_upper only for "both"/"right" and
    # control_lower only for "both"/"left", then uses the missing one, so every
    # one-sided effect-size interval stops with "object ... not found". Python
    # used to return (-inf, inf) instead; fail clearly until R is fixed.
    if confidence_interval is not None and tails != "both":
        raise ValueError(
            f"tails={tails!r} with a confidence_interval is not supported: the one-sided "
            "effect-size bound fails in R's NNS.ANOVA (object 'control_upper'/'control_lower' "
            "not found). Use tails='both', or confidence_interval=None."
        )


def _ci_alpha(confidence_interval: float | None, tails: Tail) -> float:
    interval = 0.95 if confidence_interval is None else confidence_interval
    if not 0.0 <= interval <= 1.0:
        raise ValueError("confidence_interval must be in [0, 1].")
    return (1.0 - interval) / 2.0 if tails == "both" else 1.0 - interval


def _r_min(left: float, right: float) -> float:
    return float(np.minimum(left, right))


def _r_max(left: float, right: float) -> float:
    return float(np.maximum(left, right))


def _quantile_var(percentile: float, *, upper: bool, values: NDArray[np.float64]) -> float:
    """R's ``LPM.VaR``/``UPM.VaR`` at degree 0 (type-7 quantile), NaN-tolerant.

    Degenerate groups make every certainty NaN; R's VaR then stops, while NNS
    Python keeps returning NaN bounds.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return upm_var(percentile, 0, finite) if upper else lpm_var(percentile, 0, finite)


def _as_group(value: object, name: str) -> NDArray[np.float64]:
    values = np.asarray(value, dtype=np.float64).reshape(-1)
    # Missing values (NaN) are dropped, as R's multi-group path does with na.omit
    # (R's binary path propagates NaN instead). Infinite values have no partial-
    # moment meaning and make R's VaR step fail, so they are rejected, not dropped.
    if np.isinf(values).any():
        raise ValueError(f"{name} must not contain infinite values.")
    values = values[~np.isnan(values)]
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
