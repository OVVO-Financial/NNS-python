from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from nns._rmean import r_col_means, r_mean, r_mean_logical, r_median, r_sum
from nns._rrng import RRNG
from nns.central_tendencies import nns_gravity
from nns.core import lpm_ratio, upm_ratio
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
    plot: bool = False,
) -> AnovaResult | NDArray[np.float64]:
    """Partial-moment ANOVA, matching R's non-plotting NNS.ANOVA paths.

    The bootstrap draws (effect-size interval and ``robust`` resamples) come from
    R's Mersenne-Twister stream: ``random_seed=s`` reproduces R's
    ``set.seed(s); NNS.ANOVA(...)`` draw for draw. Without a seed the stream is
    seeded from fresh entropy. ``plot=True`` also draws R's group boxplots
    (``nns.plotting.plot_nns_anova``); R's default is ``plot = TRUE``.
    """
    result = _nns_anova(
        control,
        treatment,
        means_only=means_only,
        medians=medians,
        confidence_interval=confidence_interval,
        tails=tails,
        pairwise=pairwise,
        robust=robust,
        n_boot=n_boot,
        random_seed=random_seed,
    )
    if plot:
        from nns.plotting.anova import plot_nns_anova

        if treatment is not None:
            groups = [_as_group(control, "control"), _as_group(treatment, "treatment")]
        else:
            groups = _as_groups(control)
        plot_nns_anova(groups, medians=medians)
    return result


def _nns_anova(
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
    tail = _tail(tails)
    if treatment is not None:
        control_values = _as_group(control, "control")
        treatment_values = _as_group(treatment, "treatment")
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

    pooled = np.concatenate(groups)
    grand = r_median(pooled) if medians else r_mean(pooled)

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

    # R: mean(sapply(groups, VaR)) with R's mean().
    upper_25 = r_mean(np.array([upm_var(0.25, 1, group) for group in groups]))
    lower_25 = r_mean(np.array([lpm_var(0.25, 1, group) for group in groups]))
    upper_125 = r_mean(np.array([upm_var(0.125, 1, group) for group in groups]))
    lower_125 = r_mean(np.array([lpm_var(0.125, 1, group) for group in groups]))

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
    return {"Certainty": r_mean(np.asarray(certainties, dtype=np.float64))}


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
        control_stat = r_median(control) if medians else r_mean(control)
        treatment_stat = r_median(treatment) if medians else r_mean(treatment)
        if medians:
            mean_of_means = r_median(np.concatenate((control, treatment)))
        elif (
            control.size > 0
            and treatment.size > 0
            and np.all(control == control[0])
            and np.all(treatment == treatment[0])
            and control[0] == treatment[0]
        ):
            # R14 Binary_ANOVA.R: identical constants are their own benchmark,
            # bypassing a weighted mean that may move the target by one ULP.
            mean_of_means = float(control[0])
        else:
            mean_of_means = (control.size * control_stat + treatment.size * treatment_stat) / (
                control.size + treatment.size
            )
    else:
        control_stat = r_median(control) if medians else r_mean(control)
        treatment_stat = r_median(treatment) if medians else r_mean(treatment)

    if upper_25_target is None or lower_25_target is None:
        # R: mean(c(UPM.VaR(., 1, control), UPM.VaR(., 1, treatment))), etc.
        upper_25_target = _mean2(upm_var(0.25, 1, control), upm_var(0.25, 1, treatment))
        lower_25_target = _mean2(lpm_var(0.25, 1, control), lpm_var(0.25, 1, treatment))
        upper_125_target = _mean2(upm_var(0.125, 1, control), upm_var(0.125, 1, treatment))
        lower_125_target = _mean2(lpm_var(0.125, 1, control), lpm_var(0.125, 1, treatment))
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

    # R14's median reference is an interval F(t-) <= 0.5 <= F(t).
    # For discrete/tied observations, F(t) alone need not be 0.5 even when
    # identical samples have the correct median. In the area comparisons an
    # all-constant sample exactly at its target contributes zero discrepancy
    # rather than the degenerate public step-ratio convention.
    constant_1 = float(control[0]) if control.size and np.all(control == control[0]) else None
    constant_2 = (
        float(treatment[0]) if treatment.size and np.all(treatment == treatment[0]) else None
    )

    def area_distance(r1: float, r2: float, target: float, p: float) -> float:
        d1 = 0.0 if constant_1 is not None and constant_1 == target else abs(r1 - p)
        d2 = 0.0 if constant_2 is not None and constant_2 == target else abs(r2 - p)
        return _r_min(p, _r_max(d1, d2))

    if medians:
        median_distance_1 = max(
            0.0, r_mean_logical(control < mean_of_means) - 0.5, 0.5 - lpm_ratio_1
        )
        median_distance_2 = max(
            0.0, r_mean_logical(treatment < mean_of_means) - 0.5, 0.5 - lpm_ratio_2
        )
        mad_cdf = max(median_distance_1, median_distance_2)
    else:
        mad_cdf = area_distance(lpm_ratio_1, lpm_ratio_2, mean_of_means, 0.5)
    upper_25_cdf = area_distance(upper_25_ratio_1, upper_25_ratio_2, upper_25_target, 0.25)
    lower_25_cdf = area_distance(lower_25_ratio_1, lower_25_ratio_2, lower_25_target, 0.25)
    upper_125_cdf = area_distance(upper_125_ratio_1, upper_125_ratio_2, upper_125_target, 0.125)
    lower_125_cdf = area_distance(lower_125_ratio_1, lower_125_ratio_2, lower_125_target, 0.125)

    # R squares with x * x and adds the five terms with sum(), which
    # accumulates in long double.
    if means_only:
        rho = _sq(0.5 - mad_cdf) / 0.25
    else:
        rho = (
            r_sum(
                np.array(
                    [
                        _sq(0.5 - mad_cdf) / 0.25,
                        0.5 * (_sq(0.25 - upper_25_cdf) / _sq(0.25)),
                        0.5 * (_sq(0.25 - lower_25_cdf) / _sq(0.25)),
                        0.25 * (_sq(0.125 - upper_125_cdf) / _sq(0.125)),
                        0.25 * (_sq(0.125 - lower_125_cdf) / _sq(0.125)),
                    ]
                )
            )
            / 2.5
        )

    pop_adjustment = _sq((control.size + treatment.size - 2) / (control.size + treatment.size))
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
    # R draws the resample indices first: replicate(100, sample.int(n, replace =
    # TRUE)) for the control and then for the treatment, each group from its own
    # full sample and size (the shared min(n1, n2) index matrix only
    # ever resampled the first min(n1, n2) observations of the longer group).
    control_draws = (rng.sample_int_replace(control.size, control.size * 100) - 1).reshape(100, -1)
    treatment_draws = (rng.sample_int_replace(treatment.size, treatment.size * 100) - 1).reshape(
        100, -1
    )
    certainties = np.array(
        [
            _anova_bin(
                control[control_draws[replicate]],
                treatment[treatment_draws[replicate]],
                means_only=means_only,
                medians=medians,
                confidence_interval=None,
                tails=tails,
            )["Certainty"]
            for replicate in range(100)
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
    finite = certainties[~np.isnan(certainties)]
    # R14 uses NNS.gravity, not the median, on the 100 bootstrap certainty
    # values. A robust point estimate is therefore not their middle order stat.
    base["Robust Certainty Estimate"] = nns_gravity(finite) if finite.size else math.nan
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
    # R: apply(boot, 2, median) or colMeans(boot) (long-double accumulation).
    control_stats = _r_column_stats(control_boot, medians)
    treatment_stats = _r_column_stats(treatment_boot, medians)
    alpha = _ci_alpha(confidence_interval, tails)

    # Every tail choice needs both quantiles of both groups (R used
    # to compute only one side and failed on one-sided tails).
    control_upper = _quantile_var(alpha, upper=True, values=control_stats)
    treatment_upper = _quantile_var(alpha, upper=True, values=treatment_stats)
    control_lower = _quantile_var(alpha, upper=False, values=control_stats)
    treatment_lower = _quantile_var(alpha, upper=False, values=treatment_stats)
    min_effect = treatment_lower - control_upper if tails != "right" else -math.inf
    max_effect = treatment_upper - control_lower if tails != "left" else math.inf
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
    # Missing values (NaN) are dropped, as R does with na.omit. Infinite values have
    # no partial-moment meaning and make R's VaR step fail, so they are rejected.
    if np.isinf(values).any():
        raise ValueError(f"{name} must not contain infinite values.")
    values = values[~np.isnan(values)]
    if values.size == 0:
        raise ValueError(f"{name} must contain at least one finite value.")
    return np.asarray(values, dtype=np.float64)


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


def _r_column_stats(boot: NDArray[np.float64], medians: bool) -> NDArray[np.float64]:
    """Each bootstrap replicate's (row's) R median or R mean."""
    if medians:
        return np.array([r_median(row) for row in boot], dtype=np.float64)
    return r_col_means(boot.T)


def _mean2(a: float | NDArray[np.float64], b: float | NDArray[np.float64]) -> float:
    """R's ``mean(c(a, b))`` of two scalars."""
    return r_mean(np.array([float(np.asarray(a)), float(np.asarray(b))]))


def _sq(value: float) -> float:
    """R's ``value^2``, which R evaluates as ``value * value``."""
    return value * value
