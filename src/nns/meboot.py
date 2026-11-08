"""NNS.meboot: maximum-entropy bootstrap on detrended residuals.

The port follows R 14.0's ``NNS.meboot`` step for step, including its random
draws: with ``random_seed = s`` the replicates equal R's after ``set.seed(s)``.
R draws only through ``runif`` (one per observation per replicate in the C++
``NNS.meboot.part``, then one per shrunken column in each ``expand.sd`` pass),
which :class:`nns._rrng.RRNG` reproduces. The deterministic steps use R's own
arithmetic where the last bit can matter: long-double ``mean``/``sum``/
``colSums``/``rowMeans``, R's ``sd``, ``rank`` and trimmed ``mean``, the C++
helpers' sequential double sums, ``optimize`` (Brent's ``fmin``) for the
dependence target, ``qnorm`` for the CLT spread, and ``round``.
"""

from __future__ import annotations

import math
import sys
import warnings
from typing import NotRequired, TypedDict

import numpy as np
from numpy.typing import NDArray

from nns._helpers import _fast_lm
from nns._rmean import (
    r_mean,
    r_rank_average,
    r_round_array,
    r_row_means,
    r_sd,
    r_sum,
    r_trimmed_mean,
)
from nns._rnmath import qnorm
from nns._roptim import r_optimize
from nns._rrng import RRNG, r_rng
from nns.dependence import nns_dep

_EPS = sys.float_info.epsilon


class MebootResult(TypedDict):
    """``nns_meboot`` per-``rho`` result following R's NNS.meboot diagnostics.

    Degenerate inputs return partial dictionaries (single-observation input
    yields only ``x``), hence every key is optional.
    """

    x: NotRequired[NDArray[np.float64]]
    replicates: NotRequired[NDArray[np.float64]]
    ensemble: NotRequired[NDArray[np.float64]]
    xx: NotRequired[NDArray[np.float64]]
    z: NotRequired[NDArray[np.float64]]
    dv: NotRequired[NDArray[np.float64]]
    dvtrim: NotRequired[float]
    xmin: NotRequired[float]
    xmax: NotRequired[float]
    desintxb: NotRequired[NDArray[np.float64]]
    ordxx: NotRequired[NDArray[np.float64]]
    kappa: NotRequired[float | None]


def nns_meboot(
    x: np.ndarray,
    reps: int = 999,
    rho: float | list[float] | np.ndarray | None = None,
    type: str = "spearman",
    drift: bool = True,
    target_drift: float | list[float] | np.ndarray | None = None,
    target_drift_scale: float | list[float] | np.ndarray | None = None,
    trim: float = 0.10,
    xmin: float | None = None,
    xmax: float | None = None,
    reachbnd: bool = True,
    expand_sd: bool = True,
    force_clt: bool = True,
    scl_adjustment: bool = False,
    sym: bool = False,
    elaps: bool = False,
    digits: int = 6,
    random_seed: int | None = None,
    expand_residual_sd: bool = True,
) -> MebootResult | list[MebootResult]:
    """Maximum-entropy bootstrap matching R's NNS.meboot.

    With ``random_seed = s`` the replicates are R's after ``set.seed(s)``: the
    draws come from R's Mersenne-Twister stream. Without a seed the stream is
    seeded from fresh entropy.

    ``rho``, ``target_drift`` and ``target_drift_scale`` are vectorised like R's
    ``Vectorize``/``mapply``: the given arguments are recycled elementwise to the
    longest length (not crossed) and a list with one result per position is
    returned; if all of them have length 1 a single result is returned, and if
    none is given there is nothing to map over and, as in R, the result is empty.
    Given only ``target_drift``/``target_drift_scale``, the replicates are drawn
    without a dependence target.

    When the detrended residuals are all exactly zero (a constant or exactly linear
    series) there is nothing to align and ``rho`` has no effect.
    """
    del elaps  # R elapsed-time printout flag; NNS Python emits no console output.
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("x must be a 1D numeric vector.")
    return _nns_meboot(
        values,
        reps,
        rho,
        type,
        drift,
        target_drift,
        target_drift_scale,
        trim,
        xmin,
        xmax,
        reachbnd,
        expand_sd,
        force_clt,
        scl_adjustment,
        sym,
        digits,
        expand_residual_sd,
        r_rng(random_seed),
    )


def _nns_meboot(
    values: NDArray[np.float64],
    reps: int,
    rho: float | list[float] | np.ndarray | None,
    type_: str,
    drift: bool,
    target_drift: float | list[float] | np.ndarray | None,
    target_drift_scale: float | list[float] | np.ndarray | None,
    trim: float | None,
    xmin: float | None,
    xmax: float | None,
    reachbnd: bool,
    expand_sd: bool,
    force_clt: bool,
    scl_adjustment: bool,
    sym: bool,
    digits: int,
    expand_residual_sd: bool,
    rng: RRNG,
) -> MebootResult | list[MebootResult]:
    """R's ``Vectorize``d ``NNS.meboot`` on one shared random stream."""
    rho_values = _optional_vector(rho)
    drift_values = _optional_vector(target_drift)
    scale_values = _optional_vector(target_drift_scale)
    vectorised = [v for v in (rho_values, drift_values, scale_values) if v is not None]
    if not vectorised:
        # mapply() over no vectorised argument: R returns list().
        return {}
    count = 0 if any(v.size == 0 for v in vectorised) else max(v.size for v in vectorised)
    if any(count % v.size for v in vectorised if v.size):
        warnings.warn(
            "longer argument not a multiple of length of shorter", RuntimeWarning, stacklevel=3
        )

    # R lets NaN/Inf arise silently (a degenerate replicate's sd ratios, say); so
    # does the port, rather than emitting NumPy floating-point warnings.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        results = [
            _meboot_call(
                values,
                reps,
                None if rho_values is None else float(rho_values[i % rho_values.size]),
                type_.lower(),
                drift,
                None if drift_values is None else float(drift_values[i % drift_values.size]),
                None if scale_values is None else float(scale_values[i % scale_values.size]),
                trim,
                xmin,
                xmax,
                reachbnd,
                expand_sd,
                force_clt,
                scl_adjustment,
                sym,
                digits,
                expand_residual_sd,
                rng,
            )
            for i in range(count)
        ]
    if count == 1:
        return results[0]
    return results


def _optional_vector(value: float | list[float] | np.ndarray | None) -> NDArray[np.float64] | None:
    if value is None:
        return None
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _meboot_call(
    x: NDArray[np.float64],
    reps: int,
    rho: float | None,
    type_: str,
    drift: bool,
    target_drift: float | None,
    target_drift_scale: float | None,
    trim: float | None,
    xmin_arg: float | None,
    xmax_arg: float | None,
    reachbnd: bool,
    expand_sd: bool,
    force_clt: bool,
    scl_adjustment: bool,
    sym: bool,
    digits: int,
    expand_residual_sd: bool,
    rng: RRNG,
) -> MebootResult:
    """One ``NNS.meboot`` call (one ``rho`` / drift target)."""
    n = x.size
    if n == 1:
        return {"x": x.copy()}
    if n == 0:
        raise ValueError("x must be non-empty.")
    if np.any(np.isnan(x)):
        raise ValueError("You have some missing values, please address.")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must contain only finite values.")
    if reps < 1:
        raise ValueError("reps must be positive.")
    trimval = 0.1 if trim is None else float(trim)

    # --- Fit the linear trend once (time order) and take its residuals.
    time = np.arange(1, n + 1, dtype=np.float64)
    intercept, orig_drift = _fast_lm(time, x)
    orig_res = x - (intercept + orig_drift * time)
    # All-zero residuals have no ranks for a rho target and no spread to rescale.
    zero_res = bool(np.all(orig_res == 0.0))

    if target_drift is not None or target_drift_scale is not None:
        drift = True
    if drift:
        if target_drift_scale is not None:
            target_drift = orig_drift * target_drift_scale
        elif target_drift is None:
            target_drift = orig_drift
        recon_slope = float(target_drift)
    else:
        recon_slope = 0.0
    # R keeps the original reconstruction whenever the slope is unchanged up to
    # rounding: relative to sd(x) / n (the slope of a trend spanning one sd over
    # the sample) plus the slope fit's own rounding at the level of x.
    slope_rounding = 8.0 * _EPS * float(np.max(np.abs(x))) / r_sd(time)
    same_slope = abs(recon_slope - orig_drift) <= math.sqrt(_EPS) * max(
        abs(orig_drift), r_sd(x) / n
    ) + slope_rounding
    if same_slope:
        recon_slope = orig_drift
    # A changed slope pivots about mean(x), so the replicates keep the level.
    baseline = (
        intercept + recon_slope * time
        if same_slope
        else r_mean(x) + recon_slope * (time - (n + 1) / 2.0)
    )

    # --- Meboot core on the residuals: order statistics, midpoints and tails.
    ordxx = np.argsort(orig_res, kind="stable")  # R's order() is stable
    xx = orig_res[ordxx]
    if sym:
        xx = r_mean(xx) + 0.5 * (xx - xx[::-1])
    z = (xx[1:] + xx[:-1]) / 2.0
    dv = np.abs(np.diff(orig_res))
    dvtrim = r_trimmed_mean(dv, trimval)
    xmin = xx[0] - dvtrim if xmin_arg is None else float(xmin_arg)
    xmax = xx[-1] + dvtrim if xmax_arg is None else float(xmax_arg)
    if (xmin_arg is not None or xmax_arg is not None) and force_clt:
        # R: if (isTRUE(force.clt)) { expand.sd <- FALSE; force.clt <- FALSE }
        force_clt = False
        expand_sd = False

    # Theil-Laitinen interval means: colSums of the weighted neighbours, which R
    # accumulates in long double.
    if n > 2:
        aux_ld = (xx[2:] * 0.25).astype(np.longdouble) + xx[1:-1] * 0.5 + xx[:-2] * 0.25
        aux = np.asarray(aux_ld, dtype=np.float64)
    else:
        aux = np.empty(0, dtype=np.float64)
    desintxb = np.concatenate(
        ([0.75 * xx[0] + 0.25 * xx[1]], aux, [0.25 * xx[-2] + 0.75 * xx[-1]])
    )

    # replicate(reps, NNS.meboot.part(...)): each call draws runif(n).
    draws = rng.unif_rand_array(n * reps).reshape(reps, n).T
    res_mat = _meboot_part(xx, n, z, xmin, xmax, desintxb, reachbnd, draws)
    res_mat[ordxx, :] = np.sort(res_mat, axis=0)

    if rho is not None and not zero_res:
        res_mat = _target_rho(res_mat, orig_res, rho, type_)

    # Variance expansion on the residuals (match the sd of the original residuals).
    if expand_residual_sd:
        res_mat = _meboot_expand_sd(orig_res, res_mat, rng)

    # The ME density lives on the residuals, so scale them against their own mean and sd.
    if scl_adjustment and not zero_res:
        zz = np.concatenate(([xmin], z, [xmax]))
        v = np.diff(zz * zz) / 12.0
        xb = r_mean(orig_res)
        s1 = r_sum((desintxb - xb) * (desintxb - xb))
        uv = (s1 + r_sum(v)) / n
        desired_sd = r_sd(orig_res)
        actual_me_sd = math.sqrt(uv)
        if actual_me_sd <= 0.0:
            raise ValueError("actualME.sd<=0 Error")
        kappa_value = (desired_sd / actual_me_sd) - 1.0
        res_mat = res_mat + kappa_value * (res_mat - xb)
        kappa: float | None = kappa_value
    else:
        kappa = None

    # Reconstruct the levels. (R's legacy reshuffle when ordxx equals its own
    # reverse can only fire for n = 1, which returned above.)
    ensemble = res_mat + baseline[:, np.newaxis]

    # sd(x) includes the original drift, so expand toward it only when it is kept.
    if expand_sd and same_slope:
        ensemble = _meboot_expand_sd(x, ensemble, rng)
    if force_clt and reps > 1:
        clt_ref = x if same_slope else baseline + orig_res
        if r_sd(clt_ref) > 0.0:
            ensemble = _force_clt(clt_ref, ensemble)

    if xmin_arg is not None:
        ensemble = np.maximum(ensemble, float(xmin_arg))
    if xmax_arg is not None:
        ensemble = np.minimum(ensemble, float(xmax_arg))

    return {
        "x": x.copy(),
        "replicates": r_round_array(ensemble, digits),
        "ensemble": r_row_means(ensemble),
        "xx": xx,
        "z": z,
        "dv": dv,
        "dvtrim": float(dvtrim),
        "xmin": float(xmin),
        "xmax": float(xmax),
        "desintxb": desintxb,
        "ordxx": ordxx.astype(np.float64) + 1.0,
        "kappa": kappa,
    }


def _meboot_part(
    xx: NDArray[np.float64],
    n: int,
    z: NDArray[np.float64],
    xmin: float,
    xmax: float,
    desintxb: NDArray[np.float64],
    reachbnd: bool,
    p: NDArray[np.float64],
) -> NDArray[np.float64]:
    """R's C++ ``NNS_meboot_part`` for every replicate column of the draws ``p``."""
    m = xx.size
    if m == 0:
        q = np.full(p.shape, np.nan, dtype=np.float64)
    elif m == 1:
        q = np.full(p.shape, xx[0], dtype=np.float64)
    else:
        h = 1.0 + (m - 1.0) * p
        j = np.clip(np.floor(h), 1.0, m - 1.0)
        g = h - j
        index = j.astype(np.intp)
        q = (1.0 - g) * xx[index - 1] + g * xx[index]
        q = np.where(p <= 0.0, xx[0], q)
        q = np.where(p >= 1.0, xx[m - 1], q)

    invn = 1.0 / n
    lower = xmin + (p - 0.0) * (z[0] - xmin) / (invn - 0.0)
    if not reachbnd:
        lower = lower + desintxb[0] - 0.5 * (z[0] + xmin)
    q = np.where(p <= invn, lower, q)

    edge = (n - 1.0) / n
    upper = z[n - 2] + (p - edge) * (xmax - z[n - 2]) / (1.0 - edge)
    if not reachbnd:
        upper = upper + desintxb[n - 1] - 0.5 * (z[n - 2] + xmax)
    return np.asarray(np.where(p >= edge, upper, q), dtype=np.float64)


def _target_rho(
    res_mat: NDArray[np.float64],
    orig_res: NDArray[np.float64],
    rho: float,
    type_: str,
) -> NDArray[np.float64]:
    """Blend each replicate's aligned and anti-aligned orderings to hit ``rho``.

    R mixes ``comb = t * m + (1 - t) * e`` with the single weight ``t`` from
    ``optimize(obj, c(0, 1), tol = 0.01)``; the correlation is a closed form
    against the target centred and normed once.
    """
    n = orig_res.size
    r_o = r_rank_average(orig_res)
    r_anti = float(np.max(r_o)) + 1.0 - r_o
    # Average ranks can be fractional under ties: floor and clamp to 1..n, as R's
    # integer indexing does.
    r_o_idx = np.clip(np.floor(r_o).astype(np.int64), 1, n) - 1
    r_anti_idx = np.clip(np.floor(r_anti).astype(np.int64), 1, n) - 1
    use_spearman = type_ == "spearman"
    use_corr_path = type_ in {"spearman", "pearson"}
    target_values = r_o if use_spearman else orig_res
    target_centered = target_values - r_mean(target_values)
    target_norm = math.sqrt(r_sum(target_centered * target_centered))
    if target_norm == 0.0 or not math.isfinite(target_norm):
        raise ValueError("function cannot be evaluated at initial parameters")

    out = res_mat.copy()
    for i in range(out.shape[1]):
        res_sorted = np.sort(out[:, i])
        e_values = res_sorted[r_o_idx]  # aligned with the ranks of orig_res
        m_values = res_sorted[r_anti_idx]  # anti-aligned

        def objective(
            t: float,
            e_: NDArray[np.float64] = e_values,
            m_: NDArray[np.float64] = m_values,
        ) -> float:
            comb = t * m_ + (1.0 - t) * e_
            if use_corr_path:
                corr = _fast_corr(comb, target_centered, target_norm, use_spearman)
            elif type_ == "nnsdep":
                corr = float(nns_dep(comb, orig_res)["Dependence"])
            else:
                corr = float(nns_dep(comb, orig_res)["Correlation"])
            if not math.isfinite(corr):
                return math.inf
            return abs(corr - rho)

        t_hat = r_optimize(objective, 0.0, 1.0, tol=0.01)
        out[:, i] = t_hat * m_values + (1.0 - t_hat) * e_values
    return out


def _fast_corr(
    values: NDArray[np.float64],
    target_centered: NDArray[np.float64],
    target_norm: float,
    use_spearman: bool,
) -> float:
    """R's closed-form Pearson/Spearman correlation against the fixed target."""
    xv = r_rank_average(values) if use_spearman else values
    xc = xv - r_mean(xv)
    xn = math.sqrt(r_sum(xc * xc))
    if xn == 0.0:
        return math.nan
    return r_sum(xc * target_centered) / (xn * target_norm)


def _meboot_expand_sd(
    x: NDArray[np.float64],
    ensemble: NDArray[np.float64],
    rng: RRNG,
    fiv: float = 5.0,
) -> NDArray[np.float64]:
    """R's C++ ``NNS_meboot_expand_sd`` for a vector ``x``.

    A column whose sd falls short of ``sd(x)`` is scaled up to it times a factor
    drawn by ``R::runif(1, 1 + fiv / 100)``, in column order.
    """
    out = ensemble.copy()
    sdf = np.concatenate(([_vec_sd(np.asarray(x, dtype=np.float64))], _col_sd(out)))
    with np.errstate(divide="ignore", invalid="ignore"):
        sdfa = sdf / sdf[0]
        sdfd = sdf[0] / sdf
    mx = 1.0 + (fiv / 100.0)
    for i in range(sdfa.size):
        if sdfa[i] < 1.0:
            sdfa[i] = rng.runif(1.0, mx)
    with np.errstate(invalid="ignore"):
        factors = sdfd[1:] * sdfa[1:]
    for j, factor in enumerate(factors):
        if np.floor(factor) > 0.0:
            out[:, j] *= factor
    return out


def _force_clt(x: NDArray[np.float64], ensemble: NDArray[np.float64]) -> NDArray[np.float64]:
    """R's C++ ``force_clt`` for a vector ``x``.

    The replicate means are respread like means of ``n`` observations,
    ``sd(x) / sqrt(n)`` with ``n`` the series length (R 14 fixed meboot's division
    by the number of replicates, which shrank the spread as ``reps`` grew), at the
    ``qnorm(j / (reps + 1))`` quantiles, keeping their order.
    """
    n, bigj = ensemble.shape
    gm = float(np.cumsum(x)[-1]) / x.size
    smean = _vec_sd(x) / math.sqrt(n)
    xbar = np.cumsum(ensemble, axis=0)[-1] / n
    order = np.argsort(xbar, kind="stable")
    sortxbar = np.sort(xbar)
    newbar = np.array([gm + qnorm((j + 1) / (bigj + 1)) * smean for j in range(bigj)])
    mu_nb = float(np.cumsum(newbar)[-1]) / bigj
    deviations = newbar - mu_nb
    sd_nb = math.sqrt(float(np.cumsum(deviations * deviations)[-1]) / (bigj - 1))
    out = ensemble.copy()
    for i, col in enumerate(order):
        add = (((newbar[i] - mu_nb) / sd_nb) * smean + gm) - sortxbar[i]
        out[:, col] = ensemble[:, col] + add
    return out


def _vec_sd(values: NDArray[np.float64]) -> float:
    """R's C++ ``vecSd``: sequential double mean and sum of squares."""
    n = values.size
    if n <= 1:
        return math.nan
    mu = float(np.cumsum(values)[-1]) / n
    deviations = values - mu
    return math.sqrt(float(np.cumsum(deviations * deviations)[-1]) / (n - 1))


def _col_sd(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """R's C++ ``colSd``: :func:`_vec_sd` of each column."""
    n = values.shape[0]
    if n <= 1:
        return np.full(values.shape[1], np.nan, dtype=np.float64)
    mu = np.cumsum(values, axis=0)[-1] / n
    deviations = values - mu
    return np.asarray(
        np.sqrt(np.cumsum(deviations * deviations, axis=0)[-1] / (n - 1)), dtype=np.float64
    )
