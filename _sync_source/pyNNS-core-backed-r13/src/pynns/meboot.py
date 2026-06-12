from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from pynns._helpers import _fast_lm
from pynns.dependence import nns_dep

MebootResult = dict[str, NDArray[np.float64] | float | None]


def nns_meboot(
    x: np.ndarray,
    reps: int = 999,
    rho: float | list[float] | np.ndarray | None = None,
    type: str = "spearman",
    drift: bool = True,
    target_drift: float | None = None,
    target_drift_scale: float | None = None,
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
) -> dict[str, Any] | list[dict[str, Any]]:
    """Maximum-entropy bootstrap matching R's NNS.meboot structure.

    Stochastic draws use NumPy's RNG, so exact replicate parity with R is not
    expected. Deterministic diagnostics follow the installed R algorithm.
    """
    del elaps
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("x must be a 1D numeric vector.")
    if values.size == 0:
        raise ValueError("x must be non-empty.")
    if np.any(np.isnan(values)):
        raise ValueError("You have some missing values, please address.")
    if not np.all(np.isfinite(values)):
        raise ValueError("x must contain only finite values.")
    if values.size == 1:
        return {"x": values.copy()}
    if rho is None:
        return {}
    if reps < 1:
        raise ValueError("reps must be positive.")

    rng = np.random.default_rng(random_seed)
    rho_values = np.asarray(rho, dtype=np.float64).reshape(-1)
    if rho_values.size == 1:
        return _nns_meboot_one(
            values,
            reps,
            float(rho_values[0]),
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
            rng,
        )

    return [
        _nns_meboot_one(
            values,
            reps,
            float(rho_item),
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
            rng,
        )
        for rho_item in rho_values
    ]


def _nns_meboot_one(
    x: NDArray[np.float64],
    reps: int,
    rho: float,
    type_: str,
    drift: bool,
    target_drift: float | None,
    target_drift_scale: float | None,
    trim: float,
    xmin_arg: float | None,
    xmax_arg: float | None,
    reachbnd: bool,
    expand_sd: bool,
    force_clt: bool,
    scl_adjustment: bool,
    sym: bool,
    digits: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    n = x.size
    time = np.arange(1, n + 1, dtype=np.float64)
    intercept, orig_drift = _fast_lm(time, x)
    orig_res = x - (intercept + orig_drift * time)

    if target_drift is not None or target_drift_scale is not None:
        drift = True
    if drift:
        if target_drift_scale is not None:
            target = orig_drift * target_drift_scale
        elif target_drift is None:
            target = orig_drift
        else:
            target = target_drift
        recon_slope = target
    else:
        recon_slope = 0.0
    baseline = intercept + recon_slope * time

    xx = np.sort(orig_res)
    ordxx_zero = np.argsort(orig_res, kind="stable")
    ordxx = ordxx_zero.astype(np.float64) + 1.0
    if sym:
        xx = float(np.mean(xx)) + 0.5 * (xx - xx[::-1])

    z = (xx[1:] + xx[:-1]) / 2.0
    dv = np.abs(np.diff(orig_res.astype(np.float64)))
    dvtrim = _trimmed_mean(dv, trim)
    xmin = float(xx[0] - dvtrim) if xmin_arg is None else float(xmin_arg)
    xmax = float(xx[-1] + dvtrim) if xmax_arg is None else float(xmax_arg)
    if xmin_arg is not None or xmax_arg is not None:
        force_clt = False
        expand_sd = False

    aux = 0.25 * xx[:-2] + 0.5 * xx[1:-1] + 0.25 * xx[2:]
    desintxb = np.concatenate(
        (
            np.array([0.75 * xx[0] + 0.25 * xx[1]], dtype=np.float64),
            aux,
            np.array([0.25 * xx[-2] + 0.75 * xx[-1]], dtype=np.float64),
        )
    )

    res_mat = np.column_stack(
        [_meboot_part(xx, n, z, xmin, xmax, desintxb, reachbnd, rng) for _ in range(reps)]
    )
    qseq = np.sort(res_mat, axis=0)
    res_mat[ordxx_zero, :] = qseq

    res_mat = _target_rho(res_mat, orig_res, rho, type_.lower())
    res_mat = _meboot_expand_sd(orig_res, res_mat, rng)
    ensemble = res_mat + baseline[:, np.newaxis]

    if np.array_equal(ordxx_zero[::-1], ordxx_zero) and reps > 1:
        for i in range(ensemble.shape[0]):
            ensemble[i, :] = rng.choice(ensemble[i, :], size=reps, replace=True)

    if expand_sd:
        ensemble = _meboot_expand_sd(x, ensemble, rng)
    if force_clt and reps > 1:
        ensemble = _force_clt(x, ensemble)

    if scl_adjustment:
        zz = np.concatenate(([xmin], z, [xmax]))
        v = np.diff(zz**2) / 12.0
        xb = float(np.mean(x))
        s1 = float(np.sum((desintxb - xb) ** 2))
        uv = (s1 + float(np.sum(v))) / n
        desired_sd = _sample_sd(x)
        actual_me_sd = float(np.sqrt(uv))
        if actual_me_sd <= 0.0:
            raise ValueError("actualME.sd<=0 Error")
        kappa = (desired_sd / actual_me_sd) - 1.0
        ensemble = ensemble + kappa * (ensemble - xb)
    else:
        kappa = None

    if xmin_arg is not None:
        ensemble = np.maximum(float(xmin_arg), ensemble)
    if xmax_arg is not None:
        ensemble = np.minimum(float(xmax_arg), ensemble)

    return {
        "x": x.copy(),
        "replicates": np.round(ensemble, digits),
        "ensemble": np.mean(ensemble, axis=1),
        "xx": xx,
        "z": z,
        "dv": dv,
        "dvtrim": float(dvtrim),
        "xmin": float(xmin),
        "xmax": float(xmax),
        "desintxb": desintxb,
        "ordxx": ordxx,
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
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    p = rng.random(n)
    m = xx.size
    if m == 0:
        q = np.full(n, np.nan, dtype=np.float64)
    elif m == 1:
        q = np.full(n, xx[0], dtype=np.float64)
    else:
        h = 1.0 + (m - 1.0) * p
        j = np.floor(h).astype(np.int64)
        g = h - j
        j = np.clip(j, 1, m - 1)
        q = (1.0 - g) * xx[j - 1] + g * xx[j]
        q[p <= 0.0] = xx[0]
        q[p >= 1.0] = xx[-1]

    invn = 1.0 / n
    lower = p <= invn
    if np.any(lower):
        vals = _linear_interp(p[lower], 0.0, invn, xmin, float(z[0]))
        if not reachbnd:
            vals = vals + desintxb[0] - 0.5 * (z[0] + xmin)
        q[lower] = vals

    edge = (n - 1.0) / n
    upper = p >= edge
    if np.any(upper):
        vals = _linear_interp(p[upper], edge, 1.0, float(z[n - 2]), xmax)
        if not reachbnd:
            vals = vals + desintxb[n - 1] - 0.5 * (z[n - 2] + xmax)
        q[upper] = vals

    return q


def _target_rho(
    res_mat: NDArray[np.float64],
    orig_res: NDArray[np.float64],
    rho: float,
    type_: str,
) -> NDArray[np.float64]:
    from scipy.optimize import minimize_scalar  # type: ignore[import-untyped]

    r_o = _rank_average(orig_res)
    r_anti = float(np.max(r_o)) + 1.0 - r_o
    r_o_idx = np.clip(np.floor(r_o).astype(np.int64) - 1, 0, orig_res.size - 1)
    r_anti_idx = np.clip(np.floor(r_anti).astype(np.int64) - 1, 0, orig_res.size - 1)
    out = res_mat.copy()
    target_values = _rank_average(orig_res) if type_ == "spearman" else orig_res
    target_centered = target_values - float(np.mean(target_values))
    target_norm = float(np.sqrt(np.sum(target_centered * target_centered)))
    if target_norm == 0.0 or not np.isfinite(target_norm):
        raise ValueError("function cannot be evaluated at initial parameters")

    for j in range(out.shape[1]):
        res_sorted = np.sort(out[:, j])
        e_values = res_sorted[r_o_idx]
        m_values = res_sorted[r_anti_idx]

        def objective(
            t: float,
            e_: NDArray[np.float64] = e_values,
            m_: NDArray[np.float64] = m_values,
        ) -> float:
            comb = t * m_ + (1.0 - t) * e_
            if type_ in {"spearman", "pearson"}:
                corr = _fast_corr(comb, target_centered, target_norm, type_)
            elif type_ == "nnsdep":
                corr = nns_dep(comb, orig_res)["Dependence"]
            else:
                corr = nns_dep(comb, orig_res)["Correlation"]
            if not np.isfinite(corr):
                return np.inf
            return abs(float(corr) - rho)

        if not np.isfinite(objective(0.5)):
            raise ValueError("function cannot be evaluated at initial parameters")
        opt = minimize_scalar(
            objective,
            bounds=(0.0, 1.0),
            method="bounded",
            options={"xatol": 0.01, "maxiter": 20},
        )
        if not np.isfinite(opt.fun):
            raise ValueError("function cannot be evaluated at initial parameters")
        t = float(opt.x)
        out[:, j] = t * m_values + (1.0 - t) * e_values

    return out


def _meboot_expand_sd(
    x: NDArray[np.float64],
    ensemble: NDArray[np.float64],
    rng: np.random.Generator,
    fiv: float = 5.0,
) -> NDArray[np.float64]:
    out = ensemble.copy()
    sdx = np.array([_sample_sd(np.asarray(x, dtype=np.float64))], dtype=np.float64)
    ens_sd = _col_sd(out)
    sdf = np.concatenate((sdx, ens_sd))
    with np.errstate(divide="ignore", invalid="ignore"):
        sdfa = sdf / sdf[0]
        sdfd = sdf[0] / sdf

    mx = 1.0 + (fiv / 100.0)
    low = sdfa < 1.0
    if np.any(low):
        sdfa[low] = rng.uniform(1.0, mx, size=int(np.sum(low)))

    factors = sdfd[1:] * sdfa[1:]
    for j, factor in enumerate(factors):
        if np.floor(factor) > 0.0:
            out[:, j] *= factor
    return out


def _force_clt(x: NDArray[np.float64], ensemble: NDArray[np.float64]) -> NDArray[np.float64]:
    from scipy.stats import norm  # type: ignore[import-untyped]

    out = ensemble.copy()
    n_reps = out.shape[1]
    gm = float(np.mean(x))
    smean = _sample_sd(x) / np.sqrt(n_reps)
    xbar = np.mean(out, axis=0)
    order = np.argsort(xbar, kind="stable")
    sortxbar = np.sort(xbar)
    probs = np.arange(1, n_reps + 1, dtype=np.float64) / (n_reps + 1.0)
    newbar = gm + norm.ppf(probs) * smean
    sd_newbar = _sample_sd(newbar)
    if sd_newbar == 0.0 or not np.isfinite(sd_newbar):
        return out
    scn = (newbar - np.mean(newbar)) / sd_newbar
    newm = scn * smean + gm
    meanfix = newm - sortxbar
    for i, col in enumerate(order):
        out[:, col] = ensemble[:, col] + meanfix[i]
    return out


def _trimmed_mean(values: NDArray[np.float64], trim: float) -> float:
    if values.size == 0:
        return float("nan")
    ordered = np.sort(values)
    cut = int(np.floor(values.size * trim))
    if cut > 0 and 2 * cut < values.size:
        ordered = ordered[cut:-cut]
    return float(np.mean(ordered))


def _sample_sd(values: NDArray[np.float64]) -> float:
    if values.size < 2:
        return float("nan")
    return float(np.std(values, ddof=1))


def _col_sd(values: NDArray[np.float64]) -> NDArray[np.float64]:
    if values.shape[0] < 2:
        return np.full(values.shape[1], np.nan, dtype=np.float64)
    return np.asarray(np.std(values, axis=0, ddof=1), dtype=np.float64)


def _linear_interp(
    x: NDArray[np.float64],
    x0: float,
    x1: float,
    y0: float,
    y1: float,
) -> NDArray[np.float64]:
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)


def _fast_corr(
    x: NDArray[np.float64],
    target_centered: NDArray[np.float64],
    target_norm: float,
    method: str,
) -> float:
    x_values = _rank_average(x) if method == "spearman" else x
    x_centered = x_values - float(np.mean(x_values))
    x_norm = float(np.sqrt(np.sum(x_centered * x_centered)))
    if x_norm == 0.0 or target_norm == 0.0:
        return float("nan")
    return float(np.sum(x_centered * target_centered) / (x_norm * target_norm))


def _rank_average(x: NDArray[np.float64]) -> NDArray[np.float64]:
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks = np.empty(x.size, dtype=np.float64)
    if x.size == 0:
        return ranks
    group_start = np.concatenate(([0], np.flatnonzero(sorted_x[1:] != sorted_x[:-1]) + 1))
    group_end = np.concatenate((group_start[1:], [x.size]))
    group_size = group_end - group_start
    group_rank = 0.5 * (group_start + 1 + group_end)
    ranks[order] = np.repeat(group_rank, group_size)
    return ranks
