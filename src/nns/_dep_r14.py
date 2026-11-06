"""Source-faithful R 14.0 multilevel dependence and causation levels.

Port of OVVO-Financial/NNS@7b07a837 src/NNS_dep.cpp. This is deliberately
separate from the R-13.3 native kernel until the NNS-core port is synchronized.
No statistical redefinition, softened comparisons, or fixture substitutions.
"""
from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
_EPS = sys.float_info.epsilon
_SQRT_2_PI = math.sqrt(2.0 / math.pi)


def _sum_ordered(values: FloatArray) -> float:
    # C++ double accumulation in input order, rather than np.sum's pairwise reduction.
    return float(np.cumsum(values, dtype=np.float64)[-1]) if values.size else 0.0


def _has_ties(values: FloatArray) -> bool:
    return bool(np.any(np.diff(np.sort(values)) == 0.0))


def _null_t0(m: int, a: int, b: int) -> float:
    """Hypergeometric E[|2*concordance/m-1|], R null_T0."""
    k_lo, k_hi = max(0, a + b - m), min(a, b)
    if k_lo >= k_hi:
        return 1.0
    md = float(m)
    if m > 400:
        ad, bd = float(a), float(b)
        mean_k = ad * bd / md
        var_k = bd * (ad / md) * ((md - ad) / md) * ((md - bd) / (md - 1.0))
        mu = (2.0 * (md - ad - bd + 2.0 * mean_k) - md) / md
        sd = 4.0 * math.sqrt(max(var_k, 0.0)) / md
        if sd <= 0.0:
            return abs(mu)
        z = mu / sd
        return sd * _SQRT_2_PI * math.exp(-0.5 * z * z) + mu * math.erf(z / math.sqrt(2.0))

    # Exact integer combinatorics to establish the first probability, followed
    # by the same floating-point ratio recurrence as the R/C++ oracle.
    pk = math.comb(a, k_lo) * math.comb(m - a, b - k_lo) / math.comb(m, b)
    e = 0.0
    for k in range(k_lo, k_hi + 1):
        conc = float(m - a - b + 2 * k)
        e += pk * abs(2.0 * conc / md - 1.0)
        if k < k_hi:
            pk *= (float(a - k) * float(b - k)) / (
                float(k + 1) * float(m - a - b + k + 1)
            )
    return e


def _log_arrangements(values: FloatArray) -> float:
    _, multiplicity = np.unique(values, return_counts=True)
    out = math.lgamma(float(values.size) + 1.0)
    for count in multiplicity:
        out -= math.lgamma(float(count) + 1.0)
    return out


def _next_permutation(v: list[float]) -> bool:
    """C++ next_permutation on sorted doubles; handles tied values exactly."""
    i = len(v) - 2
    while i >= 0 and v[i] >= v[i + 1]:
        i -= 1
    if i < 0:
        return False
    j = len(v) - 1
    while v[j] <= v[i]:
        j -= 1
    v[i], v[j] = v[j], v[i]
    v[i + 1 :] = reversed(v[i + 1 :])
    return True


def _null_t1(
    dx: FloatArray, dy: FloatArray, x_ties: bool, y_ties: bool, legacy: bool = False
) -> float:
    """Permutation-null expectation of the absolute degree-1 orthant ratio."""
    m = int(dx.size)
    lx, ly = 1e300, 1e300
    if m <= 6 or (x_ties and not legacy):
        lx = _log_arrangements(dx)
    if m <= 6 or (y_ties and not legacy):
        ly = _log_arrangements(dy)
    if min(lx, ly) <= math.log(720.0) + 1e-9:
        v = sorted(dx.tolist() if lx <= ly else dy.tolist())
        w = dy if lx <= ly else dx
        total = 0.0
        count = 0
        while True:
            s = 0.0
            d = 0.0
            for i, vi in enumerate(v):
                prod = float(w[i]) * vi
                s += prod
                d += abs(prod)
            total += abs(s) / d if d > 0.0 else 0.0
            count += 1
            if not _next_permutation(v):
                break
        return total / count

    sx2, sy2, sax, say = 0.0, 0.0, 0.0, 0.0
    for xx, yy in zip(dx, dy, strict=True):
        a, b = float(xx), float(yy)
        sx2 += a * a
        sy2 += b * b
        sax += abs(a)
        say += abs(b)
    if sax <= 0.0 or say <= 0.0:
        return 1.0
    md = float(m)
    return min(1.0, _SQRT_2_PI * math.sqrt(sx2 * sy2 / (md - 1.0)) * md / (sax * say))


def _debias(t: float, expectation: float) -> float:
    return 0.0 if not expectation < 1.0 - 1e-12 else (t - expectation) / (1.0 - expectation)


# Cell kind: 0 varying, 1 constant in first variable, 2 both constant.
def _cell(
    xv: FloatArray,
    yv: FloatArray,
    x_ties: bool,
    y_ties: bool,
    legacy: bool = False,
) -> tuple[float, float, int]:
    m = int(xv.size)
    if m < 2:
        return 0.0, 0.0, 0
    rx = float(np.min(xv)) if not legacy else 0.0
    ry = float(np.min(yv)) if not legacy else 0.0
    shifted_x, shifted_y = xv - rx, yv - ry
    tx, ty = _sum_ordered(shifted_x) / m, _sum_ordered(shifted_y) / m
    ax, ay = float(np.max(shifted_x)), float(np.max(shifted_y))
    tie_x = 0.0 if legacy else (m + 1) * _EPS * ax
    tie_y = 0.0 if legacy else (m + 1) * _EPS * ay
    dx, dy = shifted_x - tx, shifted_y - ty
    low_x, low_y = dx <= tie_x, dy <= tie_y
    a, b = int(np.count_nonzero(low_x)), int(np.count_nonzero(low_y))
    conc = int(np.count_nonzero(low_x == low_y))
    product = dx * dy
    s = _sum_ordered(product)
    d = _sum_ordered(np.abs(product))

    if bool(np.all(xv == xv[0])):
        return 0.0, 0.0, 2 if bool(np.all(yv == yv[0])) else 1
    s_tol = 0.0 if legacy else 1e-12 * d
    slope_sign = 1.0 if s > s_tol else (-1.0 if s < -s_tol else 0.0)
    e0 = _null_t0(m, a, b)
    if e0 < 1.0 - 1e-12:
        if conc == m:
            return 1.0, 1.0, 0
        if conc == 0:
            return -1.0, 1.0, 0
    t0 = abs(2.0 * float(conc) / float(m) - 1.0)
    t1 = abs(s) / d if d > 0.0 else 0.0
    e1 = _null_t1(dx, dy, x_ties, y_ties, legacy)
    dep = 0.5 * (_debias(t0, e0) + _debias(t1, e1))
    if slope_sign == 0.0:
        return 0.0, 0.0, 0
    return dep * slope_sign, dep, 0


def _between_sign(x: FloatArray, y: FloatArray) -> float:
    n = int(x.size)
    x0, y0 = float(x[0]), float(y[0])
    mx = _sum_ordered(x - x0) / n
    my = _sum_ordered(y - y0) / n
    p = ((x - x0) - mx) * ((y - y0) - my)
    cov = _sum_ordered(p)
    tol = 1e-12 * _sum_ordered(np.abs(p))
    return 1.0 if cov > tol else (-1.0 if cov < -tol else 0.0)


def _p_constant(
    m: int, counts: list[int], n: int, cache: dict[int, float]
) -> float:
    if m not in cache:
        denom = math.comb(n, m)
        p = sum(math.comb(c, m) / denom for c in counts if c >= m)
        cache[m] = min(1.0, p)
    return cache[m]


def _best_level(
    u: FloatArray,
    v: FloatArray,
    labels: Sequence[str],
    between_sign: float,
    conditional: bool,
    u_ties: bool,
    v_ties: bool,
    levels: list[float] | None = None,
) -> tuple[float, float]:
    """R best_level: all distinct prefix partitions, weighted cell copulas."""
    n = int(u.size)
    order = sorted(range(n), key=lambda i: labels[i])
    longest = max((len(label) for label in labels), default=1)
    v_counts = sorted(
        np.unique(v, return_counts=True)[1].tolist(), reverse=True
    ) if conditional and v_ties else []
    p0_cache: dict[int, float] = {}
    best_dep, best_corr = -1e300, 0.0
    previous_cell_count = -1

    for length in range(1, longest + 1):
        starts = [0]
        for k in range(1, n):
            if labels[order[k]][:length] != labels[order[k - 1]][:length]:
                starts.append(k)
        cell_count = len(starts)
        if cell_count == previous_cell_count:
            continue
        previous_cell_count = cell_count
        starts.append(n)

        dep, corr = 0.0, 0.0
        for i in range(cell_count):
            ids = order[starts[i] : starts[i + 1]]
            m = len(ids)
            uq = u[ids]
            vq = v[ids]
            signed, magnitude, kind = _cell(uq, vq, u_ties, v_ties)
            if kind:
                mag = 0.0
                if conditional:
                    p0 = _p_constant(m, v_counts, n, p0_cache) if v_ties else 0.0
                    score = 1.0 if kind == 2 else 0.0
                    if p0 < 1.0 - 1e-12:
                        mag = (score - p0) / (1.0 - p0)
                signed, magnitude = between_sign * mag, mag
            if not (math.isfinite(signed) and math.isfinite(magnitude)):
                signed, magnitude = 0.0, 0.0
            weight = float(m) / float(n)
            dep += weight * magnitude
            corr += weight * signed
        if levels is not None:
            levels.append(dep)
        if dep >= best_dep:
            best_dep, best_corr = dep, corr

    dep = max(0.0, best_dep)
    return min(dep, max(-dep, best_corr)), dep


def _labels(
    x: FloatArray, y: FloatArray, *, xonly: bool, obs: int
) -> list[str]:
    # Import locally: nns.part uses the shared nns.dependence._gravity and
    # must not be imported while dependence.py is still being initialized.
    from nns.part import _nns_part

    part = _nns_part(
        x, y, type="XONLY" if xonly else None, order=None,
        obs_req=obs, min_obs_stop=False, noise_reduction="off",
    )
    return [str(q) for q in part["dt"]["quadrant"]]


def dep_pair_r14(x: FloatArray, y: FloatArray, asym: bool) -> tuple[float, float]:
    """R 14.0 NNS.dep (bivariate): strongest of 1 or 3 partition families."""
    n = int(x.size)
    if n < 2 or bool(np.all(x == x[0])) or bool(np.all(y == y[0])):
        return 0.0, 0.0
    obs = max(8, n // 8)
    quad_xy = _labels(x, y, xonly=True, obs=obs)
    sign = _between_sign(x, y)
    x_ties, y_ties = _has_ties(x), _has_ties(y)
    best = _best_level(x, y, quad_xy, sign, True, x_ties, y_ties)
    if asym:
        return best
    joint = _best_level(
        x, y, _labels(x, y, xonly=False, obs=obs),
        sign, False, x_ties, y_ties,
    )
    on_y = _best_level(
        y, x, _labels(y, x, xonly=True, obs=obs),
        sign, True, y_ties, x_ties,
    )
    out = joint
    if best[1] > out[1]:
        out = best
    if on_y[1] > out[1]:
        out = on_y
    return out


def dep_levels_r14(x: FloatArray, y: FloatArray) -> list[float]:
    """R NNS_dep_levels_cpp: all distinct unfloored directional level scores."""
    n = int(x.size)
    if n < 2 or bool(np.all(x == x[0])) or bool(np.all(y == y[0])):
        return [0.0]
    scores: list[float] = []
    _best_level(
        x, y, _labels(x, y, xonly=True, obs=max(8, n // 8)),
        0.0, True, _has_ties(x), _has_ties(y), scores,
    )
    return scores
