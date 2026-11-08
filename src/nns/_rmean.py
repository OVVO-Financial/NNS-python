"""R-compatible means (long-double accumulation, as R and NNS's C++ do).

``np.mean`` sums in double with pairwise blocks, so a group of tied values
can come out one ulp off (``np.mean(np.full(30, 19.58))`` is
19.580000000000002); R returns the tied value. These helpers accumulate
sequentially in ``np.longdouble`` (80-bit extended on x86-64, the same type R
and NNS's C++ use there) so the results match R bit for bit.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def r_mean(values: NDArray[np.float64]) -> float:
    """R's ``mean()`` of a double vector (summary.c): a long-double sum / n,
    then, when finite, the second pass ``s += sum(x - s) / n``."""
    v = np.asarray(values, dtype=np.float64).ravel()
    n = v.size
    if n == 0:
        return float("nan")
    ld = v.astype(np.longdouble)
    s = (np.cumsum(ld)[-1] + 0.0) / n  # R accumulates from 0.0L
    if np.isfinite(float(s)):
        s = s + np.cumsum(ld - s)[-1] / n
    return float(s)


def r_sum(values: NDArray[np.float64]) -> float:
    """R's ``sum()`` of a double vector (summary.c ``rsum``): one sequential
    long-double accumulation, rounded to double at the end."""
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size == 0:
        return 0.0
    return float(np.cumsum(v.astype(np.longdouble))[-1] + 0.0)


def r_mean_no_na(values: NDArray[np.float64]) -> float:
    """NNS.part's C++ ``mean_no_na``: long-double sum of the non-NaN values
    divided by their count (single pass)."""
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[~np.isnan(v)]
    if v.size == 0:
        return float("nan")
    # R starts from 0.0L; "+ 0.0" reproduces that for all-(-0.0) input.
    return float((np.cumsum(v.astype(np.longdouble))[-1] + 0.0) / v.size)


def r_group_means_no_na(
    values: NDArray[np.float64], groups: NDArray[np.intp], n_groups: int
) -> NDArray[np.float64]:
    """``r_mean_no_na`` of ``values[groups == g]`` for every g, each group
    accumulated in its original element order (``np.add.at`` is sequential)."""
    v = np.asarray(values, dtype=np.float64).ravel()
    keep = ~np.isnan(v)
    sums = np.zeros(n_groups, dtype=np.longdouble)
    np.add.at(sums, groups[keep], v[keep].astype(np.longdouble))
    counts = np.bincount(groups[keep], minlength=n_groups)
    out = np.full(n_groups, np.nan, dtype=np.float64)
    has = counts > 0
    out[has] = (sums[has] / counts[has]).astype(np.float64)
    return out


def _r_cov_mean(values: NDArray[np.longdouble]) -> np.longdouble:
    """cov.c ``MEAN_``: long-double sum / n, then the refinement pass."""
    n = values.size
    tmp = (np.cumsum(values)[-1] + np.longdouble(0.0)) / n
    if np.isfinite(float(tmp)):
        tmp = tmp + np.cumsum(values - tmp)[-1] / n
    # The mean is stored as a double (xm[i]) and reloaded into a long double.
    return np.longdouble(float(tmp))


def r_cor(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """R ``stats::cor(x, y)`` (Pearson, ``use = "everything"``) of two complete
    double vectors: cov.c ``cov_na_2``.

    Deviations from the double-stored means, their products and the sums are
    all long double; each standard deviation is ``sqrtl`` of the long-double
    variance rounded to double; the ratio is formed in double and clamped to
    [-1, 1]. NaN where a standard deviation is zero or ``n < 2``.
    """
    xv = np.asarray(x, dtype=np.float64).ravel().astype(np.longdouble)
    yv = np.asarray(y, dtype=np.float64).ravel().astype(np.longdouble)
    n = xv.size
    if n <= 1 or yv.size != n:
        return float("nan")
    dx = xv - _r_cov_mean(xv)
    dy = yv - _r_cov_mean(yv)
    n1 = np.longdouble(n - 1)
    cov = float(np.cumsum(dx * dy)[-1] / n1)
    sd_x = float(np.sqrt(np.cumsum(dx * dx)[-1] / n1))
    sd_y = float(np.sqrt(np.cumsum(dy * dy)[-1] / n1))
    if sd_x == 0.0 or sd_y == 0.0:
        return float("nan")
    return min(1.0, max(-1.0, cov / (sd_x * sd_y)))


def r_cor_matrix(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """R ``stats::cor(X)`` (Pearson, ``use = "everything"``) of a complete
    double matrix: cov.c ``cov_na_1``.

    Each covariance is a long-double sum of long-double products rounded to
    double; standard deviations are double ``sqrt`` of those variances; the
    correlation is a double ratio, clamped, with a unit diagonal. NaN where a
    standard deviation is zero.
    """
    m = np.asarray(x, dtype=np.float64)
    n, p = m.shape
    out = np.full((p, p), np.nan, dtype=np.float64)
    if n <= 1:
        return out
    ld = m.astype(np.longdouble)
    dev = np.column_stack([ld[:, j] - _r_cov_mean(ld[:, j]) for j in range(p)])
    n1 = np.longdouble(n - 1)
    cov = np.empty((p, p), dtype=np.float64)
    for i in range(p):
        for j in range(i + 1):
            cov[i, j] = cov[j, i] = float(np.cumsum(dev[:, i] * dev[:, j])[-1] / n1)
    sd = np.sqrt(np.diag(cov))
    for i in range(p):
        for j in range(i):
            if sd[i] == 0.0 or sd[j] == 0.0:
                out[i, j] = out[j, i] = np.nan
            else:
                out[i, j] = out[j, i] = min(1.0, max(-1.0, cov[i, j] / (sd[i] * sd[j])))
        out[i, i] = 1.0
    return out


def r_rank_average(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """R ``rank(x)`` with the default ``ties.method = "average"``."""
    v = np.asarray(values, dtype=np.float64).ravel()
    order = np.argsort(v, kind="stable")
    sorted_values = v[order]
    ranks = np.empty(v.size, dtype=np.float64)
    if v.size == 0:
        return ranks
    breaks = np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [v.size]))
    ranks[order] = np.repeat((starts + 1 + ends) / 2.0, ends - starts)
    return ranks


def r_row_means(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """R ``rowMeans(x)`` (array.c ``do_colsum``): each row accumulated column by
    column in long double, divided by the column count, rounded to double."""
    m = np.asarray(x, dtype=np.float64)
    sums = np.zeros(m.shape[0], dtype=np.longdouble)
    for j in range(m.shape[1]):
        sums = sums + m[:, j].astype(np.longdouble)
    return np.asarray(sums / m.shape[1], dtype=np.float64)


def r_seq_by(start: float, stop: float, by: float) -> NDArray[np.float64]:
    """R ``seq(from, to, by)`` for double arguments (base ``seq.default``).

    The grid is ``from + (0:n) * by`` with ``n = as.integer((to - from)/by +
    1e-10)``, capped at ``to``; its values are not decimal-rounded, so e.g. the
    seventh value of ``seq(0.01, 0.99, by = 0.01)`` is 0.069999999999999993,
    not the double nearest 0.07.
    """
    start, stop, by = float(start), float(stop), float(by)
    delta = stop - start
    if delta == 0.0 and stop == 0.0:
        return np.array([stop], dtype=np.float64)
    n = delta / by
    if not np.isfinite(n) or n < 0:
        raise ValueError("invalid '(to - from)/by'")
    if abs(delta) / max(abs(stop), abs(start)) < 100 * np.finfo(np.float64).eps:
        return np.array([start], dtype=np.float64)
    count = int(n + 1e-10)
    grid = start + np.arange(count + 1, dtype=np.float64) * by
    return np.minimum(grid, stop) if by > 0 else np.maximum(grid, stop)


def _r_pow_di(x: float, n: int) -> float:
    """R ``R_pow_di``: x^n by repeated squaring."""
    xn = 1.0
    if n != 0:
        negative = n < 0
        n = -n if negative else n
        while True:
            if n & 1:
                xn *= x
            n >>= 1
            if n:
                x *= x
            else:
                break
        if negative:
            xn = 1.0 / xn
    return xn


_M_LOG10_2 = 0.301029995663981195213738894724


def r_round(x: float, digits: float = 0.0) -> float:
    """R ``round(x, digits)`` (nmath ``fround``, R >= 4.0.0).

    R picks the nearer of ``floor(x * 10^d) / 10^d`` and ``ceil(x * 10^d) /
    10^d`` by their double distances to ``x`` (ties to an even last digit),
    which can differ from Python's correctly rounded ``round()``.
    """
    import math

    x = float(x)
    if math.isnan(x) or math.isnan(digits):
        return x + digits
    if not math.isfinite(x) or digits > 323 or x == 0.0:
        return x
    if digits < -308:
        return 0.0
    if digits == 0.0:
        return float(np.rint(x))
    dig = math.floor(digits + 0.5)
    sgn = 1.0
    if x < 0.0:
        sgn, x = -1.0, -x
    l10x = _M_LOG10_2 * (0.5 + (math.frexp(x)[1] - 1))
    if l10x + dig > 15:
        return sgn * x
    if dig <= 308:
        pow10 = _r_pow_di(10.0, dig)
        x10 = x * pow10
        i10 = math.floor(x10)
        xd = i10 / pow10
        xu = math.ceil(x10) / pow10
    else:
        p10 = _r_pow_di(10.0, dig - 308)
        pow10 = _r_pow_di(10.0, 308)
        x10 = (x * pow10) * p10
        i10 = math.floor(x10)
        xd = i10 / pow10 / p10
        xu = math.ceil(x10) / pow10 / p10
    du = xu - x
    dd = x - xd
    return sgn * (xu if (du < dd or (math.fmod(i10, 2.0) == 1 and du == dd)) else xd)


def r_round_array(values: NDArray[np.float64], digits: float = 0.0) -> NDArray[np.float64]:
    """Elementwise :func:`r_round` (vectorised; identical double arithmetic)."""
    import math

    v = np.asarray(values, dtype=np.float64)
    if digits == 0.0:
        return np.asarray(np.rint(v), dtype=np.float64)
    if math.isnan(digits) or digits > 323 or digits < -308 or abs(digits) > 308:
        return np.asarray(np.vectorize(lambda z: r_round(float(z), digits))(v), dtype=np.float64)
    dig = math.floor(digits + 0.5)
    pow10 = _r_pow_di(10.0, dig)
    sgn = np.where(v < 0.0, -1.0, 1.0)
    a = np.abs(v)
    with np.errstate(all="ignore"):
        x10 = a * pow10
        i10 = np.floor(x10)
        xd = i10 / pow10
        xu = np.ceil(x10) / pow10
        du = xu - a
        dd = a - xd
        rounded = sgn * np.where((du < dd) | ((np.fmod(i10, 2.0) == 1) & (du == dd)), xu, xd)
        l10x = _M_LOG10_2 * (0.5 + (np.frexp(a)[1] - 1).astype(np.float64))
    keep = ~np.isfinite(v) | (v == 0.0) | (l10x + dig > 15)
    return np.asarray(np.where(keep, v, rounded), dtype=np.float64)


def r_quantile(
    values: NDArray[np.float64],
    probs: float | NDArray[np.float64] | list[float],
    type: int = 7,
    na_rm: bool = False,
) -> NDArray[np.float64]:
    """R ``stats::quantile(x, probs, type = 7 | 8, names = FALSE)``.

    Follows ``quantile.default`` operation for operation: type 7 interpolates
    ``(1 - h) * x[lo] + h * x[hi]``; type 8 uses ``a = b = 1/3`` with R's
    relative fuzz and padded order statistics. NumPy's interpolation
    (``a + (b - a) * t``) rounds differently.
    """
    x = np.asarray(values, dtype=np.float64).ravel()
    if na_rm:
        x = x[~np.isnan(x)]
    elif np.isnan(x).any():
        raise ValueError("missing values and NaN's not allowed if 'na.rm' is FALSE")
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    eps = np.finfo(np.float64).eps
    p_ok = ~np.isnan(p)
    if np.any(p_ok & ((p < -100 * eps) | (p > 1 + 100 * eps))):
        raise ValueError("'probs' outside [0,1]")
    p = np.maximum(0.0, np.minimum(1.0, p))
    n = x.size
    xs = np.sort(x)
    qs = np.full(p.size, np.nan, dtype=np.float64)
    if n == 0:
        return qs
    if type == 7:
        index = 1 + max(n - 1, 0) * p
        lo = np.floor(index)
        hi = np.ceil(index)
        lo_i = np.where(p_ok, lo, 1).astype(np.int64) - 1
        hi_i = np.where(p_ok, hi, 1).astype(np.int64) - 1
        qs = xs[lo_i].copy()
        i = p_ok & (index > lo) & (xs[hi_i] != qs)
        h = (index - lo)[i]
        qs[i] = (1 - h) * qs[i] + h * xs[hi_i[i]]
    elif type == 8:
        fuzz = 4 * eps
        a = b = 1.0 / 3.0
        nppm = a + p * (n + 1 - a - b)
        j = np.floor(nppm * (1 + fuzz))
        h = nppm - j
        h[np.abs(h) < fuzz] = 0.0
        padded = np.concatenate(([xs[0], xs[0]], xs, [xs[-1], xs[-1]]))
        j_i = np.where(p_ok, j, 0).astype(np.int64)
        left = padded[j_i + 1]
        right = padded[j_i + 2]
        qs = left.copy()
        qs[h == 1] = right[h == 1]
        other = (h > 0) & (h < 1) & (left != right)
        qs[other] = ((1 - h) * left + h * right)[other]
    else:
        raise ValueError("only quantile types 7 and 8 are used by NNS")
    qs[~p_ok] = p[~p_ok]
    return qs


def r_col_means(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """R ``colMeans(x)``: each column summed sequentially in long double,
    divided by the row count, rounded to double (no refinement pass)."""
    m = np.asarray(x, dtype=np.float64)
    if m.shape[0] == 0:
        return np.full(m.shape[1], np.nan, dtype=np.float64)
    sums = np.cumsum(m.astype(np.longdouble), axis=0)[-1] + np.longdouble(0.0)
    return np.asarray(sums / m.shape[0], dtype=np.float64)


def r_approx(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    xout: NDArray[np.float64],
    yleft: float | None = None,
    yright: float | None = None,
) -> NDArray[np.float64]:
    """R ``stats::approx(x, y, xout, method = "linear", ties = "ordered")$y``.

    ``x`` must be sorted; duplicates are kept, as with ``ties = "ordered"``.
    Mirrors the C kernel ``approx1``: bisection to ``x[i] <= v <= x[i + 1]``,
    the stored ordinate returned exactly when ``v`` equals a knot (the last of
    tied knots), else ``y[i] + (y[j] - y[i]) * ((v - x[i]) / (x[j] - x[i]))``.
    ``yleft``/``yright`` default to ``NaN`` (``rule = 1``); pass ``y[0]`` and
    ``y[-1]`` for ``rule = 2``.
    """
    xs = np.asarray(x, dtype=np.float64).ravel()
    ys = np.asarray(y, dtype=np.float64).ravel()
    v = np.asarray(xout, dtype=np.float64).ravel()
    n = xs.size
    out = np.full(v.size, np.nan, dtype=np.float64)
    if n == 0:
        return out
    low = np.nan if yleft is None else float(yleft)
    high = np.nan if yright is None else float(yright)
    finite = ~np.isnan(v)
    below = finite & (v < xs[0])
    above = finite & (v > xs[-1])
    inside = finite & ~below & ~above
    out[below] = low
    out[above] = high
    if n == 1:
        out[inside] = ys[0]
        return out
    vi = v[inside]
    i = np.clip(np.searchsorted(xs, vi, side="right") - 1, 0, n - 2)
    j = i + 1
    xi, xj, yi, yj = xs[i], xs[j], ys[i], ys[j]
    with np.errstate(invalid="ignore", divide="ignore"):
        interp = yi + (yj - yi) * ((vi - xi) / (xj - xi))
    interp = np.where(vi == xi, yi, interp)
    interp = np.where(vi == xj, yj, interp)
    out[inside] = interp
    return out


def r_tie_means(
    x: NDArray[np.float64], y: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``regularize.values(x, y, ties = mean)`` for sorted ``x``: one knot per
    distinct ``x`` with R's ``mean()`` of its ordinates (in input order)."""
    xs = np.asarray(x, dtype=np.float64).ravel()
    ys = np.asarray(y, dtype=np.float64).ravel()
    ux = np.unique(xs)
    if ux.size == xs.size:
        return xs, ys
    groups = np.searchsorted(ux, xs)
    means = np.array([r_mean(ys[groups == g]) for g in range(ux.size)])
    return ux, means
