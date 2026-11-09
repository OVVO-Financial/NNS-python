"""R's ``stats::smooth.spline(x, y, spar = )`` and its ``predict``, to the last bit.

A line-by-line port of R 4.5.0's ``smspline.R`` (grouping, scaling, knots),
``sbart.c`` (the ``spar`` given path), the Fortran ``sgram``, ``stxwx``,
``sslvrg``, ``bsplvd``/``bsplvb`` and ``bvalue``, LINPACK ``dpbfa``/``dpbsl``
with the reference BLAS ``ddot``/``daxpy`` they call, and
``predict.smooth.spline.fit``. Every expression keeps R's operation order; the
per-observation and per-prediction loops are vectorised over points, which
leaves each point's arithmetic unchanged.

Arrays named after their Fortran counterparts are 1-based (index 0 unused).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nns._rmean import r_mean, r_quantile
from nns._rnmath import r_pow_scalar


@dataclass(frozen=True)
class SmoothSplineFit:
    """R's ``smooth.spline.fit``: knots, coefficients and the x scaling."""

    knot: NDArray[np.float64]  # 1-based: knot[1..nk + 4]
    coef: NDArray[np.float64]  # 1-based: coef[1..nk]
    nk: int
    min: float
    range: float

    def predict(self, x: NDArray[np.float64], deriv: int = 0) -> NDArray[np.float64]:
        """``predict.smooth.spline.fit(object, x, deriv)$y``."""
        return _predict(self, np.asarray(x, dtype=np.float64), deriv, depth=0)


def nknots_smspl(n: int) -> int:
    """R's ``stats:::.nknots.smspl``: number of inner knots for ``n`` unique x."""
    if n < 50:
        return n
    a1, a2, a3, a4 = (math.log2(v) for v in (50.0, 100.0, 140.0, 200.0))
    if n < 200:
        value = 2.0 ** (a1 + (a2 - a1) * (n - 50) / 150)
    elif n < 800:
        value = 2.0 ** (a2 + (a3 - a2) * (n - 200) / 600)
    elif n < 3200:
        value = 2.0 ** (a3 + (a4 - a3) * (n - 800) / 2400)
    else:
        value = 200.0 + (n - 3200) ** 0.2
    return math.trunc(float(value))


def smooth_spline(x: NDArray[np.float64], y: NDArray[np.float64], spar: float) -> SmoothSplineFit:
    """``smooth.spline(x, y, spar = spar)`` with R's other defaults.

    Raises R's errors (as ``ValueError``) where R stops, and warns as R does
    when the penalised system is not positive definite at a large ``spar``.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError("'x' and 'y' lengths differ")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("missing or infinite values in inputs are not allowed")
    n = x.size
    # tol = 1e-6 * IQR(x)
    q25, q75 = r_quantile(x, [0.25, 0.75], 7)
    tol = 1e-6 * float(q75 - q25)
    if not math.isfinite(tol) or tol <= 0.0:
        raise ValueError("'tol' must be strictly positive and finite")
    # Replace y[] for the same x[] (to a precision of tol) by their mean.
    xx = np.rint((x - r_mean(x)) / tol)
    unsorted = bool(np.any(x[1:] < x[:-1]))
    order = np.argsort(x, kind="stable") if unsorted else np.arange(n)
    xxs = xx[order]
    first = np.concatenate(([True], xxs[:-1] < xxs[1:]))
    ux = x[order][first]
    nx = ux.size
    if nx <= 3:
        raise ValueError("need at least four unique 'x' values")
    if nx == n:
        wbar = np.ones(nx, dtype=np.float64)
        ybar = y[order].copy()
    else:
        # ox <- match(xx, xxs[nd]); per group, in the original order:
        # c(length(i), sum(y[i]), sum(y[i]^2)) with R's long-double sum().
        group = np.searchsorted(xxs[first], xx)
        counts = np.bincount(group, minlength=nx).astype(np.float64)
        sums = np.zeros(nx, dtype=np.longdouble)
        np.add.at(sums, group, y.astype(np.longdouble))
        wbar = counts
        ybar = np.asarray(sums, dtype=np.float64) / np.where(wbar > 0.0, wbar, 1.0)
    r_ux = float(ux[-1] - ux[0])
    xbar = (ux - ux[0]) / r_ux
    nknots = nknots_smspl(nx)
    # xbar[seq.int(1, nx, length.out = nknots)]: fractional indices truncate.
    picks = np.trunc(np.linspace(1.0, float(nx), nknots)).astype(np.intp) - 1
    knot_values = np.concatenate(([xbar[0]] * 3, xbar[picks], [xbar[-1]] * 3))
    nk = nknots + 2
    knot = np.concatenate(([0.0], knot_values))

    coef, info = _sbart_given_spar(xbar, ybar, wbar, knot, nk, float(spar))
    if info > 0:
        # R: smoothing parameter value too small/large (ier from dpbfa).
        if spar < 0.5:
            raise ValueError("smoothing parameter value too small")
        warnings.warn(
            "smoothing parameter value too large\nsetting df = 1  __use with care!__",
            UserWarning,
            stacklevel=2,
        )
    return SmoothSplineFit(knot=knot, coef=coef, nk=nk, min=float(ux[0]), range=r_ux)


# --------------------------------------------------------------------------
# sbart.c (spar supplied) and sslvrg.f
# --------------------------------------------------------------------------


def _sbart_given_spar(
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    wbar: NDArray[np.float64],
    knot: NDArray[np.float64],
    nk: int,
    spar: float,
) -> tuple[NDArray[np.float64], int]:
    # sbart squares the weights later, so it stores sqrt(w).
    ws = np.where(wbar > 0.0, np.sqrt(np.where(wbar > 0.0, wbar, 0.0)), wbar)
    sg0, sg1, sg2, sg3 = _sgram(knot, nk)
    xwy, hs0, hs1, hs2, hs3 = _stxwx(xs, ys, ws, knot, nk)
    # ratio := tr(X'WX) / tr(SIGMA) over the interior (1-based 3..nk-3).
    t1 = 0.0
    t2 = 0.0
    for i in range(3, nk - 2):
        t1 += hs0[i]
        t2 += sg0[i]
    ratio = t1 / t2
    lam = ratio * r_pow_scalar(16.0, spar * 6.0 - 2.0)
    return _sslvrg_coef(knot, nk, lam, xwy, (hs0, hs1, hs2, hs3), (sg0, sg1, sg2, sg3))


def _sslvrg_coef(
    knot: NDArray[np.float64],
    nk: int,
    lam: float,
    xwy: list[float],
    hs: tuple[list[float], list[float], list[float], list[float]],
    sg: tuple[list[float], list[float], list[float], list[float]],
) -> tuple[NDArray[np.float64], int]:
    """``sslvrg``'s coefficient solve: ``(X'WX + lambda SIGMA) coef = X'Wy``."""
    hs0, hs1, hs2, hs3 = hs
    sg0, sg1, sg2, sg3 = sg
    lda = 4
    abd = [0.0] * (lda * nk)  # Fortran abd(4, nk), column-major

    def at(row: int, col: int) -> int:
        return (row - 1) + (col - 1) * lda

    coef = [0.0] * (nk + 1)
    for i in range(1, nk + 1):
        coef[i] = xwy[i]
        abd[at(4, i)] = hs0[i] + lam * sg0[i]
    for i in range(1, nk):
        abd[at(3, i + 1)] = hs1[i] + lam * sg1[i]
    for i in range(1, nk - 1):
        abd[at(2, i + 2)] = hs2[i] + lam * sg2[i]
    for i in range(1, nk - 2):
        abd[at(1, i + 3)] = hs3[i] + lam * sg3[i]
    info = _dpbfa(abd, lda, nk, 3)
    if info == 0:
        _dpbsl(abd, lda, nk, 3, coef)
    return np.asarray(coef, dtype=np.float64), info


# --------------------------------------------------------------------------
# LINPACK dpbfa / dpbsl with reference BLAS ddot / daxpy
# --------------------------------------------------------------------------


def _ddot(n: int, a: list[float], ia: int, b: list[float], ib: int) -> float:
    """Reference BLAS ``ddot`` (unit strides): a left-to-right sum from 0."""
    total = 0.0
    for offset in range(n):
        total = total + a[ia + offset] * b[ib + offset]
    return total


def _dpbfa(abd: list[float], lda: int, n: int, m: int) -> int:
    def at(row: int, col: int) -> int:
        return (row - 1) + (col - 1) * lda

    for j in range(1, n + 1):
        s = 0.0
        ik = m + 1
        jk = max(j - m, 1)
        mu = max(m + 2 - j, 1)
        if m >= mu:
            for k in range(mu, m + 1):
                t = abd[at(k, j)] - _ddot(k - mu, abd, at(ik, jk), abd, at(mu, j))
                t = t / abd[at(m + 1, jk)]
                abd[at(k, j)] = t
                s = s + t * t
                ik = ik - 1
                jk = jk + 1
        s = abd[at(m + 1, j)] - s
        if s <= 0.0:
            return j
        abd[at(m + 1, j)] = math.sqrt(s)
    return 0


def _dpbsl(abd: list[float], lda: int, n: int, m: int, b: list[float]) -> None:
    def at(row: int, col: int) -> int:
        return (row - 1) + (col - 1) * lda

    for k in range(1, n + 1):
        lm = min(k - 1, m)
        la = m + 1 - lm
        lb = k - lm
        t = _ddot(lm, abd, at(la, k), b, lb)
        b[k] = (b[k] - t) / abd[at(m + 1, k)]
    for kb in range(1, n + 1):
        k = n + 1 - kb
        lm = min(k - 1, m)
        la = m + 1 - lm
        lb = k - lm
        b[k] = b[k] / abd[at(m + 1, k)]
        t = -b[k]
        if lm > 0 and t != 0.0:  # daxpy returns early for n <= 0 or da == 0
            base = at(la, k)
            for offset in range(lm):
                b[lb + offset] = b[lb + offset] + t * abd[base + offset]


# --------------------------------------------------------------------------
# interv (findInterval2), bsplvb / bsplvd
# --------------------------------------------------------------------------


def _interv(xt: NDArray[np.float64], n: int, x: float) -> tuple[int, int]:
    """R's ``interv(xt, n, x, rightmost_closed = 0, all_inside = 0)`` on 1-based
    ``xt[1..n]``: ``(left, mflag)`` with ``xt[left] <= x < xt[left + 1]``."""
    if n == 0:
        return 0, 0
    if x < xt[1]:
        return 0, -1
    if x >= xt[n]:
        return n, 1
    # bisect for the unique left with xt[left] <= x < xt[left + 1]
    left = int(np.searchsorted(xt[1 : n + 1], x, side="right"))
    return left, 0


def _bsplvd_values(
    t: NDArray[np.float64], x: NDArray[np.float64], left: NDArray[np.intp]
) -> list[NDArray[np.float64]]:
    """``bsplvd(t, lent, 4, x, left, work, vnikx, 1)``: the four order-4 B-spline
    values at each x (``vnikx(1..4, 1)``), vectorised over points."""
    biatx = [np.ones_like(x), np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)]
    deltar: list[NDArray[np.float64]] = []
    deltal: list[NDArray[np.float64]] = []
    for j in range(1, 4):  # bsplvb(..., jhigh = 4, index = 1)
        deltar.append(t[left + j] - x)
        deltal.append(x - t[left + 1 - j])
        saved = np.zeros_like(x)
        for i in range(1, j + 1):
            term = biatx[i - 1] / (deltar[i - 1] + deltal[j + 1 - i - 1])
            biatx[i - 1] = saved + deltar[i - 1] * term
            saved = deltal[j + 1 - i - 1] * term
        biatx[j] = saved
    return biatx


class _Bsplvb:
    """``bsplvb``'s saved state (``j``, ``deltal``, ``deltar``)."""

    def __init__(self) -> None:
        self.j = 1
        self.deltal = [0.0] * 21
        self.deltar = [0.0] * 21


def _bsplvb(
    state: _Bsplvb,
    t: NDArray[np.float64],
    jhigh: int,
    index: int,
    x: float,
    left: int,
    biatx: list[float],
) -> None:
    if index != 2:
        state.j = 1
        biatx[1] = 1.0
        if state.j >= jhigh:
            return
    while True:
        j = state.j
        jp1 = j + 1
        state.deltar[j] = float(t[left + j]) - x
        state.deltal[j] = x - float(t[left + 1 - j])
        saved = 0.0
        for i in range(1, j + 1):
            term = biatx[i] / (state.deltar[i] + state.deltal[jp1 - i])
            biatx[i] = saved + state.deltar[i] * term
            saved = state.deltal[jp1 - i] * term
        biatx[jp1] = saved
        state.j = jp1
        if state.j >= jhigh:
            return


def _bsplvd(t: NDArray[np.float64], k: int, x: float, left: int, nderiv: int) -> list[list[float]]:
    """``bsplvd``: ``dbiatx[i][m]`` (1-based) holds the (m-1)-th derivative of the
    i-th B-spline of order k that is nonzero at x."""
    mhigh = max(min(nderiv, k), 1)
    kp1 = k + 1
    state = _Bsplvb()
    column = [0.0] * (k + 2)  # dbiatx(., 1), which bsplvb works in
    dbiatx = [[0.0] * (nderiv + 1) for _ in range(k + 1)]
    _bsplvb(state, t, kp1 - mhigh, 1, x, left, column)
    if mhigh > 1:
        ideriv = mhigh
        for _ in range(2, mhigh + 1):
            jp1mid = 1
            for j in range(ideriv, k + 1):
                dbiatx[j][ideriv] = column[jp1mid]
                jp1mid += 1
            ideriv -= 1
            _bsplvb(state, t, kp1 - ideriv, 2, x, left, column)
    for i in range(1, k + 1):
        dbiatx[i][1] = column[i]
    if mhigh == 1:
        return dbiatx
    a = [[0.0] * (k + 1) for _ in range(k + 1)]  # a[row][col]
    for i in range(1, k + 1):
        a[i][i] = 1.0
    for m in range(2, mhigh + 1):
        kp1mm = kp1 - m
        fkp1mm = float(kp1mm)
        il = left
        i = k
        for _ in range(kp1mm):
            factor = fkp1mm / (float(t[il + kp1mm]) - float(t[il]))
            for j in range(1, i + 1):
                a[i][j] = (a[i][j] - a[i - 1][j]) * factor
            il -= 1
            i -= 1
        for i in range(1, k + 1):
            total = 0.0
            for j in range(max(i, m), k + 1):
                total = a[j][i] * dbiatx[j][m] + total
            dbiatx[i][m] = total
    return dbiatx


# --------------------------------------------------------------------------
# sgram.f and stxwx.f
# --------------------------------------------------------------------------


def _sgram(
    tb: NDArray[np.float64], nb: int
) -> tuple[list[float], list[float], list[float], list[float]]:
    """The banded Gram matrix of B'' (``sgram``), with R's ``.3330`` constant."""
    sg = [[0.0] * (nb + 1) for _ in range(4)]
    for i in range(1, nb + 1):
        ileft, _ = _interv(tb, nb + 1, float(tb[i]))
        lower = _bsplvd(tb, 4, float(tb[i]), ileft, 3)
        yw1 = [0.0] + [lower[ii][3] for ii in range(1, 5)]
        upper = _bsplvd(tb, 4, float(tb[i + 1]), ileft, 3)
        yw2 = [0.0] + [upper[ii][3] - yw1[ii] for ii in range(1, 5)]
        wpt = float(tb[i + 1]) - float(tb[i])
        width = min(ileft, 4)
        if width < 1:
            continue
        for ii in range(1, width + 1):
            for band in range(4):
                jj = ii + band
                if jj > width:
                    break
                value = (
                    yw1[ii] * yw1[jj]
                    + (yw2[ii] * yw1[jj] + yw2[jj] * yw1[ii]) * 0.5
                    + yw2[ii] * yw2[jj] * 0.3330
                )
                row = ileft - width + ii
                sg[band][row] = sg[band][row] + wpt * value
    return sg[0], sg[1], sg[2], sg[3]


def _stxwx(
    x: NDArray[np.float64],
    z: NDArray[np.float64],
    w: NDArray[np.float64],
    xknot: NDArray[np.float64],
    n: int,
) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
    """``X'Wz`` and the banded ``X'WX`` (``stxwx``), accumulated point by point."""
    eps = 1e-10
    left = np.empty(x.size, dtype=np.intp)
    keep = x.size
    for index, value in enumerate(x.tolist()):
        ileft, mflag = _interv(xknot, n + 1, value)
        if mflag == 1:
            if value <= float(xknot[ileft]) + eps:
                ileft -= 1
            else:
                keep = index  # Fortran returns here, keeping what it summed.
                break
        left[index] = ileft
    x, z, w, left = x[:keep], z[:keep], w[:keep], left[:keep]
    v1, v2, v3, v4 = _bsplvd_values(xknot, x, left)
    ww = w * w  # w(i)**2
    wz = ww * z
    # Contributions in Fortran's order (per point: rows ileft-3 .. ileft).
    y_terms = np.column_stack([wz * v1, wz * v2, wz * v3, wz * v4]).ravel()
    h0_terms = np.column_stack([ww * (v1 * v1), ww * (v2 * v2), ww * (v3 * v3), ww * (v4 * v4)])
    h1_terms = np.column_stack(
        [(ww * v1) * v2, (ww * v2) * v3, (ww * v3) * v4, np.zeros_like(v1)]
    )
    h2_terms = np.column_stack(
        [(ww * v1) * v3, (ww * v2) * v4, np.zeros_like(v1), np.zeros_like(v1)]
    )
    h3_terms = np.column_stack(
        [(ww * v1) * v4, np.zeros_like(v1), np.zeros_like(v1), np.zeros_like(v1)]
    )
    out = []
    for terms, slots in (
        (y_terms.reshape(-1, 4), 4),
        (h0_terms, 4),
        (h1_terms, 3),
        (h2_terms, 2),
        (h3_terms, 1),
    ):
        acc = np.zeros(n + 1, dtype=np.float64)
        used = terms[:, :slots].ravel()
        targets = np.column_stack([left - 3 + s for s in range(slots)]).ravel()
        # Unbuffered and in order: each row sums its terms point by point.
        np.add.at(acc, targets, used)
        out.append(acc.tolist())
    return out[0], out[1], out[2], out[3], out[4]


# --------------------------------------------------------------------------
# bvalue.f and predict.smooth.spline.fit
# --------------------------------------------------------------------------


def _bvalue(fit: SmoothSplineFit, x: NDArray[np.float64], jderiv: int) -> NDArray[np.float64]:
    """``bvalue(knot, coef, nk, 4, x, jderiv)`` for each x in ``[0, 1]``."""
    t = fit.knot
    n = fit.nk
    k = 4
    if jderiv >= k:
        return np.zeros_like(x)
    lefts = np.empty(x.size, dtype=np.intp)
    at_end = (x == t[n + 1]) & (t[n + 1] == t[n + k])
    for index, value in enumerate(x.tolist()):
        if at_end[index]:
            lefts[index] = n
            continue
        left, mflag = _interv(t, n + k, value)
        if mflag != 0:
            warnings.warn("bvalue()  mflag != 0: should never happen!", UserWarning, stacklevel=3)
            lefts[index] = -1
            continue
        lefts[index] = left
    valid = lefts >= k  # imk >= 0 and n - i >= 0 hold on [0, 1] with these knots
    out = np.zeros_like(x)
    if not np.any(valid):
        return out
    xv = x[valid]
    i = lefts[valid]
    km1 = k - 1
    dm = [xv - t[i + 1 - j] for j in range(1, km1 + 1)]  # dm(1..3)
    dp = [t[i + j] - xv for j in range(1, km1 + 1)]  # dp(1..3)
    aj = [fit.coef[i - k + jc] for jc in range(1, k + 1)]  # aj(1..4)
    for j in range(1, jderiv + 1):
        kmj = k - j
        fkmj = float(kmj)
        ilo = kmj
        for jj in range(1, kmj + 1):
            aj[jj - 1] = ((aj[jj] - aj[jj - 1]) / (dm[ilo - 1] + dp[jj - 1])) * fkmj
            ilo -= 1
    for j in range(jderiv + 1, km1 + 1):
        kmj = k - j
        ilo = kmj
        for jj in range(1, kmj + 1):
            aj[jj - 1] = (aj[jj] * dm[ilo - 1] + aj[jj - 1] * dp[jj - 1]) / (
                dm[ilo - 1] + dp[jj - 1]
            )
            ilo -= 1
    out[valid] = aj[0]
    return out


def _predict(
    fit: SmoothSplineFit, x: NDArray[np.float64], deriv: int, depth: int
) -> NDArray[np.float64]:
    if depth > 2:
        # R's Recall() loops when (min + range - min) / range exceeds 1.
        raise RuntimeError(
            "evaluation nested too deeply: infinite recursion / options(expressions=)?"
        )
    xs = (x - fit.min) / fit.range
    left = xs < 0.0
    right = xs > 1.0
    extrap = left | right
    interp = ~extrap
    y = xs.copy()
    if np.any(interp):
        y[interp] = _bvalue(fit, xs[interp], deriv)
    if np.any(extrap):
        xrange = np.array([fit.min, fit.min + fit.range], dtype=np.float64)
        if deriv == 0:
            end_object = _predict(fit, xrange, 0, depth + 1)
            end_slopes = _predict(fit, xrange, 1, depth + 1) * fit.range
            if np.any(left):
                y[left] = end_object[0] + end_slopes[0] * (xs[left] - 0.0)
            if np.any(right):
                y[right] = end_object[1] + end_slopes[1] * (xs[right] - 1.0)
        elif deriv == 1:
            end_slopes = _predict(fit, xrange, 1, depth + 1) * fit.range
            y[left] = end_slopes[0]
            y[right] = end_slopes[1]
        else:
            y[extrap] = 0.0
    if deriv > 0:
        y = y / r_pow_scalar(fit.range, float(deriv))
    return y
