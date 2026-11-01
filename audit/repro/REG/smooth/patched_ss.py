"""Candidate fix for nns._reg_engine._smooth_spline_predictor: reproduces R's
smooth.spline(x, y, spar=) exactly: sgram's 0.3330 constant and .nknots.smspl."""
import math
import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import solve

def nknots_smspl(n):
    if n < 50: return n
    a1, a2, a3, a4 = map(math.log2, (50, 100, 140, 200))
    if n < 200: v = 2 ** (a1 + (a2 - a1) * (n - 50) / 150)
    elif n < 800: v = 2 ** (a2 + (a3 - a2) * (n - 200) / 600)
    elif n < 3200: v = 2 ** (a3 + (a4 - a3) * (n - 800) / 2400)
    else: v = 200 + (n - 3200) ** 0.2
    return int(math.trunc(v))

def _sigma_sgram(knot, nk):
    # R sgram.f: linear approximation of B'' on each knot interval, with 1/3 -> 0.3330
    sig = np.zeros((nk, nk))
    eye = np.eye(nk)
    tb = knot
    for i in range(3, nk):  # intervals [tb(i), tb(i+1)] with i=4..nb (1-based)
        a, b = tb[i], tb[i + 1]
        if b <= a: continue
        mid = 0.5 * (a + b)
        # second derivative is linear on (a,b): evaluate on the open piece
        d_a = np.array([BSpline(tb, eye[j], 3)(a + 1e-9 * (b - a), nu=2) for j in range(nk)])
        d_b = np.array([BSpline(tb, eye[j], 3)(b - 1e-9 * (b - a), nu=2) for j in range(nk)])
        # extrapolate exactly to endpoints (linear)
        slope = (d_b - d_a) / ((b - a) * (1 - 2e-9))
        y1 = d_a - slope * 1e-9 * (b - a)
        y2 = slope * (b - a)
        w = b - a
        sig += w * (np.outer(y1, y1) + 0.5 * (np.outer(y2, y1) + np.outer(y1, y2)) + 0.333 * np.outer(y2, y2))
    return sig

def predictor(rx, ry, spar):
    x = np.asarray(rx, float); y = np.asarray(ry, float)
    o = np.argsort(x, kind="mergesort"); x, y = x[o], y[o]
    ux, inv = np.unique(x, return_inverse=True)
    w = np.bincount(inv).astype(float); yb = np.bincount(inv, weights=y) / w
    nx = ux.size
    x0, r = ux[0], ux[-1] - ux[0]
    xbar = (ux - x0) / r
    nkn = nknots_smspl(nx)
    idx = np.trunc(np.linspace(1, nx, nkn)).astype(int) - 1  # R seq.int(1, n, length.out) then index truncation
    knot = np.concatenate(([xbar[0]] * 3, xbar[idx], [xbar[-1]] * 3))
    nk = nkn + 2
    D = BSpline.design_matrix(xbar, knot, 3).toarray()
    xtwx = D.T @ (w[:, None] * D)
    sig = _sigma_sgram(knot, nk)
    ratio = np.diag(xtwx)[2:nk - 3].sum() / np.diag(sig)[2:nk - 3].sum()
    lam = ratio * 256.0 ** (3 * spar - 1)
    coef = solve(xtwx + lam * sig, D.T @ (w * yb), assume_a="sym")
    sp = BSpline(knot, coef, 3, extrapolate=False)
    e0, s0, e1, s1 = float(sp(0.0)), float(sp(0.0, nu=1)), float(sp(1.0)), float(sp(1.0, nu=1))
    def predict(xo):
        s = (np.asarray(xo, float) - x0) / r
        out = np.asarray(sp(np.clip(s, 0, 1)), float)
        lo, hi = s < 0, s > 1
        out[lo] = e0 + s[lo] * s0; out[hi] = e1 + (s[hi] - 1) * s1
        return out
    return predict
