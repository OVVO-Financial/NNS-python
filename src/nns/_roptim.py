"""R's one-dimensional minimiser, ``optimize()``.

``optimize(f, interval, tol)`` runs Brent's ``fmin`` (R 4.5.0
``src/library/stats/src/optimize.c``, ``Brent_fmin``). Its golden-section and
parabolic steps, and so the point it returns, differ from SciPy's bounded
Brent method, and NNS uses the returned point directly (``NNS.meboot`` blends
each replicate with it), so the port reproduces R's iteration. As in R, a
non-finite objective value is replaced by the largest finite double (the most
negative one for ``-Inf``) with a warning.
"""

from __future__ import annotations

import math
import sys
import warnings
from collections.abc import Callable

_DBL_EPSILON = sys.float_info.epsilon
_DBL_MAX = sys.float_info.max


def r_optimize(
    f: Callable[[float], float],
    lower: float,
    upper: float,
    tol: float = _DBL_EPSILON**0.25,
) -> float:
    """``optimize(f, c(lower, upper), tol = tol)$minimum``."""
    if not (math.isfinite(lower) and math.isfinite(upper)):
        raise ValueError("invalid 'xmin' or 'xmax' value")
    if lower >= upper:
        raise ValueError("'xmin' not less than 'xmax'")
    if not (math.isfinite(tol) and tol > 0.0):
        raise ValueError("invalid 'tol' value")

    def fcn(x: float) -> float:
        value = float(f(x))
        if math.isfinite(value):
            return value
        if value == -math.inf:
            warnings.warn("-Inf replaced by maximally negative value", RuntimeWarning, stacklevel=3)
            return -_DBL_MAX
        label = "NA/NaN" if math.isnan(value) else "Inf"
        warnings.warn(f"{label} replaced by maximum positive value", RuntimeWarning, stacklevel=3)
        return _DBL_MAX

    return _brent_fmin(lower, upper, fcn, tol)


def _brent_fmin(ax: float, bx: float, f: Callable[[float], float], tol: float) -> float:
    c = (3.0 - math.sqrt(5.0)) * 0.5
    eps = math.sqrt(_DBL_EPSILON)
    a = ax
    b = bx
    v = a + c * (b - a)
    w = v
    x = v
    d = 0.0
    e = 0.0
    fx = f(x)
    fv = fx
    fw = fx
    tol3 = tol / 3.0

    while True:
        xm = (a + b) * 0.5
        tol1 = eps * abs(x) + tol3
        t2 = tol1 * 2.0
        if abs(x - xm) <= t2 - (b - a) * 0.5:
            break
        p = 0.0
        q = 0.0
        r = 0.0
        if abs(e) > tol1:  # fit a parabola
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = (q - r) * 2.0
            if q > 0.0:
                p = -p
            else:
                q = -q
            r = e
            e = d
        if abs(p) >= abs(q * 0.5 * r) or p <= q * (a - x) or p >= q * (b - x):
            # a golden-section step
            e = b - x if x < xm else a - x
            d = c * e
        else:  # a parabolic-interpolation step
            d = p / q
            u = x + d
            # f must not be evaluated too close to ax or bx
            if u - a < t2 or b - u < t2:
                d = tol1
                if x >= xm:
                    d = -d
        # f must not be evaluated too close to x
        if abs(d) >= tol1:
            u = x + d
        elif d > 0.0:
            u = x + tol1
        else:
            u = x - tol1
        fu = f(u)
        # update a, b, v, w and x
        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            v = w
            w = x
            x = u
            fv = fw
            fw = fx
            fx = fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v = w
                fv = fw
                w = u
                fw = fu
            elif fu <= fv or v == x or v == w:
                v = u
                fv = fu
    return x
