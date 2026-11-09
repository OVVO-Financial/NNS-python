"""Bit-faithful ports of the R ``nmath`` routines the NNS kernels call.

R's C++ kernels call ``R::dhyper`` and ``R::pnorm``. Those use Loader's
saddle-point binomial (``dbinom_raw`` with ``stirlerr`` and ``bd0``) and
Cody's rational approximation, which agree with closed forms built from
``math.comb`` or ``math.erf`` only to within a few units in the last place.
NNS feeds the results into exact comparisons downstream (the dependence
coefficients become dimension-reduction weights whose index is cut by
``findInterval``), so the port reproduces R's own arithmetic.

Source: R 4.5.0 ``src/nmath/{dhyper,dbinom,stirlerr,bd0,pnorm}.c``
(Copyright (C) 2000-2024 The R Core Team; Catherine Loader; W. J. Cody;
GPL-2 or later). ``stirlerr`` follows the R >= 4.4.0 series cutoffs.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import functools
import math
import sys
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

_DBL_EPSILON = sys.float_info.epsilon
_DBL_MIN = sys.float_info.min
_M_LN_2PI = 1.837877066409345483560659472811  # log(2*pi)
_M_LN_SQRT_2PI = 0.918938533204672741780329736406  # log(sqrt(2*pi))
_M_SQRT_32 = 5.656854249492380195206754896838  # sqrt(32)
_M_1_SQRT_2PI = 0.398942280401432677939946059934  # 1/sqrt(2*pi)

_S = (
    0.083333333333333333333,  # 1/12
    0.00277777777777777777778,  # 1/360
    0.00079365079365079365079365,  # 1/1260
    0.000595238095238095238095238,  # 1/1680
    0.0008417508417508417508417508,  # 1/1188
    0.0019175269175269175269175262,  # 691/360360
    0.0064102564102564102564102561,  # 1/156
    0.029550653594771241830065352,  # 3617/122400
    0.17964437236883057316493850,  # 43867/244188
    1.3924322169059011164274315,  # 174611/125400
    13.402864044168391994478957,  # 77683/5796
    156.84828462600201730636509,  # 236364091/1506960
    2193.1033333333333333333333,  # 657931/300
    36108.771253724989357173269,  # 3392780147/93960
    691472.26885131306710839498,  # 1723168255201/2492028
    15238221.539407416192283370,  # 7709321041217/505920
    382900751.39141414141414141,  # 151628697551/396
)

_SFERR_HALVES = (
    0.0,  # n=0 - wrong, place holder only
    0.1534264097200273452913848,  # 0.5
    0.0810614667953272582196702,  # 1.0
    0.0548141210519176538961390,  # 1.5
    0.0413406959554092940938221,  # 2.0
    0.03316287351993628748511048,  # 2.5
    0.02767792568499833914878929,  # 3.0
    0.02374616365629749597132920,  # 3.5
    0.02079067210376509311152277,  # 4.0
    0.01848845053267318523077934,  # 4.5
    0.01664469118982119216319487,  # 5.0
    0.01513497322191737887351255,  # 5.5
    0.01387612882307074799874573,  # 6.0
    0.01281046524292022692424986,  # 6.5
    0.01189670994589177009505572,  # 7.0
    0.01110455975820691732662991,  # 7.5
    0.010411265261972096497478567,  # 8.0
    0.009799416126158803298389475,  # 8.5
    0.009255462182712732917728637,  # 9.0
    0.008768700134139385462952823,  # 9.5
    0.008330563433362871256469318,  # 10.0
    0.007934114564314020547248100,  # 10.5
    0.007573675487951840794972024,  # 11.0
    0.007244554301320383179543912,  # 11.5
    0.006942840107209529865664152,  # 12.0
    0.006665247032707682442354394,  # 12.5
    0.006408994188004207068439631,  # 13.0
    0.006171712263039457647532867,  # 13.5
    0.005951370112758847735624416,  # 14.0
    0.005746216513010115682023589,  # 14.5
    0.005554733551962801371038690,  # 15.0
)


def _series(n: float, nn: float, k: int) -> float:
    """R's nested ``(S0-(S1-(...-S[k-1]/nn)/nn)...)/n`` with ``k`` terms."""
    acc = _S[k - 1]
    for j in range(k - 2, -1, -1):
        acc = _S[j] - acc / nn
    return acc / n


def stirlerr(n: float) -> float:
    """R ``stirlerr(n) = log(n!) - log(sqrt(2*pi*n) * (n/e)^n)``."""
    if n <= 23.5:
        nn = n + n
        if n <= 15.0 and nn == int(nn):
            return _SFERR_HALVES[int(nn)]
        if n <= 5.25:
            if n >= 1.0:
                l_n = math.log(n)
                return clgamma(n) + n * (1 - l_n) + math.ldexp(l_n - _M_LN_2PI, -1)
            return math.lgamma(1.0 + n) - (n + 0.5) * math.log(n) + n - _M_LN_SQRT_2PI
        nn = n * n
        if n > 12.8:
            return _series(n, nn, 7)
        if n > 12.3:
            return _series(n, nn, 8)
        if n > 8.9:
            return _series(n, nn, 9)
        if n > 7.3:
            return _series(n, nn, 11)
        if n > 6.6:
            return _series(n, nn, 13)
        if n > 6.1:
            return _series(n, nn, 15)
        return _series(n, nn, 17)
    nn = n * n
    if n > 15.7e6:
        return _S[0] / n
    if n > 6180:
        return _series(n, nn, 2)
    if n > 205:
        return _series(n, nn, 3)
    if n > 86:
        return _series(n, nn, 4)
    if n > 27:
        return _series(n, nn, 5)
    return _series(n, nn, 6)


def bd0(x: float, np_: float) -> float:
    """R ``bd0(x, np) = x log(x/np) + np - x``, accurately."""
    if not (math.isfinite(x) and math.isfinite(np_)) or np_ == 0.0:
        return math.nan
    if abs(x - np_) < 0.1 * (x + np_):
        v = (x - np_) / (x + np_)
        s = (x - np_) * v
        if abs(s) < _DBL_MIN:
            return s
        ej = 2 * x * v
        v *= v
        for j in range(1, 1000):
            ej *= v
            s_prev = s
            s += ej / ((j << 1) + 1)
            if s == s_prev:
                return s
    return x * math.log(x / np_) + np_ - x


def pow1p(x: float, y: float) -> float:
    """R ``pow1p(x, y) = (1 + x)^y``, accurately for small ``|x|``."""
    if math.isnan(y):
        return 1.0 if x == 0 else y
    if 0 <= y and y == math.trunc(y) and y <= 4.0:
        k = int(y)
        if k == 0:
            return 1.0
        if k == 1:
            return x + 1.0
        if k == 2:
            return x * (x + 2.0) + 1.0
        if k == 3:
            return x * (x * (x + 3.0) + 3.0) + 1.0
        return x * (x * (x * (x + 4.0) + 6.0) + 4.0) + 1.0
    if (x + 1) - 1 == x or abs(x) > 0.5 or math.isnan(x):
        return math.pow(1 + x, y)
    return math.exp(y * math.log1p(x))


def dbinom_raw(x: float, n: float, p: float, q: float) -> float:
    """R ``dbinom_raw(x, n, p, q, give_log = FALSE)``."""
    if p == 0:
        return 1.0 if x == 0 else 0.0
    if q == 0:
        return 1.0 if x == n else 0.0
    if x == 0:
        if n == 0:
            return 1.0
        return math.pow(q, n) if p > q else pow1p(-p, n)
    if x == n:
        return pow1p(-q, n) if p > q else math.pow(p, n)
    if x < 0 or x > n:
        return 0.0
    lc = stirlerr(n) - stirlerr(x) - stirlerr(n - x) - bd0(x, n * p) - bd0(n - x, n * q)
    lf = _M_LN_2PI + math.log(x) + math.log1p(-x / n)
    return math.exp(lc - 0.5 * lf)


def dhyper(x: float, r: float, b: float, n: float) -> float:
    """R ``dhyper(x, r, b, n, log = FALSE)`` for non-negative integer arguments."""
    if r < 0 or b < 0 or n < 0 or n > r + b:
        return math.nan
    if x < 0:
        return 0.0
    if n < x or r < x or n - x > b:
        return 0.0
    if n == 0:
        return 1.0 if x == 0 else 0.0
    p = float(n) / float(r + b)
    q = float(r + b - n) / float(r + b)
    p1 = dbinom_raw(x, r, p, q)
    p2 = dbinom_raw(n - x, b, p, q)
    p3 = dbinom_raw(n, r + b, p, q)
    return p1 * p2 / p3


_A = (2.2352520354606839287, 161.02823106855587881, 1067.6894854603709582,
      18154.981253343561249, 0.065682337918207449113)
_B = (47.20258190468824187, 976.09855173777669322, 10260.932208618978205,
      45507.789335026729956)
_C = (0.39894151208813466764, 8.8831497943883759412, 93.506656132177855979,
      597.27027639480026226, 2494.5375852903726711, 6848.1904505362823326,
      11602.651437647350124, 9842.7148383839780218, 1.0765576773720192317e-8)
_D = (22.266688044328115691, 235.38790178262499861, 1519.377599407554805,
      6485.558298266760755, 18615.571640885098091, 34900.952721145977266,
      38912.003286093271411, 19685.429676859990727)
_P = (0.21589853405795699, 0.1274011611602473639, 0.022235277870649807,
      0.001421619193227893466, 2.9112874951168792e-5, 0.02307344176494017303)
_Q = (1.28426009614491121, 0.468238212480865118, 0.0659881378689285515,
      0.00378239633202758244, 7.29751555083966205e-5)


def _do_del(x: float, temp: float) -> tuple[float, float]:
    xsq = math.ldexp(math.trunc(math.ldexp(x, 4)), -4)
    delta = (x - xsq) * (x + xsq)
    cum = math.exp(-xsq * math.ldexp(xsq, -1)) * math.exp(-math.ldexp(delta, -1)) * temp
    return cum, 1.0 - cum


def pnorm(x: float) -> float:
    """R ``pnorm(x, 0, 1, lower.tail = TRUE, log.p = FALSE)`` (``pnorm_both``)."""
    if math.isnan(x):
        return x
    if math.isinf(x):
        return 0.0 if x < 0 else 1.0
    eps = _DBL_EPSILON * 0.5
    y = abs(x)
    if y <= 0.67448975:
        if y > eps:
            xsq = x * x
            xnum = _A[4] * xsq
            xden = xsq
            for i in range(3):
                xnum = (xnum + _A[i]) * xsq
                xden = (xden + _B[i]) * xsq
        else:
            xnum = xden = 0.0
        temp = x * (xnum + _A[3]) / (xden + _B[3])
        return 0.5 + temp
    if y <= _M_SQRT_32:
        xnum = _C[8] * y
        xden = y
        for i in range(7):
            xnum = (xnum + _C[i]) * y
            xden = (xden + _D[i]) * y
        temp = (xnum + _C[7]) / (xden + _D[7])
        cum, ccum = _do_del(y, temp)
        return ccum if x > 0.0 else cum
    if -38.4674 < x < 8.2924:
        xsq = 1.0 / (x * x)
        xnum = _P[5] * xsq
        xden = xsq
        for i in range(4):
            xnum = (xnum + _P[i]) * xsq
            xden = (xden + _Q[i]) * xsq
        temp = xsq * (xnum + _P[4]) / (xden + _Q[4])
        temp = (_M_1_SQRT_2PI - temp) / y
        cum, ccum = _do_del(x, temp)
        return ccum if x > 0.0 else cum
    return 1.0 if x > 0 else 0.0


@functools.cache
def _libm_lgamma() -> Callable[[float], float]:
    """The C library's ``lgamma``, as R's C++ kernels call it (``std::lgamma``).

    CPython's ``math.lgamma`` is its own Lanczos approximation and differs from
    the C library's in the last bit for about half of all integer arguments.
    Falls back to ``math.lgamma`` where no C math library can be loaded.
    """
    names = [ctypes.util.find_library("m"), ctypes.util.find_library("c"), "ucrtbase"]
    for name in names:
        if not name:
            continue
        try:
            fn = ctypes.CDLL(name).lgamma
        except (OSError, AttributeError):
            continue
        fn.restype = ctypes.c_double
        fn.argtypes = [ctypes.c_double]
        if fn(1.0) == 0.0 and abs(fn(10.0) - math.lgamma(10.0)) < 1e-12:
            return lambda x: float(fn(x))
    return math.lgamma


def clgamma(x: float) -> float:
    """``std::lgamma(x)`` as evaluated by R's compiled code on this platform."""
    return _libm_lgamma()(x)



_M_SQRT2 = 1.41421356237309504880  # sqrt(2)
_M_2PI = 6.283185307179586476925286766559  # 2*pi


def qnorm(p: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """R's ``qnorm(p, mu, sigma)`` on the lower tail and probability scale.

    Wichura's AS 241 with R's extension beyond ``r = 27``, as in R 4.5.0
    ``src/nmath/qnorm.c``; ``scipy.stats.norm.ppf`` (Cephes ``ndtri``) differs
    from it in the last bits.
    """
    if math.isnan(p) or math.isnan(mu) or math.isnan(sigma):
        return p + mu + sigma
    if p < 0.0 or p > 1.0:
        return math.nan
    if p == 0.0:
        return -math.inf
    if p == 1.0:
        return math.inf
    if sigma < 0.0:
        return math.nan
    if sigma == 0.0:
        return mu
    q = p - 0.5
    if abs(q) <= 0.425:
        r = 0.180625 - q * q
        val = (
            q
            * (
                (
                    (
                        (
                            (
                                ((r * 2509.0809287301226727 + 33430.575583588128105) * r
                                 + 67265.770927008700853)
                                * r
                                + 45921.953931549871457
                            )
                            * r
                            + 13731.693765509461125
                        )
                        * r
                        + 1971.5909503065514427
                    )
                    * r
                    + 133.14166789178437745
                )
                * r
                + 3.387132872796366608
            )
            / (
                (
                    (
                        (
                            (
                                ((r * 5226.495278852854561 + 28729.085735721942674) * r
                                 + 39307.89580009271061)
                                * r
                                + 21213.794301586595867
                            )
                            * r
                            + 5394.1960214247511077
                        )
                        * r
                        + 687.1870074920579083
                    )
                    * r
                    + 42.313330701600911252
                )
                * r
                + 1.0
            )
        )
        return mu + sigma * val
    # R_DT_CIv(p) on the lower tail is R_D_Cval(p) = 0.5 - p + 0.5.
    lp = math.log((0.5 - p + 0.5) if q > 0 else p)
    r = math.sqrt(-lp)
    if r <= 5.0:
        r = r + -1.6
        val = (
            (
                (
                    (
                        (
                            ((r * 7.7454501427834140764e-4 + 0.0227238449892691845833) * r
                             + 0.24178072517745061177)
                            * r
                            + 1.27045825245236838258
                        )
                        * r
                        + 3.64784832476320460504
                    )
                    * r
                    + 5.7694972214606914055
                )
                * r
                + 4.6303378461565452959
            )
            * r
            + 1.42343711074968357734
        ) / (
            (
                (
                    (
                        (
                            ((r * 1.05075007164441684324e-9 + 5.475938084995344946e-4) * r
                             + 0.0151986665636164571966)
                            * r
                            + 0.14810397642748007459
                        )
                        * r
                        + 0.68976733498510000455
                    )
                    * r
                    + 1.6763848301838038494
                )
                * r
                + 2.05319162663775882187
            )
            * r
            + 1.0
        )
    elif r <= 27.0:
        r = r + -5.0
        val = (
            (
                (
                    (
                        (
                            ((r * 2.01033439929228813265e-7 + 2.71155556874348757815e-5) * r
                             + 0.0012426609473880784386)
                            * r
                            + 0.026532189526576123093
                        )
                        * r
                        + 0.29656057182850489123
                    )
                    * r
                    + 1.7848265399172913358
                )
                * r
                + 5.4637849111641143699
            )
            * r
            + 6.6579046435011037772
        ) / (
            (
                (
                    (
                        (
                            ((r * 2.04426310338993978564e-15 + 1.4215117583164458887e-7) * r
                             + 1.8463183175100546818e-5)
                            * r
                            + 7.868691311456132591e-4
                        )
                        * r
                        + 0.0148753612908506148525
                    )
                    * r
                    + 0.13692988092273580531
                )
                * r
                + 0.59983220655588793769
            )
            * r
            + 1.0
        )
    elif r >= 6.4e8:
        val = r * _M_SQRT2
    else:
        s2 = -math.ldexp(lp, 1)
        x2 = s2 - math.log(_M_2PI * s2)
        if r < 36000.0:
            x2 = s2 - math.log(_M_2PI * x2) - 2.0 / (2.0 + x2)
            if r < 840.0:
                x2 = s2 - math.log(_M_2PI * x2) + 2 * math.log1p(
                    -(1 - 1 / (4 + x2)) / (2.0 + x2)
                )
                if r < 109.0:
                    x2 = s2 - math.log(_M_2PI * x2) + 2 * math.log1p(
                        -(1 - (1 - 5 / (6 + x2)) / (4.0 + x2)) / (2.0 + x2)
                    )
                    if r < 55.0:
                        x2 = s2 - math.log(_M_2PI * x2) + 2 * math.log1p(
                            -(1 - (1 - (5 - 9 / (8.0 + x2)) / (6.0 + x2)) / (4.0 + x2))
                            / (2.0 + x2)
                        )
        val = math.sqrt(x2)
    if q < 0.0:
        val = -val
    return mu + sigma * val


def dt(x: float, n: float) -> float:
    """R's ``dt(x, n)`` (density, not log): Loader's factorisation through
    ``bd0`` and ``stirlerr``, as in R 4.5.0 ``src/nmath/dt.c``."""
    if math.isnan(x) or math.isnan(n):
        return x + n
    if n <= 0:
        return math.nan
    if not math.isfinite(x):
        return 0.0
    if not math.isfinite(n):
        return dnorm(x, 0.0, 1.0)
    t = -bd0(n / 2.0, (n + 1) / 2.0) + stirlerr((n + 1) / 2.0) - stirlerr(n / 2.0)
    x2n = x * x / n
    large = x2n > 1.0 / _DBL_EPSILON
    if large:
        ax = abs(x)
        l_x2n = math.log(ax) - math.log(n) / 2.0
        u = n * l_x2n
    elif x2n > 0.2:
        ax = 0.0
        l_x2n = math.log(1 + x2n) / 2.0
        u = n * l_x2n
    else:
        ax = 0.0
        l_x2n = math.log1p(x2n) / 2.0
        u = -bd0(n / 2.0, (n + x * x) / 2.0) + x * x / 2.0
    i_sqrt = math.sqrt(n) / ax if large else math.exp(-l_x2n)
    return math.exp(t - u) * _M_1_SQRT_2PI * i_sqrt


def dnorm(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """R's ``dnorm(x, mu, sigma)`` (``dnorm4``, density), as in R 4.5.0
    ``src/nmath/dnorm.c`` with its split evaluation beyond ``|x| = 5``."""
    if math.isnan(x) or math.isnan(mu) or math.isnan(sigma):
        return x + mu + sigma
    if sigma < 0:
        return math.nan
    if not math.isfinite(sigma):
        return 0.0
    if not math.isfinite(x) and mu == x:
        return math.nan
    if sigma == 0:
        return math.inf if x == mu else 0.0
    z = (x - mu) / sigma
    if not math.isfinite(z):
        return 0.0
    z = abs(z)
    if z >= 2 * math.sqrt(sys.float_info.max):
        return 0.0
    if z < 5:
        return _M_1_SQRT_2PI * math.exp(-0.5 * z * z) / sigma
    if z > math.sqrt(-2 * _M_LN2 * (sys.float_info.min_exp + 1 - sys.float_info.mant_dig)):
        return 0.0
    x1 = math.ldexp(_r_forceint(math.ldexp(z, 16)), -16)
    x2 = z - x1
    return _M_1_SQRT_2PI / sigma * (math.exp(-0.5 * x1 * x1) * math.exp((-0.5 * x2 - x1) * x2))


def dlnorm_log(x: float, meanlog: float, sdlog: float) -> float:
    """R's ``dlnorm(x, meanlog, sdlog, log = TRUE)`` (R 4.5.0 ``dlnorm.c``)."""
    if math.isnan(x) or math.isnan(meanlog) or math.isnan(sdlog):
        return x + meanlog + sdlog
    if sdlog < 0:
        return math.nan
    if sdlog == 0:
        return math.inf if x > 0 and math.log(x) == meanlog else -math.inf
    if x <= 0:
        return -math.inf
    y = (math.log(x) - meanlog) / sdlog
    return -(_M_LN_SQRT_2PI + 0.5 * y * y + math.log(x * sdlog))


def dexp(x: float, scale: float) -> float:
    """R's ``dexp(x, rate = 1 / scale)`` as C ``dexp(x, scale)`` (density)."""
    if math.isnan(x) or math.isnan(scale):
        return x + scale
    if scale <= 0.0:
        return math.nan
    if x < 0.0:
        return 0.0
    return math.exp(-x / scale) / scale


_M_LN2 = 0.693147180559945309417232121458  # log(2)


def _r_forceint(x: float) -> float:
    """R_forceint: C ``nearbyint`` in the default round-half-even mode."""
    return float(round(x))


def _c_exp(x: float) -> float:
    """C ``exp``: ``math.exp`` is libm's, but raises where C returns ``inf``."""
    try:
        return math.exp(x)
    except OverflowError:
        return math.inf


_libm_exp_ufunc = np.frompyfunc(_c_exp, 1, 1)


def libm_exp(values: Any) -> NDArray[np.float64]:
    """Elementwise libm ``exp``, as R evaluates ``exp()`` on a vector.

    NumPy's vectorised ``exp`` (SIMD on AVX2/AVX-512 builds) is not libm's and
    differs from it in the last bit for a few percent of inputs, which matters
    wherever R's result is compared exactly or feeds a discrete choice.
    """
    return np.asarray(_libm_exp_ufunc(np.asarray(values, dtype=np.float64)), dtype=np.float64)


def _c_log(x: float) -> float:
    """C ``log`` (R's ``R_log``): ``-inf`` at 0 and NaN below, where ``math.log`` raises."""
    if x > 0.0:
        return math.log(x)
    if x == 0.0:
        return -math.inf
    return math.nan


_libm_log_ufunc = np.frompyfunc(_c_log, 1, 1)


def libm_log(values: Any) -> NDArray[np.float64]:
    """Elementwise libm ``log``, as R evaluates ``log()`` on a vector.

    NumPy's vectorised ``log`` is not libm's on SIMD builds and can differ from
    it in the last bit.
    """
    return np.asarray(_libm_log_ufunc(np.asarray(values, dtype=np.float64)), dtype=np.float64)


def r_pow_scalar(x: float, y: float) -> float:
    """R's ``x ^ y`` (arithmetic.c ``R_POW``/``R_pow``) for doubles: ``x * x``
    for 2, 1 for a base of 1 or an exponent of 0, R's zero-base rules, else
    libm ``pow``."""
    if y == 2.0:
        return x * x
    if x == 1.0 or y == 0.0:
        return 1.0
    if x == 0.0:
        if y > 0.0:
            return 0.0
        return math.inf if y < 0.0 else y
    if math.isfinite(x) and math.isfinite(y):
        try:
            return math.pow(x, y)
        except ValueError:  # negative base with a fractional exponent
            return math.nan
        except OverflowError:
            return math.copysign(math.inf, x) if float(y).is_integer() and y % 2 else math.inf
    return float(np.power(np.float64(x), np.float64(y)))


_libm_pow_ufunc = np.frompyfunc(r_pow_scalar, 2, 1)


def libm_pow(values: Any, exponent: Any) -> NDArray[np.float64]:
    """Elementwise R ``^``: ``x * x`` for 2, else libm ``pow``. NumPy's
    vectorised power is not libm's and can differ from it in the last bit."""
    base = np.asarray(values, dtype=np.float64)
    power = np.asarray(exponent, dtype=np.float64)
    return np.asarray(_libm_pow_ufunc(base, power), dtype=np.float64)
