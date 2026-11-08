// Bit-faithful ports of the R nmath routines called by the NNS C++ kernels.
//
// R's src/NNS_dep.cpp calls R::dhyper and R::pnorm. They use Loader's
// saddle-point binomial (dbinom_raw with stirlerr and bd0) and Cody's
// rational approximation; closed forms from binomial products or erfc agree
// with them only to a few units in the last place, which NNS later feeds into
// exact comparisons. These are the R routines themselves, evaluated with the
// same libm calls.
//
// Source: R 4.5.0 src/nmath/{dhyper,dbinom,stirlerr,bd0,pnorm}.c
// Copyright (C) 2000-2024 The R Core Team; Catherine Loader; W. J. Cody.
// GPL-2 or later. stirlerr follows the R >= 4.4.0 series cutoffs.
// Mirrored in src/nns/_rnmath.py for the pure-Python fallback.
// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include <cfloat>
#include <cmath>

namespace nns_rmath {

static const double kLn2Pi = 1.837877066409345483560659472811;      // log(2*pi)
static const double kLnSqrt2Pi = 0.918938533204672741780329736406;  // log(sqrt(2*pi))
static const double kSqrt32 = 5.656854249492380195206754896838;     // sqrt(32)
static const double k1SqrtTwoPi = 0.398942280401432677939946059934; // 1/sqrt(2*pi)

inline double stirlerr(double n) {
  static const double S[17] = {
      0.083333333333333333333,        // 1/12
      0.00277777777777777777778,      // 1/360
      0.00079365079365079365079365,   // 1/1260
      0.000595238095238095238095238,  // 1/1680
      0.0008417508417508417508417508, // 1/1188
      0.0019175269175269175269175262, // 691/360360
      0.0064102564102564102564102561, // 1/156
      0.029550653594771241830065352,  // 3617/122400
      0.17964437236883057316493850,   // 43867/244188
      1.3924322169059011164274315,    // 174611/125400
      13.402864044168391994478957,    // 77683/5796
      156.84828462600201730636509,    // 236364091/1506960
      2193.1033333333333333333333,    // 657931/300
      36108.771253724989357173269,    // 3392780147/93960
      691472.26885131306710839498,    // 1723168255201/2492028
      15238221.539407416192283370,    // 7709321041217/505920
      382900751.39141414141414141};   // 151628697551/396
  static const double sferr_halves[31] = {
      0.0, /* n=0 - wrong, place holder only */
      0.1534264097200273452913848,   0.0810614667953272582196702,
      0.0548141210519176538961390,   0.0413406959554092940938221,
      0.03316287351993628748511048,  0.02767792568499833914878929,
      0.02374616365629749597132920,  0.02079067210376509311152277,
      0.01848845053267318523077934,  0.01664469118982119216319487,
      0.01513497322191737887351255,  0.01387612882307074799874573,
      0.01281046524292022692424986,  0.01189670994589177009505572,
      0.01110455975820691732662991,  0.010411265261972096497478567,
      0.009799416126158803298389475, 0.009255462182712732917728637,
      0.008768700134139385462952823, 0.008330563433362871256469318,
      0.007934114564314020547248100, 0.007573675487951840794972024,
      0.007244554301320383179543912, 0.006942840107209529865664152,
      0.006665247032707682442354394, 0.006408994188004207068439631,
      0.006171712263039457647532867, 0.005951370112758847735624416,
      0.005746216513010115682023589, 0.005554733551962801371038690};
  // (S0-(S1-(...-S[k-1]/nn)/nn)...)/n with k terms, as R spells it out.
  auto series = [&](double nn, int k) {
    double acc = S[k - 1];
    for (int j = k - 2; j >= 0; --j) acc = S[j] - acc / nn;
    return acc / n;
  };
  double nn;
  if (n <= 23.5) {
    nn = n + n;
    if (n <= 15.0 && (nn == static_cast<int>(nn))) return sferr_halves[static_cast<int>(nn)];
    if (n <= 5.25) {
      if (n >= 1.0) {
        const double l_n = std::log(n);
        return std::lgamma(n) + n * (1 - l_n) + std::ldexp(l_n - kLn2Pi, -1);
      }
      return std::lgamma(1.0 + n) - (n + 0.5) * std::log(n) + n - kLnSqrt2Pi;
    }
    nn = n * n;
    if (n > 12.8) return series(nn, 7);
    if (n > 12.3) return series(nn, 8);
    if (n > 8.9) return series(nn, 9);
    if (n > 7.3) return series(nn, 11);
    if (n > 6.6) return series(nn, 13);
    if (n > 6.1) return series(nn, 15);
    return series(nn, 17);
  }
  nn = n * n;
  if (n > 15.7e6) return S[0] / n;
  if (n > 6180) return series(nn, 2);
  if (n > 205) return series(nn, 3);
  if (n > 86) return series(nn, 4);
  if (n > 27) return series(nn, 5);
  return series(nn, 6);
}

inline double bd0(double x, double np) {
  if (!std::isfinite(x) || !std::isfinite(np) || np == 0.0) return NAN;
  if (std::fabs(x - np) < 0.1 * (x + np)) {
    double v = (x - np) / (x + np);
    double s = (x - np) * v;
    if (std::fabs(s) < DBL_MIN) return s;
    double ej = 2 * x * v;
    v *= v;
    for (int j = 1; j < 1000; j++) {
      ej *= v;
      const double s_ = s;
      s += ej / ((j << 1) + 1);
      if (s == s_) return s;
    }
  }
  return x * std::log(x / np) + np - x;
}

inline double pow1p(double x, double y) {
  if (std::isnan(y)) return (x == 0) ? 1. : y;
  if (0 <= y && y == std::trunc(y) && y <= 4.) {
    switch (static_cast<int>(y)) {
      case 0: return 1;
      case 1: return x + 1.;
      case 2: return x * (x + 2.) + 1.;
      case 3: return x * (x * (x + 3.) + 3.) + 1.;
      case 4: return x * (x * (x * (x + 4.) + 6.) + 4.) + 1.;
    }
  }
  if ((x + 1) - 1 == x || std::fabs(x) > 0.5 || std::isnan(x)) return std::pow(1 + x, y);
  return std::exp(y * std::log1p(x));
}

inline double dbinom_raw(double x, double n, double p, double q) {
  if (p == 0) return (x == 0) ? 1. : 0.;
  if (q == 0) return (x == n) ? 1. : 0.;
  if (x == 0) {
    if (n == 0) return 1.;
    return (p > q) ? std::pow(q, n) : pow1p(-p, n);
  }
  if (x == n) return (p > q) ? pow1p(-q, n) : std::pow(p, n);
  if (x < 0 || x > n) return 0.;
  const double lc =
      stirlerr(n) - stirlerr(x) - stirlerr(n - x) - bd0(x, n * p) - bd0(n - x, n * q);
  const double lf = kLn2Pi + std::log(x) + std::log1p(-x / n);
  return std::exp(lc - 0.5 * lf);
}

// R::dhyper(x, r, b, n, 0) for non-negative integer arguments.
inline double dhyper(double x, double r, double b, double n) {
  if (r < 0 || b < 0 || n < 0 || n > r + b) return NAN;
  if (x < 0) return 0.;
  if (n < x || r < x || n - x > b) return 0.;
  if (n == 0) return (x == 0) ? 1. : 0.;
  const double p = n / (r + b);
  const double q = (r + b - n) / (r + b);
  const double p1 = dbinom_raw(x, r, p, q);
  const double p2 = dbinom_raw(n - x, b, p, q);
  const double p3 = dbinom_raw(n, r + b, p, q);
  return p1 * p2 / p3;
}

// R::pnorm(x, 0, 1, 1, 0): the lower tail of pnorm_both.
inline double pnorm(double x) {
  static const double a[5] = {2.2352520354606839287, 161.02823106855587881,
                              1067.6894854603709582, 18154.981253343561249,
                              0.065682337918207449113};
  static const double b[4] = {47.20258190468824187, 976.09855173777669322,
                              10260.932208618978205, 45507.789335026729956};
  static const double c[9] = {0.39894151208813466764, 8.8831497943883759412,
                              93.506656132177855979,  597.27027639480026226,
                              2494.5375852903726711,  6848.1904505362823326,
                              11602.651437647350124,  9842.7148383839780218,
                              1.0765576773720192317e-8};
  static const double d[8] = {22.266688044328115691, 235.38790178262499861,
                              1519.377599407554805,  6485.558298266760755,
                              18615.571640885098091, 34900.952721145977266,
                              38912.003286093271411, 19685.429676859990727};
  static const double p[6] = {0.21589853405795699,  0.1274011611602473639,
                              0.022235277870649807, 0.001421619193227893466,
                              2.9112874951168792e-5, 0.02307344176494017303};
  static const double q[5] = {1.28426009614491121, 0.468238212480865118,
                              0.0659881378689285515, 0.00378239633202758244,
                              7.29751555083966205e-5};
  if (std::isnan(x)) return x;
  if (!std::isfinite(x)) return x < 0 ? 0. : 1.;
  const double eps = DBL_EPSILON * 0.5;
  const double y = std::fabs(x);
  double xnum, xden, temp, xsq, del, cum, ccum;
  auto do_del = [&](double X) {
    xsq = std::ldexp(std::trunc(std::ldexp(X, 4)), -4);
    del = (X - xsq) * (X + xsq);
    cum = std::exp(-xsq * std::ldexp(xsq, -1)) * std::exp(-std::ldexp(del, -1)) * temp;
    ccum = 1.0 - cum;
  };
  if (y <= 0.67448975) {
    if (y > eps) {
      xsq = x * x;
      xnum = a[4] * xsq;
      xden = xsq;
      for (int i = 0; i < 3; ++i) {
        xnum = (xnum + a[i]) * xsq;
        xden = (xden + b[i]) * xsq;
      }
    } else {
      xnum = xden = 0.0;
    }
    temp = x * (xnum + a[3]) / (xden + b[3]);
    return 0.5 + temp;
  }
  if (y <= kSqrt32) {
    xnum = c[8] * y;
    xden = y;
    for (int i = 0; i < 7; ++i) {
      xnum = (xnum + c[i]) * y;
      xden = (xden + d[i]) * y;
    }
    temp = (xnum + c[7]) / (xden + d[7]);
    do_del(y);
    return x > 0. ? ccum : cum;
  }
  if (-38.4674 < x && x < 8.2924) {
    xsq = 1.0 / (x * x);
    xnum = p[5] * xsq;
    xden = xsq;
    for (int i = 0; i < 4; ++i) {
      xnum = (xnum + p[i]) * xsq;
      xden = (xden + q[i]) * xsq;
    }
    temp = xsq * (xnum + p[4]) / (xden + q[4]);
    temp = (k1SqrtTwoPi - temp) / y;
    do_del(x);
    return x > 0. ? ccum : cum;
  }
  return x > 0 ? 1. : 0.;
}

}  // namespace nns_rmath
