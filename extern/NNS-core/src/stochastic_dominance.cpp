// src/stochastic_dominance.cpp
//
// Faithful port of original_src/SD.cpp and original_src/stoch_sup.cpp.
// Decoupled from Rcpp.
//
// This file reproduces, exactly:
//   - ColPre prefix-sum precompute (sorted values, P1, P2, min, mean)
//   - identical_samples() short-circuit (identical series never dominate)
//   - for_each_threshold(): the MERGED grid of both series' values
//   - sd_dom_pair():
//       FSD  gate X.mn >= Y.mn; discrete -> ECDF compare,
//            continuous -> LPM1/(LPM1+UPM1) ratio compare; strict '>' fails
//       SSD  gates X.mn >= Y.mn and !(Y.mean > X.mean); LPM degree-1 compare
//       TSD  same gates; LPM degree-2 compare
//       (no epsilon tolerances anywhere, matching upstream)
//   - sd_dom_matrix (port of sd_dom_matrix_prefix_parallel)
//   - the efficient-set sweep (port of NNS_SD_efficient_set_parallel_cpp):
//       order columns by LPM(degree, tmax, .) ascending (stable tie-break by
//       index), then keep a column only if it is not dominated by any
//       previously KEPT column.
//   - stoch_superiority (p_gt / p_tie / p_star), two-pointer exact count
//
// NA semantics match upstream: NaN ("missing values") is rejected with the
// original error message; +/-Inf values are permitted, as in the Rcpp code
// (NumericVector::is_na is true for NA/NaN only).
//
// SPDX-License-Identifier: GPL-3.0-only
#include "nns/stochastic_dominance.hpp"
#include "nns/parallel.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace nns {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

inline double at(const double* M, std::size_t rows, std::size_t r, std::size_t c) {
  return M[c * rows + r];
}

// Upstream uses Rcpp::NumericVector::is_na, which is true for NA/NaN but
// false for +/-Inf.  Mirror that exactly.
inline void check_no_nan(const double* x, std::size_t len) {
  for (std::size_t k = 0; k < len; ++k) {
    if (std::isnan(x[k])) {
      throw std::invalid_argument("You have some missing values, please address.");
    }
  }
}

// small inline helper: repeated multiplication for integer exponents
inline double repeat_multiplication(double value, int n) {
  double result = 1.0;
  for (int i = 0; i < n; ++i) result *= value;
  return result;
}

// =====================================================================
// Per-column precompute: sorted values, prefix sums, basic stats
// =====================================================================
struct ColPre {
  std::vector<double> vals;      // sorted ascending, length m
  std::vector<double> P1;        // prefix sum of vals; length m+1, P1[0]=0
  std::vector<double> P2;        // prefix sum of vals^2; length m+1
  double S1{0.0}, S2{0.0};
  double mn{kInf}, mean{kNaN};
  int m{0};
};

ColPre precompute_ptr(const double* x, std::size_t n, std::size_t stride_rows,
                      std::size_t col) {
  ColPre c;
  c.m = static_cast<int>(n);
  c.vals.resize(n);
  for (std::size_t i = 0; i < n; ++i) c.vals[i] = at(x, stride_rows, i, col);
  std::sort(c.vals.begin(), c.vals.end());

  c.P1.assign(n + 1, 0.0);
  c.P2.assign(n + 1, 0.0);
  for (std::size_t k = 1; k <= n; ++k) {
    double v = c.vals[k - 1];
    c.P1[k] = c.P1[k - 1] + v;
    c.P2[k] = c.P2[k - 1] + v * v;
  }
  c.S1 = c.P1[n];
  c.S2 = c.P2[n];
  if (n > 0) {
    c.mn = c.vals.front();
    c.mean = c.S1 / static_cast<double>(n);
  }
  return c;
}

ColPre precompute_vec(const double* x, std::size_t n) {
  return precompute_ptr(x, n, n, 0);
}

inline bool identical_samples(const ColPre& a, const ColPre& b) {
  if (a.m != b.m) return false;
  for (int i = 0; i < a.m; ++i) {
    if (a.vals[i] != b.vals[i]) return false;
  }
  return true;
}

// =====================================================================
// O(1) evaluators from prefix sums
// =====================================================================
inline void lpm_upm_deg1(const ColPre& c, int k, double t, double& L1, double& U1) {
  // L1 = mean(max(t - x,0)) = (k*t - P1[k]) / m
  // U1 = mean(max(x - t,0)) = (S1 - P1[k] - (m-k)*t) / m
  double m = static_cast<double>(c.m);
  L1 = (k * t - c.P1[k]) / m;
  U1 = ((c.S1 - c.P1[k]) - (c.m - k) * t) / m;
}

inline double lpm_deg2(const ColPre& c, int k, double t) {
  // L2 = mean(max(t-x,0)^2) = (k*t^2 - 2t*P1[k] + P2[k]) / m
  double m = static_cast<double>(c.m);
  return (k * t * t - 2.0 * t * c.P1[k] + c.P2[k]) / m;
}

// Walk merged grid of both series' values and apply functor at each
// threshold t.  Preserves the upstream quirk of bounding both walkers by
// a.m (columns of one matrix always share the same row count).
template <class F>
inline void for_each_threshold(const ColPre& a, const ColPre& b, F f) {
  int ia = 0, ib = 0, m = a.m;
  while (ia < m || ib < m) {
    double next_a = (ia < m ? a.vals[ia] : kInf);
    double next_b = (ib < m ? b.vals[ib] : kInf);
    double t = (next_a < next_b ? next_a : next_b);
    while (ia < m && a.vals[ia] <= t) ++ia;  // k_a = ia
    while (ib < m && b.vals[ib] <= t) ++ib;  // k_b = ib
    f(t, ia, ib);
  }
}

// =====================================================================
// Pairwise dominance via prefix sums (O(m) per pair)
// degree: 1=FSD, 2=SSD, 3=TSD. 'discrete' only matters for FSD.
// Returns 1 iff X dominates Y, else 0.
// =====================================================================
inline int sd_dom_pair(const ColPre& X, const ColPre& Y, int degree, bool discrete) {
  if (degree == 1) {  // FSD
    if (!(X.mn >= Y.mn)) return 0;            // FSD gate
    if (identical_samples(X, Y)) return 0;    // identical series -> 0

    bool x_gt_y = false;
    int deg = (discrete ? 0 : 1);  // discrete->0, continuous->1
    for_each_threshold(X, Y, [&](double t, int kx, int ky) {
      double Rx, Ry;
      if (deg == 0) {
        // L0/(L0+U0) == ECDF
        Rx = static_cast<double>(kx) / static_cast<double>(X.m);
        Ry = static_cast<double>(ky) / static_cast<double>(Y.m);
      } else {
        double Lx, Ux, Ly, Uy;
        lpm_upm_deg1(X, kx, t, Lx, Ux);
        lpm_upm_deg1(Y, ky, t, Ly, Uy);
        double Ax = Lx + Ux, Ay = Ly + Uy;
        Rx = (Ax > 0.0 ? Lx / Ax : 0.0);
        Ry = (Ay > 0.0 ? Ly / Ay : 0.0);
      }
      if (Rx > Ry) x_gt_y = true;
    });
    return x_gt_y ? 0 : 1;                    // 1 iff "X FSD Y"
  }

  // SSD/TSD gates
  if (!(X.mn >= Y.mn) || (Y.mean > X.mean)) return 0;
  if (identical_samples(X, Y)) return 0;      // identical series -> 0

  if (degree == 2) {  // SSD: compare LPM degree 1
    bool x_gt_y = false;
    for_each_threshold(X, Y, [&](double t, int kx, int ky) {
      double Lx, Ux, Ly, Uy;
      (void)Ux; (void)Uy;  // not used beyond calc
      lpm_upm_deg1(X, kx, t, Lx, Ux);
      lpm_upm_deg1(Y, ky, t, Ly, Uy);
      if (Lx > Ly) x_gt_y = true;
    });
    return x_gt_y ? 0 : 1;                    // 1 iff "X SSD Y"
  }

  // TSD: compare LPM degree 2
  bool x_gt_y = false;
  for_each_threshold(X, Y, [&](double t, int kx, int ky) {
    double Lx2 = lpm_deg2(X, kx, t);
    double Ly2 = lpm_deg2(Y, ky, t);
    if (Lx2 > Ly2) x_gt_y = true;
  });
  return x_gt_y ? 0 : 1;                      // 1 iff "X TSD Y"
}

// Dominance matrix over an arbitrary column ordering (parallel over rows).
// dom is p x p column-major: dom[j * p + i] = 1 iff column ord[i] dominates
// column ord[j].
std::vector<int> dom_matrix_for_order(const std::vector<ColPre>& cols,
                                      const std::vector<int>& ord,
                                      int degree, bool discrete, int nthreads) {
  const std::size_t p = ord.size();
  std::vector<int> dom(p * p, 0);
  parallel_for(0, p, [&](std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; ++i) {
      for (std::size_t j = 0; j < p; ++j) {
        dom[j * p + i] =
            (i == j) ? 0
                     : sd_dom_pair(cols[static_cast<std::size_t>(ord[i])],
                                   cols[static_cast<std::size_t>(ord[j])],
                                   degree, discrete);
      }
    }
  }, nthreads);
  return dom;
}

// Port of NNS_SD_efficient_set_parallel_cpp.  Returns the surviving
// ORIGINAL 0-based column indices, in ascending-LPM(degree, tmax) order
// (the same order in which upstream returns column names).
std::vector<int> efficient_set(const double* X, std::size_t n, std::size_t p,
                               int degree, bool discrete, int nthreads) {
  if (p == 0) return {};
  if (!(degree == 1 || degree == 2 || degree == 3)) {
    throw std::invalid_argument("degree must be 1, 2, or 3");
  }
  // The upstream pipeline always reaches sd_dom_matrix_prefix_parallel,
  // which stops on any missing value; observable behavior is a hard error.
  check_no_nan(X, n * p);

  // global max for ordering key
  double tmax = -kInf;
  for (std::size_t k = 0; k < n * p; ++k) {
    if (X[k] > tmax) tmax = X[k];
  }

  // precompute columns
  std::vector<ColPre> cols;
  cols.reserve(p);
  for (std::size_t j = 0; j < p; ++j) cols.push_back(precompute_ptr(X, n, n, j));

  // ===== order by LPM(degree, tmax, .) =====
  std::vector<double> lpm_vals(p, 0.0);
  for (std::size_t j = 0; j < p; ++j) {
    double sum = 0.0;
    int cnt = 0;
    for (std::size_t i = 0; i < n; ++i) {
      double xv = at(X, n, i, j);
      double diff = tmax - xv;
      if (diff > 0.0) {
        sum += repeat_multiplication(diff, degree);
      }
      cnt++;
    }
    lpm_vals[j] = (cnt > 0) ? sum / static_cast<double>(cnt) : kInf;
  }

  std::vector<int> ord(p);
  for (std::size_t j = 0; j < p; ++j) ord[j] = static_cast<int>(j);
  std::sort(ord.begin(), ord.end(), [&](int a, int b) {
    if (lpm_vals[static_cast<std::size_t>(a)] ==
        lpm_vals[static_cast<std::size_t>(b)]) {
      return a < b;  // stable tie-break by index
    }
    return lpm_vals[static_cast<std::size_t>(a)] <
           lpm_vals[static_cast<std::size_t>(b)];
  });

  // dominance matrix in the sorted order
  const std::vector<int> D = dom_matrix_for_order(cols, ord, degree, discrete, nthreads);

  // single pass to keep maximal elements: a column is dropped only if a
  // previously KEPT column dominates it.
  std::vector<char> keep(p, 0);
  for (std::size_t k = 0; k < p; ++k) {
    bool dominated = false;
    for (std::size_t i = 0; i < k; ++i) {
      if (keep[i] && D[k * p + i] == 1) {  // D(i, k) == 1
        dominated = true;
        break;
      }
    }
    keep[k] = dominated ? 0 : 1;
  }

  std::vector<int> out;
  out.reserve(p);
  for (std::size_t k = 0; k < p; ++k) {
    if (keep[k]) out.push_back(ord[k]);
  }
  return out;
}

}  // namespace

// ---------- Dominance Matrix (port of sd_dom_matrix_prefix_parallel) -------

std::vector<int> sd_dom_matrix(const double* X, std::size_t n, std::size_t p,
                               int degree, bool discrete, int nthreads) {
  if (!(degree == 1 || degree == 2 || degree == 3)) {
    throw std::invalid_argument("degree must be 1, 2, or 3");
  }
  check_no_nan(X, n * p);

  // 'discrete' only matters for degree 1 (upstream forces discrete = true
  // for degrees 2 and 3 regardless of the supplied type).
  const bool disc = (degree == 1) ? discrete : true;

  std::vector<ColPre> cols;
  cols.reserve(p);
  for (std::size_t j = 0; j < p; ++j) cols.push_back(precompute_ptr(X, n, n, j));

  std::vector<int> ord(p);
  for (std::size_t j = 0; j < p; ++j) ord[j] = static_cast<int>(j);
  return dom_matrix_for_order(cols, ord, degree, disc, nthreads);
}

// ---------- Univariate Wrappers ----------

int fsd_uni(const double* x, const double* y, std::size_t n, bool discrete) {
  check_no_nan(x, n);
  check_no_nan(y, n);
  ColPre X = precompute_vec(x, n);
  ColPre Y = precompute_vec(y, n);
  return sd_dom_pair(X, Y, 1, discrete);
}

int ssd_uni(const double* x, const double* y, std::size_t n) {
  check_no_nan(x, n);
  check_no_nan(y, n);
  ColPre X = precompute_vec(x, n);
  ColPre Y = precompute_vec(y, n);
  return sd_dom_pair(X, Y, 2, true);  // discrete flag irrelevant past FSD
}

int tsd_uni(const double* x, const double* y, std::size_t n) {
  check_no_nan(x, n);
  check_no_nan(y, n);
  ColPre X = precompute_vec(x, n);
  ColPre Y = precompute_vec(y, n);
  return sd_dom_pair(X, Y, 3, true);  // discrete flag irrelevant past FSD
}

// ---------- Multivariate Efficient-Set Wrappers ----------

std::vector<int> fsd(const double* X, std::size_t n, std::size_t p, bool discrete, int nthreads) {
  return efficient_set(X, n, p, 1, discrete, nthreads);
}

std::vector<int> ssd(const double* X, std::size_t n, std::size_t p, int nthreads) {
  return efficient_set(X, n, p, 2, true, nthreads);
}

std::vector<int> tsd(const double* X, std::size_t n, std::size_t p, int nthreads) {
  return efficient_set(X, n, p, 3, true, nthreads);
}

// ---------- Stochastic Superiority (port of stoch_superiority_cpp) ---------

StochSupResult stochastic_superiority(const double* x, std::size_t n_x,
                                      const double* y, std::size_t n_y) {
  if (n_x == 0 || n_y == 0) {
    throw std::invalid_argument("x and y must both have positive length.");
  }

  // Clone and sort the arrays natively
  std::vector<double> xs(x, x + n_x);
  std::vector<double> ys(y, y + n_y);

  std::sort(xs.begin(), xs.end());
  std::sort(ys.begin(), ys.end());

  long double less_count = 0.0L;
  long double tie_count = 0.0L;

  std::size_t left = 0;   // number of elements in y strictly less than x[i]
  std::size_t right = 0;  // number of elements in y less than or equal to x[i]

  for (std::size_t i = 0; i < n_x; ++i) {
    const double xi = xs[i];

    while (left < n_y && ys[left] < xi) {
      ++left;
    }
    while (right < n_y && ys[right] <= xi) {
      ++right;
    }

    less_count += left;
    tie_count += (right - left);
  }

  const long double denom =
      static_cast<long double>(n_x) * static_cast<long double>(n_y);

  const double p_gt = static_cast<double>(less_count / denom);
  const double p_tie = static_cast<double>(tie_count / denom);
  const double p_star = p_gt + 0.5 * p_tie;

  return {p_gt, p_tie, p_star};
}

}  // namespace nns