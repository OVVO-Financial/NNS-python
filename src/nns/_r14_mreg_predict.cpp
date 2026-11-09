// src/nns/_r14_mreg_predict.cpp: R14 overlay of NNS-core src/mreg_predict.cpp
//
// Identical to the vendored kernel except for its density components: the
// Student-t and normal weights call R's own nmath dt and dnorm4
// (src/nns/_rnmath.hpp) instead of the vendor's lgamma-based closed form, so
// the ensemble weights, and the predictions, match R's NNS_mreg_predict_v2_cpp
// bit for bit.
//
// Pure C++ port of R NNS 13.3 src/NNS_mreg_predict.cpp
// (NNS_mreg_predict_v2_cpp / NNS_mreg_predict_path_v2_cpp, i.e. the
// PredictPathWorker kernel). Decoupled from Rcpp and Rmath:
//   - range-normalised NNS / L2 / L1 or Hamming (FACTOR) distances,
//   - stable ascending order with index tie-break via partial selection,
//   - k = 1 aggregates all exact nearest-distance ties (gravity / class mode),
//   - k > 1 uses the eight-component ensemble weights,
//   - exact weighted class mode (max total, tie -> smallest value),
//   - the O(log n + k) 1-D fast path for duplicated coordinates.
//
// The R kernel evaluates its density components with Rmath (Rf_dt, Rf_dexp,
// Rf_dlnorm, Rf_dnorm4) inside RcppParallel workers. Here dexp, dlnorm and
// dnorm are the same closed forms Rmath uses, and dt follows Rmath's
// factorisation with lgamma for the constant. No R API is touched from worker
// threads, and results agree with R to floating-point rounding.
//
// SPDX-License-Identifier: GPL-3.0-only
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "_rnmath.hpp"
#include "nns/central_tendencies.hpp"
#include "nns/mreg.hpp"
#include "nns/parallel.hpp"

namespace nns {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kLnSqrt2Pi = 0.918938533204672741780329736406;  // log(sqrt(2*pi))

// ---- densities (closed forms; see file comment) -----------------------------

// Rmath dexp(x, scale, give_log = 0).
inline double dexp_scale(double x, double scale) {
  return (x < 0.0) ? 0.0 : std::exp(-x / scale) / scale;
}

// Rmath dlnorm(x, meanlog = 0, sdlog, give_log = 1) for x > 0, sdlog > 0.
inline double dlnorm_log(double x, double sdlog) {
  const double y = std::log(x) / sdlog;
  return -(kLnSqrt2Pi + 0.5 * y * y + std::log(x * sdlog));
}

// ---- weight helpers ----------------------------------------------------------

inline void normalize_weights(std::vector<double>& w) {
  const std::size_t k = w.size();
  double s = 0.0;
  for (double& v : w) {
    if (!std::isfinite(v) || v < 0.0) v = 0.0;
    s += v;
  }
  if (s <= 0.0) {
    for (double& v : w) v = 1.0 / static_cast<double>(k);
  } else {
    for (double& v : w) v /= s;
  }
}

inline double sample_sd(const std::vector<double>& v) {
  const std::size_t n = v.size();
  if (n < 2) return kNaN;
  double mu = 0.0;
  for (double x : v) mu += x;
  mu /= static_cast<double>(n);
  double acc = 0.0;
  for (double x : v) {
    const double d = x - mu;
    acc += d * d;
  }
  return std::sqrt(acc / static_cast<double>(n - 1));
}

// Totals per distinct value, max total, tie -> smallest value.
inline double weighted_mode(const std::vector<double>& vals, const std::vector<double>& w) {
  std::vector<std::pair<double, double>> items;
  items.reserve(vals.size());
  for (std::size_t i = 0; i < vals.size(); ++i) {
    if (std::isfinite(vals[i]) && std::isfinite(w[i]) && w[i] >= 0.0)
      items.push_back(std::make_pair(vals[i], w[i]));
  }
  if (items.empty()) return kNaN;
  std::sort(items.begin(), items.end());
  double best_val = items[0].first;
  double best_tot = -1.0;
  std::size_t i = 0;
  while (i < items.size()) {
    const double v = items[i].first;
    double tot = 0.0;
    while (i < items.size() && items[i].first == v) {
      tot += items[i].second;
      ++i;
    }
    if (tot > best_tot) {
      best_tot = tot;
      best_val = v;
    }
  }
  return best_val;
}

void rank_components(int k, std::vector<double>& uniform, std::vector<double>& exponential,
                     std::vector<double>& lognormal, std::vector<double>& power) {
  uniform.assign(k, 1.0 / static_cast<double>(k));
  exponential.resize(k);
  lognormal.assign(k, 0.0);
  power.resize(k);
  const double kk = static_cast<double>(k);
  for (int i = 0; i < k; ++i) exponential[i] = dexp_scale(static_cast<double>(i + 1), kk);
  normalize_weights(exponential);
  std::vector<double> ranks(k);
  for (int i = 0; i < k; ++i) ranks[i] = static_cast<double>(i + 1);
  const double rank_sd = sample_sd(ranks);
  if (std::isfinite(rank_sd) && rank_sd > 0.0) {
    for (int i = 0; i < k; ++i) lognormal[i] = std::fabs(dlnorm_log(ranks[i], rank_sd));
    normalize_weights(lognormal);
    std::reverse(lognormal.begin(), lognormal.end());
  }
  for (int i = 0; i < k; ++i) {
    const double r = static_cast<double>(i + 1);
    power[i] = 1.0 / (r * r);
  }
  normalize_weights(power);
}

// Rank weights for k in [kmin, kmax]; every k up to kmax is O(kmax^2) memory.
struct RankComponents {
  std::vector<std::vector<double>> uniform, exponential, lognormal, power;
  RankComponents(int kmin, int kmax)
      : uniform(kmax + 1), exponential(kmax + 1), lognormal(kmax + 1), power(kmax + 1) {
    for (int k = std::max(kmin, 2); k <= kmax; ++k) {
      rank_components(k, uniform[k], exponential[k], lognormal[k], power[k]);
    }
  }
};

inline void normalize_prefix(std::vector<double>& values, int k) {
  double s = 0.0;
  for (int i = 0; i < k; ++i) {
    if (!std::isfinite(values[i]) || values[i] < 0.0) values[i] = 0.0;
    s += values[i];
  }
  if (s <= 0.0) {
    const double u = 1.0 / static_cast<double>(k);
    for (int i = 0; i < k; ++i) values[i] = u;
  } else {
    for (int i = 0; i < k; ++i) values[i] /= s;
  }
}

inline double weighted_mode_prefix(const std::vector<double>& yk,
                                   const std::vector<double>& weights, int k,
                                   std::vector<double>& cls, std::vector<double>& totals) {
  int ncls = 0;
  for (int i = 0; i < k; ++i) {
    const double v = yk[i], w = weights[i];
    if (!std::isfinite(v) || !std::isfinite(w) || w < 0.0) continue;
    int pos = -1;
    for (int c = 0; c < ncls; ++c)
      if (cls[c] == v) {
        pos = c;
        break;
      }
    if (pos < 0) {
      pos = ncls++;
      cls[pos] = v;
      totals[pos] = 0.0;
    }
    totals[pos] += w;
  }
  if (ncls == 0) return kNaN;
  double best_val = cls[0], best_tot = totals[0];
  for (int c = 1; c < ncls; ++c) {
    if (totals[c] > best_tot || (totals[c] == best_tot && cls[c] < best_val)) {
      best_tot = totals[c];
      best_val = cls[c];
    }
  }
  return best_val;
}

struct Scratch {
  std::vector<double> student, inverse, normal, rbf, total, cls, cls_totals;
  explicit Scratch(int kmax)
      : student(kmax), inverse(kmax), normal(kmax), rbf(kmax), total(kmax), cls(kmax),
        cls_totals(kmax) {}
};

double predict_sorted_k(const std::vector<double>& dk, const std::vector<double>& yk, int k,
                        bool is_class, const RankComponents& rank, Scratch& s) {
  const double kk = static_cast<double>(k);
  for (int i = 0; i < k; ++i) s.student[i] = nns_rmath::dt(dk[i], kk);
  normalize_prefix(s.student, k);
  for (int i = 0; i < k; ++i) s.inverse[i] = 1.0 / std::max(dk[i], 1e-12);
  normalize_prefix(s.inverse, k);

  double mu = 0.0;
  for (int i = 0; i < k; ++i) mu += dk[i];
  mu /= kk;
  double acc = 0.0;
  for (int i = 0; i < k; ++i) {
    const double z = dk[i] - mu;
    acc += z * z;
  }
  const double sd = (k > 1) ? std::sqrt(acc / static_cast<double>(k - 1)) : kNaN;
  const double var = std::isfinite(sd) ? sd * sd : kNaN;
  if (std::isfinite(sd) && sd > 0.0) {
    for (int i = 0; i < k; ++i) s.normal[i] = nns_rmath::dnorm(dk[i], 0.0, sd);
    normalize_prefix(s.normal, k);
  } else {
    for (int i = 0; i < k; ++i) s.normal[i] = 0.0;
  }
  if (std::isfinite(var) && var > 0.0) {
    for (int i = 0; i < k; ++i) s.rbf[i] = std::exp(-dk[i] / (2.0 * var));
    normalize_prefix(s.rbf, k);
  } else {
    for (int i = 0; i < k; ++i) s.rbf[i] = 0.0;
  }

  for (int i = 0; i < k; ++i) {
    s.total[i] = rank.uniform[k][i] + s.student[i] + s.inverse[i] + rank.exponential[k][i] +
                 rank.lognormal[k][i] + rank.power[k][i] + s.normal[i] + s.rbf[i];
  }
  normalize_prefix(s.total, k);

  if (is_class) return weighted_mode_prefix(yk, s.total, k, s.cls, s.cls_totals);
  double dot = 0.0;
  for (int i = 0; i < k; ++i) dot += yk[i] * s.total[i];
  return dot;
}

// ---- the prediction kernel (port of PredictPathWorker) ----------------------

class PredictKernel {
 public:
  PredictKernel(const double* rpm_x, std::size_t n, std::size_t p, const double* yhat,
                const double* Xtest, std::size_t m, int kmax, int dist_code,
                const double* mins, const double* maxs, bool is_class,
                const RankComponents& rank, double* out, int only_k)
      : rpm_x_(rpm_x), yhat_(yhat), Xtest_(Xtest), out_(out), n_(static_cast<int>(n)),
        p_(static_cast<int>(p)), m_(m), kmax_(kmax), dist_code_(dist_code),
        only_k_(only_k), is_class_(is_class), rank_(rank) {
    for (int j = 0; j < p_; ++j) {
      const double range = maxs[j] - mins[j];
      if (std::isfinite(range) && range > 0.0) {
        active_.push_back(j);
        col_range_.push_back(range);
      }
    }
    // Effectively 1-D design: all active columns identical in rpm_x and Xtest
    // (FACTOR / Hamming excluded).
    if (dist_code_ != kMregDistFactor && !active_.empty()) {
      axis_ = active_[0];
      bool identical = true;
      for (std::size_t a = 1; a < active_.size() && identical; ++a) {
        const int j = active_[a];
        for (int i = 0; i < n_; ++i)
          if (rx(i, j) != rx(i, axis_)) {
            identical = false;
            break;
          }
        for (std::size_t i = 0; identical && i < m_; ++i)
          if (xt(i, j) != xt(i, axis_)) {
            identical = false;
            break;
          }
      }
      if (identical) {
        is_1d_ = true;
        std::vector<double> coord(n_);
        for (int i = 0; i < n_; ++i) coord[i] = rx(i, axis_);
        so_.resize(n_);
        for (int i = 0; i < n_; ++i) so_[i] = i;
        std::sort(so_.begin(), so_.end(), [&coord](int a, int b) {
          return coord[a] < coord[b] || (coord[a] == coord[b] && a < b);
        });
        sc_.resize(n_);
        for (int i = 0; i < n_; ++i) sc_[i] = coord[so_[i]];
      }
    }
  }

  void operator()(std::size_t begin, std::size_t end) const {
    std::vector<double> d(n_), dk(kmax_), yk(kmax_), tied, one_weights;
    std::vector<int> idx(n_), tk;
    Scratch s(kmax_);
    auto cmp = [&d](int a, int b) { return d[a] < d[b] || (d[a] == d[b] && a < b); };
    auto out = [&](std::size_t r, std::size_t c) -> double& { return out_[r + c * m_]; };
    for (std::size_t r = begin; r < end; ++r) {
      if (is_1d_) {
        const double t = xt(r, axis_);
        if (only_k_ > 0) {
          if (only_k_ == 1) {
            out(r, 0) = min_ties_1d(r, t, tied, one_weights);
          } else {
            topk_1d(r, t, only_k_, tk, dk);
            for (int i = 0; i < only_k_; ++i) yk[i] = yhat_[tk[i]];
            out(r, 0) = predict_sorted_k(dk, yk, only_k_, is_class_, rank_, s);
          }
        } else {
          out(r, 0) = min_ties_1d(r, t, tied, one_weights);
          if (kmax_ >= 2) {
            topk_1d(r, t, kmax_, tk, dk);
            for (int i = 0; i < kmax_; ++i) yk[i] = yhat_[tk[i]];
            for (int k = 2; k <= kmax_; ++k)
              out(r, static_cast<std::size_t>(k - 1)) =
                  predict_sorted_k(dk, yk, k, is_class_, rank_, s);
          }
        }
        continue;
      }
      distances_for_row(r, d);
      for (int i = 0; i < n_; ++i) idx[i] = i;
      if (kmax_ < n_) {
        std::nth_element(idx.begin(), idx.begin() + kmax_, idx.end(), cmp);
        std::sort(idx.begin(), idx.begin() + kmax_, cmp);
      } else {
        std::sort(idx.begin(), idx.end(), cmp);
      }
      for (int i = 0; i < kmax_; ++i) {
        dk[i] = d[idx[i]];
        yk[i] = yhat_[idx[i]];
      }
      if (only_k_ > 0) {
        out(r, 0) = (only_k_ == 1) ? min_ties(d, tied, one_weights)
                                   : predict_sorted_k(dk, yk, only_k_, is_class_, rank_, s);
      } else {
        out(r, 0) = min_ties(d, tied, one_weights);
        for (int k = 2; k <= kmax_; ++k)
          out(r, static_cast<std::size_t>(k - 1)) =
              predict_sorted_k(dk, yk, k, is_class_, rank_, s);
      }
    }
  }

 private:
  double rx(int i, int j) const { return rpm_x_[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * n_]; }
  double xt(std::size_t i, int j) const { return Xtest_[i + static_cast<std::size_t>(j) * m_]; }

  // Range-normalised distance of training row i to query r. The 1-D path uses
  // it too, so it selects neighbours and ties exactly as the generic path does.
  double distance(int i, std::size_t r) const {
    double acc = 0.0;
    for (std::size_t a = 0; a < active_.size(); ++a) {
      const int j = active_[a];
      // Divide as R's reference does; a precomputed 1/range can flip exact ties.
      const double z = (rx(i, j) - xt(r, j)) / col_range_[a];
      if (dist_code_ == kMregDistL1) acc += std::fabs(z);
      else if (dist_code_ == kMregDistNNS) acc += std::fabs(z) + z * z;
      else acc += z * z;
    }
    return dist_code_ == kMregDistL2 ? std::sqrt(acc) : acc;
  }

  // distance() is non-decreasing in |coord - t|, so walking outward in that
  // order and extending over distance() ties at the boundary finds the k nearest.
  void topk_1d(std::size_t row, double t, int k, std::vector<int>& out_idx,
               std::vector<double>& out_d) const {
    const int nn = static_cast<int>(sc_.size());
    k = std::min(k, nn);
    int r = static_cast<int>(std::lower_bound(sc_.begin(), sc_.end(), t) - sc_.begin());
    int lo = r - 1;
    std::vector<std::pair<double, int>> win;
    while (static_cast<int>(win.size()) < k && (lo >= 0 || r < nn)) {
      const double dl = (lo >= 0) ? std::fabs(sc_[lo] - t) : kInf;
      const double dr = (r < nn) ? std::fabs(sc_[r] - t) : kInf;
      const int pos = (dl <= dr) ? lo-- : r++;
      win.emplace_back(distance(so_[pos], row), so_[pos]);
    }
    double thr = 0.0;
    for (const auto& w : win) thr = std::max(thr, w.first);
    for (double dd; lo >= 0 && (dd = distance(so_[lo], row)) <= thr; --lo)
      win.emplace_back(dd, so_[lo]);
    for (double dd; r < nn && (dd = distance(so_[r], row)) <= thr; ++r)
      win.emplace_back(dd, so_[r]);
    std::sort(win.begin(), win.end());  // (distance, original index)
    out_idx.clear();
    for (int i = 0; i < k; ++i) {
      out_idx.push_back(win[i].second);
      out_d[i] = win[i].first;
    }
  }

  double min_ties_1d(std::size_t row, double t, std::vector<double>& tied,
                     std::vector<double>& weights) const {
    const int nn = static_cast<int>(sc_.size());
    const int r = static_cast<int>(std::lower_bound(sc_.begin(), sc_.end(), t) - sc_.begin());
    const int lo = r - 1;
    const double dl = (lo >= 0) ? distance(so_[lo], row) : kInf;
    const double dr = (r < nn) ? distance(so_[r], row) : kInf;
    const double mn = std::min(dl, dr);
    tied.clear();
    for (int li = lo; li >= 0 && distance(so_[li], row) == mn; --li)
      if (std::isfinite(yhat_[so_[li]])) tied.push_back(yhat_[so_[li]]);
    for (int ri = r; ri < nn && distance(so_[ri], row) == mn; ++ri)
      if (std::isfinite(yhat_[so_[ri]])) tied.push_back(yhat_[so_[ri]]);
    if (is_class_) {
      weights.assign(tied.size(), 1.0);
      return weighted_mode(tied, weights);
    }
    return gravity(tied.data(), tied.size(), false);
  }

  void distances_for_row(std::size_t r, std::vector<double>& d) const {
    if (dist_code_ == kMregDistFactor) {
      for (int i = 0; i < n_; ++i) {
        double mismatches = 0.0;
        for (int j = 0; j < p_; ++j)
          if (rx(i, j) != xt(r, j)) mismatches += 1.0;
        d[i] = mismatches / static_cast<double>(p_);
      }
    } else if (active_.empty()) {
      std::fill(d.begin(), d.end(), 0.0);
    } else {
      for (int i = 0; i < n_; ++i) d[i] = distance(i, r);
    }
  }

  double min_ties(const std::vector<double>& d, std::vector<double>& tied,
                  std::vector<double>& weights) const {
    double dmin = d[0];
    for (int i = 1; i < n_; ++i)
      if (d[i] < dmin) dmin = d[i];
    tied.clear();
    for (int i = 0; i < n_; ++i)
      if (d[i] == dmin && std::isfinite(yhat_[i])) tied.push_back(yhat_[i]);
    if (is_class_) {
      weights.assign(tied.size(), 1.0);
      return weighted_mode(tied, weights);
    }
    return gravity(tied.data(), tied.size(), false);
  }

  const double* rpm_x_;
  const double* yhat_;
  const double* Xtest_;
  double* out_;
  int n_, p_;
  std::size_t m_;
  int kmax_, dist_code_, only_k_;
  bool is_class_;
  const RankComponents& rank_;
  std::vector<int> active_;
  std::vector<double> col_range_;
  bool is_1d_ = false;
  int axis_ = -1;
  std::vector<double> sc_;
  std::vector<int> so_;
};

void validate(std::size_t n, std::size_t p, std::size_t m, int k, int dist_code,
              const double* mins, const double* maxs, const char* kname) {
  (void)m;
  if (n == 0) throw std::invalid_argument("mreg_predict: rpm_x must have at least one row");
  if (p == 0) throw std::invalid_argument("mreg_predict: rpm_x must have at least one column");
  if (mins == nullptr || maxs == nullptr)
    throw std::invalid_argument("mreg_predict: mins/maxs must have one value per column");
  if (k < 1) throw std::invalid_argument(std::string("mreg_predict: ") + kname + " must be >= 1");
  if (dist_code < 0 || dist_code > 3)
    throw std::invalid_argument("mreg_predict: unknown NNS multivariate distance code");
}

}  // namespace

std::vector<double> mreg_predict(const double* rpm_x, std::size_t n, std::size_t p,
                                 const double* yhat, const double* Xtest, std::size_t m,
                                 int k, int dist_code, const double* mins, const double* maxs,
                                 bool is_class, int n_threads) {
  validate(n, p, m, k, dist_code, mins, maxs, "k");
  if (static_cast<std::size_t>(k) > n) k = static_cast<int>(n);
  RankComponents rank(k, k);
  std::vector<double> out(m);
  PredictKernel kernel(rpm_x, n, p, yhat, Xtest, m, k, dist_code, mins, maxs, is_class, rank,
                       out.data(), /*only_k=*/k);
  parallel_for(0, m, kernel, n_threads, n * p + static_cast<std::size_t>(k) * 16U);
  return out;
}

std::vector<double> mreg_predict_path(const double* rpm_x, std::size_t n, std::size_t p,
                                      const double* yhat, const double* Xtest, std::size_t m,
                                      int kmax, int dist_code, const double* mins,
                                      const double* maxs, bool is_class, int n_threads) {
  validate(n, p, m, kmax, dist_code, mins, maxs, "kmax");
  if (static_cast<std::size_t>(kmax) > n) kmax = static_cast<int>(n);
  RankComponents rank(2, kmax);
  std::vector<double> out(m * static_cast<std::size_t>(kmax));
  PredictKernel kernel(rpm_x, n, p, yhat, Xtest, m, kmax, dist_code, mins, maxs, is_class,
                       rank, out.data(), /*only_k=*/0);
  parallel_for(0, m, kernel, n_threads,
               n * p + static_cast<std::size_t>(kmax) * static_cast<std::size_t>(kmax) * 8U);
  return out;
}

}  // namespace nns
