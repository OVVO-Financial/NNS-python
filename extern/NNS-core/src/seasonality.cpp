// src/seasonality.cpp
//
// Implementation extracted from NNS NNS_seas.cpp (R 13.3, 82a92eb), scoring every
// component series of each lag as later R does. Decoupled from Rcpp.
//
// SPDX-License-Identifier: GPL-3.0-only
#include "nns/seasonality.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <vector>

namespace nns {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

inline bool any_na_or_inf(const double* x, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    if (!std::isfinite(x[i])) return true;
  }
  return false;
}

inline double vec_mean(const std::vector<double>& x) {
  if (x.empty()) return kNaN;
  double s = 0.0;
  for (double val : x) s += val;
  return s / static_cast<double>(x.size());
}

inline double vec_sd(const std::vector<double>& x) {
  std::size_t n = x.size();
  if (n < 2) return kNaN;
  double m = vec_mean(x);
  double ss = 0.0;
  for (double val : x) {
    double d = val - m;
    ss += d * d;
  }
  return std::sqrt(ss / static_cast<double>(n - 1));
}

} // namespace

// ---------- Core Seasonality API ----------

SeasonalityResult seasonality(const double* x, std::size_t n,
                              const int* modulo, std::size_t mod_len,
                              bool mod_only) {
                              
  if (n == 0) throw std::invalid_argument("Variable must be numeric and non-empty");
  if (any_na_or_inf(x, n)) throw std::invalid_argument("You have some missing or infinite values, please address.");

  if (n < 5) {
    return {
      {0}, {0.0}, {0.0}, // all.periods (DataFrame cols)
      0,                 // best.period
      {0}                // periods (upstream returns c(0), not empty)
    };
  }

  std::vector<double> variable(x, x + n);

  const int ni = static_cast<int>(n);
  const int half_n = ni / 2;
  const double mean_var = vec_mean(variable);
  const double sd_var = vec_sd(variable);
  // CV is meaningless for a mean that is zero up to rounding; compare sds instead
  const double tol = std::sqrt(DBL_EPSILON);
  const bool use_cv = std::fabs(mean_var) > tol * sd_var;
  const double var_cov = use_cv ? std::fabs(sd_var / mean_var) : sd_var;

  // Lag m: length-weighted mean CV of its m component series x[j], x[j + m], ...
  std::vector<int> periods_vec;
  std::vector<double> cvmean_vec;
  for (int m = 1; m <= half_n; ++m) {
    double wsum = 0.0;
    for (int j = 0; j < m; ++j) {
      int len = (ni - 1 - j) / m + 1;
      double mu = 0.0, ss = 0.0;
      for (int i = j; i < ni; i += m) mu += x[i];
      mu /= static_cast<double>(len);
      for (int i = j; i < ni; i += m) {
        double d = x[i] - mu;
        ss += d * d;
      }
      double sd = std::sqrt(ss / static_cast<double>(len - 1));
      // a constant component has CV 0; a near-zero mean caps CV at 1/tol, where the series switches to sd
      wsum += len * (use_cv ? (sd == 0.0 ? 0.0 : sd / std::max(std::fabs(mu), tol * sd)) : sd);
    }
    double cv_m = wsum / static_cast<double>(ni);
    if (cv_m < var_cov) {
      periods_vec.push_back(m);
      cvmean_vec.push_back(cv_m);
    }
  }

  std::vector<int> Period;
  std::vector<double> CoefVar;
  std::vector<double> VarCoefVar;

  if (!periods_vec.empty()) {
    int m = static_cast<int>(periods_vec.size());
    Period = periods_vec;
    CoefVar = cvmean_vec;
    VarCoefVar.assign(m, var_cov);

    std::vector<int> ord(m);
    std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(), [&](int a, int b) { return CoefVar[a] < CoefVar[b]; });

    std::vector<int> sortedP(m);
    std::vector<double> sortedCV(m);
    for (int k = 0; k < m; ++k) {
      sortedP[k] = Period[ord[k]];
      sortedCV[k] = CoefVar[ord[k]];
    }
    Period = std::move(sortedP);
    CoefVar = std::move(sortedCV);
  } else {
    Period = {1};
    CoefVar = {var_cov};
    VarCoefVar = {var_cov};
  }

  // Modulo Handling
  if (modulo != nullptr && mod_len > 0) {
    std::set<int> per_set;
    for (std::size_t i = 0; i < Period.size(); ++i) {
      for (std::size_t j = 0; j < mod_len; ++j) {
        int m_val = modulo[j];
        if (m_val <= 0) continue;
        int minus = Period[i] - (Period[i] % m_val);
        int plus  = Period[i] + (m_val - (Period[i] % m_val));
        if (minus > 0) per_set.insert(minus);
        if (plus > 0) per_set.insert(plus);
      }
    }

    if (mod_only) {
      std::set<int> curr(Period.begin(), Period.end());
      std::vector<int> keptP;
      std::vector<double> keptCV;

      for (std::size_t i = 0; i < Period.size(); ++i) {
        if (per_set.count(Period[i])) {
          keptP.push_back(Period[i]);
          keptCV.push_back(CoefVar[i]);
        }
      }
      for (int s : per_set) {
        if (!curr.count(s)) {
          keptP.push_back(s);
          keptCV.push_back(var_cov);
        }
      }

      if (keptP.empty()) {
        Period = {1};
        CoefVar = {var_cov};
        VarCoefVar = {var_cov};
      } else {
        int m = static_cast<int>(keptP.size());
        Period = keptP;
        CoefVar = keptCV;
        VarCoefVar.assign(m, var_cov);

        std::vector<int> ord(m);
        std::iota(ord.begin(), ord.end(), 0);
        std::sort(ord.begin(), ord.end(), [&](int a, int b) { return CoefVar[a] < CoefVar[b]; });

        std::vector<int> sortedP(m);
        std::vector<double> sortedCV(m);
        for (int k = 0; k < m; ++k) {
          sortedP[k] = Period[ord[k]];
          sortedCV[k] = CoefVar[ord[k]];
        }
        Period = std::move(sortedP);
        CoefVar = std::move(sortedCV);
      }
    } else {
      per_set.insert(1);
      std::set<int> curr(Period.begin(), Period.end());
      std::vector<int> add;
      for (int s : per_set) {
        if (!curr.count(s)) add.push_back(s);
      }

      if (!add.empty()) {
        for (int a : add) {
          Period.push_back(a);
          CoefVar.push_back(var_cov);
          VarCoefVar.push_back(var_cov);
        }

        int m = static_cast<int>(Period.size());
        std::vector<int> ord(m);
        std::iota(ord.begin(), ord.end(), 0);
        std::sort(ord.begin(), ord.end(), [&](int a, int b) { return CoefVar[a] < CoefVar[b]; });

        std::vector<int> sortedP(m);
        std::vector<double> sortedCV(m);
        for (int k = 0; k < m; ++k) {
          sortedP[k] = Period[ord[k]];
          sortedCV[k] = CoefVar[ord[k]];
        }
        Period = std::move(sortedP);
        CoefVar = std::move(sortedCV);
      }
    }
  }

  // Strict cap: Period < n/2
  {
    std::vector<int> P;
    std::vector<double> CV;
    std::vector<double> VCV;
    for (std::size_t i = 0; i < Period.size(); ++i) {
      if (Period[i] < static_cast<int>(n) / 2) {
        P.push_back(Period[i]);
        CV.push_back(CoefVar[i]);
        VCV.push_back(VarCoefVar[i]);
      }
    }

    if (!P.empty()) {
      int m = static_cast<int>(P.size());
      Period = P;
      CoefVar = CV;
      VarCoefVar = VCV;

      std::vector<int> ord(m);
      std::iota(ord.begin(), ord.end(), 0);
      std::sort(ord.begin(), ord.end(), [&](int a, int b) { return CoefVar[a] < CoefVar[b]; });

      std::vector<int> sortedP(m);
      std::vector<double> sortedCV(m);
      for (int k = 0; k < m; ++k) {
        sortedP[k] = Period[ord[k]];
        sortedCV[k] = CoefVar[ord[k]];
      }
      Period = std::move(sortedP);
      CoefVar = std::move(sortedCV);
    } else {
      Period = {1};
      CoefVar = {var_cov};
      VarCoefVar = {var_cov};
    }
  }

  SeasonalityResult res;
  res.all_periods = Period;
  res.all_coef_var = CoefVar;
  res.all_var_coef_var = VarCoefVar;
  res.best_period = Period.empty() ? 0 : Period[0];
  res.periods = Period;

  return res;
}

} // namespace nns