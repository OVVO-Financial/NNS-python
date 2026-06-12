// src/seasonality.cpp
//
// Implementation extracted from NNS 13.0 NNS_seas.cpp. Decoupled from Rcpp.
//
// SPDX-License-Identifier: GPL-3.0-only
#include "nns/seasonality.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <vector>

namespace nns {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

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

// lag-1 Pearson autocorrelation
inline double acf1(const std::vector<double>& x) {
  std::size_t n = x.size();
  if (n < 2) return kNaN;
  double m = vec_mean(x);
  double num = 0.0;
  double den = 0.0;
  for (std::size_t t = 1; t < n; ++t) num += (x[t] - m) * (x[t - 1] - m);
  for (std::size_t t = 0; t < n; ++t) { 
    double d = x[t] - m; 
    den += d * d; 
  }
  if (den == 0.0) return kNaN;
  return num / den;
}

inline double cv_or_fallback(const std::vector<double>& x, bool use_cv, double var_cov) {
  std::size_t n = x.size();
  if (n < 2) return var_cov;
  double z;
  if (use_cv) {
    double m = vec_mean(x);
    double s = vec_sd(x);
    z = std::fabs(s / m);
  } else {
    double a1 = acf1(x);
    if (!std::isfinite(a1)) return var_cov;
    z = std::pow(std::fabs(a1), -1.0);
  }
  if (!std::isfinite(z)) return var_cov;
  return z;
}

// 0-based indices stepping backwards
inline std::vector<int> rev_step_indices(int n, int step) {
  int len = (n - 1) / step + 1;
  std::vector<int> out(len);
  int v = n - 1; // 0-based max index
  for (int k = 0; k < len; ++k, v -= step) out[k] = v;
  return out;
}

inline std::vector<double> take_by_index(const std::vector<double>& x, const std::vector<int>& idx) {
  std::size_t m = idx.size();
  std::vector<double> out(m);
  for (std::size_t i = 0; i < m; ++i) {
    int j = idx[i];
    out[i] = (j >= 0 && j < static_cast<int>(x.size())) ? x[j] : kNaN;
  }
  return out;
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
  std::vector<double> variable_1(x, x + n - 1);
  std::vector<double> variable_2;
  if (n - 1 >= 2) variable_2.assign(x, x + n - 2);

  const int half_n = static_cast<int>(n) / 2;
  const double mean_var = vec_mean(variable);
  const bool use_cv = (mean_var != 0.0);
  
  double var_cov = use_cv ? std::fabs(vec_sd(variable) / mean_var) : std::pow(std::fabs(acf1(variable)), -1.0);
  if (!std::isfinite(var_cov)) var_cov = kInf;

  std::vector<double> out(half_n), out1(half_n), out2(half_n);
  std::vector<int> inst(half_n, 0), inst1(half_n, 0), inst2(half_n, 0);

  const int n1 = static_cast<int>(n) - 1;
  const int n2 = static_cast<int>(variable_2.size());

  for (int i = 1; i <= half_n; ++i) {
    std::vector<int> idx  = rev_step_indices(static_cast<int>(n), i);
    std::vector<int> idx1 = rev_step_indices(n1, i);
    std::vector<int> idx2 = (n2 > 0) ? rev_step_indices(n2, i) : std::vector<int>();

    double t  = cv_or_fallback(take_by_index(variable, idx), use_cv, var_cov);
    double t1 = cv_or_fallback(take_by_index(variable_1, idx1), use_cv, var_cov);
    double t2 = cv_or_fallback(take_by_index(variable_2, idx2), use_cv, var_cov);

    if (t  <= var_cov) { inst[i - 1]  = i; out[i - 1]  = t;  }
    if (t1 <= var_cov) { inst1[i - 1] = i; out1[i - 1] = t1; }
    if (t2 <= var_cov) { inst2[i - 1] = i; out2[i - 1] = t2; }
  }

  std::vector<int> periods_vec;
  std::vector<double> cvmean_vec;
  for (int i = 0; i < half_n; ++i) {
    if (inst[i] > 0 && inst1[i] > 0 && inst2[i] > 0) {
      periods_vec.push_back(inst[i]);
      cvmean_vec.push_back((out[i] + out1[i] + out2[i]) / 3.0);
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