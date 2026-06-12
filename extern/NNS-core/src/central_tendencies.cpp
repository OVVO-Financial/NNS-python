// src/central_tendencies.cpp
//
// Implementation reconstructed from original_src/central_tendencies.cpp; covers NNS_gravity_cpp, NNS_rescale_cpp, and NNS_mode_cpp. Decoupled from Rcpp.
//
// SPDX-License-Identifier: GPL-3.0-only
#include "nns/central_tendencies.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace nns {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// ---------- helpers ----------

inline double frac_part(double x) {
  return x - std::floor(x);
}

inline double mean_vec(const std::vector<double>& v) {
  if (v.empty()) return kNaN;
  long double s = 0.0L;
  for (double x : v) s += x;
  return static_cast<double>(s / v.size());
}

inline double nearest_int_half_up(double x) {
  double f = std::floor(x);
  return ((x - f) < 0.5) ? f : std::ceil(x);
}

// Given a sorted vector xs, reproduce the q1, q2, q3 *exactly* as in the R code.
void quartiles_like_R_code(const std::vector<double>& xs, double& q1, double& q2, double& q3) {
  const int l = static_cast<int>(xs.size());
  const double l25 = l * 0.25;
  const double l50 = l * 0.50;
  const double l75 = l * 0.75;
  
  if (l % 2 == 0) {
    int i25 = std::max(1, static_cast<int>(std::floor(l25))) - 1;
    int i50 = std::max(1, static_cast<int>(std::floor(l50))) - 1;
    int i75 = std::max(1, static_cast<int>(std::floor(l75))) - 1;
    q1 = xs[i25];
    q2 = xs[i50];
    q3 = xs[i75];
  } else {
    int f25 = static_cast<int>(std::floor(l25));
    int c25 = static_cast<int>(std::ceil(l25));
    f25 = std::min(std::max(1, f25), l);
    c25 = std::min(std::max(1, c25), l);
    double w25 = frac_part(l25);
    q1 = xs[f25 - 1] + w25 * (xs[c25 - 1] - xs[f25 - 1]);
    
    int f50 = static_cast<int>(std::floor(l50));
    int c50 = static_cast<int>(std::ceil(l50));
    f50 = std::min(std::max(1, f50), l);
    c50 = std::min(std::max(1, c50), l);
    q2 = 0.5 * (xs[f50 - 1] + xs[c50 - 1]);
    
    int f75 = static_cast<int>(std::floor(l75));
    int c75 = static_cast<int>(std::ceil(l75));
    f75 = std::min(std::max(1, f75), l);
    c75 = std::min(std::max(1, c75), l);
    double w75 = frac_part(l75);
    q3 = xs[f75 - 1] + w75 * (xs[c75 - 1] - xs[f75 - 1]);
  }
}

// Minimal replacement for NNS_bin used by mode/gravity
void simple_bin_counts(const std::vector<double>& xs, double width, double origin,
                       std::vector<double>& bin_names, std::vector<int>& counts) {
  const int l = static_cast<int>(xs.size());
  if (l == 0) { bin_names.clear(); counts.clear(); return; }
  
  const double xmax = xs.back();
  int nbins = static_cast<int>(std::floor((xmax - origin) / width + 1e-12)) + 1;
  if (nbins < 1) nbins = 1;
  
  bin_names.resize(nbins);
  for (int k = 0; k < nbins; ++k) bin_names[k] = origin + k * width;
  
  counts.assign(nbins, 0);
  for (double v : xs) {
    int idx = static_cast<int>(std::floor((v - origin) / width));
    if (idx < 0) idx = 0;
    if (idx >= nbins) idx = nbins - 1;
    counts[idx] += 1;
  }
}

// Triangular smoothing helper: 7-tap [1,2,3,4,3,2,1] with mirrored edges
void smooth_counts_tri7(const std::vector<int>& counts, std::vector<double>& smooth) {
  static const int w[7] = {1, 2, 3, 4, 3, 2, 1};
  static const int Wsum = 16;
  const int n = static_cast<int>(counts.size());
  smooth.assign(n, 0.0);
  if (n == 0) return;
  
  auto at = [&](int idx) -> int {
    if (idx < 0) return counts[-idx];
    if (idx >= n) return counts[2 * n - 2 - idx];
    return counts[idx];
  };
  
  for (int i = 0; i < n; ++i) {
    int acc = 0;
    acc += w[0] * at(i - 3); acc += w[1] * at(i - 2); acc += w[2] * at(i - 1);
    acc += w[3] * at(i);
    acc += w[4] * at(i + 1); acc += w[5] * at(i + 2); acc += w[6] * at(i + 3);
    smooth[i] = static_cast<double>(acc) / static_cast<double>(Wsum);
  }
}

} // namespace

// ---------- NNS.gravity ----------

double gravity(const double* x_in, std::size_t n, bool discrete) {
  std::vector<double> x;
  x.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    if (std::isfinite(x_in[i])) x.push_back(x_in[i]);
  }
  
  const int l = static_cast<int>(x.size());
  if (l == 0) return kNaN;
  if (l <= 3) {
    std::vector<double> t = x;
    std::sort(t.begin(), t.end());
    double med = (l % 2) ? t[l / 2] : 0.5 * (t[l / 2 - 1] + t[l / 2]);
    if (discrete) return nearest_int_half_up(med);
    return med;
  }
  
  bool all_eq = true;
  for (int i = 1; i < l; ++i) {
    if (x[i] != x[0]) { all_eq = false; break; }
  }
  if (all_eq) return x[0];
  
  std::sort(x.begin(), x.end());
  double range = std::fabs(x.back() - x.front());
  if (range == 0.0) return x.front();
  
  double q1, q2, q3;
  quartiles_like_R_code(x, q1, q2, q3);
  
  double width = (q3 - q1) * std::pow(static_cast<double>(l), -0.5);
  if (!(width > 0.0) || !std::isfinite(width)) width = range / 128.0;
  
  std::vector<double> z_names;
  std::vector<int> counts;
  simple_bin_counts(x, width, x.front(), z_names, counts);
  const int lz = static_cast<int>(counts.size());
  
  int maxc = 0;
  for (int c : counts) if (c > maxc) maxc = c;
  int ties = 0;
  for (int c : counts) if (c == maxc) ++ties;
  
  int lo = 0, hi = lz - 1;
  if (ties == 1) {
    int zc = 0;
    for (int i = 0; i < lz; ++i) {
      if (counts[i] == maxc) { zc = i; break; }
    }
    lo = std::max(0, zc - 1);
    hi = std::min(lz - 1, zc + 1);
  }
  
  long double num = 0.0L, den = 0.0L;
  for (int i = lo; i <= hi; ++i) {
    num += static_cast<long double>(z_names[i]) * static_cast<long double>(counts[i]);
    den += static_cast<long double>(counts[i]);
  }
  double m = (den > 0.0L) ? static_cast<double>(num / den) : z_names[(lo + hi) / 2];
  
  double mu = mean_vec(x);
  double mid = 0.25 * (q2 + m + mu + 0.5 * (q1 + q3));
  
  double out = std::isfinite(mid) ? mid : q2;
  if (discrete) out = nearest_int_half_up(out);
  return out;
}

// ---------- NNS.rescale ----------

std::vector<double> rescale(const double* x_in, std::size_t n, double a, double b,
                            const std::string& method,
                            std::optional<double> T,
                            const std::string& type) {
  std::vector<double> out(n, kNaN);
  
  std::string method_lower = method;
  std::transform(method_lower.begin(), method_lower.end(), method_lower.begin(),
                 [](unsigned char c){ return std::tolower(c); });
                 
  std::string type_lower = type;
  std::transform(type_lower.begin(), type_lower.end(), type_lower.begin(),
                 [](unsigned char c){ return std::tolower(c); });
  
  if (method_lower == "minmax") {
    double xmin = std::numeric_limits<double>::infinity();
    double xmax = -std::numeric_limits<double>::infinity();
    
    for (std::size_t i = 0; i < n; ++i) {
      if (std::isfinite(x_in[i])) {
        if (x_in[i] < xmin) xmin = x_in[i];
        if (x_in[i] > xmax) xmax = x_in[i];
      }
    }
    
    // Fallback if all values identical or no valid values
    if (!std::isfinite(xmin) || !std::isfinite(xmax) || xmax == xmin) {
      for (std::size_t i = 0; i < n; ++i) out[i] = (a + b) / 2.0;
      return out;
    }
    
    for (std::size_t i = 0; i < n; ++i) {
      out[i] = a + (b - a) * ((x_in[i] - xmin) / (xmax - xmin));
    }
    return out;
  }
  
  if (method_lower == "riskneutral") {
    if (!T.has_value()) {
      throw std::invalid_argument("T (time to maturity) must be provided for riskneutral method");
    }
    double T_val = T.value();
    
    if (!(a > 0.0)) {
      throw std::invalid_argument("S_0 (a) must be positive for riskneutral method");
    }
    
    double S0 = a;
    double r = b;
    
    long double s = 0.0L; 
    int cnt = 0;
    for (std::size_t i = 0; i < n; ++i) {
      if (std::isfinite(x_in[i])) { s += x_in[i]; ++cnt; }
    }
    double mx = (cnt > 0) ? static_cast<double>(s / cnt) : kNaN;
    
    if (!std::isfinite(mx) || mx <= 0.0) {
      throw std::invalid_argument("Mean(x) must be positive/finite for riskneutral scaling");
    }
    
    double target = (type_lower == "discounted") ? S0 : (S0 * std::exp(r * T_val));
    double theta = std::log(target / mx);
    
    for (std::size_t i = 0; i < n; ++i) {
      out[i] = x_in[i] * std::exp(theta);
    }
    return out;
  }
  
  throw std::invalid_argument("Invalid method: use 'minmax' or 'riskneutral'");
}

// ---------- NNS.mode ----------

std::vector<double> mode(const double* x_in, std::size_t n, bool discrete, bool multi) {
  std::vector<double> xnum; 
  xnum.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    if (std::isfinite(x_in[i])) xnum.push_back(x_in[i]);
  }
  
  const int l = static_cast<int>(xnum.size());
  if (l == 0) return {kNaN};
  
  // ====================== DISCRETE PATH ======================
  if (discrete) {
    if (l <= 3) {
      std::vector<double> tmp = xnum; 
      std::sort(tmp.begin(), tmp.end());
      double med = (l % 2 == 1) ? tmp[l / 2] : 0.5 * (tmp[l / 2 - 1] + tmp[l / 2]);
      return {nearest_int_half_up(med)};
    }
    
    std::unordered_map<int, int> freq; 
    freq.reserve(l * 2u);
    for (double v : xnum) ++freq[static_cast<int>(nearest_int_half_up(v))];
    
    int maxf = 0; 
    for (const auto& kv : freq) if (kv.second > maxf) maxf = kv.second;
    
    std::vector<int> modes_int;
    for (const auto& kv : freq) if (kv.second == maxf) modes_int.push_back(kv.first);
    std::sort(modes_int.begin(), modes_int.end());
    
    if (multi) {
      std::vector<double> out(modes_int.size());
      for (std::size_t i = 0; i < modes_int.size(); ++i) out[i] = static_cast<double>(modes_int[i]);
      return out;
    } else {
      long double sum = 0.0L;
      for (int m : modes_int) sum += static_cast<long double>(m);
      double mean_modes = modes_int.empty() ? kNaN : static_cast<double>(sum / static_cast<long double>(modes_int.size()));
      return {mean_modes};
    }
  }
  
  // ====================== CONTINUOUS PATH ======================
  if (l <= 3) {
    std::vector<double> tmp = xnum; 
    std::sort(tmp.begin(), tmp.end());
    double med = (l % 2 == 1) ? tmp[l / 2] : 0.5 * (tmp[l / 2 - 1] + tmp[l / 2]);
    return {med};
  }
  
  bool all_eq = true;
  for (int i = 1; i < l; ++i) {
    if (xnum[i] != xnum[0]) { all_eq = false; break; }
  }
  if (all_eq) return {xnum[0]};
  
  std::sort(xnum.begin(), xnum.end());
  double range = std::fabs(xnum.back() - xnum.front());
  if (range == 0.0) return {xnum.front()};
  
  double q1, q2, q3;
  quartiles_like_R_code(xnum, q1, q2, q3);
  double width = (q3 - q1) * std::pow(static_cast<double>(l), -0.5);
  if (!(width > 0.0) || !std::isfinite(width)) width = range / 128.0;
  
  std::vector<double> z_names;
  std::vector<int> counts;
  if (width <= 0.0 || !std::isfinite(width)) width = range / 128.0;
  simple_bin_counts(xnum, width, xnum.front(), z_names, counts);
  
  const int lz = static_cast<int>(counts.size());
  if (lz == 0) return {kNaN};
  
  int maxc = 0; 
  for (int c : counts) if (c > maxc) maxc = c;
  
  std::vector<double> cs; 
  smooth_counts_tri7(counts, cs);
  
  const double MARGIN = 0.0;
  std::vector<int> peak_idx; 
  peak_idx.reserve(lz);
  
  for (int i = 3; i <= lz - 4; ++i) {
    double ci = cs[i];
    if (ci <= 0.0) continue;
    
    double Ls = std::max(std::max(cs[i - 1], cs[i - 2]), cs[i - 3]);
    double Rs = std::max(std::max(cs[i + 1], cs[i + 2]), cs[i + 3]);
    if (!(ci > Ls + MARGIN && ci > Rs + MARGIN)) continue;
    
    double curv = cs[i - 1] - 2.0 * cs[i] + cs[i + 1];
    if (!(curv < 0.0)) continue;
    
    peak_idx.push_back(i);
  }
  
  if (!peak_idx.empty()) {
    std::sort(peak_idx.begin(), peak_idx.end(), [&](int a, int b){ return cs[a] > cs[b]; });
    std::vector<int> kept;
    for (int idx : peak_idx) {
      bool too_close = false;
      for (int jdx : kept) if (std::abs(idx - jdx) <= 3) { too_close = true; break; }
      if (!too_close) kept.push_back(idx);
    }
    
    if (!kept.empty()) {
      std::vector<double> centers(kept.size());
      for (std::size_t t = 0; t < kept.size(); ++t) {
        int zc = kept[t];
        int lo = std::max(0, zc - 3);
        int hi = std::min(lz - 1, zc + 3);
        long double num = 0.0L, den = 0.0L;
        for (int j = lo; j <= hi; ++j) {
          if (std::abs(j - zc) <= 3) {
            num += static_cast<long double>(z_names[j]) * static_cast<long double>(counts[j]);
            den += static_cast<long double>(counts[j]);
          }
        }
        centers[t] = (den > 0.0L) ? static_cast<double>(num / den) : z_names[zc];
      }
      
      if (multi) {
        std::vector<double> out = centers;
        std::sort(out.begin(), out.end());
        return out;
      } else {
        int best_t = 0;
        for (std::size_t t = 1; t < kept.size(); ++t) {
          if (cs[kept[t]] > cs[kept[best_t]]) best_t = t;
        }
        return {centers[best_t]};
      }
    }
  }
  
  int ties = 0; 
  for (int c : counts) if (c == maxc) ++ties;
  
  if (ties > 1) {
    if (multi) {
      std::vector<double> out;
      out.reserve(ties);
      for (int i = 0; i < lz; ++i) if (counts[i] == maxc) out.push_back(z_names[i]);
      std::sort(out.begin(), out.end());
      return out;
    } else {
      long double sum = 0.0L;
      int pos = 0;
      for (int i = 0; i < lz; ++i) {
        if (counts[i] == maxc) { sum += static_cast<long double>(z_names[i]); ++pos; }
      }
      double mean_modes = (pos > 0) ? static_cast<double>(sum / static_cast<long double>(pos)) : kNaN;
      return {mean_modes};
    }
  }
  
  int zc = 0; 
  for (int i = 0; i < lz; ++i) if (counts[i] == maxc) { zc = i; break; }
  
  int lo = std::max(0, zc - 1);
  int hi = std::min(lz - 1, zc + 1);
  long double num = 0.0L, den = 0.0L;
  for (int j = lo; j <= hi; ++j) {
    num += static_cast<long double>(z_names[j]) * static_cast<long double>(counts[j]);
    den += static_cast<long double>(counts[j]);
  }
  
  double finalv = (den > 0.0L) ? static_cast<double>(num / den) : z_names[zc];
  return {finalv};
}

} // namespace nns