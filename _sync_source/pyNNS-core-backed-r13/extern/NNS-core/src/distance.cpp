// src/distance.cpp
//
// Implementation extracted from NNS 13.0 NNS_distance.cpp. Decoupled from Rcpp.
//
// SPDX-License-Identifier: GPL-3.0-only
#include "nns/distance.hpp"
#include "nns/parallel.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nns {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kEps = 1e-12;
constexpr double M_SQRT2PI = 2.5066282746310005024; // sqrt(2 * pi)

inline double safe_eps() { return kEps; }

// Column-major indexing helper
inline double at(const double* M, std::size_t rows, std::size_t r, std::size_t c) {
  return M[c * rows + r];
}

inline void set_at(std::vector<double>& M, std::size_t rows, std::size_t r, std::size_t c, double val) {
  M[c * rows + r] = val;
}

// --- Math & Statistical Helpers ---

inline double mean_vec(const std::vector<double>& v) {
  if (v.empty()) return kNaN;
  double s = 0.0;
  for (double x : v) s += x;
  return s / static_cast<double>(v.size());
}

inline double sd_vec(const std::vector<double>& v) {
  std::size_t n = v.size(); 
  if (n < 2) return kNaN;
  double mu = mean_vec(v), acc = 0.0;
  for (double x : v) { double d = x - mu; acc += d * d; }
  return std::sqrt(acc / static_cast<double>(n - 1));
}

inline double var_vec(const std::vector<double>& v) {
  double s = sd_vec(v); 
  return std::isfinite(s) ? s * s : kNaN;
}

// --- Pure C++ Probability Density Functions (Replaces RMath C-API) ---

// Mirrors the R C-API ::Rf_dexp(x, scale, 0): R's C-level dexp is
// SCALE-parameterized, density = exp(-x/scale)/scale.  All call sites pass
// 1.0/k exactly as the original Rf_dexp(r, 1.0/k, 0), which therefore
// evaluates to k * exp(-r * k).
inline double pdf_exp(double x, double scale) {
  return std::exp(-x / scale) / scale;
}

inline double pdf_t_prop(double x, double df) {
  return std::pow(1.0 + (x * x) / df, -(df + 1.0) / 2.0);
}

inline double pdf_norm_prop(double x, double mu, double sigma) {
  double z = (x - mu) / sigma;
  return std::exp(-0.5 * z * z);
}

inline double pdf_lnorm_log(double x, double meanlog, double sdlog) {
  if (x <= 0.0) return -std::numeric_limits<double>::infinity();
  return -std::log(x * sdlog * M_SQRT2PI) - 0.5 * std::pow((std::log(x) - meanlog) / sdlog, 2.0);
}

// --- Class Weighting ---

double mode_class_weighted(const std::vector<double>& y, const std::vector<double>& w) {
  int n = static_cast<int>(y.size());
  if (n == 0) return kNaN;
  if (n == 1) return y[0];
  
  std::vector<std::pair<double, long long>> items;
  items.reserve(n);
  for (int i = 0; i < n; ++i) {
    long long c = static_cast<long long>(std::ceil(100.0 * w[i]));
    if (c > 0) items.push_back({y[i], c});
  }
  if (items.empty()) return kNaN;
  
  std::sort(items.begin(), items.end(), [](const std::pair<double, long long>& a, const std::pair<double, long long>& b) {
    return a.first < b.first;
  });
  
  double best_val = items[0].first;
  long long best_cnt = items[0].second;
  double cur_val = items[0].first;
  long long cur_cnt = items[0].second;
  
  for (std::size_t i = 1; i < items.size(); ++i) {
    if (items[i].first == cur_val) {
      cur_cnt += items[i].second;
    } else {
      if (cur_cnt > best_cnt) { best_cnt = cur_cnt; best_val = cur_val; }
      cur_val = items[i].first;
      cur_cnt = items[i].second;
    }
  }
  if (cur_cnt > best_cnt) { best_val = cur_val; }
  return best_val;
}

inline void compute_distances(const double* rpm, int n, int p,
                              const std::vector<double>& test_row,
                              std::vector<double>& dist_out) {
  for (int i = 0; i < n; ++i) {
    double acc = 0.0;
    for (int j = 0; j < p; ++j) {
      const double d = at(rpm, n, i, j) - test_row[j];
      acc += d * d + std::fabs(d);
    }
    dist_out[i] = (acc == 0.0 ? safe_eps() : acc);
  }
}

inline void argsort_by_distance(const std::vector<double>& dist, std::vector<int>& idx) {
  const int n = static_cast<int>(dist.size());
  idx.resize(n);
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&dist](int a, int b){ return dist[a] < dist[b]; });
}

} // namespace

// ---------- Core Distance API ----------

double distance(const double* X, std::size_t l, std::size_t n,
                const double* yhat, const double* dest,
                int k, bool use_class) {
  if (l == 0 || n == 0) throw std::invalid_argument("Empty matrix");
  
  std::vector<double> invR(n, 0.0);
  for (std::size_t j = 0; j < n; ++j) {
    double cmin = dest[j], cmax = dest[j];
    for (std::size_t i = 0; i < l; ++i) {
      double v = at(X, l, i, j);
      if (std::isfinite(v)) { if (v < cmin) cmin = v; if (v > cmax) cmax = v; }
    }
    double range = cmax - cmin;
    if (std::isfinite(range) && range > 0.0) invR[j] = 1.0 / range;
  }
  
  std::vector<double> S(l, 0.0);
  for (std::size_t i = 0; i < l; ++i) {
    double acc = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
      double a = at(X, l, i, j), b = dest[j];
      if (std::isfinite(a) && std::isfinite(b) && invR[j] > 0.0) {
        double diff = (a - b) * invR[j];
        acc += diff * diff + std::fabs(diff);
      }
    }
    S[i] = (acc == 0.0 ? 1e-10 : acc);
  }
  
  int ll = std::min(k, static_cast<int>(l));
  std::vector<int> idx(l);
  std::iota(idx.begin(), idx.end(), 0);
  auto cmp = [&](int a, int b){ return S[a] < S[b]; };
  if (ll < static_cast<int>(l)) std::partial_sort(idx.begin(), idx.begin()+ll, idx.end(), cmp);
  else std::sort(idx.begin(), idx.end(), cmp);
  
  idx.resize(ll);
  std::vector<double> Ssel(ll), ysel(ll);
  for (int t = 0; t < ll; ++t) {
    int i = idx[t];
    Ssel[t] = S[i];
    ysel[t] = yhat[i];
  }
  
  if (ll == 1) return ysel[0];
  if (k == 1) {
    double smin = *std::min_element(Ssel.begin(), Ssel.end());
    std::vector<double> yties;
    for (int t = 0; t < ll; ++t) if (Ssel[t] == smin) yties.push_back(ysel[t]);
    if (yties.size() == 1) return yties[0];
    std::vector<double> fake_w(yties.size(), 1.0);
    return mode_class_weighted(yties, fake_w);
  }
  
  std::vector<double> uni(ll, 1.0 / static_cast<double>(ll));
  
  std::vector<double> tw(ll, 0.0);
  for (int i = 0; i < ll; ++i) {
    double dens = pdf_t_prop(Ssel[i], static_cast<double>(ll));
    tw[i] = std::isfinite(dens) ? dens : 0.0;
  }
  double twsum = std::accumulate(tw.begin(), tw.end(), 0.0);
  if (twsum > 0) for (double &v: tw) v /= twsum; else std::fill(tw.begin(), tw.end(), 0.0);
  
  std::vector<double> emp(ll, 0.0);
  for (int i = 0; i < ll; ++i){ double v = Ssel[i]; emp[i] = (v>0) ? 1.0/v : 0.0; }
  double empsum = std::accumulate(emp.begin(), emp.end(), 0.0);
  if (empsum > 0) for (double &v: emp) v /= empsum; else std::fill(emp.begin(), emp.end(), 0.0);
  
  std::vector<double> exw(ll, 0.0);
  for (int i = 0; i < ll; ++i){
    double dens = pdf_exp(static_cast<double>(i+1), 1.0/static_cast<double>(ll));
    exw[i] = std::isfinite(dens) ? dens : 0.0;
  }
  double exsum = std::accumulate(exw.begin(), exw.end(), 0.0);
  if (exsum > 0) for (double &v: exw) v /= exsum; else std::fill(exw.begin(), exw.end(), 0.0);
  
  std::vector<double> lnorm(ll, 0.0);
  double sd_ranks = kNaN;
  if (ll >= 2){
    std::vector<double> ranks(ll); for(int i=0; i<ll; ++i) ranks[i] = static_cast<double>(i+1);
    sd_ranks = sd_vec(ranks);
  }
  if (std::isfinite(sd_ranks)){
    for (int i = 0; i < ll; ++i){
      double lp = pdf_lnorm_log(static_cast<double>(i+1), 0.0, sd_ranks);
      lnorm[i] = std::fabs(lp);
    }
    std::reverse(lnorm.begin(), lnorm.end());
  } else {
    std::fill(lnorm.begin(), lnorm.end(), 0.0);
  }
  double lnsum = std::accumulate(lnorm.begin(), lnorm.end(), 0.0);
  if (lnsum > 0) for (double &v: lnorm) v /= lnsum; else std::fill(lnorm.begin(), lnorm.end(), 0.0);
  
  std::vector<double> pl(ll, 0.0);
  for (int i = 0; i < ll; ++i){ double r = static_cast<double>(i+1); pl[i] = std::pow(r, -2.0); }
  double plsum = std::accumulate(pl.begin(), pl.end(), 0.0);
  if (plsum > 0) for (double &v: pl) v /= plsum; else std::fill(pl.begin(), pl.end(), 0.0);
  
  std::vector<double> normw(ll, 0.0);
  double sdS = sd_vec(Ssel);
  if (std::isfinite(sdS) && sdS > 0){
    for (int i = 0; i < ll; ++i){
      double dens = pdf_norm_prop(Ssel[i], 0.0, sdS);
      normw[i] = std::isfinite(dens) ? dens : 0.0;
    }
    double nsum = std::accumulate(normw.begin(), normw.end(), 0.0);
    if (nsum > 0) for (double &v: normw) v /= nsum; else std::fill(normw.begin(), normw.end(), 0.0);
  }
  
  std::vector<double> rbf(ll, 0.0);
  double varS = var_vec(Ssel);
  if (std::isfinite(varS) && varS > 0){
    for (int i = 0; i < ll; ++i) rbf[i] = std::exp(- Ssel[i] / (2.0*varS));
    double rsum = std::accumulate(rbf.begin(), rbf.end(), 0.0);
    if (rsum > 0) for (double &v: rbf) v /= rsum; else std::fill(rbf.begin(), rbf.end(), 0.0);
  }
  
  std::vector<double> w(ll, 0.0);
  double tot = 0.0;
  for (int i = 0; i < ll; ++i){
    double wi = uni[i] + tw[i] + emp[i] + exw[i] + lnorm[i] + pl[i] + normw[i] + rbf[i];
    w[i] = wi; tot += wi;
  }
  if (tot > 0) for (double &v: w) v /= tot; else for (double &v: w) v = 1.0/static_cast<double>(ll);
  
  if (!use_class){
    double dot = 0.0;
    for (int i = 0; i < ll; ++i) dot += ysel[i] * w[i];
    return dot;
  } else {
    return mode_class_weighted(ysel, w);
  }
}

// ---------- Distance Path (Sequential) ----------

std::vector<double> distance_path(const double* RPM, std::size_t n, std::size_t p,
                                  const double* yhat, const double* Xtest, std::size_t m,
                                  int kmax, bool is_class) {
  if (kmax > static_cast<int>(n)) kmax = static_cast<int>(n);
  
  std::vector<double> out(m * kmax, 0.0);
  std::vector<double> dist(n), y_sorted(n), d_sorted(n), tr(p);
  std::vector<int> ord(n);
  
  for (std::size_t r = 0; r < m; ++r) {
    for (std::size_t j = 0; j < p; ++j) tr[j] = at(Xtest, m, r, j);
    
    compute_distances(RPM, n, p, tr, dist);
    argsort_by_distance(dist, ord);
    
    for (std::size_t i = 0; i < n; ++i) {
      const int j = ord[i];
      y_sorted[i] = yhat[j];
      d_sorted[i] = (dist[j] <= 0.0 ? safe_eps() : dist[j]);
    }
    
    double csum_w = 0.0, csum_yw = 0.0;
    for (int k = 1; k <= kmax; ++k) {
      const double w = 1.0 / d_sorted[k - 1];
      csum_w  += w;
      csum_yw += w * y_sorted[k - 1];
      double val = (csum_w > 0.0) ? (csum_yw / csum_w) : 0.0;
      set_at(out, m, r, k - 1, val);
    }
  }
  return out;
}

// ---------- Distance Bulk (Sequential) ----------

std::vector<double> distance_bulk(const double* RPM, std::size_t n, std::size_t p,
                                  const double* yhat, const double* Xtest, std::size_t m,
                                  int k, bool is_class) {
  if (k > static_cast<int>(n)) k = static_cast<int>(n);
  
  std::vector<double> out(m, 0.0);
  std::vector<double> dist(n), tr(p);
  std::vector<int> ord(n);
  
  for (std::size_t r = 0; r < m; ++r) {
    for (std::size_t j = 0; j < p; ++j) tr[j] = at(Xtest, m, r, j);
    
    compute_distances(RPM, n, p, tr, dist);
    argsort_by_distance(dist, ord);
    
    double csum_w = 0.0, csum_yw = 0.0;
    for (int i = 0; i < k; ++i) {
      const int j = ord[i];
      const double dj = (dist[j] <= 0.0 ? safe_eps() : dist[j]);
      const double w  = 1.0 / dj;
      csum_w  += w;
      csum_yw += w * yhat[j];
    }
    out[r] = (csum_w > 0.0) ? (csum_yw / csum_w) : 0.0;
  }
  return out;
}

// ---------- Parallel Path ----------

std::vector<double> distance_path_parallel(const double* RPM, std::size_t l, std::size_t n,
                                           const double* yhat, const double* Xtest, std::size_t m,
                                           int kmax, bool is_class, int nthreads) {
  if (kmax <= 0) kmax = static_cast<int>(l);
  if (kmax > static_cast<int>(l)) kmax = static_cast<int>(l);
  
  std::vector<double> minRPM(n, std::numeric_limits<double>::infinity());
  std::vector<double> maxRPM(n, -std::numeric_limits<double>::infinity());
  for (std::size_t j = 0; j < n; ++j){
    for (std::size_t i = 0; i < l; ++i){
      double v = at(RPM, l, i, j);
      if (std::isfinite(v)) { if(v < minRPM[j]) minRPM[j] = v; if(v > maxRPM[j]) maxRPM[j] = v; }
    }
    if (!std::isfinite(minRPM[j])) { minRPM[j] = 0.0; maxRPM[j] = 0.0; }
  }
  
  std::vector<std::vector<double>> uniW(kmax+1), expW(kmax+1), lnormW(kmax+1), plW(kmax+1);
  for (int k = 1; k <= kmax; ++k){
    uniW[k].assign(k, 1.0 / static_cast<double>(k));
    
    std::vector<double> ex(k);
    for (int r = 1; r <= k; ++r) ex[r-1] = pdf_exp(static_cast<double>(r), 1.0 / static_cast<double>(k));
    double exs = std::accumulate(ex.begin(), ex.end(), 0.0);
    if (exs > 0) for (double &v: ex) v /= exs; else std::fill(ex.begin(), ex.end(), 0.0);
    expW[k] = std::move(ex);
    
    std::vector<double> pl(k);
    for (int r = 1; r <= k; ++r) pl[r-1] = std::pow(static_cast<double>(r), -2.0);
    double pls = std::accumulate(pl.begin(), pl.end(), 0.0);
    if (pls > 0) for (double &v: pl) v /= pls; else std::fill(pl.begin(), pl.end(), 0.0);
    plW[k] = std::move(pl);
    
    std::vector<double> ln(k, 0.0);
    if (k >= 2){
      double sdlog = std::sqrt((static_cast<double>(k) * static_cast<double>(k) - 1.0) / 12.0);
      for (int r = 1; r <= k; ++r){ 
        double lp = pdf_lnorm_log(static_cast<double>(r), 0.0, sdlog);
        ln[r-1] = std::fabs(lp); 
      }
      std::reverse(ln.begin(), ln.end());
      double lns = std::accumulate(ln.begin(), ln.end(), 0.0);
      if (lns > 0) for (double &v: ln) v /= lns; else std::fill(ln.begin(), ln.end(), 0.0);
    }
    lnormW[k] = std::move(ln);
  }
  
  std::vector<double> out(m * kmax, 0.0);
  
  parallel_for(0, m, [&](std::size_t begin, std::size_t end) {
    std::vector<double> invR(n), S(l), topS, topY;
    std::vector<int> idx(l);
    
    for (std::size_t r = begin; r < end; ++r) {
      for (std::size_t j = 0; j < n; ++j){
        double t = at(Xtest, m, r, j);
        double mn = std::min(minRPM[j], t);
        double mx = std::max(maxRPM[j], t);
        double range = mx - mn;
        invR[j] = (std::isfinite(range) && range > 0.0) ? (1.0 / range) : 0.0;
      }
      
      for (std::size_t i = 0; i < l; ++i){
        double acc = 0.0; 
        for (std::size_t j = 0; j < n; ++j){
          double a = at(RPM, l, i, j), b = at(Xtest, m, r, j);
          if (std::isfinite(a) && std::isfinite(b) && invR[j] > 0.0){
            double diff = (a - b) * invR[j];
            acc += diff * diff + std::fabs(diff);
          }
        }
        S[i] = (acc == 0.0 ? 1e-10 : acc);
      }
      
      std::iota(idx.begin(), idx.end(), 0);
      auto cmp = [&](int a, int b){ return S[a] < S[b]; };
      if (kmax < static_cast<int>(l)) std::partial_sort(idx.begin(), idx.begin()+kmax, idx.end(), cmp);
      else std::sort(idx.begin(), idx.end(), cmp);
      
      auto cmp2 = [&](int a, int b){
        if (S[a] < S[b]) return true;
        if (S[b] < S[a]) return false;
        return a < b;
      };
      std::stable_sort(idx.begin(), idx.begin()+kmax, cmp2);
      
      topS.resize(kmax); topY.resize(kmax);
      for (int t = 0; t < kmax; ++t){ int i = idx[t]; topS[t] = S[i]; topY[t] = yhat[i]; }
      
      for (int k = 1; k <= kmax; ++k){
        const double* Ssel = topS.data();
        const double* Ysel = topY.data();
        if (k == 1){ set_at(out, m, r, k-1, Ysel[0]); continue; }
        
        std::vector<double> tw(k,0.0), emp(k,0.0), normw(k,0.0), rbf(k,0.0);
        for (int i = 0; i < k; ++i){
          tw[i] = pdf_t_prop(Ssel[i], static_cast<double>(k));
          emp[i] = (Ssel[i] > 0) ? 1.0 / Ssel[i] : 0.0;
        }
        double tws = std::accumulate(tw.begin(), tw.end(), 0.0);
        if (tws > 0) for(double &v: tw) v /= tws; else std::fill(tw.begin(), tw.end(), 0.0);
        
        double emps = std::accumulate(emp.begin(), emp.end(), 0.0);
        if (emps > 0) for(double &v: emp) v /= emps; else std::fill(emp.begin(), emp.end(), 0.0);
        
        double sdS = sd_vec(std::vector<double>(topS.begin(), topS.begin() + k));
        if (std::isfinite(sdS) && sdS > 0){
          for (int i = 0; i < k; ++i) normw[i] = pdf_norm_prop(Ssel[i], 0.0, sdS);
          double ns = std::accumulate(normw.begin(), normw.end(), 0.0);
          if (ns > 0) for(double &v: normw) v /= ns; else std::fill(normw.begin(), normw.end(), 0.0);
        }
        
        double vS = var_vec(std::vector<double>(topS.begin(), topS.begin() + k));
        if (std::isfinite(vS) && vS > 0){
          for (int i = 0; i < k; ++i) rbf[i] = std::exp(-Ssel[i] / (2.0 * vS));
          double rs = std::accumulate(rbf.begin(), rbf.end(), 0.0);
          if (rs > 0) for(double &v: rbf) v /= rs; else std::fill(rbf.begin(), rbf.end(), 0.0);
        }
        
        double dot = 0.0, tot = 0.0;
        for (int i = 0; i < k; ++i){
          double wi = uniW[k][i] + expW[k][i] + lnormW[k][i] + plW[k][i] + tw[i] + emp[i] + normw[i] + rbf[i];
          tot += wi;
          if (!is_class) dot += Ysel[i] * wi;
        }
        double invTot = (tot > 0.0) ? (1.0 / tot) : (1.0 / static_cast<double>(k));
        
        if (!is_class){
          double val = (tot > 0.0) ? (dot * invTot) : (std::accumulate(topY.begin(), topY.begin()+k, 0.0) / static_cast<double>(k));
          set_at(out, m, r, k-1, val);
        } else {
          std::vector<double> w(k);
          if (tot > 0.0) {
            for (int i = 0; i < k; ++i) w[i] = (uniW[k][i]+expW[k][i]+lnormW[k][i]+plW[k][i]+tw[i]+emp[i]+normw[i]+rbf[i]) * invTot;
          } else {
            std::fill(w.begin(), w.end(), 1.0/static_cast<double>(k));
          }
          set_at(out, m, r, k-1, mode_class_weighted(std::vector<double>(topY.begin(), topY.begin()+k), w));
        }
      }
    }
  }, nthreads);
  
  return out;
}

// ---------- Parallel Single Path ----------

std::vector<double> distance_path_single_parallel(const double* RPM, std::size_t l, std::size_t n,
                                                  const double* yhat, const double* Xtest, std::size_t m,
                                                  int k, bool is_class, int nthreads) {
  if (k <= 0) k = static_cast<int>(l);
  if (k > static_cast<int>(l)) k = static_cast<int>(l);
  
  std::vector<double> minRPM(n, std::numeric_limits<double>::infinity());
  std::vector<double> maxRPM(n, -std::numeric_limits<double>::infinity());
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t i = 0; i < l; ++i) {
      double v = at(RPM, l, i, j);
      if (std::isfinite(v)) { if (v < minRPM[j]) minRPM[j] = v; if (v > maxRPM[j]) maxRPM[j] = v; }
    }
    if (!std::isfinite(minRPM[j])) { minRPM[j] = 0.0; maxRPM[j] = 0.0; }
  }
  
  std::vector<double> uniW(k, 1.0 / static_cast<double>(k));
  
  std::vector<double> expW(k);
  for (int r = 1; r <= k; ++r) expW[r - 1] = pdf_exp(static_cast<double>(r), 1.0 / static_cast<double>(k));
  double exs = std::accumulate(expW.begin(), expW.end(), 0.0);
  if (exs > 0.0) for (double &v : expW) v /= exs; else std::fill(expW.begin(), expW.end(), 0.0);
  
  std::vector<double> plW(k);
  for (int r = 1; r <= k; ++r) plW[r - 1] = std::pow(static_cast<double>(r), -2.0);
  double pls = std::accumulate(plW.begin(), plW.end(), 0.0);
  if (pls > 0.0) for (double &v : plW) v /= pls; else std::fill(plW.begin(), plW.end(), 0.0);
  
  std::vector<double> lnormW(k, 0.0);
  if (k >= 2) {
    double sdlog = std::sqrt((static_cast<double>(k) * static_cast<double>(k) - 1.0) / 12.0);
    for (int r = 1; r <= k; ++r) {
      double lp = pdf_lnorm_log(static_cast<double>(r), 0.0, sdlog);
      lnormW[r - 1] = std::fabs(lp);
    }
    std::reverse(lnormW.begin(), lnormW.end());
    double lns = std::accumulate(lnormW.begin(), lnormW.end(), 0.0);
    if (lns > 0.0) for (double &v : lnormW) v /= lns; else std::fill(lnormW.begin(), lnormW.end(), 0.0);
  }
  
  std::vector<double> out(m, 0.0);
  
  parallel_for(0, m, [&](std::size_t begin, std::size_t end) {
    std::vector<double> invR(n), S(l), topS(k), topY(k);
    std::vector<int> idx(l);
    
    for (std::size_t r = begin; r < end; ++r) {
      for (std::size_t j = 0; j < n; ++j) {
        double t = at(Xtest, m, r, j);
        double mn = std::min(minRPM[j], t);
        double mx = std::max(maxRPM[j], t);
        double range = mx - mn;
        invR[j] = (std::isfinite(range) && range > 0.0) ? (1.0 / range) : 0.0;
      }
      
      for (std::size_t i = 0; i < l; ++i) {
        double acc = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
          double a = at(RPM, l, i, j), b = at(Xtest, m, r, j);
          if (std::isfinite(a) && std::isfinite(b) && invR[j] > 0.0) {
            double diff = (a - b) * invR[j];
            acc += diff * diff + std::fabs(diff);
          }
        }
        S[i] = (acc == 0.0 ? 1e-10 : acc);
      }
      
      std::iota(idx.begin(), idx.end(), 0);
      auto cmp = [&](int a, int b) { return S[a] < S[b]; };
      if (k < static_cast<int>(l)) std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), cmp);
      else std::sort(idx.begin(), idx.end(), cmp);
      
      auto cmp2 = [&](int a, int b) {
        if (S[a] < S[b]) return true;
        if (S[b] < S[a]) return false;
        return a < b;
      };
      std::stable_sort(idx.begin(), idx.begin() + k, cmp2);
      
      for (int t = 0; t < k; ++t) { int i = idx[t]; topS[t] = S[i]; topY[t] = yhat[i]; }
      
      if (k == 1) { out[r] = topY[0]; continue; }
      
      std::vector<double> tw(k, 0.0), emp(k, 0.0), normw(k, 0.0), rbf(k, 0.0);
      for (int i = 0; i < k; ++i) {
        tw[i] = pdf_t_prop(topS[i], static_cast<double>(k));
        emp[i] = (topS[i] > 0.0) ? 1.0 / topS[i] : 0.0;
      }
      
      double tws = std::accumulate(tw.begin(), tw.end(), 0.0);
      if (tws > 0.0) for (double &v : tw) v /= tws; else std::fill(tw.begin(), tw.end(), 0.0);
      
      double emps = std::accumulate(emp.begin(), emp.end(), 0.0);
      if (emps > 0.0) for (double &v : emp) v /= emps; else std::fill(emp.begin(), emp.end(), 0.0);
      
      double sdS = sd_vec(topS);
      if (std::isfinite(sdS) && sdS > 0.0) {
        for (int i = 0; i < k; ++i) normw[i] = pdf_norm_prop(topS[i], 0.0, sdS);
        double ns = std::accumulate(normw.begin(), normw.end(), 0.0);
        if (ns > 0.0) for (double &v : normw) v /= ns; else std::fill(normw.begin(), normw.end(), 0.0);
      }
      
      double vS = var_vec(topS);
      if (std::isfinite(vS) && vS > 0.0) {
        for (int i = 0; i < k; ++i) rbf[i] = std::exp(-topS[i] / (2.0 * vS));
        double rs = std::accumulate(rbf.begin(), rbf.end(), 0.0);
        if (rs > 0.0) for (double &v : rbf) v /= rs; else std::fill(rbf.begin(), rbf.end(), 0.0);
      }
      
      double dot = 0.0, tot = 0.0;
      for (int i = 0; i < k; ++i) {
        double wi = uniW[i] + expW[i] + lnormW[i] + plW[i] + tw[i] + emp[i] + normw[i] + rbf[i];
        tot += wi;
        if (!is_class) dot += topY[i] * wi;
      }
      
      double invTot = (tot > 0.0) ? (1.0 / tot) : (1.0 / static_cast<double>(k));
      
      if (!is_class) {
        out[r] = (tot > 0.0) ? (dot * invTot) : (std::accumulate(topY.begin(), topY.end(), 0.0) / static_cast<double>(k));
      } else {
        std::vector<double> w(k);
        if (tot > 0.0) {
          for (int i = 0; i < k; ++i) w[i] = (uniW[i] + expW[i] + lnormW[i] + plW[i] + tw[i] + emp[i] + normw[i] + rbf[i]) * invTot;
        } else {
          std::fill(w.begin(), w.end(), 1.0 / static_cast<double>(k));
        }
        out[r] = mode_class_weighted(topY, w);
      }
    }
  }, nthreads);
  
  return out;
}

} // namespace nns
