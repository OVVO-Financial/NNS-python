// src/dependence.cpp
//
// Implementation extracted from NNS 13.0 NNS_dep.cpp. Decoupled from Rcpp.
//
// SPDX-License-Identifier: GPL-3.0-only
#include "nns/dependence.hpp"
#include "nns/parallel.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace nns {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

inline double at(const double* M, std::size_t rows, std::size_t r, std::size_t c) {
  return M[c * rows + r];
}

inline double gravity_pure(const std::vector<double>& v) {
  std::size_t n = v.size();
  if (n == 0) return kNaN;
  if (n == 1) return v[0];
  if (n == 2) return (v[0] + v[1]) / 2.0;
  
  double sum = 0.0;
  for (double val : v) sum += val;
  return sum / static_cast<double>(n);
}

inline int n_unique(const double* v, std::size_t n) {
  std::unordered_map<double, int> seen;
  seen.reserve(n);
  for (std::size_t i = 0; i < n; ++i) seen[v[i]] = 1;
  return static_cast<int>(seen.size());
}

double copula_signed(const std::vector<double>& xv, const std::vector<double>& yv) {
  int n = static_cast<int>(xv.size());
  if (n < 2) return 0.0;
  
  double tx = 0.0, ty = 0.0;
  for (int i = 0; i < n; ++i) { tx += xv[i]; ty += yv[i]; }
  tx /= static_cast<double>(n); 
  ty /= static_cast<double>(n);
  
  double d0_cupm = 0.0, d0_clpm = 0.0, dpm_d0_count = 0.0;
  double c1_cupm = 0.0, c1_clpm = 0.0, c1_dpm = 0.0;
  double cov = 0.0, varx = 0.0;
  
  for (int i = 0; i < n; ++i) {
    double dx = xv[i] - tx;
    double dy = yv[i] - ty;
    
    if (dx > 0.0 && dy > 0.0) d0_cupm += 1.0;
    if (dx <= 0.0 && dy <= 0.0) d0_clpm += 1.0;
    if (!((dx < 0.0 && dy < 0.0) || (dx > 0.0 && dy > 0.0)))
      dpm_d0_count += 1.0;
    
    if (dx >= 0.0 && dy >= 0.0) {
      c1_cupm += dx * dy;
    } else if (dx <= 0.0 && dy <= 0.0) {
      c1_clpm += dx * dy;
    } else {
      c1_dpm += std::abs(dx) * std::abs(dy);
    }
    
    cov += dx * dy;
    varx += dx * dx;
  }
  
  double inv_n = 1.0 / static_cast<double>(n);
  double d0_Co = (d0_cupm + d0_clpm) * inv_n;
  if (d0_Co == 1.0 || d0_Co == 0.0) return 1.0;
  
  double c1_total = c1_cupm + c1_clpm + c1_dpm;
  double co_d1 = c1_total > 0.0 ? (c1_cupm + c1_clpm) / c1_total : 0.0;
  double dpm_d0 = dpm_d0_count * inv_n;
  double dpm_d1 = c1_total > 0.0 ? c1_dpm / c1_total : 0.0;
  
  constexpr double indep_Co = 0.5;
  constexpr double indep_D  = 0.75;
  
  double discrete_dep   = std::min(1.0, std::max(0.0, std::abs(d0_Co - indep_Co) / indep_Co));
  double continuous_dep = std::min(1.0, std::max(0.0, std::abs(co_d1 - indep_Co) / indep_Co));
  double nd_disc_dep    = std::abs(dpm_d0 - indep_D) / indep_D;
  double nd_cont_dep    = std::abs(dpm_d1 - indep_D) / indep_D;
  
  double copula_val = std::sqrt((discrete_dep + continuous_dep + nd_disc_dep + nd_cont_dep) / 4.0);
  double slope_sign = varx == 0.0 ? 0.0 : ((cov > 0.0) ? 1.0 : (cov < 0.0) ? -1.0 : 0.0);
  return copula_val * slope_sign;
}

double copula_degree0_unsigned(const std::vector<double>& xv, const std::vector<double>& yv) {
  int n = static_cast<int>(xv.size());
  if (n < 2) return 0.0;
  
  double tx = 0.0, ty = 0.0;
  for (int i = 0; i < n; ++i) { tx += xv[i]; ty += yv[i]; }
  tx /= static_cast<double>(n); 
  ty /= static_cast<double>(n);
  
  double d0_cupm = 0.0, d0_clpm = 0.0, dpm_d0_count = 0.0;
  for (int i = 0; i < n; ++i) {
    double dx = xv[i] - tx;
    double dy = yv[i] - ty;
    if (dx > 0.0 && dy > 0.0) d0_cupm += 1.0;
    if (dx <= 0.0 && dy <= 0.0) d0_clpm += 1.0;
    if (!((dx < 0.0 && dy < 0.0) || (dx > 0.0 && dy > 0.0)))
      dpm_d0_count += 1.0;
  }
  
  double inv_n = 1.0 / static_cast<double>(n);
  double d0_Co = (d0_cupm + d0_clpm) * inv_n;
  double dpm_d0 = dpm_d0_count * inv_n;
  
  constexpr double indep_Co = 0.5;
  constexpr double indep_D  = 0.75;
  
  double disc_dep = std::min(1.0, std::max(0.0, std::abs(d0_Co - indep_Co) / indep_Co));
  double nd_disc  = std::abs(dpm_d0 - indep_D) / indep_D;
  
  return std::sqrt((disc_dep + nd_disc) / 2.0);
}

} // namespace

// ---------- Pairwise Dependence Kernel ----------

DepResult dep_pair(const double* xv, const double* yv, std::size_t n,
                   const uint64_t* quad_xy, const uint64_t* quad_yx, bool asym) {
  
  bool cx = true, cy = true;
  for (std::size_t i = 1; i < n; ++i) {
    if (xv[i] != xv[0]) cx = false;
    if (yv[i] != yv[0]) cy = false;
    if (!cx && !cy) break;
  }
  if (cx || cy) return {0.0, 0.0};
  
  std::unordered_map<uint64_t, std::vector<int>> grp_xy;
  grp_xy.reserve(n);
  for (std::size_t i = 0; i < n; ++i) grp_xy[quad_xy[i]].push_back(static_cast<int>(i));
  
  std::unordered_map<uint64_t, std::vector<int>> grp_yx;
  grp_yx.reserve(n);
  for (std::size_t i = 0; i < n; ++i) grp_yx[quad_yx[i]].push_back(static_cast<int>(i));
  
  std::vector<double> xv_vec(xv, xv + n);
  std::vector<double> yv_vec(yv, yv + n);
  
  double global_cop = copula_signed(xv_vec, yv_vec);
  if (!std::isfinite(global_cop)) global_cop = 0.0;
  
  double corr_xy = 0.0, dep_xy = 0.0;
  for (const auto& kv : grp_xy) {
    const auto& idx = kv.second;
    int nq = static_cast<int>(idx.size());
    if (nq < 1) continue;
    
    std::vector<double> xq(nq), yq(nq);
    for (int k = 0; k < nq; ++k) { xq[k] = xv[idx[k]]; yq[k] = yv[idx[k]]; }
    
    double cop = copula_signed(xq, yq);
    if (!std::isfinite(cop)) cop = global_cop;
    
    double w = static_cast<double>(nq) / static_cast<double>(n);
    corr_xy += cop           * w;
    dep_xy  += std::abs(cop) * w;
  }
  
  double corr_yx = 0.0, dep_yx = 0.0;
  for (const auto& kv : grp_yx) {
    const auto& idx = kv.second;
    int nq = static_cast<int>(idx.size());
    if (nq < 1) continue;
    
    std::vector<double> yq(nq), xq(nq);
    for (int k = 0; k < nq; ++k) { yq[k] = yv[idx[k]]; xq[k] = xv[idx[k]]; }
    
    double cop = copula_signed(yq, xq);
    if (!std::isfinite(cop)) cop = global_cop;
    
    double w = static_cast<double>(nq) / static_cast<double>(n);
    corr_yx += cop           * w;
    dep_yx  += std::abs(cop) * w;
  }
  
  int lx = n_unique(xv, n);
  int ly = n_unique(yv, n);
  bool discrete_case = (lx < std::sqrt(static_cast<double>(n))) &&
                       (ly < std::sqrt(static_cast<double>(n)));
  
  if (discrete_case) {
    double disc_cop = copula_degree0_unsigned(xv_vec, yv_vec);
    if (!std::isfinite(disc_cop)) disc_cop = std::max(dep_xy, dep_yx);
    
    if (asym) {
      std::vector<double> gv = {dep_xy, disc_cop};
      dep_xy = gravity_pure(gv);
    } else {
      double dep_sym = std::max(dep_xy, dep_yx);
      std::vector<double> gv = {dep_sym, disc_cop};
      double blended = gravity_pure(gv);
      dep_xy = blended;
      dep_yx = blended;
    }
  }
  
  if (asym) {
    return {corr_xy, dep_xy};
  }
  return {std::max(corr_xy, corr_yx), std::max(dep_xy, dep_yx)};
}

// ---------- Full Dependence Matrix Kernel ----------

DepMatrixResult dep_matrix(const double* X, std::size_t n, std::size_t p, 
                           bool asym, int nthreads) {
  
  if (p < 2) throw std::invalid_argument("dep_matrix: X must have at least 2 columns");
  
  std::size_t n_pairs = p * (p - 1) / 2;
  int obs_req = std::max(8, static_cast<int>(n) / 8);
  std::vector<std::vector<uint64_t>> all_quads(p);
  
  // Phase 1: Precompute Partitions
  parallel_for(0, p, [&](std::size_t begin, std::size_t end) {
    for (std::size_t j = begin; j < end; ++j) {
      int max_order = std::max(1, static_cast<int>(std::floor(std::log2(std::max(1, static_cast<int>(n))))));
      std::vector<uint64_t> quad(n, 1);
      
      for (int depth = 0; depth < max_order; ++depth) {
        std::unordered_map<uint64_t, std::vector<int>> grp;
        grp.reserve(n);
        for (std::size_t i = 0; i < n; ++i) grp[quad[i]].push_back(static_cast<int>(i));
        
        bool any_split = false;
        for (const auto& kv : grp) {
          const auto& idx = kv.second;
          if (static_cast<int>(idx.size()) <= obs_req) continue;
          
          double cx = 0.0;
          for (int i : idx) cx += at(X, n, i, j);
          cx /= static_cast<double>(idx.size());
          
          for (int i : idx) {
            quad[i] = (quad[i] << 2) | ((at(X, n, i, j) > cx) ? 2 : 1);
          }
          any_split = true;
        }
        if (!any_split) break;
      }
      all_quads[j] = std::move(quad);
    }
  }, nthreads);
  
  // Phase 2: Compute Pairwise Dependence
  std::vector<double> corr_upper(n_pairs, 0.0);
  std::vector<double> dep_upper(n_pairs, 0.0);
  std::vector<double> corr_lower(n_pairs, 0.0);
  std::vector<double> dep_lower(n_pairs, 0.0);
  
  std::vector<int> pair_i, pair_j;
  pair_i.reserve(n_pairs); pair_j.reserve(n_pairs);
  for (std::size_t i = 0; i < p - 1; ++i) {
    for (std::size_t j = i + 1; j < p; ++j) {
      pair_i.push_back(static_cast<int>(i));
      pair_j.push_back(static_cast<int>(j));
    }
  }
  
  parallel_for(0, n_pairs, [&](std::size_t begin, std::size_t end) {
    for (std::size_t idx = begin; idx < end; ++idx) {
      int ci = pair_i[idx];
      int cj = pair_j[idx];
      
      const std::vector<uint64_t>& q_xy = all_quads[ci];
      const std::vector<uint64_t>& q_yx = all_quads[cj];
      
      std::vector<double> xnv(n), ynv(n);
      for (std::size_t r = 0; r < n; ++r) {
        xnv[r] = at(X, n, r, ci);
        ynv[r] = at(X, n, r, cj);
      }
      
      DepResult res_ij = dep_pair(xnv.data(), ynv.data(), n, q_xy.data(), q_yx.data(), asym);
      corr_upper[idx] = res_ij.correlation;
      dep_upper[idx]  = res_ij.dependence;
      
      if (asym) {
        DepResult res_ji = dep_pair(ynv.data(), xnv.data(), n, q_yx.data(), q_xy.data(), true);
        corr_lower[idx] = res_ji.correlation;
        dep_lower[idx]  = res_ji.dependence;
      } else {
        corr_lower[idx] = corr_upper[idx];
        dep_lower[idx]  = dep_upper[idx];
      }
    }
  }, nthreads);
  
  // Phase 3: Construct the final Column-Major output matrices
  DepMatrixResult result;
  result.p = p;
  result.correlation.assign(p * p, 0.0);
  result.dependence.assign(p * p, 0.0);
  
  for (std::size_t i = 0; i < p; ++i) {
    result.correlation[i * p + i] = 1.0;
    result.dependence[i * p + i] = 1.0;
  }
  
  std::size_t idx = 0;
  for (std::size_t i = 0; i < p - 1; ++i) {
    for (std::size_t j = i + 1; j < p; ++j, ++idx) {
      if (!asym) {
        double r = (corr_upper[idx] + corr_lower[idx]) / 2.0;
        double d = (dep_upper[idx]  + dep_lower[idx])  / 2.0;
        
        result.correlation[j * p + i] = r; // Row i, Col j
        result.correlation[i * p + j] = r; // Row j, Col i
        
        result.dependence[j * p + i] = d;
        result.dependence[i * p + j] = d;
      } else {
        result.correlation[j * p + i] = corr_upper[idx];
        result.dependence[j * p + i]  = dep_upper[idx];
        
        result.correlation[i * p + j] = corr_lower[idx];
        result.dependence[i * p + j]  = dep_lower[idx];
      }
    }
  }
  
  return result;
}

} // namespace nns