// SPDX-License-Identifier: GPL-3.0-only
#include "_r14_partial_moments.hpp"

#include <climits>
#include <cmath>
#include <stdexcept>

#include "nns/parallel.hpp"

namespace nns_r14 {

double nd_moment(NDMethod moment, const double* data, std::size_t n, std::size_t d,
                 const double* target, double degree, bool norm, int threads) {
  if (!norm || degree == 0.0) return moment(data, n, d, target, degree, false, threads);
  const double lower = nns::clpm_nd(data, n, d, target, degree, false, threads);
  const double upper = nns::cupm_nd(data, n, d, target, degree, false, threads);
  const double div = nns::dpm_nd(data, n, d, target, degree, false, threads);
  const double total = lower + upper + div;
  if (total == 0.0) return moment(data, n, d, target, 0.0, false, threads);
  const double raw = moment == nns::clpm_nd ? lower : (moment == nns::cupm_nd ? upper : div);
  return total > 0.0 ? raw / total : 0.0;
}

static double power(double value, double degree) {
  if (std::trunc(degree) != degree || degree > INT_MAX) return std::pow(value, degree);
  double term = 1.0;
  for (int k = 0; k < static_cast<int>(degree); ++k) term *= value;
  return term;
}

void clpm_nd_batch(const double* data, std::size_t n, std::size_t d, const double* targets,
                   std::size_t m, double degree, bool norm, bool strict, double* out, int threads) {
  if (!(degree >= 0.0)) throw std::invalid_argument("`degree` must be a non-negative number");
  if (n == 0) throw std::invalid_argument("`data` must have at least one row");
  const double inv_n = 1.0 / static_cast<double>(n);
  nns::parallel_for(0, m, [&](std::size_t begin, std::size_t end) {
    for (std::size_t r = begin; r < end; ++r) {
      auto lower_share = [&] {
        std::size_t count = 0;
        for (std::size_t i = 0; i < n; ++i) {
          std::size_t j = 0;
          for (; j < d; ++j) {
            const double x = data[j * n + i], t = targets[j * m + r];
            if (strict ? x >= t : x > t) break;
          }
          if (j == d) ++count;
        }
        return static_cast<double>(count) / static_cast<double>(n);
      };
      if (degree == 0.0) {
        out[r] = lower_share();
        continue;
      }
      double lower = 0.0, upper = 0.0, div = 0.0;
      for (std::size_t i = 0; i < n; ++i) {
        double lo = 1.0, hi = 1.0, abs_prod = 1.0;
        bool below = true, above = true;
        for (std::size_t j = 0; j < d; ++j) {
          const double diff = data[j * n + i] - targets[j * m + r];
          const double term = power(std::abs(diff), degree);
          lo *= diff > 0.0 ? 0.0 : term;
          hi *= diff < 0.0 ? 0.0 : term;
          abs_prod *= term;
          if (diff >= 0.0) below = false;
          if (diff <= 0.0) above = false;
        }
        lower += lo;
        upper += hi;
        if (!below && !above) div += abs_prod;
      }
      lower *= inv_n;
      if (!norm) {
        out[r] = lower;
        continue;
      }
      const double total = lower + upper * inv_n + div * inv_n;
      if (total == 0.0) out[r] = lower_share();
      else out[r] = total > 0.0 ? lower / total : 0.0;
    }
  }, threads, n * d);
}

nns::PMMatrixResult pm_matrix(double lower, double upper, const double* target, const double* data,
                              std::size_t n, std::size_t d, bool pop_adj, bool norm, int threads) {
  auto result = nns::pm_matrix(lower, upper, target, data, n, d, pop_adj, false, threads);
  if (!norm) return result;
  nns::PMMatrixResult pm0;
  for (std::size_t i = 0; i < result.dim; ++i) {
    for (std::size_t j = i; j < result.dim; ++j) {
      const auto k = j * d + i, mirror = i * d + j;
      double cu = result.cupm[k], du = result.dupm[k], dl = result.dlpm[k], cl = result.clpm[k];
      const double total = cu + du + dl + cl;
      if (total > 0.0) {
        cu /= total;
        du /= total;
        dl /= total;
        cl /= total;
      } else if (total == 0.0) {
        if (pm0.clpm.empty()) pm0 = nns::pm_matrix(0, 0, target, data, n, d, false, true, threads);
        cu = pm0.cupm[k];
        du = pm0.dupm[k];
        dl = pm0.dlpm[k];
        cl = pm0.clpm[k];
      } else {
        cu = du = dl = cl = 0.0;
      }
      result.cupm[k] = result.cupm[mirror] = cu;
      result.clpm[k] = result.clpm[mirror] = cl;
      result.dupm[k] = result.dlpm[mirror] = du;
      result.dlpm[k] = result.dupm[mirror] = dl;
      result.cov[k] = result.cov[mirror] = cu + cl - du - dl;
    }
  }
  return result;
}

}  // namespace nns_r14
