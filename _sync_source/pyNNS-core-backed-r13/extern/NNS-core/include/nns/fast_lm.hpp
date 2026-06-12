// include/nns/fast_lm.hpp
//
// SPDX-License-Identifier: GPL-3.0-only
#ifndef NNS_FAST_LM_HPP
#define NNS_FAST_LM_HPP

#include <cstddef>
#include <vector>

namespace nns {

struct FastLmResult {
  std::vector<double> coef;          // length 2: [intercept, slope]
  std::vector<double> fitted_values; // original `fitted.values`
  std::vector<double> residuals;
  long long df_residual;             // original `df.residual = n - 2`
};

struct FastLmMultResult {
  std::vector<double> coefficients;  // intercept then slopes
  std::vector<double> fitted_values;
  std::vector<double> residuals;
  double r_squared;
};

FastLmResult fast_lm(const double* x, const double* y, std::size_t n);

/// Multiple OLS. X is an n x p column-major matrix, matching R NumericMatrix.
FastLmMultResult fast_lm_mult(const double* X, const double* y,
                              std::size_t n, std::size_t p);

} // namespace nns

#endif // NNS_FAST_LM_HPP
