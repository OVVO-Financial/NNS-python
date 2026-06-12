// include/nns/seasonality.hpp
//
// SPDX-License-Identifier: GPL-3.0-only
#ifndef NNS_SEASONALITY_HPP
#define NNS_SEASONALITY_HPP

#include <cstddef>
#include <vector>

namespace nns {

struct SeasonalityResult {
  std::vector<int> all_periods;           // Equivalents to DataFrame columns
  std::vector<double> all_coef_var;
  std::vector<double> all_var_coef_var;
  
  int best_period;                        // Scalar best period
  std::vector<int> periods;               // The chosen periods vector
};

/// Detect seasonality periods within a time series.
///
/// @param x Pointer to the numeric time series array.
/// @param n Length of the array.
/// @param modulo Pointer to an optional array of integer modulos to enforce.
/// @param mod_len Length of the modulo array (0 if none).
/// @param mod_only Flag to keep only periods matching the modulo set.
/// @return SeasonalityResult containing the detected periods and coefficients of variation.
SeasonalityResult seasonality(const double* x, std::size_t n,
                              const int* modulo = nullptr, std::size_t mod_len = 0,
                              bool mod_only = true);

} // namespace nns

#endif // NNS_SEASONALITY_HPP