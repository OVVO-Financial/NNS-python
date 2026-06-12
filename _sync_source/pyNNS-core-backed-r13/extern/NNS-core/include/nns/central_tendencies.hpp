// include/nns/central_tendencies.hpp
//
// SPDX-License-Identifier: GPL-3.0-only
#ifndef NNS_CENTRAL_TENDENCIES_HPP
#define NNS_CENTRAL_TENDENCIES_HPP

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace nns {

/// Compute the "center of gravity" statistic used by NNS.
///
/// @param x Pointer to the input data array.
/// @param n Length of the input array.
/// @param discrete Whether to coerce the result to the discrete analogue.
/// @return The estimated center of gravity (NaN if empty).
double gravity(const double* x, std::size_t n, bool discrete);

/// Rescale a vector using either min-max or risk-neutral methods.
///
/// @param x Pointer to the input data array.
/// @param n Length of the input array.
/// @param a Minimum target (minmax) or S_0 (riskneutral).
/// @param b Maximum target (minmax) or r (riskneutral).
/// @param method The scaling method: "minmax" or "riskneutral".
/// @param T Time to maturity (required for riskneutral).
/// @param type Terminal or discounted (used for riskneutral).
/// @return A new vector containing the rescaled values.
std::vector<double> rescale(const double* x, std::size_t n, double a, double b,
                            const std::string& method = "minmax",
                            std::optional<double> T = std::nullopt,
                            const std::string& type = "Terminal");

/// Compute the mode (or modal class) depending on the supplied flags.
///
/// @param x Pointer to the input data array.
/// @param n Length of the input array.
/// @param discrete Treat data as discrete values.
/// @param multi Return the multi-modal result (all tied modes).
/// @return A vector of modes. If multi=false, the vector contains exactly one element.
std::vector<double> mode(const double* x, std::size_t n, bool discrete, bool multi);

} // namespace nns

#endif // NNS_CENTRAL_TENDENCIES_HPP