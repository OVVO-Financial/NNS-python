// include/nns/dependence.hpp
//
// SPDX-License-Identifier: GPL-3.0-only
#ifndef NNS_DEPENDENCE_HPP
#define NNS_DEPENDENCE_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nns {

// --- Output Data Structures ---

struct DepResult {
  double correlation;
  double dependence;
};

struct DepMatrixResult {
  std::vector<double> correlation; // p x p column-major matrix
  std::vector<double> dependence;  // p x p column-major matrix
  std::size_t p;
};

// --- Core API ---

/// Compute bivariate dependence between two vectors
///
/// @param x Pointer to the first array.
/// @param y Pointer to the second array.
/// @param n Length of the arrays.
/// @param quad_xy Pointer to the pre-hashed partition labels for x given y.
/// @param quad_yx Pointer to the pre-hashed partition labels for y given x.
/// @param asym Calculate asymmetric dependence (true/false).
/// @return DepResult containing the scalar correlation and dependence.
DepResult dep_pair(const double* x, const double* y, std::size_t n,
                   const uint64_t* quad_xy, const uint64_t* quad_yx, bool asym = false);

/// Compute the full pairwise dependence matrix
///
/// @param X Pointer to the column-major data matrix.
/// @param n Number of rows in X.
/// @param p Number of columns in X.
/// @param asym Calculate asymmetric dependence (true/false).
/// @param nthreads Number of parallel threads to use (-1 for hardware max).
/// @return DepMatrixResult containing the column-major correlation and dependence matrices.
DepMatrixResult dep_matrix(const double* X, std::size_t n, std::size_t p, 
                           bool asym = false, int nthreads = -1);

} // namespace nns

#endif // NNS_DEPENDENCE_HPP