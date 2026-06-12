// include/nns/distance.hpp
//
// SPDX-License-Identifier: GPL-3.0-only
#ifndef NNS_DISTANCE_HPP
#define NNS_DISTANCE_HPP

#include <cstddef>
#include <vector>

namespace nns {

/// Single row NNS Distance evaluation
///
/// @param X Pointer to the column-major predictor matrix.
/// @param l Number of rows in X.
/// @param n Number of columns in X.
/// @param yhat Pointer to the target variable array (length l).
/// @param dest Pointer to the destination vector (length n).
/// @param k K-nearest neighbors parameter.
/// @param use_class Treat predictions as discrete classes.
/// @return The estimated distance or classification target.
double distance(const double* X, std::size_t l, std::size_t n,
                const double* yhat, const double* dest,
                int k, bool use_class);

/// Sequential NNS Distance Path (Multi-target path evaluation)
///
/// @param RPM Pointer to the column-major Partial Moments matrix (n x p).
/// @param n Number of rows in RPM.
/// @param p Number of columns in RPM.
/// @param yhat Pointer to the target variable array (length n).
/// @param Xtest Pointer to the column-major test matrix (m x p).
/// @param m Number of rows in Xtest.
/// @param kmax Maximum k to evaluate the path up to.
/// @param is_class Treat predictions as discrete classes.
/// @return A column-major matrix represented as a flat vector (m x kmax).
std::vector<double> distance_path(const double* RPM, std::size_t n, std::size_t p,
                                  const double* yhat, const double* Xtest, std::size_t m,
                                  int kmax, bool is_class);

/// Sequential NNS Distance Bulk (Multi-target, fixed k)
///
/// @param RPM Pointer to the column-major Partial Moments matrix (n x p).
/// @param n Number of rows in RPM.
/// @param p Number of columns in RPM.
/// @param yhat Pointer to the target variable array (length n).
/// @param Xtest Pointer to the column-major test matrix (m x p).
/// @param m Number of rows in Xtest.
/// @param k K-nearest neighbors parameter.
/// @param is_class Treat predictions as discrete classes.
/// @return A flat vector of length m containing the predictions.
std::vector<double> distance_bulk(const double* RPM, std::size_t n, std::size_t p,
                                  const double* yhat, const double* Xtest, std::size_t m,
                                  int k, bool is_class);

/// Multi-threaded NNS Distance Path (Evaluates all k up to kmax)
///
/// @param nthreads Number of parallel threads to use (-1 for hardware max).
/// @return A column-major matrix represented as a flat vector (m x kmax).
std::vector<double> distance_path_parallel(const double* RPM, std::size_t l, std::size_t n,
                                           const double* yhat, const double* Xtest, std::size_t m,
                                           int kmax, bool is_class, int nthreads = -1);

/// Multi-threaded NNS Distance Path (Evaluates ONLY a single specified k)
///
/// @param nthreads Number of parallel threads to use (-1 for hardware max).
/// @return A flat vector of length m containing the predictions.
std::vector<double> distance_path_single_parallel(const double* RPM, std::size_t l, std::size_t n,
                                                  const double* yhat, const double* Xtest, std::size_t m,
                                                  int k, bool is_class, int nthreads = -1);

} // namespace nns

#endif // NNS_DISTANCE_HPP