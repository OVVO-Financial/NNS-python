// include/nns/stochastic_dominance.hpp
//
// SPDX-License-Identifier: GPL-3.0-only
#ifndef NNS_STOCHASTIC_DOMINANCE_HPP
#define NNS_STOCHASTIC_DOMINANCE_HPP

#include <cstddef>
#include <vector>

namespace nns {

// --- Univariate Dominance Tests ---
// Faithful ports of NNS_FSD_uni_cpp / NNS_SSD_uni_cpp / NNS_TSD_uni_cpp.
// Return 1 if x dominates y, otherwise 0.  Identical samples never dominate.
// NaN (missing) values raise std::invalid_argument, matching the upstream
// "You have some missing values, please address." stop; +/-Inf is permitted.

/// First-degree Stochastic Dominance (Univariate)
/// @param x Pointer to the first array.
/// @param y Pointer to the second array.
/// @param n Length of the arrays.
/// @param discrete Treat data as discrete (ECDF compare) or continuous
///        (degree-1 LPM-ratio compare), matching upstream type =
///        "discrete"/"continuous".
int fsd_uni(const double* x, const double* y, std::size_t n, bool discrete);

/// Second-degree Stochastic Dominance (Univariate)
int ssd_uni(const double* x, const double* y, std::size_t n);

/// Third-degree Stochastic Dominance (Univariate)
int tsd_uni(const double* x, const double* y, std::size_t n);

// --- Pairwise Dominance Matrix ---

/// Faithful port of sd_dom_matrix_prefix_parallel.
///
/// @param X Pointer to the column-major data matrix (n x p), no NaN.
/// @param n Number of rows in X.
/// @param p Number of columns in X.
/// @param degree 1 (FSD), 2 (SSD) or 3 (TSD).
/// @param discrete Only meaningful for degree 1 (forced true otherwise,
///        as upstream).
/// @param nthreads Number of parallel threads to use (-1 for hardware max).
/// @return A p x p column-major matrix M with M[j * p + i] = 1 iff column i
///         dominates column j, else 0 (diagonal is 0).
std::vector<int> sd_dom_matrix(const double* X, std::size_t n, std::size_t p,
                               int degree, bool discrete, int nthreads = -1);

// --- Multivariate Efficient-Set Filters ---
// Faithful ports of NNS_SD_efficient_set_parallel_cpp: columns are ordered
// by LPM(degree, global-max, .) ascending (stable tie-break by original
// index); a column is then dropped only if it is dominated by a previously
// KEPT column.  The returned vector contains the surviving ORIGINAL 0-based
// column indices, in that sorted order (the same order in which upstream
// returns column names).

/// First-degree Stochastic Dominance efficient set.
/// @param discrete Treat data as discrete (true) or continuous (false).
std::vector<int> fsd(const double* X, std::size_t n, std::size_t p, bool discrete, int nthreads = -1);

/// Second-degree Stochastic Dominance efficient set.
std::vector<int> ssd(const double* X, std::size_t n, std::size_t p, int nthreads = -1);

/// Third-degree Stochastic Dominance efficient set.
std::vector<int> tsd(const double* X, std::size_t n, std::size_t p, int nthreads = -1);

// --- Stochastic Superiority ---

struct StochSupResult {
  double p_gt;   // Probability that X > Y
  double p_tie;  // Probability that X == Y
  double p_star; // p_gt + 0.5 * p_tie
};

/// Compute the stochastic superiority of array X over array Y.
///
/// @param x Pointer to the first numeric array (X).
/// @param n_x Length of array X.
/// @param y Pointer to the second numeric array (Y).
/// @param n_y Length of array Y.
/// @return StochSupResult containing the exact probabilities.
StochSupResult stochastic_superiority(const double* x, std::size_t n_x,
                                      const double* y, std::size_t n_y);

} // namespace nns

#endif // NNS_STOCHASTIC_DOMINANCE_HPP