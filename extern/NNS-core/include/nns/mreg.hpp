// include/nns/mreg.hpp
//
// Portable ports of the R 13.1-13.3 NNS.M.reg / NNS.stack native kernels:
//   src/NNS_mreg_setup.cpp   -> duplicate_column_map, mreg_setup
//   src/NNS_mreg_predict.cpp -> mreg_predict, mreg_predict_path
//   src/NNS_stack_fast.cpp   -> xstar_path
//
// All matrices are column-major (element (i, j) at i + j * nrow), like R.
// Indices returned by these functions are 0-based (R returns 1-based ones).
//
// SPDX-License-Identifier: GPL-3.0-only
#ifndef NNS_MREG_HPP
#define NNS_MREG_HPP

#include <cstddef>
#include <string>
#include <vector>

namespace nns {

/// Port of NNS_duplicate_column_map_cpp. For each column j of the n x p
/// matrix X, the index of the first column whose values are byte-identical
/// to column j (j itself if there is none). Columns containing NaN never
/// match another column, exactly as upstream.
std::vector<std::size_t> duplicate_column_map(const double* X, std::size_t n,
                                              std::size_t p);

/// Reducer codes of NNS_mreg_setup_cpp (R: noise.reduction).
enum MregReducer : int {
  kMregMean = 0,     // "mean"
  kMregMedian = 1,   // "median"
  kMregMode = 2,     // "mode": NNS.mode (class responses: most frequent value)
  kMregGravity = 3,  // "off": NNS.gravity
};

struct MregSetupResult {
  std::size_t n_groups = 0;
  std::size_t p = 0;
  /// n_groups x (p + 1) column-major regression point matrix: the reduced
  /// X1..Xp of each group, then the reduced y.
  std::vector<double> rpm;
  /// Group id of every input row ("i1.i2...ip", findInterval ids).
  std::vector<std::string> ids;
  /// Group id of every RPM row, sorted lexicographically (as upstream).
  std::vector<std::string> row_ids;
};

/// Port of NNS_mreg_setup_cpp. X is n x p, y has length n, and
/// boundaries[j] holds the sorted partition boundaries of column j. Each row
/// is assigned the id built from findInterval(X[i, j], boundaries[j],
/// rightmost.closed = TRUE) over the columns; rows sharing an id are reduced
/// with `reducer_code` (y uses the exact-frequency mode when is_class).
/// Non-finite values are dropped before reducing (NaN if none remain).
MregSetupResult mreg_setup(const double* X, std::size_t n, std::size_t p,
                           const double* y,
                           const std::vector<std::vector<double>>& boundaries,
                           int reducer_code, bool is_class);

/// Distance codes shared by the M.reg kernels (R: dist).
enum MregDistance : int {
  kMregDistNNS = 0,     // sum(|z| + z^2) over range-normalised columns
  kMregDistL2 = 1,      // sqrt(sum(z^2))
  kMregDistL1 = 2,      // sum(|z|)
  kMregDistFactor = 3,  // Hamming distance over encoded columns
};

/// Port of NNS_mreg_predict_v2_cpp: the k-nearest-neighbour prediction for
/// each of the m rows of Xtest (m x p) against the regression points rpm_x
/// (n x p) with fitted values yhat (length n). k = 1 aggregates all exact
/// nearest-distance ties (gravity, or the class mode); k > 1 uses the
/// eight-component ensemble weights. mins/maxs (length p) give the column
/// ranges used for normalisation. k is clipped to n.
///
/// The density components use closed forms instead of Rmath (dt is used up
/// to its normalising constant, which cancels), so results agree with R to
/// floating-point rounding (about 1e-15), not bit for bit.
///
/// n_threads follows nns::parallel_for (<= 0: library default).
/// Throws std::invalid_argument on inconsistent arguments.
std::vector<double> mreg_predict(const double* rpm_x, std::size_t n, std::size_t p,
                                 const double* yhat, const double* Xtest,
                                 std::size_t m, int k, int dist_code,
                                 const double* mins, const double* maxs,
                                 bool is_class, int n_threads = -1);

/// Port of NNS_mreg_predict_path_v2_cpp: predictions for every k = 1..kmax
/// (kmax clipped to n). Returns an m x kmax column-major matrix; column k-1
/// equals mreg_predict(..., k, ...).
std::vector<double> mreg_predict_path(const double* rpm_x, std::size_t n, std::size_t p,
                                      const double* yhat, const double* Xtest,
                                      std::size_t m, int kmax, int dist_code,
                                      const double* mins, const double* maxs,
                                      bool is_class, int n_threads = -1);

struct XstarPathResult {
  std::size_t n_train = 0, n_test = 0, p = 0;
  std::vector<double> train;           // n_train x p, column-major
  std::vector<double> test;            // n_test x p, column-major
  std::vector<double> denominator;     // length p
  std::vector<std::size_t> column_order;    // 0-based, as passed in
  std::vector<std::size_t> representative;  // 0-based first identical column
};

/// Port of NNS_xstar_path_cpp (NNS.stack X* projection path). Column m of
/// the output is the running sum over the first m + 1 columns in
/// `column_order` (0-based) of design * coefficient, divided by the number of
/// non-zero coefficients so far (all coefficients count as 1 when every one
/// is zero). representative[m] is the first earlier output column identical
/// to column m in both train and test.
XstarPathResult xstar_path(const double* train_design, std::size_t n_train,
                           const double* test_design, std::size_t n_test,
                           std::size_t p, const double* coefficients,
                           const std::size_t* column_order, int n_threads = -1);

}  // namespace nns

#endif  // NNS_MREG_HPP
