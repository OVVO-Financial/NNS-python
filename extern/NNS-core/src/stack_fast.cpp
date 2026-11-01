// src/stack_fast.cpp
//
// Pure C++ port of R NNS 13.3 src/NNS_stack_fast.cpp (NNS_xstar_path_cpp,
// the NNS.stack X* projection path). Decoupled from Rcpp / RcppParallel.
//
// SPDX-License-Identifier: GPL-3.0-only
#include <stdexcept>
#include <vector>

#include "nns/mreg.hpp"
#include "nns/parallel.hpp"

namespace nns {

namespace {

// Running weighted sum over the ordered columns, divided by the number of
// non-zero coefficients so far (port of FillProjectionWorker).
void fill_projection(const double* design, std::size_t rows, std::size_t p,
                     const std::vector<std::size_t>& order, const double* coef,
                     const std::vector<double>& denom, bool all_zero, double* out,
                     int n_threads) {
  parallel_for(
      0, rows,
      [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
          double numerator = 0.0;
          for (std::size_t m = 0; m < p; ++m) {
            const std::size_t j = order[m];
            const double c = all_zero ? 1.0 : coef[j];
            numerator += design[i + j * rows] * c;
            out[i + m * rows] = numerator / denom[m];
          }
        }
      },
      n_threads, p);
}

}  // namespace

XstarPathResult xstar_path(const double* train_design, std::size_t n_train,
                           const double* test_design, std::size_t n_test, std::size_t p,
                           const double* coefficients, const std::size_t* column_order,
                           int n_threads) {
  XstarPathResult res;
  res.n_train = n_train;
  res.n_test = n_test;
  res.p = p;
  res.column_order.assign(column_order, column_order + p);
  for (std::size_t m = 0; m < p; ++m)
    if (res.column_order[m] >= p)
      throw std::invalid_argument("xstar_path: column_order entries must be in [0, p)");

  bool all_zero = true;
  for (std::size_t j = 0; j < p; ++j)
    if (coefficients[j] != 0.0) all_zero = false;
  res.denominator.resize(p);
  double den = 0.0;
  for (std::size_t m = 0; m < p; ++m) {
    const std::size_t j = res.column_order[m];
    if (all_zero || coefficients[j] != 0.0) den += 1.0;
    res.denominator[m] = den;
  }

  res.train.assign(n_train * p, 0.0);
  res.test.assign(n_test * p, 0.0);
  fill_projection(train_design, n_train, p, res.column_order, coefficients, res.denominator,
                  all_zero, res.train.data(), n_threads);
  fill_projection(test_design, n_test, p, res.column_order, coefficients, res.denominator,
                  all_zero, res.test.data(), n_threads);

  res.representative.resize(p);
  for (std::size_t m = 0; m < p; ++m) {
    res.representative[m] = m;
    for (std::size_t r = 0; r < m; ++r) {
      bool same = true;
      for (std::size_t i = 0; i < n_train && same; ++i)
        same = (res.train[i + m * n_train] == res.train[i + r * n_train]);
      for (std::size_t i = 0; i < n_test && same; ++i)
        same = (res.test[i + m * n_test] == res.test[i + r * n_test]);
      if (same) {
        res.representative[m] = r;
        break;
      }
    }
  }
  return res;
}

}  // namespace nns
