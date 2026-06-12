// src/fast_lm.cpp
//
// Pure C++ port of NNS 13.0 fast_lm.cpp. Decoupled from Rcpp.
//
// This file preserves the numerical rules and return payloads of the original
// Rcpp functions:
//   fast_lm      -> coef, residuals, fitted.values, df.residual
//   fast_lm_mult -> coefficients, fitted.values, residuals, r.squared
//
// SPDX-License-Identifier: GPL-3.0-only
#include "nns/fast_lm.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace nns {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

inline bool is_pos(double x) {
  return x > 0.0 && std::isfinite(x);
}

inline double at(const double* M, std::size_t rows, std::size_t r, std::size_t c) {
  return M[c * rows + r];
}

inline double design_value(const double* x, std::size_t n, std::size_t row,
                           std::size_t col_with_intercept) {
  return (col_with_intercept == 0) ? 1.0 : at(x, n, row, col_with_intercept - 1U);
}

inline double mean_vec(const double* x, std::size_t n) {
  if (n == 0U) return kNaN;
  double s = 0.0;
  for (std::size_t i = 0; i < n; ++i) s += x[i];
  return s / static_cast<double>(n);
}

std::string dim_msg(const char* prefix, std::size_t a, std::size_t b) {
  return std::string(prefix) + " (got " + std::to_string(static_cast<unsigned long long>(a)) +
         " vs " + std::to_string(static_cast<unsigned long long>(b)) + ").";
}

// Cholesky decomposition of a symmetric positive-definite matrix A.
// A is column-major n x n. Returns lower triangular L such that A = L * L^T.
std::vector<double> cholesky_decomposition(const std::vector<double>& A, std::size_t n) {
  if (A.size() != n * n) {
    throw std::invalid_argument("cholesky_decomposition: matrix must be square.");
  }

  std::vector<double> L(n * n, 0.0);

  for (std::size_t i = 0; i < n; ++i) {
    double sum = A[i * n + i];
    for (std::size_t k = 0; k < i; ++k) {
      sum -= L[k * n + i] * L[k * n + i];
    }
    if (!is_pos(sum)) {
      throw std::runtime_error(
          "cholesky_decomposition: matrix not positive-definite (nonpositive pivot at " +
          std::to_string(static_cast<unsigned long long>(i + 1U)) + ").");
    }
    L[i * n + i] = std::sqrt(sum);

    const double Lii = L[i * n + i];
    for (std::size_t j = i + 1U; j < n; ++j) {
      double s = A[i * n + j];
      for (std::size_t k = 0; k < i; ++k) {
        s -= L[k * n + j] * L[k * n + i];
      }
      L[i * n + j] = s / Lii;
    }
  }

  return L;
}

// Solve L * z = b, where L is lower triangular in column-major storage.
std::vector<double> forward_substitution(const std::vector<double>& L,
                                         const std::vector<double>& b,
                                         std::size_t n) {
  if (b.size() != n || L.size() != n * n) {
    throw std::invalid_argument("forward_substitution: incompatible dimensions.");
  }

  std::vector<double> z(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    double sum = b[i];
    for (std::size_t j = 0; j < i; ++j) {
      sum -= L[j * n + i] * z[j];
    }
    const double Lii = L[i * n + i];
    if (Lii == 0.0 || !std::isfinite(Lii)) {
      throw std::runtime_error("forward_substitution: singular pivot.");
    }
    z[i] = sum / Lii;
  }
  return z;
}

// Solve L^T * x = z, where L is lower triangular in column-major storage.
std::vector<double> back_substitution(const std::vector<double>& L,
                                      const std::vector<double>& z,
                                      std::size_t n) {
  if (z.size() != n || L.size() != n * n) {
    throw std::invalid_argument("back_substitution: incompatible dimensions.");
  }

  std::vector<double> x(n, 0.0);
  for (std::size_t ii = n; ii-- > 0U;) {
    double sum = z[ii];
    for (std::size_t j = ii + 1U; j < n; ++j) {
      // L^T(ii, j) = L(j, ii).
      sum -= L[ii * n + j] * x[j];
    }
    const double Lii = L[ii * n + ii];
    if (Lii == 0.0 || !std::isfinite(Lii)) {
      throw std::runtime_error("back_substitution: singular pivot.");
    }
    x[ii] = sum / Lii;
  }
  return x;
}

}  // namespace

FastLmResult fast_lm(const double* x, const double* y, std::size_t n) {
  const double mean_x = mean_vec(x, n);
  const double mean_y = mean_vec(y, n);

  double var_x = 0.0;
  double cov_xy = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double dx = x[i] - mean_x;
    const double dy = y[i] - mean_y;
    var_x += dx * dx;
    cov_xy += dx * dy;
  }

  FastLmResult out;
  out.coef.assign(2U, 0.0);
  out.fitted_values.assign(n, kNaN);
  out.residuals.assign(n, kNaN);

  if (var_x == 0.0) {
    // Original behavior: all x identical -> slope = 0, intercept = mean(y).
    out.coef[0] = mean_y;
    out.coef[1] = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
      out.fitted_values[i] = mean_y;
      out.residuals[i] = y[i] - mean_y;
    }
  } else {
    const double slope = cov_xy / var_x;
    const double intercept = mean_y - slope * mean_x;

    out.coef[0] = intercept;
    out.coef[1] = slope;

    for (std::size_t i = 0; i < n; ++i) {
      out.fitted_values[i] = intercept + slope * x[i];
      out.residuals[i] = y[i] - out.fitted_values[i];
    }
  }

  // Match original integer rule: ny - 2, even for short inputs.
  out.df_residual = static_cast<long long>(n) - 2LL;
  return out;
}

FastLmMultResult fast_lm_mult(const double* x, const double* y,
                              std::size_t n, std::size_t p) {
  if (n == 0U) {
    throw std::invalid_argument("fast_lm_mult: 'x' has zero rows.");
  }
  if (p == 0U) {
    throw std::invalid_argument("fast_lm_mult: 'x' has zero columns.");
  }
  const std::size_t q = p + 1U;

  // Compute X'X and X'y for the design matrix [1, x].  Storage is column-major,
  // matching R's NumericMatrix memory layout and the rest of the rendered core.
  std::vector<double> XtX(q * q, 0.0);
  std::vector<double> Xty(q, 0.0);

  for (std::size_t i = 0; i < q; ++i) {
    for (std::size_t j = 0; j <= i; ++j) {
      double s = 0.0;
      for (std::size_t k = 0; k < n; ++k) {
        s += design_value(x, n, k, i) * design_value(x, n, k, j);
      }
      XtX[j * q + i] = s;
      if (i != j) XtX[i * q + j] = s;
    }

    double sy = 0.0;
    for (std::size_t k = 0; k < n; ++k) {
      sy += design_value(x, n, k, i) * y[k];
    }
    Xty[i] = sy;
  }

  const std::vector<double> L = cholesky_decomposition(XtX, q);
  const std::vector<double> z = forward_substitution(L, Xty, q);
  std::vector<double> coef = back_substitution(L, z, q);

  std::vector<double> fitted_values(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    double s = 0.0;
    for (std::size_t j = 0; j < q; ++j) {
      s += coef[j] * design_value(x, n, i, j);
    }
    fitted_values[i] = s;
  }

  std::vector<double> residuals(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) residuals[i] = y[i] - fitted_values[i];

  const double y_mean = mean_vec(y, n);
  double TSS = 0.0;
  double RSS = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double dy = y[i] - y_mean;
    TSS += dy * dy;
    const double re = residuals[i];
    RSS += re * re;
  }

  FastLmMultResult out;
  out.coefficients = std::move(coef);
  out.fitted_values = std::move(fitted_values);
  out.residuals = std::move(residuals);
  out.r_squared = (TSS == 0.0) ? kNaN : (1.0 - RSS / TSS);
  return out;
}

}  // namespace nns
