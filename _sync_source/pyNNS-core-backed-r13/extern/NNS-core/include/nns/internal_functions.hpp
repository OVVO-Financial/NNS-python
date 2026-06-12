// include/nns/internal_functions.hpp
//
// SPDX-License-Identifier: GPL-3.0-only
#ifndef NNS_INTERNAL_FUNCTIONS_HPP
#define NNS_INTERNAL_FUNCTIONS_HPP

#include <cstddef>
#include <string>
#include <vector>

namespace nns {

// --- Basic Utilities ---

// Pure-C++ representation of the original R factor/string/logical class check.
enum class ValueKind { Numeric, Integer, Logical, String, Factor };

bool is_fcl(ValueKind kind);

// Pure-C++ factor representation. Codes are 1-based like R factors; code 0
// represents NA. Levels preserve original R ordering.
struct Factor {
  std::vector<int> codes;
  std::vector<std::string> levels;
};

struct DummyMatrix {
  std::vector<double> data;      // column-major matrix data
  std::vector<std::string> names; // column names in original R level order
  std::size_t nrow = 0;
  std::size_t ncol = 0;
};

// Equivalent to factor_2_dummy: drops the first level when more than one
// factor level is present in the data, preserving R's 1-based factor codes.
DummyMatrix factor_2_dummy(const Factor& factor);

// Equivalent to factor_2_dummy_FR: full-rank dummy expansion retaining every
// level column, preserving level/column ordering.
DummyMatrix factor_2_dummy_fr(const Factor& factor);


double vec_sd(const double* x, std::size_t n);
std::vector<double> col_sd(const double* X, std::size_t n, std::size_t p);
bool is_discrete(const double* x, std::size_t n);

// --- Time Series Vector Generation ---

struct TimeSeriesVectors {
  std::vector<std::vector<double>> series;
  std::vector<std::vector<int>> index;
};

struct ForecastVectors {
  std::vector<std::vector<double>> series;
  std::vector<std::vector<int>> index;
  std::vector<std::vector<double>> forecast_values;
  std::vector<std::vector<int>> forecast_index;
};

TimeSeriesVectors generate_vectors(const double* x, std::size_t n, const int* lags, std::size_t num_lags);
ForecastVectors generate_lin_vectors(const double* x, std::size_t n, int l, int h = 1);

// --- ARMA Seasonality Weighting ---

struct ARMAWeights {
  std::vector<double> lags;
  std::vector<double> weights;
};

/// Computes ARMA seasonality weighting. 
/// Replaces the dynamic R DataFrame lookup with explicit arrays.
ARMAWeights arma_seas_weighting(const double* periods, const double* covar, const double* varcovar, std::size_t m);

// --- Maximum Entropy Bootstrap (MEBoot) ---

std::vector<double> meboot_part(const double* xx, std::size_t m, std::size_t n, 
                                const double* z, std::size_t z_len,
                                double xmin, double xmax, 
                                const double* desintxb, bool reachbnd, int seed = 123);

void meboot_expand_sd(double* ensemble, std::size_t n, std::size_t J, 
                      const double* orig_sd, std::size_t orig_p, double fiv = 5.0, int seed = 123);

void force_clt(double* ensemble, std::size_t n, std::size_t J, 
               double orig_gm, const double* orig_sd, std::size_t orig_p);

// --- Class Sampling ---

struct SampleResult {
  std::vector<double> x; // Balanced column-major matrix
  std::vector<int> y;    // Balanced class labels
  std::size_t n;         // New number of rows
  std::size_t p;         // Number of columns
};

SampleResult up_sample(const double* X, const int* y, std::size_t n, std::size_t p, int seed = 123);
SampleResult down_sample(const double* X, const int* y, std::size_t n, std::size_t p, int seed = 123);

} // namespace nns

#endif // NNS_INTERNAL_FUNCTIONS_HPP
