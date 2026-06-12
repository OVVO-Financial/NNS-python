// include/nns/partial_moments.hpp
//
// SPDX-License-Identifier: GPL-3.0-only
#ifndef NNS_PARTIAL_MOMENTS_HPP
#define NNS_PARTIAL_MOMENTS_HPP

#include <cstddef>
#include <vector>

namespace nns {

struct PMMatrixResult {
  std::vector<double> cupm; // column-major d x d
  std::vector<double> dupm; // column-major d x d
  std::vector<double> dlpm; // column-major d x d
  std::vector<double> clpm; // column-major d x d
  std::vector<double> cov;  // column-major d x d
  std::size_t dim = 0;
};

double lpm(double degree, double target, const double* x, std::size_t n);
double upm(double degree, double target, const double* x, std::size_t n);

void lpm_v(double degree, const double* target, std::size_t n_targets,
           const double* x, std::size_t n, double* out, int n_threads = -1);
void upm_v(double degree, const double* target, std::size_t n_targets,
           const double* x, std::size_t n, double* out, int n_threads = -1);
void lpm_ratio_v(double degree, const double* target, std::size_t n_targets,
                 const double* x, std::size_t n, double* out,
                 int n_threads = -1);
void upm_ratio_v(double degree, const double* target, std::size_t n_targets,
                 const double* x, std::size_t n, double* out,
                 int n_threads = -1);

double co_upm(double degree_x, double degree_y, const double* x,
              const double* y, std::size_t n_x, std::size_t n_y,
              double target_x, double target_y);
double co_lpm(double degree_x, double degree_y, const double* x,
              const double* y, std::size_t n_x, std::size_t n_y,
              double target_x, double target_y);
double d_lpm(double degree_lpm, double degree_upm, const double* x,
             const double* y, std::size_t n_x, std::size_t n_y,
             double target_x, double target_y);
double d_upm(double degree_lpm, double degree_upm, const double* x,
             const double* y, std::size_t n_x, std::size_t n_y,
             double target_x, double target_y);

void co_lpm_v(double degree_x, double degree_y, const double* x,
              const double* y, std::size_t n_x, std::size_t n_y,
              const double* target_x, std::size_t n_target_x,
              const double* target_y, std::size_t n_target_y, double* out,
              int n_threads = -1);
void co_upm_v(double degree_x, double degree_y, const double* x,
              const double* y, std::size_t n_x, std::size_t n_y,
              const double* target_x, std::size_t n_target_x,
              const double* target_y, std::size_t n_target_y, double* out,
              int n_threads = -1);
void d_lpm_v(double degree_lpm, double degree_upm, const double* x,
             const double* y, std::size_t n_x, std::size_t n_y,
             const double* target_x, std::size_t n_target_x,
             const double* target_y, std::size_t n_target_y, double* out,
             int n_threads = -1);
void d_upm_v(double degree_lpm, double degree_upm, const double* x,
             const double* y, std::size_t n_x, std::size_t n_y,
             const double* target_x, std::size_t n_target_x,
             const double* target_y, std::size_t n_target_y, double* out,
             int n_threads = -1);

/// n-dimensional partial moments. data is n x d column-major.
double clpm_nd(const double* data, std::size_t n, std::size_t d,
               const double* target, double degree, bool norm,
               int n_threads = -1);
double cupm_nd(const double* data, std::size_t n, std::size_t d,
               const double* target, double degree, bool norm,
               int n_threads = -1);
double dpm_nd(const double* data, std::size_t n, std::size_t d,
              const double* target, double degree, bool norm,
              int n_threads = -1);

void clpm_nd_batch(const double* data, std::size_t n, std::size_t d,
                   const double* targets, std::size_t n_targets, double degree,
                   bool norm, double* out, int n_threads = -1);

PMMatrixResult pm_matrix(double degree_lpm, double degree_upm,
                         const double* target, const double* variable,
                         std::size_t n, std::size_t d, bool pop_adj, bool norm,
                         int n_threads = -1);

} // namespace nns

#endif // NNS_PARTIAL_MOMENTS_HPP
