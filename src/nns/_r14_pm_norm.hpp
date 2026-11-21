// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "nns/partial_moments.hpp"

namespace nns_r14 {

using NDMethod = decltype(&nns::clpm_nd);
double nd_moment(NDMethod moment, const double* data, std::size_t n, std::size_t d,
                 const double* target, double degree, bool norm, int threads);
void clpm_nd_batch(const double* data, std::size_t n, std::size_t d, const double* targets,
                   std::size_t m, double degree, bool norm, bool strict, double* out, int threads);
nns::PMMatrixResult pm_matrix(double lower, double upper, const double* target, const double* data,
                              std::size_t n, std::size_t d, bool pop_adj, bool norm, int threads);

}  // namespace nns_r14
