// R14 NNS.seas lag scan: declaration. The definition lives in _r14_seas.cpp,
// which builds into the NNS-core library target with its Release optimisation
// (the nanobind module itself is compiled for size).
// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include <cstddef>
#include <vector>

namespace nns_r14 {

// R's seas_score (src/NNS_seas.cpp) for each requested lag of the series x.
std::vector<double> seas_scores(const double* x, int n, const int* lags, std::size_t n_lags,
                                bool use_cv, int n_threads);

}  // namespace nns_r14
