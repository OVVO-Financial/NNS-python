// R14 NNS.seas lag scan.
// Source: OVVO-Financial/NNS@7b07a837c130ba7220dbe16c584221508757d5ae
// src/NNS_seas.cpp: SeasComponent, seas_score and the SeasScanWorker split,
// copied verbatim. RcppParallel::parallelFor becomes nns::parallel_for and
// R vectors become plain arrays; each lag's arithmetic is R's, in R's order,
// and every lag writes its own output slot, so results never depend on the
// thread count.
// SPDX-License-Identifier: GPL-3.0-only
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

#include "nns/parallel.hpp"
#include "_r14_seas.hpp"

namespace nns_r14 {

// Running state of one component series while the sorted series is walked.
struct SeasComponent { int rank; int len; double first; double last; double sum; double acc; };

// Score of lag m: the length-weighted mean over its m component series of the
// Gini mean difference, divided by the component mean when use_cv. xs holds the
// series in increasing order and pos the original index of each value, so each
// value lands in its component (pos mod m) at that component's next rank.
static double seas_score(int m, int n, const double* xs, const int* pos, bool use_cv,
                         std::vector<SeasComponent>& comp){
  if ((int)comp.size() < m) comp.resize(m);
  const double inv_m = 1.0 / (double)m;
  for(int j=0;j<m;++j){ SeasComponent& c = comp[j]; c.rank = 0; c.len = (n-1-j)/m + 1; c.sum = 0.0; c.acc = 0.0; }
  for(int t=0;t<n;++t){
    const int i = pos[t];
    int j = i - m * (int)((double)i * inv_m);   // i mod m without a division
    if (j >= m) j -= m; else if (j < 0) j += m;
    const double v = xs[t];
    SeasComponent& c = comp[j];
    if (c.rank == 0) c.first = v;
    c.last = v;
    // The weights sum to 0, so measuring from the component minimum avoids cancelling a large level.
    c.acc += (2.0 * c.rank + 1.0 - c.len) * (v - c.first);
    c.sum += v;
    ++c.rank;
  }
  double wsum = 0.0;
  for(int j=0;j<m;++j){
    const SeasComponent& c = comp[j];
    const double k = (double)c.len;
    // a constant component (smallest value = largest) has dispersion 0
    const double g = (c.len < 2 || c.first == c.last) ? 0.0 : std::max(0.0, 2.0 * c.acc / (k * (k - 1.0)));
    wsum += k * (use_cv ? (g == 0.0 ? 0.0 : g / std::fabs(c.sum / k)) : g);
  }
  return wsum / (double)n;
}

// seas_score for each requested lag. One stable sort of the series gives every
// component's order statistics (R's NNS_seas_cpp); the lags are then scored in
// parallel, each chunk with its own component buffer (R's SeasScanWorker).
std::vector<double> seas_scores(const double* x, int n, const int* lags,
                                       std::size_t n_lags, bool use_cv, int n_threads){
  std::vector<int> ord(n);
  std::iota(ord.begin(), ord.end(), 0);
  std::stable_sort(ord.begin(), ord.end(), [&](int a, int b){ return x[a] < x[b]; });
  std::vector<double> xs(n);
  for(int t=0;t<n;++t) xs[t] = x[ord[t]];
  std::vector<double> out(n_lags);
  nns::parallel_for(
      0, n_lags,
      [&](std::size_t begin, std::size_t end){
        std::vector<SeasComponent> comp;
        for(std::size_t idx = begin; idx < end; ++idx)
          out[idx] = seas_score((int)lags[idx], n, xs.data(), ord.data(), use_cv, comp);
      },
      n_threads, static_cast<std::size_t>(n));
  return out;
}

}  // namespace nns_r14
