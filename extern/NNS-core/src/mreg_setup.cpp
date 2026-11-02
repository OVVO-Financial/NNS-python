// src/mreg_setup.cpp
//
// Pure C++ port of R NNS 13.3 src/NNS_mreg_setup.cpp
// (NNS_duplicate_column_map_cpp, NNS_mreg_setup_cpp). Decoupled from Rcpp.
// The reducers, findInterval ids and lexicographic RPM row order are
// preserved verbatim, including the exact-frequency "mode" reducer.
//
// SPDX-License-Identifier: GPL-3.0-only
#include "nns/mreg.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "nns/central_tendencies.hpp"

namespace nns {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

double reduce_numeric(std::vector<double>& v, int reducer, bool is_class) {
  v.erase(std::remove_if(v.begin(), v.end(), [](double x) { return !std::isfinite(x); }),
          v.end());
  if (v.empty()) return kNaN;
  std::sort(v.begin(), v.end());
  if (is_class || reducer == kMregMode) {
    double best = v[0];
    int bestn = 0;
    for (std::size_t i = 0; i < v.size();) {
      const double val = v[i];
      int cnt = 0;
      while (i < v.size() && v[i] == val) {
        ++i;
        ++cnt;
      }
      if (cnt > bestn) {
        bestn = cnt;
        best = val;
      }
    }
    return best;
  }
  if (reducer == kMregMedian) {
    const std::size_t n = v.size();
    return (n % 2) ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
  }
  if (reducer == kMregGravity) return gravity(v.data(), v.size(), false);
  double s = 0.0;
  for (double x : v) s += x;
  return s / static_cast<double>(v.size());
}

// One-based findInterval(x, b, left.open = FALSE, rightmost.closed = TRUE):
// below the first boundary -> 0; exactly the last boundary -> n - 1; above
// the last -> n. The ids are later sorted as strings, so parity here
// preserves the RPM row order.
int find_interval_one(double x, const std::vector<double>& b) {
  const int n = static_cast<int>(b.size());
  if (n == 0) return 0;
  if (x < b[0]) return 0;
  if (x == b[static_cast<std::size_t>(n - 1)]) return std::max(0, n - 1);
  return static_cast<int>(std::upper_bound(b.begin(), b.end(), x) - b.begin());
}

}  // namespace

std::vector<std::size_t> duplicate_column_map(const double* X, std::size_t n,
                                              std::size_t p) {
  std::vector<std::size_t> rep(p);
  std::unordered_map<std::string, std::vector<std::size_t>> seen;
  for (std::size_t j = 0; j < p; ++j) {
    const double* col = X + j * n;
    std::string key(reinterpret_cast<const char*>(col), n * sizeof(double));
    std::size_t r = j;
    auto it = seen.find(key);
    if (it != seen.end()) {
      for (std::size_t cand : it->second) {
        const double* other = X + cand * n;
        bool same = true;
        for (std::size_t i = 0; i < n && same; ++i) same = (col[i] == other[i]);
        if (same) {
          r = cand;
          break;
        }
      }
      it->second.push_back(j);
    } else {
      seen.emplace(std::move(key), std::vector<std::size_t>(1, j));
    }
    rep[j] = r;
  }
  return rep;
}

MregSetupResult mreg_setup(const double* X, std::size_t n, std::size_t p,
                           const double* y,
                           const std::vector<std::vector<double>>& boundaries,
                           int reducer_code, bool is_class) {
  if (boundaries.size() != p)
    throw std::invalid_argument("mreg_setup: need one boundary vector per column");

  MregSetupResult out;
  out.p = p;
  out.ids.resize(n);
  std::unordered_map<std::string, std::vector<std::size_t>> groups;
  groups.reserve(n * 2);
  for (std::size_t i = 0; i < n; ++i) {
    std::string id;
    for (std::size_t j = 0; j < p; ++j) {
      if (j) id += '.';
      id += std::to_string(find_interval_one(X[i + j * n], boundaries[j]));
    }
    groups[id].push_back(i);
    out.ids[i] = std::move(id);
  }

  out.row_ids.reserve(groups.size());
  for (const auto& kv : groups) out.row_ids.push_back(kv.first);
  std::sort(out.row_ids.begin(), out.row_ids.end());

  const std::size_t g_n = out.row_ids.size();
  out.n_groups = g_n;
  out.rpm.assign(g_n * (p + 1), 0.0);
  std::vector<double> vals;
  for (std::size_t g = 0; g < g_n; ++g) {
    const std::vector<std::size_t>& idx = groups[out.row_ids[g]];
    for (std::size_t j = 0; j < p; ++j) {
      vals.clear();
      for (std::size_t row : idx) vals.push_back(X[row + j * n]);
      out.rpm[g + j * g_n] = reduce_numeric(vals, reducer_code, false);
    }
    vals.clear();
    for (std::size_t row : idx) vals.push_back(y[row]);
    out.rpm[g + p * g_n] = reduce_numeric(vals, reducer_code, is_class);
  }
  return out;
}

}  // namespace nns
