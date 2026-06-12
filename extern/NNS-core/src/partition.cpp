// src/partition.cpp
//
// Pure C++ port of NNS 13.0 NNS_part.cpp. Decoupled from Rcpp.
//
// This file preserves the original NNS_part_cpp semantics:
//   - arguments: x, y, type, order_in, obs_req, min_obs_stop,
//     noise_reduction, quadrants_only
//   - quadrant labels and x-only labels
//   - prior.quadrant tracking
//   - xonly detection from non-null type, exactly as upstream
//   - order_in and min_obs_stop stopping rules
//   - noise_reduction choices: mean, median, mode, mode_class, gravity default
//   - return payload: order, dt, regression.points, segments_h, segments_v,
//     vlines, or only quadrant when quadrants_only is true
//
// SPDX-License-Identifier: GPL-3.0-only
#include "nns/partition.hpp"
#include "nns/central_tendencies.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace nns {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

inline double mean_no_na(const std::vector<double>& v) {
  long double s = 0.0L;
  std::size_t m = 0;
  for (double xi : v) {
    if (std::isfinite(xi)) {
      s += xi;
      ++m;
    }
  }
  return m ? static_cast<double>(s / static_cast<long double>(m)) : kNaN;
}

inline double median_no_na(const std::vector<double>& v) {
  std::vector<double> a;
  a.reserve(v.size());
  for (double xi : v) {
    if (std::isfinite(xi)) a.push_back(xi);
  }
  if (a.empty()) return kNaN;

  const std::size_t n = a.size();
  std::nth_element(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(n / 2U), a.end());
  const double hi = a[n / 2U];
  if (n & 1U) return hi;

  const auto lm = std::max_element(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(n / 2U));
  return (*lm + hi) * 0.5;
}

inline std::string lower_ascii(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

struct Agg {
  std::string noise;

  double mode_disc_single(const std::vector<double>& v) const {
    const std::vector<double> m = nns::mode(v.data(), v.size(), true, false);
    return m.empty() ? kNaN : m[0];
  }

  double gravity_cont(const std::vector<double>& v, bool discrete = false) const {
    return nns::gravity(v.data(), v.size(), discrete);
  }

  double for_x(const std::vector<double>& v) const {
    if (noise == "mean") return mean_no_na(v);
    if (noise == "median") return median_no_na(v);
    if (noise == "mode") return mode_disc_single(v);
    if (noise == "mode_class") return gravity_cont(v, false);
    return gravity_cont(v, false);
  }

  double for_y(const std::vector<double>& v) const {
    if (noise == "mean") return mean_no_na(v);
    if (noise == "median") return median_no_na(v);
    if (noise == "mode") return mode_disc_single(v);
    if (noise == "mode_class") return mode_disc_single(v);
    return gravity_cont(v, false);
  }
};

struct Pair {
  double x;
  double y;
};

}  // namespace

PartitionResult partition(const double* x,
                          const double* y,
                          std::size_t n,
                          const std::optional<std::string>& type,
                          const std::optional<int>& order_in,
                          int obs_req,
                          bool min_obs_stop,
                          const std::string& noise_reduction,
                          bool quadrants_only) {
  PartitionResult out;

  const int ni = static_cast<int>(n);
  const int default_order = std::max(static_cast<int>(std::ceil(std::log2(std::max(1, ni)))), 1);
  int max_order = order_in.has_value() ? *order_in : default_order;
  if (max_order == 0) max_order = 1;

  // Upstream uses type.isNotNull() only. The value of type is irrelevant here.
  const bool xonly = type.has_value();
  const Agg agg{lower_ascii(noise_reduction)};

  std::vector<std::string> quadrant(n, "q");
  std::vector<std::string> prior_quadrant(n, "pq");
  int depth = 0;

  std::vector<double> H_x0;
  std::vector<double> H_x1;
  std::vector<double> H_y;
  std::vector<double> V_x;
  std::vector<double> V_y0;
  std::vector<double> V_y1;
  std::vector<double> V_lines;

  while (true) {
    if (depth >= max_order) break;
    if (depth >= static_cast<int>(std::floor(std::log2(std::max(1, ni))))) break;

    std::unordered_map<std::string, std::vector<int>> grp;
    grp.reserve(n * 2U);
    for (std::size_t i = 0; i < n; ++i) {
      grp[quadrant[i]].push_back(static_cast<int>(i));
    }

    std::vector<std::string> to_split;
    to_split.reserve(grp.size());
    for (auto& kv : grp) {
      if (static_cast<int>(kv.second.size()) > obs_req) to_split.push_back(kv.first);
    }
    if (to_split.empty()) break;

    std::unordered_map<std::string, Pair> centers;
    centers.reserve(to_split.size());

    for (const auto& q : to_split) {
      const auto& idx = grp[q];

      std::vector<double> xv(idx.size());
      std::vector<double> yv(idx.size());

      double minx = std::numeric_limits<double>::infinity();
      double maxx = -std::numeric_limits<double>::infinity();
      double miny = std::numeric_limits<double>::infinity();
      double maxy = -std::numeric_limits<double>::infinity();

      for (std::size_t k = 0; k < idx.size(); ++k) {
        const int i = idx[k];
        const double xi = x[i];
        const double yi = y[i];
        xv[k] = xi;
        yv[k] = yi;

        if (std::isfinite(xi)) {
          if (xi < minx) minx = xi;
          if (xi > maxx) maxx = xi;
        }
        if (std::isfinite(yi)) {
          if (yi < miny) miny = yi;
          if (yi > maxy) maxy = yi;
        }
      }

      const Pair c{agg.for_x(xv), agg.for_y(yv)};
      centers[q] = c;

      if (!xonly) {
        if (std::isfinite(c.y) && std::isfinite(minx) && std::isfinite(maxx)) {
          H_x0.push_back(minx);
          H_x1.push_back(maxx);
          H_y.push_back(c.y);
        }
        if (std::isfinite(c.x) && std::isfinite(miny) && std::isfinite(maxy)) {
          V_x.push_back(c.x);
          V_y0.push_back(miny);
          V_y1.push_back(maxy);
        }
      }
    }

    if (xonly && !quadrants_only) {
      for (auto& kv : grp) {
        const auto& idx = kv.second;
        double minx = std::numeric_limits<double>::infinity();
        double maxx = -std::numeric_limits<double>::infinity();
        for (int i : idx) {
          const double xi = x[i];
          if (std::isfinite(xi)) {
            if (xi < minx) minx = xi;
            if (xi > maxx) maxx = xi;
          }
        }
        if (std::isfinite(minx)) V_lines.push_back(minx);
        if (std::isfinite(maxx)) V_lines.push_back(maxx);
      }
    }

    for (const auto& q : to_split) {
      const Pair c = centers[q];
      for (int i : grp[q]) {
        prior_quadrant[static_cast<std::size_t>(i)] = quadrant[static_cast<std::size_t>(i)];

        int qn = 1;
        if (!xonly) {
          const int lox = (std::isfinite(x[i]) && std::isfinite(c.x)) ? (x[i] <= c.x) : 0;
          const int loy = (std::isfinite(y[i]) && std::isfinite(c.y)) ? (y[i] <= c.y) : 0;
          qn = 1 + lox + 2 * loy;
        } else {
          const int lox = (std::isfinite(x[i]) && std::isfinite(c.x)) ? (x[i] > c.x) : 0;
          qn = 1 + lox;
        }

        quadrant[static_cast<std::size_t>(i)] += static_cast<char>('0' + qn);
      }
    }

    ++depth;

    if (min_obs_stop) {
      std::unordered_map<std::string, int> cnt;
      cnt.reserve(n * 2U);
      for (const auto& qstr : quadrant) ++cnt[qstr];

      int minc = ni;
      for (auto& kv : cnt) {
        if (kv.second < minc) minc = kv.second;
      }
      if (minc <= obs_req) break;
    }
  }

  out.order = depth;
  out.quadrant = quadrant;

  if (quadrants_only) {
    out.quadrants_only = true;
    return out;
  }

  out.quadrants_only = false;
  out.dt.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    out.dt.push_back({x[i], y[i], quadrant[i], prior_quadrant[i]});
  }

  std::unordered_map<std::string, std::vector<int>> by_prior;
  by_prior.reserve(n * 2U);
  for (std::size_t i = 0; i < n; ++i) {
    by_prior[prior_quadrant[i]].push_back(static_cast<int>(i));
  }

  out.regression_points.reserve(by_prior.size());
  for (auto& kv : by_prior) {
    const auto& idx = kv.second;
    std::vector<double> xv(idx.size());
    std::vector<double> yv(idx.size());
    for (std::size_t k = 0; k < idx.size(); ++k) {
      const int i = idx[k];
      xv[k] = x[i];
      yv[k] = y[i];
    }
    out.regression_points.push_back({kv.first, agg.for_x(xv), agg.for_y(yv)});
  }

  out.segments_h.reserve(H_x0.size());
  for (std::size_t i = 0; i < H_x0.size(); ++i) {
    out.segments_h.push_back({H_x0[i], H_x1[i], H_y[i]});
  }

  out.segments_v.reserve(V_x.size());
  for (std::size_t i = 0; i < V_x.size(); ++i) {
    out.segments_v.push_back({V_x[i], V_y0[i], V_y1[i]});
  }

  out.vlines = std::move(V_lines);
  return out;
}

}  // namespace nns
