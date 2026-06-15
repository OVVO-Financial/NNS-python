#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "nns/nns.hpp"

namespace nb = nanobind;

namespace {

using Vector = nb::ndarray<const double, nb::ndim<1>, nb::c_contig>;
using IntVector = nb::ndarray<const int, nb::ndim<1>, nb::c_contig>;
using Matrix = nb::ndarray<nb::numpy, double, nb::shape<-1, -1>, nb::f_contig>;

std::size_t checked_size(const Vector& x, const char* name) {
  const std::size_t n = x.shape(0);
  if (n == 0U) {
    throw std::invalid_argument(std::string(name) + " must be non-empty.");
  }
  return n;
}

Matrix matrix_from_column_major_vector(std::vector<double>&& values, std::size_t dim) {
  auto* storage = new std::vector<double>(std::move(values));
  nb::capsule owner(storage, [](void* p) noexcept {
    delete static_cast<std::vector<double>*>(p);
  });
  return Matrix(storage->data(), {dim, dim}, owner, {1, static_cast<int64_t>(dim)});
}

void check_same_size(const Vector& x, const Vector& y, const char* x_name, const char* y_name) {
  if (checked_size(x, x_name) != checked_size(y, y_name)) {
    throw std::invalid_argument(std::string(x_name) + " and " + y_name + " must have the same length.");
  }
}


double lpm_sequence(double degree, double target, const std::vector<double>& x) {
  if (x.empty()) {
    throw std::invalid_argument("x must be non-empty.");
  }
  return nns::lpm(degree, target, x.data(), x.size());
}

double upm_sequence(double degree, double target, const std::vector<double>& x) {
  if (x.empty()) {
    throw std::invalid_argument("x must be non-empty.");
  }
  return nns::upm(degree, target, x.data(), x.size());
}

std::size_t checked_flat_matrix_size(const Vector& x, std::size_t n, std::size_t p, const char* name) {
  if (n == 0U || p == 0U) {
    throw std::invalid_argument(std::string(name) + " dimensions must be non-empty.");
  }
  const std::size_t expected = n * p;
  if (x.shape(0) != expected) {
    throw std::invalid_argument(std::string(name) + " length must equal n * p.");
  }
  return expected;
}

std::vector<double> moment_ratio_vector(bool lower, double degree, const Vector& targets, const Vector& x) {
  const std::size_t n = checked_size(x, "x");
  const std::size_t n_targets = targets.shape(0);
  std::vector<double> out(n_targets, 0.0);
  if (n_targets == 0U) {
    return out;
  }
  if (lower) {
    nns::lpm_ratio_v(degree, targets.data(), n_targets, x.data(), n, out.data());
  } else {
    nns::upm_ratio_v(degree, targets.data(), n_targets, x.data(), n, out.data());
  }
  return out;
}

std::vector<double> moment_vector(bool lower, double degree, const Vector& targets, const Vector& x) {
  const std::size_t n = checked_size(x, "x");
  const std::size_t n_targets = targets.shape(0);
  std::vector<double> out(n_targets, 0.0);
  if (n_targets == 0U) {
    return out;
  }
  if (lower) {
    nns::lpm_v(degree, targets.data(), n_targets, x.data(), n, out.data());
  } else {
    nns::upm_v(degree, targets.data(), n_targets, x.data(), n, out.data());
  }
  return out;
}

std::vector<double> co_moment_vector(const std::string& kind,
                                     double degree_x,
                                     double degree_y,
                                     const Vector& x,
                                     const Vector& y,
                                     const Vector& target_x,
                                     const Vector& target_y) {
  const std::size_t n_x = checked_size(x, "x");
  const std::size_t n_y = checked_size(y, "y");
  if (n_x != n_y) {
    throw std::invalid_argument("x and y must have the same length.");
  }
  const std::size_t n_target_x = target_x.shape(0);
  const std::size_t n_target_y = target_y.shape(0);
  const std::size_t n_out = n_target_x > n_target_y ? n_target_x : n_target_y;
  std::vector<double> out(n_out, 0.0);
  if (n_out == 0U) {
    return out;
  }
  if (kind == "co_lpm") {
    nns::co_lpm_v(degree_x, degree_y, x.data(), y.data(), n_x, n_y, target_x.data(), n_target_x,
                  target_y.data(), n_target_y, out.data());
  } else if (kind == "co_upm") {
    nns::co_upm_v(degree_x, degree_y, x.data(), y.data(), n_x, n_y, target_x.data(), n_target_x,
                  target_y.data(), n_target_y, out.data());
  } else if (kind == "d_lpm") {
    nns::d_lpm_v(degree_x, degree_y, x.data(), y.data(), n_x, n_y, target_x.data(), n_target_x,
                 target_y.data(), n_target_y, out.data());
  } else if (kind == "d_upm") {
    nns::d_upm_v(degree_x, degree_y, x.data(), y.data(), n_x, n_y, target_x.data(), n_target_x,
                 target_y.data(), n_target_y, out.data());
  } else {
    throw std::invalid_argument("unknown co-moment kind.");
  }
  return out;
}

nb::dict pm_matrix_dict(double degree_lpm,
                        double degree_upm,
                        const Vector& target,
                        const Vector& variable,
                        std::size_t n,
                        std::size_t d,
                        bool pop_adj,
                        bool norm) {
  if (target.shape(0) != d) {
    throw std::invalid_argument("target length must equal d.");
  }
  checked_flat_matrix_size(variable, n, d, "variable");
  nns::PMMatrixResult result = nns::pm_matrix(degree_lpm, degree_upm, target.data(),
                                              variable.data(), n, d, pop_adj, norm);
  nb::dict out;
  out["cupm"] = matrix_from_column_major_vector(std::move(result.cupm), result.dim);
  out["dupm"] = matrix_from_column_major_vector(std::move(result.dupm), result.dim);
  out["dlpm"] = matrix_from_column_major_vector(std::move(result.dlpm), result.dim);
  out["clpm"] = matrix_from_column_major_vector(std::move(result.clpm), result.dim);
  out["cov.matrix"] = matrix_from_column_major_vector(std::move(result.cov), result.dim);
  out["dim"] = result.dim;
  return out;
}

nb::dict fast_lm_dict(const Vector& x, const Vector& y) {
  const std::size_t n = checked_size(x, "x");
  if (y.shape(0) != n) {
    throw std::invalid_argument("x and y must have the same length.");
  }
  const nns::FastLmResult result = nns::fast_lm(x.data(), y.data(), n);
  nb::dict out;
  out["coef"] = result.coef;
  out["fitted_values"] = result.fitted_values;
  out["residuals"] = result.residuals;
  out["df_residual"] = result.df_residual;
  return out;
}

nb::dict fast_lm_mult_dict(const Vector& x, const Vector& y, std::size_t n, std::size_t p) {
  checked_flat_matrix_size(x, n, p, "x");
  if (y.shape(0) != n) {
    throw std::invalid_argument("y length must equal n.");
  }
  const nns::FastLmMultResult result = nns::fast_lm_mult(x.data(), y.data(), n, p);
  nb::dict out;
  out["coefficients"] = result.coefficients;
  out["fitted_values"] = result.fitted_values;
  out["residuals"] = result.residuals;
  out["r_squared"] = result.r_squared;
  return out;
}

nb::dict dummy_matrix_dict(const std::vector<int>& codes, const std::vector<std::string>& levels, bool full_rank) {
  const nns::Factor factor{codes, levels};
  const nns::DummyMatrix result = full_rank ? nns::factor_2_dummy_fr(factor) : nns::factor_2_dummy(factor);
  nb::dict out;
  out["data"] = result.data;
  out["names"] = result.names;
  out["nrow"] = result.nrow;
  out["ncol"] = result.ncol;
  return out;
}

nb::dict time_series_vectors_dict(const Vector& x, const IntVector& lags) {
  const nns::TimeSeriesVectors result = nns::generate_vectors(x.data(), checked_size(x, "x"),
                                                              lags.data(), lags.shape(0));
  nb::dict out;
  out["series"] = result.series;
  out["index"] = result.index;
  return out;
}

nb::dict forecast_vectors_dict(const Vector& x, int l, int h) {
  const nns::ForecastVectors result = nns::generate_lin_vectors(x.data(), checked_size(x, "x"), l, h);
  nb::dict out;
  out["series"] = result.series;
  out["index"] = result.index;
  out["forecast_values"] = result.forecast_values;
  out["forecast_index"] = result.forecast_index;
  return out;
}

nb::dict stochastic_superiority_dict(const Vector& x, const Vector& y) {
  const nns::StochSupResult result = nns::stochastic_superiority(
      x.data(), checked_size(x, "x"), y.data(), checked_size(y, "y"));
  nb::dict out;
  out["p_gt"] = result.p_gt;
  out["p_tie"] = result.p_tie;
  out["p_star"] = result.p_star;
  return out;
}

// Bit-faithful port of the pure-Python NNS gravity central tendency
// (`nns.dependence._gravity` and its helpers `_quartiles_like_r_code`,
// `_interpolate_position`, `_simple_bin_counts`). This is intentionally a
// separate symbol from the vendored `nns::gravity`, which derives from a
// different revision and diverges from the R-faithful Python the parity suite
// enforces. Non-finite values are filtered and the remainder is sorted, exactly
// as the Python does, so the caller may pass a raw array.
// Forward declarations: gravity reuses numpy-faithful pairwise summation so its
// internal mean/weighted sums are bit-identical to np.mean/np.sum (a ~1e-11
// difference is enough to flip a discrete partition split downstream).
double np_pairwise_sum(const double* a, std::size_t n);
double np_mean(const double* a, std::size_t n);

double gravity_at_position(const std::vector<double>& sorted, double position) {
  const std::size_t n = sorted.size();
  const long floor_pos = std::min<long>(static_cast<long>(n),
                                         std::max<long>(1L, static_cast<long>(std::floor(position))));
  const long ceil_pos = std::min<long>(static_cast<long>(n),
                                        std::max<long>(1L, static_cast<long>(std::ceil(position))));
  const double weight = position - std::floor(position);
  const double lo = sorted[static_cast<std::size_t>(floor_pos) - 1U];
  const double hi = sorted[static_cast<std::size_t>(ceil_pos) - 1U];
  return lo + weight * (hi - lo);
}

double gravity_exact_impl(const double* data, std::size_t raw_n) {
  std::vector<double> values;
  values.reserve(raw_n);
  for (std::size_t i = 0; i < raw_n; ++i) {
    if (std::isfinite(data[i])) {
      values.push_back(data[i]);
    }
  }
  const std::size_t n = values.size();
  if (n == 0U) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  std::sort(values.begin(), values.end());
  if (n <= 3U) {
    if (n == 1U) return values[0];
    if (n == 2U) return 0.5 * (values[0] + values[1]);
    return values[1];  // numpy median of 3 sorted values
  }
  if (values.front() == values.back()) {
    return values[0];
  }
  const double value_range = values.back() - values.front();
  if (value_range == 0.0) {
    return values[0];
  }

  // _quartiles_like_r_code
  const double dn = static_cast<double>(n);
  const double p25 = dn * 0.25, p50 = dn * 0.50, p75 = dn * 0.75;
  double q1, q2, q3;
  if (n % 2U == 0U) {
    q1 = values[static_cast<std::size_t>(std::max<long>(1L, static_cast<long>(std::floor(p25)))) - 1U];
    q2 = values[static_cast<std::size_t>(std::max<long>(1L, static_cast<long>(std::floor(p50)))) - 1U];
    q3 = values[static_cast<std::size_t>(std::max<long>(1L, static_cast<long>(std::floor(p75)))) - 1U];
  } else {
    q1 = gravity_at_position(values, p25);
    const long f50 = std::min<long>(static_cast<long>(n), std::max<long>(1L, static_cast<long>(std::floor(p50))));
    const long c50 = std::min<long>(static_cast<long>(n), std::max<long>(1L, static_cast<long>(std::ceil(p50))));
    q2 = 0.5 * (values[static_cast<std::size_t>(f50) - 1U] + values[static_cast<std::size_t>(c50) - 1U]);
    q3 = gravity_at_position(values, p75);
  }

  double width = (q3 - q1) * std::pow(dn, -0.5);
  if (!(width > 0.0) || !std::isfinite(width)) {
    width = value_range / 128.0;
  }

  // _simple_bin_counts
  const double origin = values.front();
  const double int_max = static_cast<double>(std::numeric_limits<int32_t>::max());
  long bin_count;
  if (!(width > 0.0) || !std::isfinite(width)) {
    bin_count = 1;
  } else {
    const double bin_ratio = (values.back() - origin) / width + 1e-12;
    if (!std::isfinite(bin_ratio) || bin_ratio > int_max) {
      bin_count = 1;
    } else {
      bin_count = static_cast<long>(std::floor(bin_ratio)) + 1;
    }
  }
  bin_count = std::min<long>(std::max<long>(1L, bin_count), static_cast<long>(4U * n));
  const std::size_t nb = static_cast<std::size_t>(bin_count);

  std::vector<int64_t> counts(nb, 0);
  if (nb == 1U) {
    counts[0] = static_cast<int64_t>(n);
  } else {
    for (const double v : values) {
      long idx = static_cast<long>(std::floor((v - origin) / width));
      idx = std::min<long>(std::max<long>(0L, idx), bin_count - 1);
      counts[static_cast<std::size_t>(idx)] += 1;
    }
  }

  int64_t max_count = counts[0];
  for (const int64_t c : counts) {
    if (c > max_count) max_count = c;
  }
  std::size_t n_max = 0, last_max = 0;
  for (std::size_t i = 0; i < nb; ++i) {
    if (counts[i] == max_count) {
      ++n_max;
      last_max = i;
    }
  }
  std::size_t lo, hi;
  if (n_max == 1U) {
    const long center = static_cast<long>(last_max);
    lo = static_cast<std::size_t>(std::max<long>(0L, center - 1));
    hi = static_cast<std::size_t>(std::min<long>(bin_count - 1, center + 1));
  } else {
    lo = 0;
    hi = nb - 1U;
  }

  std::vector<double> products;
  products.reserve(hi - lo + 1U);
  double denom = 0.0;
  for (std::size_t i = lo; i <= hi; ++i) {
    denom += static_cast<double>(counts[i]);
    products.push_back((origin + static_cast<double>(i) * width) * static_cast<double>(counts[i]));
  }
  const double weighted = np_pairwise_sum(products.data(), products.size());
  const double mode_gravity =
      denom > 0.0 ? weighted / denom : origin + static_cast<double>((lo + hi) / 2U) * width;

  const double mean = np_mean(values.data(), values.size());

  return 0.25 * (q2 + mode_gravity + mean + 0.5 * (q1 + q3));
}

// Bit-faithful port of the pure-Python NNS.part recursive partitioner
// (`nns.part.nns_part`) for the noise_reduction="off" path that the NNS.reg
// numeric regression / nns_arma hot path uses exclusively. Centers and
// regression points are aggregated with gravity_exact, matching the Python
// `_gravity`. The caller (Python wrapper) resolves max_order exactly as
// nns_part does and falls back to pure Python for the mean/median/mode paths,
// so this only implements the gravity ("off") aggregation.
nb::dict nns_part_off(const Vector& xa, const Vector& ya, bool xonly, int max_order, int obs_req,
                      bool min_obs_stop) {
  const std::size_t n = checked_size(xa, "x");
  if (ya.shape(0) != n) {
    throw std::invalid_argument("x and y must have the same length.");
  }
  const double* x = xa.data();
  const double* y = ya.data();

  // floor_order = floor(log2(max(1, n))), computed exactly via bit position.
  int floor_order = 0;
  {
    std::size_t v = (n < 1U) ? 1U : n;
    while ((static_cast<std::size_t>(1) << (floor_order + 1)) <= v) {
      ++floor_order;
    }
  }
  if (max_order == 0) {
    max_order = 1;
  }

  std::vector<std::string> quad(n, "q");
  std::vector<std::string> prior(n, "pq");
  int depth = 0;

  while (depth < max_order && depth < floor_order) {
    // np.unique(quad) -> sorted labels, inverse, counts (std::map sorts keys).
    std::map<std::string, int> label_idx;
    for (const std::string& s : quad) {
      label_idx.emplace(s, 0);
    }
    int g = 0;
    std::vector<std::string> labels;
    labels.reserve(label_idx.size());
    for (auto& kv : label_idx) {
      kv.second = g++;
      labels.push_back(kv.first);
    }
    const int ngroups = g;
    std::vector<int> counts(ngroups, 0);
    std::vector<std::vector<std::size_t>> members(ngroups);
    for (std::size_t i = 0; i < n; ++i) {
      const int gi = label_idx[quad[i]];
      counts[gi] += 1;
      members[gi].push_back(i);
    }

    bool any_split = false;
    for (int gi = 0; gi < ngroups; ++gi) {
      if (counts[gi] <= obs_req) {
        continue;
      }
      any_split = true;
      std::vector<double> gx;
      std::vector<double> gy;
      gx.reserve(members[gi].size());
      gy.reserve(members[gi].size());
      for (const std::size_t i : members[gi]) {
        gx.push_back(x[i]);
        gy.push_back(y[i]);
      }
      const double cx = gravity_exact_impl(gx.data(), gx.size());
      const double cy = xonly ? 0.0 : gravity_exact_impl(gy.data(), gy.size());
      for (const std::size_t i : members[gi]) {
        prior[i] = labels[gi];
        if (xonly) {
          const bool low_x = std::isfinite(x[i]) && std::isfinite(cx) && (x[i] > cx);
          quad[i] += (low_x ? '2' : '1');
        } else {
          const bool low_x = std::isfinite(x[i]) && std::isfinite(cx) && (x[i] <= cx);
          const bool low_y = std::isfinite(y[i]) && std::isfinite(cy) && (y[i] <= cy);
          const int qn = 1 + (low_x ? 1 : 0) + 2 * (low_y ? 1 : 0);
          quad[i] += static_cast<char>('0' + qn);
        }
      }
    }
    if (!any_split) {
      break;
    }
    ++depth;

    if (min_obs_stop) {
      std::map<std::string, int> post_counts;
      for (const std::string& s : quad) {
        post_counts[s] += 1;
      }
      int min_count = INT_MAX;
      for (const auto& kv : post_counts) {
        min_count = std::min(min_count, kv.second);
      }
      if (min_count <= obs_req) {
        break;
      }
    }
  }

  // regression points grouped by prior.quadrant (sorted labels via std::map).
  std::map<std::string, std::vector<std::size_t>> prior_groups;
  for (std::size_t i = 0; i < n; ++i) {
    prior_groups[prior[i]].push_back(i);
  }
  std::vector<std::string> rp_quadrant;
  std::vector<double> rp_x;
  std::vector<double> rp_y;
  rp_quadrant.reserve(prior_groups.size());
  rp_x.reserve(prior_groups.size());
  rp_y.reserve(prior_groups.size());
  for (const auto& kv : prior_groups) {
    std::vector<double> gx;
    std::vector<double> gy;
    gx.reserve(kv.second.size());
    gy.reserve(kv.second.size());
    for (const std::size_t i : kv.second) {
      gx.push_back(x[i]);
      gy.push_back(y[i]);
    }
    rp_quadrant.push_back(kv.first);
    rp_x.push_back(gravity_exact_impl(gx.data(), gx.size()));
    rp_y.push_back(gravity_exact_impl(gy.data(), gy.size()));
  }

  // _is_discrete_like_r(x) -> round regression-point x half-up.
  bool any_finite = false;
  bool all_integral = true;
  for (std::size_t i = 0; i < n; ++i) {
    if (std::isfinite(x[i])) {
      any_finite = true;
      if (x[i] != std::floor(x[i])) {
        all_integral = false;
        break;
      }
    }
  }
  if (any_finite && all_integral) {
    for (double& v : rp_x) {
      const double f = std::floor(v);
      v = (v - f < 0.5) ? f : std::ceil(v);
    }
  }

  std::vector<double> dt_x(x, x + n);
  std::vector<double> dt_y(y, y + n);
  nb::dict dt;
  dt["x"] = std::move(dt_x);
  dt["y"] = std::move(dt_y);
  dt["quadrant"] = std::move(quad);
  dt["prior.quadrant"] = std::move(prior);

  nb::dict regression_points;
  regression_points["quadrant"] = std::move(rp_quadrant);
  regression_points["x"] = std::move(rp_x);
  regression_points["y"] = std::move(rp_y);

  nb::dict out;
  out["order"] = depth;
  out["dt"] = dt;
  out["regression.points"] = regression_points;
  return out;
}

// --- NNS.dep faithful port (nns.dependence.nns_dep) -------------------------
// Matches numpy's pairwise summation so means/sums used as comparison
// thresholds are bit-identical to np.mean/np.sum. Co-moments reuse the
// verified nns:: kernels; the n-d partial moments and copula logic mirror the
// pure-Python formulas exactly.

double np_pairwise_sum(const double* a, std::size_t n) {
  if (n < 8U) {
    double res = 0.0;
    for (std::size_t i = 0; i < n; ++i) res += a[i];
    return res;
  }
  if (n <= 128U) {
    double r0 = a[0], r1 = a[1], r2 = a[2], r3 = a[3];
    double r4 = a[4], r5 = a[5], r6 = a[6], r7 = a[7];
    const std::size_t limit = n - (n % 8U);
    std::size_t i = 8;
    for (; i < limit; i += 8U) {
      r0 += a[i + 0]; r1 += a[i + 1]; r2 += a[i + 2]; r3 += a[i + 3];
      r4 += a[i + 4]; r5 += a[i + 5]; r6 += a[i + 6]; r7 += a[i + 7];
    }
    double res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
    for (; i < n; ++i) res += a[i];
    return res;
  }
  std::size_t n2 = n / 2U;
  n2 -= n2 % 8U;
  return np_pairwise_sum(a, n2) + np_pairwise_sum(a + n2, n - n2);
}

double np_mean(const double* a, std::size_t n) { return np_pairwise_sum(a, n) / static_cast<double>(n); }

double clamp01(double v) { return std::min(std::max(v, 0.0), 1.0); }

std::size_t unique_count(const double* a, std::size_t n) {
  std::vector<double> v(a, a + n);
  std::sort(v.begin(), v.end());
  return static_cast<std::size_t>(std::distance(v.begin(), std::unique(v.begin(), v.end())));
}

double dep_ols_sign(const double* x, const double* y, std::size_t n) {
  if (n < 2U) return 0.0;
  const double mx = np_mean(x, n);
  std::vector<double> dx(n), dxy(n);
  for (std::size_t i = 0; i < n; ++i) dx[i] = (x[i] - mx) * (x[i] - mx);
  const double denom = np_pairwise_sum(dx.data(), n);
  if (denom == 0.0) return 0.0;
  const double my = np_mean(y, n);
  for (std::size_t i = 0; i < n; ++i) dxy[i] = (x[i] - mx) * (y[i] - my);
  const double slope = np_pairwise_sum(dxy.data(), n) / denom;
  if (slope > 0.0) return 1.0;
  if (slope < 0.0) return -1.0;
  return 0.0;
}

// _dpm_nd(degree=0, norm=True) == mean(discordant)
double nd_dpm_deg0(const double* x, const double* y, std::size_t n, double tx, double ty) {
  std::vector<double> disc(n);
  for (std::size_t i = 0; i < n; ++i) {
    const double dxi = x[i] - tx, dyi = y[i] - ty;
    const bool below = (dxi < 0.0) && (dyi < 0.0);
    const bool above = (dxi > 0.0) && (dyi > 0.0);
    disc[i] = (below || above) ? 0.0 : 1.0;
  }
  return np_mean(disc.data(), n);
}

double nd_clpm_deg1(const double* x, const double* y, std::size_t n, double tx, double ty) {
  std::vector<double> v(n);
  for (std::size_t i = 0; i < n; ++i) {
    const double cdx = tx - x[i], cdy = ty - y[i];
    v[i] = (cdx >= 0.0 && cdy >= 0.0) ? cdx * cdy : 0.0;
  }
  return np_mean(v.data(), n);
}

double nd_cupm_deg1(const double* x, const double* y, std::size_t n, double tx, double ty) {
  std::vector<double> v(n);
  for (std::size_t i = 0; i < n; ++i) {
    const double ddx = x[i] - tx, ddy = y[i] - ty;
    v[i] = (ddx >= 0.0 && ddy >= 0.0) ? ddx * ddy : 0.0;
  }
  return np_mean(v.data(), n);
}

// _dpm_nd(degree=1, norm=True)
double nd_dpm_deg1_norm(const double* x, const double* y, std::size_t n, double tx, double ty) {
  std::vector<double> v(n);
  for (std::size_t i = 0; i < n; ++i) {
    const double dxi = x[i] - tx, dyi = y[i] - ty;
    const bool below = (dxi < 0.0) && (dyi < 0.0);
    const bool above = (dxi > 0.0) && (dyi > 0.0);
    const bool discordant = !(below || above);
    v[i] = discordant ? std::abs(dxi) * std::abs(dyi) : 0.0;
  }
  const double dpm = np_mean(v.data(), n);
  const double clpm = nd_clpm_deg1(x, y, n, tx, ty);
  const double cupm = nd_cupm_deg1(x, y, n, tx, ty);
  const double total = clpm + cupm + dpm;
  return total > 0.0 ? dpm / total : 0.0;
}

double dep_copula_signed(const double* x, const double* y, std::size_t n) {
  if (n < 2U) return 0.0;
  const double tx = np_mean(x, n), ty = np_mean(y, n);
  const double d0_cupm = nns::co_upm(0.0, 0.0, x, y, n, n, tx, ty);
  const double d0_clpm = nns::co_lpm(0.0, 0.0, x, y, n, n, tx, ty);
  const double d0_co = d0_cupm + d0_clpm;
  if (d0_co == 1.0 || d0_co == 0.0) return 1.0;

  double c1_cupm = nns::co_upm(1.0, 1.0, x, y, n, n, tx, ty);
  double c1_clpm = nns::co_lpm(1.0, 1.0, x, y, n, n, tx, ty);
  double c1_dlpm = nns::d_lpm(1.0, 1.0, x, y, n, n, tx, ty);
  double c1_dupm = nns::d_upm(1.0, 1.0, x, y, n, n, tx, ty);
  const double adjust = static_cast<double>(n) / (static_cast<double>(n) - 1.0);
  c1_cupm *= adjust; c1_clpm *= adjust; c1_dlpm *= adjust; c1_dupm *= adjust;
  const double total = c1_cupm + c1_dupm + c1_dlpm + c1_clpm;
  if (total > 0.0) { c1_cupm /= total; c1_clpm /= total; }

  const double dpm_d0 = nd_dpm_deg0(x, y, n, tx, ty);
  const double dpm_d1 = nd_dpm_deg1_norm(x, y, n, tx, ty);
  const double discrete_dep = clamp01(std::abs(d0_co - 0.5) / 0.5);
  const double continuous_dep = clamp01(std::abs(c1_cupm + c1_clpm - 0.5) / 0.5);
  const double nd_disc = std::abs(dpm_d0 - 0.75) / 0.75;
  const double nd_cont = std::abs(dpm_d1 - 0.75) / 0.75;
  const double copula = std::sqrt((discrete_dep + continuous_dep + nd_disc + nd_cont) / 4.0);
  return copula * dep_ols_sign(x, y, n);
}

double dep_copula_degree0_unsigned(const double* x, const double* y, std::size_t n) {
  const double tx = np_mean(x, n), ty = np_mean(y, n);
  const double d0_co =
      nns::co_upm(0.0, 0.0, x, y, n, n, tx, ty) + nns::co_lpm(0.0, 0.0, x, y, n, n, tx, ty);
  const double dpm_d0 = nd_dpm_deg0(x, y, n, tx, ty);
  const double disc_dep = clamp01(std::abs(d0_co - 0.5) / 0.5);
  const double nd_disc = std::abs(dpm_d0 - 0.75) / 0.75;
  return std::sqrt((disc_dep + nd_disc) / 2.0);
}

std::vector<std::string> dep_xonly_partition(const double* x, std::size_t n, int obs_req) {
  int max_order = 0;
  while ((static_cast<std::size_t>(1) << max_order) < n) ++max_order;  // ceil(log2(max(1,n)))
  if (max_order < 1) max_order = 1;
  int floor_order = 0;
  {
    std::size_t v = (n < 1U) ? 1U : n;
    while ((static_cast<std::size_t>(1) << (floor_order + 1)) <= v) ++floor_order;
  }
  std::vector<std::string> quad(n, "q");
  for (int depth = 0; depth < max_order; ++depth) {
    if (depth >= floor_order) break;
    std::map<std::string, std::vector<std::size_t>> groups;
    for (std::size_t i = 0; i < n; ++i) groups[quad[i]].push_back(i);
    bool any_split = false;
    for (auto& kv : groups) {
      if (static_cast<int>(kv.second.size()) <= obs_req) continue;
      any_split = true;
      std::vector<double> gx;
      gx.reserve(kv.second.size());
      for (const std::size_t i : kv.second) gx.push_back(x[i]);
      const double center = gravity_exact_impl(gx.data(), gx.size());
      for (const std::size_t i : kv.second) quad[i] += (x[i] > center) ? '2' : '1';
    }
    if (!any_split) break;
  }
  return quad;
}

void dep_directional(const double* x, const double* y, const std::vector<std::string>& quad,
                     double fallback, double& corr_out, double& dep_out) {
  // group by first-seen quadrant order (matches Python defaultdict iteration).
  std::map<std::string, std::size_t> seen;
  std::vector<std::vector<std::size_t>> groups;
  for (std::size_t i = 0; i < quad.size(); ++i) {
    auto it = seen.find(quad[i]);
    if (it == seen.end()) {
      seen.emplace(quad[i], groups.size());
      groups.emplace_back();
      groups.back().push_back(i);
    } else {
      groups[it->second].push_back(i);
    }
  }
  const double n = static_cast<double>(quad.size());
  double corr = 0.0, dep = 0.0;
  std::vector<double> gx, gy;
  for (const auto& idx : groups) {
    gx.clear(); gy.clear();
    gx.reserve(idx.size()); gy.reserve(idx.size());
    for (const std::size_t i : idx) { gx.push_back(x[i]); gy.push_back(y[i]); }
    double cop = dep_copula_signed(gx.data(), gy.data(), gx.size());
    if (!std::isfinite(cop)) cop = fallback;
    const double weight = static_cast<double>(idx.size()) / n;
    corr += cop * weight;
    dep += std::abs(cop) * weight;
  }
  corr_out = corr;
  dep_out = dep;
}

nb::dict nns_dep_pair(const Vector& xa, const Vector& ya, bool asym) {
  const std::size_t n = checked_size(xa, "x");
  if (ya.shape(0) != n) throw std::invalid_argument("x and y must have the same length.");
  const double* x = xa.data();
  const double* y = ya.data();

  nb::dict out;
  bool x_const = true, y_const = true;
  for (std::size_t i = 1; i < n; ++i) {
    if (x[i] != x[0]) x_const = false;
    if (y[i] != y[0]) y_const = false;
  }
  if (x_const || y_const) {
    out["Correlation"] = 0.0;
    out["Dependence"] = 0.0;
    return out;
  }

  const int obs_req = std::max<int>(8, static_cast<int>(n / 8U));
  const std::vector<std::string> quad_xy = dep_xonly_partition(x, n, obs_req);
  const std::vector<std::string> quad_yx = dep_xonly_partition(y, n, obs_req);

  double global_cop = dep_copula_signed(x, y, n);
  if (!std::isfinite(global_cop)) global_cop = 0.0;

  double corr_xy, dep_xy, corr_yx, dep_yx;
  dep_directional(x, y, quad_xy, global_cop, corr_xy, dep_xy);
  dep_directional(y, x, quad_yx, global_cop, corr_yx, dep_yx);

  const bool discrete_case =
      static_cast<double>(unique_count(x, n)) < std::sqrt(static_cast<double>(n)) &&
      static_cast<double>(unique_count(y, n)) < std::sqrt(static_cast<double>(n));
  if (discrete_case) {
    double disc_cop = dep_copula_degree0_unsigned(x, y, n);
    if (!std::isfinite(disc_cop)) disc_cop = std::max(dep_xy, dep_yx);
    if (asym) {
      const double pair[2] = {dep_xy, disc_cop};
      dep_xy = gravity_exact_impl(pair, 2);
    } else {
      const double pair[2] = {std::max(dep_xy, dep_yx), disc_cop};
      const double dep_sym = gravity_exact_impl(pair, 2);
      dep_xy = dep_sym;
      dep_yx = dep_sym;
    }
  }

  if (asym) {
    out["Correlation"] = corr_xy;
    out["Dependence"] = dep_xy;
  } else {
    out["Correlation"] = std::max(corr_xy, corr_yx);
    out["Dependence"] = std::max(dep_xy, dep_yx);
  }
  return out;
}

}  // namespace

NB_MODULE(_nnscore, m) {
  m.doc() = "Private nanobind bindings for the vendored NNS-core C++ backend.";

  m.def("lpm", [](double degree, double target, const Vector& x) {
    return nns::lpm(degree, target, x.data(), checked_size(x, "x"));
  });
  m.def("lpm", &lpm_sequence);
  m.def("lpm", [](double degree, const Vector& target, const Vector& x) {
    return moment_vector(true, degree, target, x);
  });
  m.def("lpm_v", [](double degree, const Vector& target, const Vector& x) {
    return moment_vector(true, degree, target, x);
  });

  m.def("upm", [](double degree, double target, const Vector& x) {
    return nns::upm(degree, target, x.data(), checked_size(x, "x"));
  });
  m.def("upm", &upm_sequence);
  m.def("upm", [](double degree, const Vector& target, const Vector& x) {
    return moment_vector(false, degree, target, x);
  });
  m.def("upm_v", [](double degree, const Vector& target, const Vector& x) {
    return moment_vector(false, degree, target, x);
  });
  m.def("lpm_ratio_v", [](double degree, const Vector& target, const Vector& x) {
    return moment_ratio_vector(true, degree, target, x);
  });
  m.def("upm_ratio_v", [](double degree, const Vector& target, const Vector& x) {
    return moment_ratio_vector(false, degree, target, x);
  });

  m.def("co_lpm", [](double degree_x, double degree_y, const Vector& x, const Vector& y,
                     double target_x, double target_y) {
    check_same_size(x, y, "x", "y");
    return nns::co_lpm(degree_x, degree_y, x.data(), y.data(), x.shape(0), y.shape(0), target_x,
                       target_y);
  });
  m.def("co_upm", [](double degree_x, double degree_y, const Vector& x, const Vector& y,
                     double target_x, double target_y) {
    check_same_size(x, y, "x", "y");
    return nns::co_upm(degree_x, degree_y, x.data(), y.data(), x.shape(0), y.shape(0), target_x,
                       target_y);
  });
  m.def("d_lpm", [](double degree_lpm, double degree_upm, const Vector& x, const Vector& y,
                    double target_x, double target_y) {
    check_same_size(x, y, "x", "y");
    return nns::d_lpm(degree_lpm, degree_upm, x.data(), y.data(), x.shape(0), y.shape(0), target_x,
                      target_y);
  });
  m.def("d_upm", [](double degree_lpm, double degree_upm, const Vector& x, const Vector& y,
                    double target_x, double target_y) {
    check_same_size(x, y, "x", "y");
    return nns::d_upm(degree_lpm, degree_upm, x.data(), y.data(), x.shape(0), y.shape(0), target_x,
                      target_y);
  });
  m.def("co_lpm_v", [](double degree_x, double degree_y, const Vector& x, const Vector& y,
                       const Vector& target_x, const Vector& target_y) {
    return co_moment_vector("co_lpm", degree_x, degree_y, x, y, target_x, target_y);
  });
  m.def("co_upm_v", [](double degree_x, double degree_y, const Vector& x, const Vector& y,
                       const Vector& target_x, const Vector& target_y) {
    return co_moment_vector("co_upm", degree_x, degree_y, x, y, target_x, target_y);
  });
  m.def("d_lpm_v", [](double degree_lpm, double degree_upm, const Vector& x, const Vector& y,
                      const Vector& target_x, const Vector& target_y) {
    return co_moment_vector("d_lpm", degree_lpm, degree_upm, x, y, target_x, target_y);
  });
  m.def("d_upm_v", [](double degree_lpm, double degree_upm, const Vector& x, const Vector& y,
                      const Vector& target_x, const Vector& target_y) {
    return co_moment_vector("d_upm", degree_lpm, degree_upm, x, y, target_x, target_y);
  });

  m.def("clpm_nd", [](const Vector& data, std::size_t n, std::size_t d, const Vector& target,
                      double degree, bool norm) {
    checked_flat_matrix_size(data, n, d, "data");
    if (target.shape(0) != d) throw std::invalid_argument("target length must equal d.");
    return nns::clpm_nd(data.data(), n, d, target.data(), degree, norm);
  });
  m.def("cupm_nd", [](const Vector& data, std::size_t n, std::size_t d, const Vector& target,
                      double degree, bool norm) {
    checked_flat_matrix_size(data, n, d, "data");
    if (target.shape(0) != d) throw std::invalid_argument("target length must equal d.");
    return nns::cupm_nd(data.data(), n, d, target.data(), degree, norm);
  });
  m.def("dpm_nd", [](const Vector& data, std::size_t n, std::size_t d, const Vector& target,
                    double degree, bool norm) {
    checked_flat_matrix_size(data, n, d, "data");
    if (target.shape(0) != d) throw std::invalid_argument("target length must equal d.");
    return nns::dpm_nd(data.data(), n, d, target.data(), degree, norm);
  });
  m.def("clpm_nd_batch", [](const Vector& data, std::size_t n, std::size_t d,
                            const Vector& targets, std::size_t n_targets, double degree, bool norm) {
    checked_flat_matrix_size(data, n, d, "data");
    if (targets.shape(0) != n_targets * d) {
      throw std::invalid_argument("targets length must equal n_targets * d.");
    }
    std::vector<double> out(n_targets, 0.0);
    nns::clpm_nd_batch(data.data(), n, d, targets.data(), n_targets, degree, norm, out.data());
    return out;
  });
  m.def("pm_matrix", &pm_matrix_dict);

  m.def("fast_lm", &fast_lm_dict);
  m.def("fast_lm_mult", &fast_lm_mult_dict);

  m.def("is_discrete", [](const Vector& x) { return nns::is_discrete(x.data(), checked_size(x, "x")); });
  m.def("vec_sd", [](const Vector& x) { return nns::vec_sd(x.data(), checked_size(x, "x")); });
  m.def("col_sd", [](const Vector& x, std::size_t n, std::size_t p) {
    checked_flat_matrix_size(x, n, p, "x");
    return nns::col_sd(x.data(), n, p);
  });
  m.def("factor_2_dummy", [](const std::vector<int>& codes, const std::vector<std::string>& levels) {
    return dummy_matrix_dict(codes, levels, false);
  });
  m.def("factor_2_dummy_fr", [](const std::vector<int>& codes, const std::vector<std::string>& levels) {
    return dummy_matrix_dict(codes, levels, true);
  });
  m.def("generate_vectors", &time_series_vectors_dict);
  m.def("generate_lin_vectors", &forecast_vectors_dict);

  m.def("gravity", [](const Vector& x, bool discrete) {
    return nns::gravity(x.data(), checked_size(x, "x"), discrete);
  });

  m.def("mode", [](const Vector& x, bool discrete, bool multi) {
    return nns::mode(x.data(), checked_size(x, "x"), discrete, multi);
  });

  m.def("stochastic_superiority", &stochastic_superiority_dict);
  m.def("gravity_exact", [](const Vector& x) {
    return gravity_exact_impl(x.data(), x.shape(0));
  });
  m.def("nns_part_off", &nns_part_off, nb::arg("x"), nb::arg("y"), nb::arg("xonly"),
        nb::arg("max_order"), nb::arg("obs_req"), nb::arg("min_obs_stop"));
  m.def("nns_dep_pair", &nns_dep_pair, nb::arg("x"), nb::arg("y"), nb::arg("asym"));
}
