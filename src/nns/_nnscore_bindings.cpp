#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <functional>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "nns/nns.hpp"

namespace nb = nanobind;

namespace {

// Worker threads for the NNS-core kernels called from this OS thread: 0 means
// the library default (nns::get_num_threads(): set_num_threads(), else the
// NNS_NUM_THREADS environment variable, else the hardware thread count).
// nns._native.native_threads(ncores) sets it for the duration of one public
// call, so Python threads with different ncores do not interfere. Results never
// depend on it: every parallel kernel writes disjoint output slots.
thread_local int t_call_threads = 0;

int call_threads() { return t_call_threads > 0 ? t_call_threads : -1; }

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

Matrix matrix_from_column_major(std::vector<double>&& values, std::size_t rows,
                                 std::size_t cols) {
  auto* storage = new std::vector<double>(std::move(values));
  nb::capsule owner(storage, [](void* p) noexcept {
    delete static_cast<std::vector<double>*>(p);
  });
  return Matrix(storage->data(), {rows, cols}, owner, {1, static_cast<int64_t>(rows)});
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
  nb::gil_scoped_release release;
  if (lower) {
    nns::lpm_ratio_v(degree, targets.data(), n_targets, x.data(), n, out.data(), call_threads());
  } else {
    nns::upm_ratio_v(degree, targets.data(), n_targets, x.data(), n, out.data(), call_threads());
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
  nb::gil_scoped_release release;
  if (lower) {
    nns::lpm_v(degree, targets.data(), n_targets, x.data(), n, out.data(), call_threads());
  } else {
    nns::upm_v(degree, targets.data(), n_targets, x.data(), n, out.data(), call_threads());
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
  nb::gil_scoped_release release;
  if (kind == "co_lpm") {
    nns::co_lpm_v(degree_x, degree_y, x.data(), y.data(), n_x, n_y, target_x.data(), n_target_x,
                  target_y.data(), n_target_y, out.data(), call_threads());
  } else if (kind == "co_upm") {
    nns::co_upm_v(degree_x, degree_y, x.data(), y.data(), n_x, n_y, target_x.data(), n_target_x,
                  target_y.data(), n_target_y, out.data(), call_threads());
  } else if (kind == "d_lpm") {
    nns::d_lpm_v(degree_x, degree_y, x.data(), y.data(), n_x, n_y, target_x.data(), n_target_x,
                 target_y.data(), n_target_y, out.data(), call_threads());
  } else if (kind == "d_upm") {
    nns::d_upm_v(degree_x, degree_y, x.data(), y.data(), n_x, n_y, target_x.data(), n_target_x,
                 target_y.data(), n_target_y, out.data(), call_threads());
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
  nns::PMMatrixResult result;
  {
    nb::gil_scoped_release release;
    result = nns::pm_matrix(degree_lpm, degree_upm, target.data(), variable.data(), n, d, pop_adj,
                            norm, call_threads());
  }
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

// NNS.gravity as R's NNS_gravity_cpp (non-discrete), used by the partitioners
// below and by nns.dependence._gravity. This is NNS-core's nns::gravity: R's
// long-double mean and weighted bin average, and an O(n) histogram that only
// shortens runs of empty bins. The previous port here clipped the
// histogram at 1e6 bins, which put the mode term on the outlier, and
// used pairwise double sums that differed from R in the last ulp.
// Non-finite values are dropped, so the caller may pass a raw array.
double gravity_exact_impl(const double* data, std::size_t raw_n) {
  return nns::gravity(data, raw_n, false);
}

// Bit-faithful port of the pure-Python NNS.part recursive partitioner
// (`nns.part.nns_part`) for the noise_reduction="off" path that the NNS.reg
// numeric regression / nns_arma hot path uses exclusively. Centers and
// regression points are aggregated with gravity_exact, matching the Python
// `_gravity`. The caller (Python wrapper) resolves max_order exactly as
// nns_part does and falls back to pure Python for the mean/median/mode paths,
// so this only implements the gravity ("off") aggregation.

struct PartitionLabels {
  std::vector<std::string> quadrant;
  std::vector<std::string> prior;
  int depth;
};

// Shared partition loop (noise="off"): assigns quadrant and prior.quadrant
// labels exactly as nns.part.nns_part does for the gravity path.
PartitionLabels partition_labels(const double* x, const double* y, std::size_t n, bool xonly,
                                 int max_order, int obs_req, bool min_obs_stop) {
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
      if (!xonly) gy.reserve(members[gi].size());
      for (const std::size_t i : members[gi]) {
        gx.push_back(x[i]);
        if (!xonly) gy.push_back(y[i]);
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
  return {std::move(quad), std::move(prior), depth};
}

// regression points grouped by prior.quadrant (sorted labels via std::map): the
// (x, y) gravity centroid of each cell. x is not rounded for integer-valued x
// (R fix "NNS.part: do not round regression-point x for integer-valued x").
void regression_points_from_prior(const double* x, const double* y, std::size_t n,
                                  const std::vector<std::string>& prior,
                                  std::vector<std::string>* rp_quadrant, std::vector<double>* rp_x,
                                  std::vector<double>* rp_y) {
  std::map<std::string, std::vector<std::size_t>> prior_groups;
  for (std::size_t i = 0; i < n; ++i) {
    prior_groups[prior[i]].push_back(i);
  }
  for (const auto& kv : prior_groups) {
    std::vector<double> gx;
    std::vector<double> gy;
    gx.reserve(kv.second.size());
    gy.reserve(kv.second.size());
    for (const std::size_t i : kv.second) {
      gx.push_back(x[i]);
      gy.push_back(y[i]);
    }
    if (rp_quadrant != nullptr) rp_quadrant->push_back(kv.first);
    rp_x->push_back(gravity_exact_impl(gx.data(), gx.size()));
    rp_y->push_back(gravity_exact_impl(gy.data(), gy.size()));
  }
}

nb::dict nns_part_off(const Vector& xa, const Vector& ya, bool xonly, int max_order, int obs_req,
                      bool min_obs_stop) {
  const std::size_t n = checked_size(xa, "x");
  if (ya.shape(0) != n) {
    throw std::invalid_argument("x and y must have the same length.");
  }
  const double* x = xa.data();
  const double* y = ya.data();

  PartitionLabels pl;
  std::vector<std::string> rp_quadrant;
  std::vector<double> rp_x;
  std::vector<double> rp_y;
  {
    nb::gil_scoped_release release;
    pl = partition_labels(x, y, n, xonly, max_order, obs_req, min_obs_stop);
    regression_points_from_prior(x, y, n, pl.prior, &rp_quadrant, &rp_x, &rp_y);
  }

  std::vector<double> dt_x(x, x + n);
  std::vector<double> dt_y(y, y + n);
  nb::dict dt;
  dt["x"] = std::move(dt_x);
  dt["y"] = std::move(dt_y);
  dt["quadrant"] = std::move(pl.quadrant);
  dt["prior.quadrant"] = std::move(pl.prior);

  nb::dict regression_points;
  regression_points["quadrant"] = std::move(rp_quadrant);
  regression_points["x"] = std::move(rp_x);
  regression_points["y"] = std::move(rp_y);

  nb::dict out;
  out["order"] = pl.depth;
  out["dt"] = dt;
  out["regression.points"] = regression_points;
  return out;
}

// --- NumPy-faithful sums and the NNS.dep partition -------------------------
// np_pairwise_sum matches numpy's pairwise summation, so means/sums used as
// comparison thresholds by the copula port below are bit-identical to
// np.mean/np.sum.

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

// NNS.dep = R NNS_dep_pair_cpp on the X-only partitions of x and y. The
// labels are hashed with std::hash<std::string> exactly as the R wrapper does,
// and nns::dep_pair is the NNS-core port of R's NNS_dep_pair_core: cell
// means are sequential double sums / n, the moments are one-pass sums, and the
// cell weights are accumulated in the same unordered_map order as R. A
// pairwise (NumPy) cell mean differs from R's in the last ulp on evenly spaced
// or tied cells, and a point sitting exactly on the mean then changes
// quadrant counts (R seq(0, 4*pi, pi/100), sin: dependence 0.92402 vs R
// 0.92466).
std::vector<uint64_t> hash_labels(const std::vector<std::string>& labels) {
  const std::hash<std::string> hasher;
  std::vector<uint64_t> out(labels.size());
  for (std::size_t i = 0; i < labels.size(); ++i) out[i] = hasher(labels[i]);
  return out;
}

nb::dict nns_dep_pair(const Vector& xa, const Vector& ya, bool asym) {
  const std::size_t n = checked_size(xa, "x");
  if (ya.shape(0) != n) throw std::invalid_argument("x and y must have the same length.");
  const double* x = xa.data();
  const double* y = ya.data();

  nns::DepResult res;
  {
    nb::gil_scoped_release release;
    const int obs_req = std::max<int>(8, static_cast<int>(n / 8U));
    const std::vector<uint64_t> quad_xy = hash_labels(dep_xonly_partition(x, n, obs_req));
    const std::vector<uint64_t> quad_yx = hash_labels(dep_xonly_partition(y, n, obs_req));
    res = nns::dep_pair(x, y, n, quad_xy.data(), quad_yx.data(), asym);
  }

  nb::dict out;
  out["Correlation"] = res.correlation;
  out["Dependence"] = res.dependence;
  return out;
}

// --- NNS.copula faithful port (nns.copula._copula) --------------------------
// General-d n-dimensional partial moments mirroring the pure-Python
// _clpm_nd/_cupm_nd/_dpm_nd, with numpy-faithful pairwise means. data is
// column-major n x d; target is the d-vector of per-column targets supplied by
// the caller (np.mean), so it is already bit-identical. pm covariance matrices
// reuse the verified nns::pm_matrix kernel.

double nd_clpm_general(const double* data, std::size_t n, std::size_t d, const double* target,
                       double degree) {
  std::vector<double> v(n);
  for (std::size_t i = 0; i < n; ++i) {
    bool valid = true;
    double prod = 1.0;
    for (std::size_t c = 0; c < d; ++c) {
      const double cd = target[c] - data[i + c * n];
      if (cd < 0.0) valid = false;
      prod *= cd;
    }
    v[i] = (degree == 0.0) ? (valid ? 1.0 : 0.0) : (valid ? prod : 0.0);
  }
  return np_mean(v.data(), n);
}

double nd_cupm_general(const double* data, std::size_t n, std::size_t d, const double* target,
                       double degree) {
  std::vector<double> v(n);
  for (std::size_t i = 0; i < n; ++i) {
    bool valid = true;
    double prod = 1.0;
    for (std::size_t c = 0; c < d; ++c) {
      const double dd = data[i + c * n] - target[c];
      if (dd < 0.0) valid = false;
      prod *= dd;
    }
    v[i] = (degree == 0.0) ? (valid ? 1.0 : 0.0) : (valid ? prod : 0.0);
  }
  return np_mean(v.data(), n);
}

double nd_dpm_general(const double* data, std::size_t n, std::size_t d, const double* target,
                      double degree, bool norm) {
  std::vector<double> v(n);
  for (std::size_t i = 0; i < n; ++i) {
    bool below = true, above = true;
    double prod_abs = 1.0;
    for (std::size_t c = 0; c < d; ++c) {
      const double diff = data[i + c * n] - target[c];
      // Degree 0 partitions the sample like CLPM (all <= target) and CUPM (all >
      // target), as R's DPM_nD; at degree 1 a tied coordinate adds a
      // zero factor either way.
      if (!(diff <= 0.0)) below = false;
      if (!(diff > 0.0)) above = false;
      prod_abs *= std::abs(diff);
    }
    const bool discordant = !(below || above);
    v[i] = (degree == 0.0) ? (discordant ? 1.0 : 0.0) : (discordant ? prod_abs : 0.0);
  }
  const double dpm = np_mean(v.data(), n);
  if (degree == 0.0) return dpm;  // Python returns mean(discordant) before norm
  if (!norm) return dpm;
  const double clpm = nd_clpm_general(data, n, d, target, 1.0);
  const double cupm = nd_cupm_general(data, n, d, target, 1.0);
  const double total = clpm + cupm + dpm;
  return total > 0.0 ? dpm / total : 0.0;
}

double upper_tri_pairwise(const std::vector<double>& m, std::size_t d) {
  std::vector<double> vals;
  vals.reserve(d * (d - 1U) / 2U);
  for (std::size_t i = 0; i < d; ++i)
    for (std::size_t j = i + 1U; j < d; ++j) vals.push_back(m[i + j * d]);
  return np_pairwise_sum(vals.data(), vals.size());
}

double copula_nd(const double* data, std::size_t n, std::size_t d, const double* target,
                 bool continuous) {
  const nns::PMMatrixResult disc = nns::pm_matrix(0.0, 0.0, target, data, n, d, false, false, call_threads());
  const double disc_co = upper_tri_pairwise(disc.cupm, d) + upper_tri_pairwise(disc.clpm, d);
  // Perfect dependence: every pair fully concordant (1) or fully discordant (0).
  bool perfect = true;
  for (std::size_t i = 0; i < d && perfect; ++i) {
    for (std::size_t j = i + 1U; j < d; ++j) {
      const double pair = disc.cupm[i + j * d] + disc.clpm[i + j * d];
      if (!(std::abs(pair - 1.0) < 1e-9 || std::abs(pair) < 1e-9)) {
        perfect = false;
        break;
      }
    }
  }
  if (perfect) return 1.0;
  const double disc_d = nd_dpm_general(data, n, d, target, 0.0, true);

  double cont_co, cont_d;
  if (continuous) {
    const nns::PMMatrixResult cont = nns::pm_matrix(1.0, 1.0, target, data, n, d, true, true, call_threads());
    cont_co = upper_tri_pairwise(cont.cupm, d) + upper_tri_pairwise(cont.clpm, d);
    cont_d = nd_dpm_general(data, n, d, target, 1.0, true);
  } else {
    cont_co = disc_co;
    cont_d = disc_d;
  }

  const double dd = static_cast<double>(d);
  const double indep_co = 0.25 * (dd * dd - dd);
  const double discrete_dep = clamp01(std::abs(disc_co - indep_co) / indep_co);
  const double continuous_dep = clamp01(std::abs(cont_co - indep_co) / indep_co);
  // DPM_nD counts a point concordant when all-below OR all-above the target
  // (both fully-aligned orthants), so under independence
  // P(discordant) = 1 - 2*0.5^d (0.5 when d == 2), not 1 - 0.5^d.
  const double indep_d = 1.0 - 2.0 * std::pow(0.5, dd);
  const double nd_disc = std::abs(disc_d - indep_d) / indep_d;
  const double nd_cont = std::abs(cont_d - indep_d) / indep_d;
  return std::sqrt((discrete_dep + continuous_dep + nd_disc + nd_cont) / 4.0);
}

// --- NNS.M.reg prediction kernels (NNS-core ports of R 13.3) ----------------
// Matrices are passed flattened column-major (element (i, j) at i + j * rows).
// NNS-core's mreg_setup and xstar_path are not bound: the NumPy partition, RPM
// reduction and X* path already take a few ms, and mreg_setup's "mode" reducer
// is the exact-frequency mode rather than NNS.mode.

void check_mreg_predict_args(const Vector& rpm_x, std::size_t n, std::size_t p,
                             const Vector& yhat, const Vector& xtest, std::size_t m,
                             const Vector& mins, const Vector& maxs) {
  checked_flat_matrix_size(rpm_x, n, p, "rpm_x");
  if (yhat.shape(0) != n) throw std::invalid_argument("yhat length must equal n.");
  if (xtest.shape(0) != m * p) throw std::invalid_argument("xtest length must equal m * p.");
  if (mins.shape(0) != p || maxs.shape(0) != p) {
    throw std::invalid_argument("mins and maxs must have one value per column.");
  }
}

std::vector<double> mreg_predict_vector(const Vector& rpm_x, std::size_t n, std::size_t p,
                                        const Vector& yhat, const Vector& xtest, std::size_t m,
                                        int k, int dist_code, const Vector& mins,
                                        const Vector& maxs, bool is_class) {
  check_mreg_predict_args(rpm_x, n, p, yhat, xtest, m, mins, maxs);
  if (m == 0U) return {};
  nb::gil_scoped_release release;
  return nns::mreg_predict(rpm_x.data(), n, p, yhat.data(), xtest.data(), m, k, dist_code,
                           mins.data(), maxs.data(), is_class, call_threads());
}

Matrix mreg_predict_path_matrix(const Vector& rpm_x, std::size_t n, std::size_t p,
                                const Vector& yhat, const Vector& xtest, std::size_t m, int kmax,
                                int dist_code, const Vector& mins, const Vector& maxs,
                                bool is_class) {
  check_mreg_predict_args(rpm_x, n, p, yhat, xtest, m, mins, maxs);
  const std::size_t cols = std::min<std::size_t>(static_cast<std::size_t>(std::max(kmax, 1)), n);
  std::vector<double> out;
  if (m > 0U) {
    nb::gil_scoped_release release;
    out = nns::mreg_predict_path(rpm_x.data(), n, p, yhat.data(), xtest.data(), m, kmax,
                                 dist_code, mins.data(), maxs.data(), is_class, call_threads());
  }
  return matrix_from_column_major(std::move(out), m, cols);
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
    nb::gil_scoped_release release;
    return nns::clpm_nd(data.data(), n, d, target.data(), degree, norm, call_threads());
  });
  m.def("cupm_nd", [](const Vector& data, std::size_t n, std::size_t d, const Vector& target,
                      double degree, bool norm) {
    checked_flat_matrix_size(data, n, d, "data");
    if (target.shape(0) != d) throw std::invalid_argument("target length must equal d.");
    nb::gil_scoped_release release;
    return nns::cupm_nd(data.data(), n, d, target.data(), degree, norm, call_threads());
  });
  m.def("dpm_nd", [](const Vector& data, std::size_t n, std::size_t d, const Vector& target,
                    double degree, bool norm) {
    checked_flat_matrix_size(data, n, d, "data");
    if (target.shape(0) != d) throw std::invalid_argument("target length must equal d.");
    nb::gil_scoped_release release;
    return nns::dpm_nd(data.data(), n, d, target.data(), degree, norm, call_threads());
  });
  m.def("clpm_nd_batch", [](const Vector& data, std::size_t n, std::size_t d,
                            const Vector& targets, std::size_t n_targets, double degree, bool norm) {
    checked_flat_matrix_size(data, n, d, "data");
    if (targets.shape(0) != n_targets * d) {
      throw std::invalid_argument("targets length must equal n_targets * d.");
    }
    std::vector<double> out(n_targets, 0.0);
    nb::gil_scoped_release release;
    nns::clpm_nd_batch(data.data(), n, d, targets.data(), n_targets, degree, norm, out.data(),
                       call_threads());
    return out;
  });
  m.def("pm_matrix", &pm_matrix_dict);

  m.def(
      "set_num_threads", [](int n) { nns::set_num_threads(n); }, nb::arg("n"),
      "Default NNS-core worker threads (n <= 0: NNS_NUM_THREADS or the hardware count).");
  m.def("get_num_threads", []() { return nns::get_num_threads(); });
  m.def(
      "_set_call_threads",
      [](int n) {
        const int previous = t_call_threads;
        t_call_threads = n > 0 ? n : 0;
        return previous;
      },
      nb::arg("n"));

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
    const std::size_t n = checked_size(x, "x");
    nb::gil_scoped_release release;
    return nns::gravity(x.data(), n, discrete);
  });

  m.def("mode", [](const Vector& x, bool discrete, bool multi) {
    const std::size_t n = checked_size(x, "x");
    nb::gil_scoped_release release;
    return nns::mode(x.data(), n, discrete, multi);
  });

  m.def("stochastic_superiority", &stochastic_superiority_dict);
  m.def("gravity_exact", [](const Vector& x) {
    nb::gil_scoped_release release;
    return gravity_exact_impl(x.data(), x.shape(0));
  });
  m.def("nns_part_off", &nns_part_off, nb::arg("x"), nb::arg("y"), nb::arg("xonly"),
        nb::arg("max_order"), nb::arg("obs_req"), nb::arg("min_obs_stop"));
  m.def("mreg_predict", &mreg_predict_vector, nb::arg("rpm_x"), nb::arg("n"), nb::arg("p"),
        nb::arg("yhat"), nb::arg("xtest"), nb::arg("m"), nb::arg("k"), nb::arg("dist_code"),
        nb::arg("mins"), nb::arg("maxs"), nb::arg("is_class"));
  m.def("mreg_predict_path", &mreg_predict_path_matrix, nb::arg("rpm_x"), nb::arg("n"),
        nb::arg("p"), nb::arg("yhat"), nb::arg("xtest"), nb::arg("m"), nb::arg("kmax"),
        nb::arg("dist_code"), nb::arg("mins"), nb::arg("maxs"), nb::arg("is_class"));
  m.def("nns_dep_pair", &nns_dep_pair, nb::arg("x"), nb::arg("y"), nb::arg("asym"));
  m.def(
      "copula_nd",
      [](const Vector& data, std::size_t n, std::size_t d, const Vector& target, bool continuous) {
        if (data.shape(0) != n * d) {
          throw std::invalid_argument("data length must equal n * d.");
        }
        if (target.shape(0) != d) {
          throw std::invalid_argument("target length must equal d.");
        }
        nb::gil_scoped_release release;
        return copula_nd(data.data(), n, d, target.data(), continuous);
      },
      nb::arg("data"), nb::arg("n"), nb::arg("d"), nb::arg("target"), nb::arg("continuous"));
}
