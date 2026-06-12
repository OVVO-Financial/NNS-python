// src/partial_moments.cpp
//
// Implementation extracted from NNS 13.0 src/partial_moments.{h,cpp}.
// Every numerical rule (>= vs > boundaries, integer-degree fast powers,
// median shift in the prefix backend, min/max-length recycling, population
// adjustment gating, crossed DUPM/DLPM mirroring) is preserved verbatim.
//
// SPDX-License-Identifier: GPL-3.0-only
#include "nns/partial_moments.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

#include "nns/parallel.hpp"

namespace nns {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// --- shared helpers (ports of the static helpers in partial_moments.cpp) ---

inline double repeat_multiplication(double value, int n) {
  double result = 1.0;
  for (int i = 0; i < n; ++i) result *= value;
  return result;
}

inline bool is_integer(double v) { return v == static_cast<double>(static_cast<int>(v)); }

inline double lower_component(double diff, double degree, bool degree_is_int) {
  if (degree == 0) return diff >= 0.0 ? 1.0 : 0.0;
  if (diff < 0.0) return 0.0;
  return degree_is_int ? repeat_multiplication(diff, static_cast<int>(degree))
                       : std::pow(diff, degree);
}

inline double upper_component(double diff, double degree, bool degree_is_int) {
  if (degree == 0) return diff > 0.0 ? 1.0 : 0.0;
  if (diff < 0.0) return 0.0;
  return degree_is_int ? repeat_multiplication(diff, static_cast<int>(degree))
                       : std::pow(diff, degree);
}

// --- prefix-power backend (port of nns_pm_detail, NNS 13.0) ----------------

constexpr int kPrefixMaxDegree = 32;
constexpr std::size_t kDirectPathMaxTargets = 32;  // NNS_DIRECT_PATH_MAX_TARGETS

bool prefix_supported_degree(double degree, int& degree_int) {
  if (!std::isfinite(degree) || degree < 0.0) return false;
  const double rounded = std::round(degree);
  if (std::fabs(degree - rounded) > 1e-12) return false;
  if (rounded > static_cast<double>(kPrefixMaxDegree)) return false;
  degree_int = static_cast<int>(rounded);
  return true;
}

std::vector<double> binomial_coefficients(int degree) {
  std::vector<double> choose(static_cast<std::size_t>(degree) + 1U, 1.0);
  for (int j = 1; j < degree; ++j) {
    choose[static_cast<std::size_t>(j)] =
        choose[static_cast<std::size_t>(j - 1)] *
        static_cast<double>(degree - j + 1) / static_cast<double>(j);
  }
  return choose;
}

struct PrefixBackend {
  std::vector<double> sorted;
  std::vector<std::vector<double>> prefix_power;
  std::vector<double> total_power;
  std::vector<double> choose;
  std::size_t n;
  int degree;
  double shift;

  PrefixBackend(const double* variable, std::size_t n_, int degree_)
      : sorted(variable, variable + n_),
        prefix_power(static_cast<std::size_t>(degree_) + 1U),
        total_power(static_cast<std::size_t>(degree_) + 1U, 0.0),
        choose(binomial_coefficients(degree_)),
        n(n_),
        degree(degree_),
        shift(0.0) {
    for (std::size_t i = 0; i < n; ++i) {
      if (!std::isfinite(sorted[i])) {  // defensive guard, as upstream
        sorted.clear();
        n = 0;
        return;
      }
    }
    std::sort(sorted.begin(), sorted.end());
    shift = sorted[n / 2U];
    for (int p = 0; p <= degree; ++p)
      prefix_power[static_cast<std::size_t>(p)].assign(n + 1U, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      const double x = sorted[i] - shift;
      double x_power = 1.0;
      for (int p = 0; p <= degree; ++p) {
        const std::size_t ps = static_cast<std::size_t>(p);
        prefix_power[ps][i + 1U] = prefix_power[ps][i] + x_power;
        x_power *= x;
      }
    }
    for (int p = 0; p <= degree; ++p) {
      const std::size_t ps = static_cast<std::size_t>(p);
      total_power[ps] = prefix_power[ps][n];
    }
  }

  bool ok() const { return n > 0U; }

  std::size_t count_leq(double target) const {
    return static_cast<std::size_t>(
        std::upper_bound(sorted.begin(), sorted.end(), target) -
        sorted.begin());
  }

  double lpm(double target) const {
    if (!std::isfinite(target)) return kNaN;
    const std::size_t k = count_leq(target);
    const double tc = target - shift;
    const double nd = static_cast<double>(n);
    if (degree == 0) return static_cast<double>(k) / nd;
    if (degree == 1)
      return (static_cast<double>(k) * tc - prefix_power[1][k]) / nd;
    if (degree == 2) {
      const double t2 = tc * tc;
      return (static_cast<double>(k) * t2 - 2.0 * tc * prefix_power[1][k] +
              prefix_power[2][k]) /
             nd;
    }
    double out = 0.0;
    for (int j = 0; j <= degree; ++j) {
      const std::size_t js = static_cast<std::size_t>(j);
      const double sign = (j % 2 == 0) ? 1.0 : -1.0;
      out += choose[js] * sign * std::pow(tc, static_cast<double>(degree - j)) *
             prefix_power[js][k];
    }
    return out / nd;
  }

  double upm(double target) const {
    if (!std::isfinite(target)) return kNaN;
    const std::size_t k = count_leq(target);
    const double tc = target - shift;
    const std::size_t above = n - k;
    const double nd = static_cast<double>(n);
    if (degree == 0) return static_cast<double>(above) / nd;
    const double suffix1 = total_power[1] - prefix_power[1][k];
    if (degree == 1) return (suffix1 - static_cast<double>(above) * tc) / nd;
    if (degree == 2) {
      const double suffix2 = total_power[2] - prefix_power[2][k];
      const double t2 = tc * tc;
      return (suffix2 - 2.0 * tc * suffix1 + static_cast<double>(above) * t2) /
             nd;
    }
    double out = 0.0;
    for (int j = 0; j <= degree; ++j) {
      const std::size_t js = static_cast<std::size_t>(j);
      const double suffix_j = total_power[js] - prefix_power[js][k];
      const double sign = ((degree - j) % 2 == 0) ? 1.0 : -1.0;
      out += choose[js] * sign * std::pow(tc, static_cast<double>(degree - j)) *
             suffix_j;
    }
    return out / nd;
  }

  std::pair<double, double> both(double target) const {
    return std::make_pair(lpm(target), upm(target));
  }
};

std::shared_ptr<const PrefixBackend> make_prefix_backend(double degree,
                                                         const double* x,
                                                         std::size_t n) {
  int degree_int = 0;
  if (n == 0) return nullptr;
  if (!prefix_supported_degree(degree, degree_int)) return nullptr;
  for (std::size_t i = 0; i < n; ++i)
    if (!std::isfinite(x[i])) return nullptr;
  return std::make_shared<const PrefixBackend>(x, n, degree_int);
}

}  // namespace

// --- univariate scalar kernels (ports of LPM_C / UPM_C) --------------------

double lpm(double degree, double target, const double* x, std::size_t n) {
  double out = 0;
  const bool deg_is_int = is_integer(degree);
  for (std::size_t i = 0; i < n; ++i) {
    const double value = target - x[i];
    if (value >= 0) {
      if (deg_is_int) {
        if (degree == 0)
          out += 1;
        else if (degree == 1)
          out += value;
        else
          out += repeat_multiplication(value, static_cast<int>(degree));
      } else {
        out += std::pow(value, degree);
      }
    }
  }
  out /= static_cast<double>(n);
  return out;
}

double upm(double degree, double target, const double* x, std::size_t n) {
  double out = 0;
  const bool deg_is_int = is_integer(degree);
  for (std::size_t i = 0; i < n; ++i) {
    const double value = x[i] - target;
    if (value > 0) {
      if (deg_is_int) {
        if (degree == 0)
          out += 1;
        else if (degree == 1)
          out += value;
        else
          out += repeat_multiplication(value, static_cast<int>(degree));
      } else {
        out += std::pow(value, degree);
      }
    }
  }
  out /= static_cast<double>(n);
  return out;
}

// --- vectorized univariate (ports of LPM_CPv / UPM_CPv / ratio kernels) ----

namespace {

enum class PMKind { Lower, Upper, LowerRatio, UpperRatio };

template <PMKind K>
void pm_vectorized(double degree, const double* target, std::size_t n_targets,
                   const double* x, std::size_t n, double* out,
                   int n_threads) {
  // Direct path for few targets: bit-identical to pre-13.0 semantics and
  // avoids building the prefix backend (port of the <=32 guard).
  const bool few_targets = n_targets <= kDirectPathMaxTargets;
  std::shared_ptr<const PrefixBackend> prefix =
      few_targets ? nullptr : make_prefix_backend(degree, x, n);

  auto body = [&](std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; ++i) {
      const double t = target[i];
      double l = 0.0, u = 0.0;
      if (prefix && std::isfinite(t)) {
        if (K == PMKind::Lower) {
          out[i] = prefix->lpm(t);
          continue;
        }
        if (K == PMKind::Upper) {
          out[i] = prefix->upm(t);
          continue;
        }
        const std::pair<double, double> pm = prefix->both(t);
        l = pm.first;
        u = pm.second;
      } else {
        if (K == PMKind::Lower) {
          out[i] = lpm(degree, t, x, n);
          continue;
        }
        if (K == PMKind::Upper) {
          out[i] = upm(degree, t, x, n);
          continue;
        }
        l = lpm(degree, t, x, n);
        u = upm(degree, t, x, n);
      }
      out[i] = (K == PMKind::LowerRatio) ? l / (l + u) : u / (l + u);
    }
  };

  if (few_targets) {
    body(0, n_targets);  // serial direct path, as upstream
  } else {
    parallel_for(0, n_targets, body, n_threads);
  }
}

}  // namespace

void lpm_v(double degree, const double* target, std::size_t n_targets,
           const double* x, std::size_t n, double* out, int n_threads) {
  pm_vectorized<PMKind::Lower>(degree, target, n_targets, x, n, out,
                               n_threads);
}

void upm_v(double degree, const double* target, std::size_t n_targets,
           const double* x, std::size_t n, double* out, int n_threads) {
  pm_vectorized<PMKind::Upper>(degree, target, n_targets, x, n, out,
                               n_threads);
}

void lpm_ratio_v(double degree, const double* target, std::size_t n_targets,
                 const double* x, std::size_t n, double* out, int n_threads) {
  if (degree > 0) {
    pm_vectorized<PMKind::LowerRatio>(degree, target, n_targets, x, n, out,
                                      n_threads);
  } else {
    lpm_v(degree, target, n_targets, x, n, out, n_threads);
  }
}

void upm_ratio_v(double degree, const double* target, std::size_t n_targets,
                 const double* x, std::size_t n, double* out, int n_threads) {
  if (degree > 0) {
    pm_vectorized<PMKind::UpperRatio>(degree, target, n_targets, x, n, out,
                                      n_threads);
  } else {
    upm_v(degree, target, n_targets, x, n, out, n_threads);
  }
}

// --- bivariate co-moments (ports of CoUPM_C/CoLPM_C/DLPM_C/DUPM_C) ---------

double co_upm(double degree_x, double degree_y, const double* x,
              const double* y, std::size_t n_x, std::size_t n_y,
              double target_x, double target_y) {
  const std::size_t max_size = (n_x > n_y ? n_x : n_y);
  const std::size_t min_size = (n_x < n_y ? n_x : n_y);
  if (min_size == 0) return 0;
  double out = 0;
  const bool d_x_0 = (degree_x == 0), d_y_0 = (degree_y == 0);
  const bool x_is_int = is_integer(degree_x), y_is_int = is_integer(degree_y);
  for (std::size_t i = 0; i < min_size; ++i) {
    double x1 = (x[i] - target_x);
    double y1 = (y[i] - target_y);
    if (d_x_0)
      x1 = (x1 > 0 ? 1 : 0);
    else
      x1 = (x1 < 0 ? 0 : x1);
    if (d_y_0)
      y1 = (y1 > 0 ? 1 : 0);
    else
      y1 = (y1 < 0 ? 0 : y1);
    if (!d_x_0)
      x1 = x_is_int ? repeat_multiplication(x1, static_cast<int>(degree_x))
                    : std::pow(x1, degree_x);
    if (!d_y_0)
      y1 = y_is_int ? repeat_multiplication(y1, static_cast<int>(degree_y))
                    : std::pow(y1, degree_y);
    out += x1 * y1;
  }
  return out / static_cast<double>(max_size);
}

double co_lpm(double degree_x, double degree_y, const double* x,
              const double* y, std::size_t n_x, std::size_t n_y,
              double target_x, double target_y) {
  const std::size_t max_size = (n_x > n_y ? n_x : n_y);
  const std::size_t min_size = (n_x < n_y ? n_x : n_y);
  if (min_size == 0) return 0;
  double out = 0;
  const bool d_x_0 = (degree_x == 0), d_y_0 = (degree_y == 0);
  const bool x_is_int = is_integer(degree_x), y_is_int = is_integer(degree_y);
  for (std::size_t i = 0; i < min_size; ++i) {
    double x1 = (target_x - x[i]);
    double y1 = (target_y - y[i]);
    if (d_x_0)
      x1 = (x1 >= 0 ? 1 : 0);
    else
      x1 = (x1 < 0 ? 0 : x1);
    if (d_y_0)
      y1 = (y1 >= 0 ? 1 : 0);
    else
      y1 = (y1 < 0 ? 0 : y1);
    if (!d_x_0)
      x1 = x_is_int ? repeat_multiplication(x1, static_cast<int>(degree_x))
                    : std::pow(x1, degree_x);
    if (!d_y_0)
      y1 = y_is_int ? repeat_multiplication(y1, static_cast<int>(degree_y))
                    : std::pow(y1, degree_y);
    out += x1 * y1;
  }
  return out / static_cast<double>(max_size);
}

double d_lpm(double degree_lpm, double degree_upm, const double* x,
             const double* y, std::size_t n_x, std::size_t n_y,
             double target_x, double target_y) {
  const std::size_t max_size = (n_x > n_y ? n_x : n_y);
  const std::size_t min_size = (n_x < n_y ? n_x : n_y);
  if (min_size == 0) return 0;
  double out = 0;
  const bool dont_use_pow_lpm = is_integer(degree_lpm),
             dont_use_pow_upm = is_integer(degree_upm),
             d_lpm_0 = (degree_lpm == 0), d_upm_0 = (degree_upm == 0);
  for (std::size_t i = 0; i < min_size; ++i) {
    double x1 = (x[i] - target_x);
    double y1 = (target_y - y[i]);
    if (d_upm_0)
      x1 = (x1 > 0 ? 1 : 0);
    else
      x1 = (x1 < 0 ? 0 : x1);
    if (d_lpm_0)
      y1 = (y1 >= 0 ? 1 : 0);
    else
      y1 = (y1 < 0 ? 0 : y1);
    if (dont_use_pow_lpm && dont_use_pow_upm) {
      if (!d_upm_0) x1 = repeat_multiplication(x1, static_cast<int>(degree_upm));
      if (!d_lpm_0) y1 = repeat_multiplication(y1, static_cast<int>(degree_lpm));
      out += x1 * y1;
    } else if (dont_use_pow_lpm && !dont_use_pow_upm) {
      if (!d_lpm_0) y1 = repeat_multiplication(y1, static_cast<int>(degree_lpm));
      out += std::pow(x1, degree_upm) * y1;
    } else if (dont_use_pow_upm && !dont_use_pow_lpm) {
      if (!d_upm_0) x1 = repeat_multiplication(x1, static_cast<int>(degree_upm));
      out += x1 * std::pow(y1, degree_lpm);
    } else {
      out += std::pow(x1, degree_upm) * std::pow(y1, degree_lpm);
    }
  }
  return out / static_cast<double>(max_size);
}

double d_upm(double degree_lpm, double degree_upm, const double* x,
             const double* y, std::size_t n_x, std::size_t n_y,
             double target_x, double target_y) {
  const std::size_t max_size = (n_x > n_y ? n_x : n_y);
  const std::size_t min_size = (n_x < n_y ? n_x : n_y);
  if (min_size == 0) return 0;
  double out = 0;
  const bool dont_use_pow_lpm = is_integer(degree_lpm),
             dont_use_pow_upm = is_integer(degree_upm),
             d_lpm_0 = (degree_lpm == 0), d_upm_0 = (degree_upm == 0);
  for (std::size_t i = 0; i < min_size; ++i) {
    double x1 = (target_x - x[i]);
    double y1 = (y[i] - target_y);
    if (d_lpm_0)
      x1 = (x1 >= 0 ? 1 : 0);
    else
      x1 = (x1 < 0 ? 0 : x1);
    if (d_upm_0)
      y1 = (y1 > 0 ? 1 : 0);
    else
      y1 = (y1 < 0 ? 0 : y1);
    if (dont_use_pow_lpm && dont_use_pow_upm) {
      if (!d_lpm_0) x1 = repeat_multiplication(x1, static_cast<int>(degree_lpm));
      if (!d_upm_0) y1 = repeat_multiplication(y1, static_cast<int>(degree_upm));
      out += x1 * y1;
    } else if (dont_use_pow_lpm && !dont_use_pow_upm) {
      if (!d_upm_0) y1 = repeat_multiplication(y1, static_cast<int>(degree_upm));
      out += std::pow(x1, degree_lpm) * y1;
    } else if (dont_use_pow_upm && !dont_use_pow_lpm) {
      if (!d_lpm_0) x1 = repeat_multiplication(x1, static_cast<int>(degree_lpm));
      out += x1 * std::pow(y1, degree_upm);
    } else {
      out += std::pow(x1, degree_lpm) * std::pow(y1, degree_upm);
    }
  }
  return out / static_cast<double>(max_size);
}

// --- vectorized bivariate (port of NNS_PM_TWO_VARIABLES_WORKER macro) ------

namespace {

template <typename ScalarFn>
void two_var_vectorized(ScalarFn&& scalar, const double* target_x,
                        std::size_t n_tx, const double* target_y,
                        std::size_t n_ty, double* out, int n_threads) {
  const std::size_t n_out = (n_tx > n_ty ? n_tx : n_ty);
  parallel_for(
      0, n_out,
      [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i)
          out[i] = scalar(target_x[i % n_tx], target_y[i % n_ty]);
      },
      n_threads);
}

}  // namespace

void co_lpm_v(double degree_x, double degree_y, const double* x,
              const double* y, std::size_t n_x, std::size_t n_y,
              const double* target_x, std::size_t n_tx, const double* target_y,
              std::size_t n_ty, double* out, int n_threads) {
  two_var_vectorized(
      [&](double tx, double ty) {
        return co_lpm(degree_x, degree_y, x, y, n_x, n_y, tx, ty);
      },
      target_x, n_tx, target_y, n_ty, out, n_threads);
}

void co_upm_v(double degree_x, double degree_y, const double* x,
              const double* y, std::size_t n_x, std::size_t n_y,
              const double* target_x, std::size_t n_tx, const double* target_y,
              std::size_t n_ty, double* out, int n_threads) {
  two_var_vectorized(
      [&](double tx, double ty) {
        return co_upm(degree_x, degree_y, x, y, n_x, n_y, tx, ty);
      },
      target_x, n_tx, target_y, n_ty, out, n_threads);
}

void d_lpm_v(double degree_lpm, double degree_upm, const double* x,
             const double* y, std::size_t n_x, std::size_t n_y,
             const double* target_x, std::size_t n_tx, const double* target_y,
             std::size_t n_ty, double* out, int n_threads) {
  two_var_vectorized(
      [&](double tx, double ty) {
        return d_lpm(degree_lpm, degree_upm, x, y, n_x, n_y, tx, ty);
      },
      target_x, n_tx, target_y, n_ty, out, n_threads);
}

void d_upm_v(double degree_lpm, double degree_upm, const double* x,
             const double* y, std::size_t n_x, std::size_t n_y,
             const double* target_x, std::size_t n_tx, const double* target_y,
             std::size_t n_ty, double* out, int n_threads) {
  two_var_vectorized(
      [&](double tx, double ty) {
        return d_upm(degree_lpm, degree_upm, x, y, n_x, n_y, tx, ty);
      },
      target_x, n_tx, target_y, n_ty, out, n_threads);
}

// --- n-dimensional co-partial moments (ports of *_nD_cpp) ------------------

namespace {

// data: column-major n x d -> element (i, j) = data[j * n + i]
inline double at(const double* m, std::size_t n, std::size_t i,
                 std::size_t j) {
  return m[j * n + i];
}

void check_nd_args(std::size_t n, std::size_t d) {
  if (d == 0) throw std::invalid_argument("`data` must have at least one column");
  if (n == 0) throw std::invalid_argument("`data` must have at least one row");
}

// Parallel row reduction: each chunk fills disjoint slots, then a serial sum.
// Mirrors R's parallelFor-into-vector + sum() pattern (same summation order
// as the upstream NumericVector accumulation).
template <typename RowFn>
double reduce_rows(std::size_t n, int n_threads, RowFn&& row_value) {
  std::vector<double> out(n);
  parallel_for(
      0, n,
      [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) out[i] = row_value(i);
      },
      n_threads);
  double total = 0.0;
  for (std::size_t i = 0; i < n; ++i) total += out[i];
  return total;
}

double clpm_nd_impl(const double* data, std::size_t n, std::size_t d,
                    const double* target, double degree, int n_threads) {
  const bool deg_is_int = is_integer(degree);
  if (degree == 0.0) {
    const double count = reduce_rows(n, n_threads, [&](std::size_t i) {
      for (std::size_t j = 0; j < d; ++j)
        if (at(data, n, i, j) > target[j]) return 0.0;
      return 1.0;
    });
    return count / static_cast<double>(n);
  }
  const double s = reduce_rows(n, n_threads, [&](std::size_t i) {
    double prod = 1.0;
    for (std::size_t j = 0; j < d; ++j) {
      const double diff = target[j] - at(data, n, i, j);
      if (diff < 0.0) return 0.0;
      prod *= deg_is_int ? repeat_multiplication(diff, static_cast<int>(degree))
                         : std::pow(diff, degree);
    }
    return prod;
  });
  return s / static_cast<double>(n);
}

double cupm_nd_impl(const double* data, std::size_t n, std::size_t d,
                    const double* target, double degree, int n_threads) {
  const bool deg_is_int = is_integer(degree);
  if (degree == 0.0) {
    const double count = reduce_rows(n, n_threads, [&](std::size_t i) {
      for (std::size_t j = 0; j < d; ++j)
        if (at(data, n, i, j) < target[j]) return 0.0;
      return 1.0;
    });
    return count / static_cast<double>(n);
  }
  const double s = reduce_rows(n, n_threads, [&](std::size_t i) {
    double prod = 1.0;
    for (std::size_t j = 0; j < d; ++j) {
      const double diff = at(data, n, i, j) - target[j];
      if (diff < 0.0) return 0.0;
      prod *= deg_is_int ? repeat_multiplication(diff, static_cast<int>(degree))
                         : std::pow(diff, degree);
    }
    return prod;
  });
  return s / static_cast<double>(n);
}

double dpm_nd_impl(const double* data, std::size_t n, std::size_t d,
                   const double* target, double degree, int n_threads) {
  const bool deg_is_int = is_integer(degree);
  if (degree == 0.0) {
    const double count = reduce_rows(n, n_threads, [&](std::size_t i) {
      bool all_below = true, all_above = true;
      for (std::size_t j = 0; j < d; ++j) {
        const double diff = at(data, n, i, j) - target[j];
        if (diff >= 0.0) all_below = false;
        if (diff <= 0.0) all_above = false;
        if (!all_below && !all_above) break;
      }
      return (!all_below && !all_above) ? 1.0 : 0.0;
    });
    return count / static_cast<double>(n);
  }
  const double s = reduce_rows(n, n_threads, [&](std::size_t i) {
    bool all_below = true, all_above = true;
    for (std::size_t j = 0; j < d; ++j) {
      const double diff = at(data, n, i, j) - target[j];
      if (diff >= 0.0) all_below = false;
      if (diff <= 0.0) all_above = false;
      if (!all_below && !all_above) break;
    }
    if (all_below || all_above) return 0.0;
    double prod = 1.0;
    for (std::size_t j = 0; j < d; ++j) {
      const double abs_dev = std::abs(at(data, n, i, j) - target[j]);
      prod *= deg_is_int
                  ? repeat_multiplication(abs_dev, static_cast<int>(degree))
                  : std::pow(abs_dev, degree);
    }
    return prod;
  });
  return s / static_cast<double>(n);
}

}  // namespace

double clpm_nd(const double* data, std::size_t n, std::size_t d,
               const double* target, double degree, bool norm, int n_threads) {
  check_nd_args(n, d);
  const double clpm_un = clpm_nd_impl(data, n, d, target, degree, n_threads);
  if (degree == 0.0 || !norm) return clpm_un;
  const double cupm_un = cupm_nd_impl(data, n, d, target, degree, n_threads);
  const double dpm_un = dpm_nd_impl(data, n, d, target, degree, n_threads);
  const double norm_const = clpm_un + cupm_un + dpm_un;
  return norm_const > 0.0 ? clpm_un / norm_const : 0.0;
}

double cupm_nd(const double* data, std::size_t n, std::size_t d,
               const double* target, double degree, bool norm, int n_threads) {
  check_nd_args(n, d);
  const double cupm_un = cupm_nd_impl(data, n, d, target, degree, n_threads);
  if (degree == 0.0 || !norm) return cupm_un;
  const double clpm_un = clpm_nd_impl(data, n, d, target, degree, n_threads);
  const double dpm_un = dpm_nd_impl(data, n, d, target, degree, n_threads);
  const double norm_const = clpm_un + cupm_un + dpm_un;
  return norm_const > 0.0 ? cupm_un / norm_const : 0.0;
}

double dpm_nd(const double* data, std::size_t n, std::size_t d,
              const double* target, double degree, bool norm, int n_threads) {
  check_nd_args(n, d);
  const double dpm_un = dpm_nd_impl(data, n, d, target, degree, n_threads);
  if (degree == 0.0 || !norm) return dpm_un;
  const double clpm_un = clpm_nd_impl(data, n, d, target, degree, n_threads);
  const double cupm_un = cupm_nd_impl(data, n, d, target, degree, n_threads);
  const double norm_const = clpm_un + cupm_un + dpm_un;
  return norm_const > 0.0 ? dpm_un / norm_const : 0.0;
}

// --- batched nD CoLPM (port of CoLPM_nD_batch_RCPP, new in 13.0) ------------

void clpm_nd_batch(const double* data, std::size_t n, std::size_t d,
                   const double* targets, std::size_t n_targets, double degree,
                   bool norm, double* out, int n_threads) {
  if (n == 0) throw std::invalid_argument("`data` must have at least one row");
  const bool deg_is_int = is_integer(degree);

  parallel_for(
      0, n_targets,
      [&](std::size_t begin, std::size_t end) {
        for (std::size_t r = begin; r < end; ++r) {
          if (degree == 0.0) {
            double count = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
              bool below_all = true;
              for (std::size_t j = 0; j < d; ++j) {
                if (at(data, n, i, j) > at(targets, n_targets, r, j)) {
                  below_all = false;
                  break;
                }
              }
              if (below_all) count += 1.0;
            }
            out[r] = count / static_cast<double>(n);
            continue;
          }
          double clpm_sum = 0.0, cupm_sum = 0.0, dpm_sum = 0.0;
          for (std::size_t i = 0; i < n; ++i) {
            double lower_prod = 1.0, upper_prod = 1.0, dpm_prod = 1.0;
            bool all_below_strict = true, all_above_strict = true;
            for (std::size_t j = 0; j < d; ++j) {
              const double diff =
                  at(data, n, i, j) - at(targets, n_targets, r, j);
              lower_prod *= lower_component(-diff, degree, deg_is_int);
              upper_prod *= upper_component(diff, degree, deg_is_int);
              if (diff >= 0.0) all_below_strict = false;
              if (diff <= 0.0) all_above_strict = false;
              dpm_prod *=
                  deg_is_int
                      ? repeat_multiplication(std::abs(diff),
                                              static_cast<int>(degree))
                      : std::pow(std::abs(diff), degree);
            }
            clpm_sum += lower_prod;
            cupm_sum += upper_prod;
            if (!(all_below_strict || all_above_strict)) dpm_sum += dpm_prod;
          }
          const double inv_n = 1.0 / static_cast<double>(n);
          const double clpm_un = clpm_sum * inv_n;
          if (!norm) {
            out[r] = clpm_un;
          } else {
            const double cupm_un = cupm_sum * inv_n;
            const double dpm_un = dpm_sum * inv_n;
            const double norm_const = clpm_un + cupm_un + dpm_un;
            out[r] = norm_const > 0.0 ? clpm_un / norm_const : 0.0;
          }
        }
      },
      n_threads);
}

// --- PM matrix (port of the 13.0 tensorized PMMatrix_CPv) -------------------

PMMatrixResult pm_matrix(double degree_lpm, double degree_upm,
                         const double* target, const double* variable,
                         std::size_t n, std::size_t d, bool pop_adj, bool norm,
                         int n_threads) {
  // Mirrors Rcpp::stop("variable matrix cols != target vector length") —
  // bindings pass target length separately, so enforce d > 0 here and let
  // the binding layer validate target length against d before calling.
  PMMatrixResult res;
  if (n == 0) return res;
  res.dim = d;
  res.cupm.assign(d * d, 0.0);
  res.dupm.assign(d * d, 0.0);
  res.dlpm.assign(d * d, 0.0);
  res.clpm.assign(d * d, 0.0);
  res.cov.assign(d * d, 0.0);

  const bool lpm_is_int = is_integer(degree_lpm);
  const bool upm_is_int = is_integer(degree_upm);

  // Step 1: precompute deviation matrices once per element (column-parallel).
  std::vector<double> D_lower(n * d), D_upper(n * d);
  parallel_for(
      0, d,
      [&](std::size_t begin, std::size_t end) {
        for (std::size_t j = begin; j < end; ++j) {
          const double t_j = target[j];
          for (std::size_t i = 0; i < n; ++i) {
            const double val = at(variable, n, i, j);
            D_lower[j * n + i] =
                lower_component(t_j - val, degree_lpm, lpm_is_int);
            D_upper[j * n + i] =
                upper_component(val - t_j, degree_upm, upm_is_int);
          }
        }
      },
      n_threads);

  double adjust = 1.0;
  if (pop_adj && n > 1)
    adjust = static_cast<double>(n) / static_cast<double>(n - 1);
  const bool apply_adj = pop_adj && n > 1 && degree_lpm > 0 && degree_upm > 0;
  const double inv_rows = 1.0 / static_cast<double>(n);

  auto M = [d](std::vector<double>& m, std::size_t i,
               std::size_t j) -> double& { return m[j * d + i]; };

  // Step 2: fused contraction over the upper triangle, crossed DUPM/DLPM
  // mirror — identical to FusedMatrixMultiplicationWorker.
  parallel_for(
      0, d,
      [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
          for (std::size_t j = i; j < d; ++j) {
            double sum_cupm = 0.0, sum_clpm = 0.0, sum_dupm = 0.0,
                   sum_dlpm = 0.0;
            const double* u_i = &D_upper[i * n];
            const double* l_i = &D_lower[i * n];
            const double* u_j = &D_upper[j * n];
            const double* l_j = &D_lower[j * n];
            for (std::size_t k = 0; k < n; ++k) {
              sum_cupm += u_i[k] * u_j[k];
              sum_clpm += l_i[k] * l_j[k];
              sum_dupm += l_i[k] * u_j[k];
              sum_dlpm += u_i[k] * l_j[k];
            }
            sum_cupm *= inv_rows;
            sum_clpm *= inv_rows;
            sum_dupm *= inv_rows;
            sum_dlpm *= inv_rows;
            if (apply_adj) {
              sum_cupm *= adjust;
              sum_clpm *= adjust;
              sum_dupm *= adjust;
              sum_dlpm *= adjust;
            }
            const double cov_ij = sum_cupm + sum_clpm - sum_dupm - sum_dlpm;
            M(res.cupm, i, j) = sum_cupm;
            M(res.clpm, i, j) = sum_clpm;
            M(res.dupm, i, j) = sum_dupm;
            M(res.dlpm, i, j) = sum_dlpm;
            M(res.cov, i, j) = cov_ij;
            if (j != i) {
              M(res.cupm, j, i) = sum_cupm;
              M(res.clpm, j, i) = sum_clpm;
              M(res.dupm, j, i) = sum_dlpm;  // crossed mirror
              M(res.dlpm, j, i) = sum_dupm;  // crossed mirror
              M(res.cov, j, i) = cov_ij;
            }
          }
        }
      },
      n_threads);

  // Step 3: cellular normalization, preserving the crossed mirror.
  if (norm) {
    for (std::size_t i = 0; i < d; ++i) {
      for (std::size_t j = i; j < d; ++j) {
        double cupm_ij = M(res.cupm, i, j);
        double dupm_ij = M(res.dupm, i, j);
        double dlpm_ij = M(res.dlpm, i, j);
        double clpm_ij = M(res.clpm, i, j);
        const double total = cupm_ij + dupm_ij + dlpm_ij + clpm_ij;
        if (total > 0.0) {
          cupm_ij /= total;
          dupm_ij /= total;
          dlpm_ij /= total;
          clpm_ij /= total;
        } else {
          cupm_ij = dupm_ij = dlpm_ij = clpm_ij = 0.0;
        }
        const double cov_ij = cupm_ij + clpm_ij - dupm_ij - dlpm_ij;
        M(res.cupm, i, j) = cupm_ij;
        M(res.clpm, i, j) = clpm_ij;
        M(res.dupm, i, j) = dupm_ij;
        M(res.dlpm, i, j) = dlpm_ij;
        M(res.cov, i, j) = cov_ij;
        if (j != i) {
          M(res.cupm, j, i) = cupm_ij;
          M(res.clpm, j, i) = clpm_ij;
          M(res.dupm, j, i) = dlpm_ij;  // crossed mirror after norm too
          M(res.dlpm, j, i) = dupm_ij;
          M(res.cov, j, i) = cov_ij;
        }
      }
    }
  }
  return res;
}

}  // namespace nns