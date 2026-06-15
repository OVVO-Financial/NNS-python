#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <limits>
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

  double denom = 0.0, weighted = 0.0;
  for (std::size_t i = lo; i <= hi; ++i) {
    const double bin_name = origin + static_cast<double>(i) * width;
    denom += static_cast<double>(counts[i]);
    weighted += bin_name * static_cast<double>(counts[i]);
  }
  const double mode_gravity =
      denom > 0.0 ? weighted / denom : origin + static_cast<double>((lo + hi) / 2U) * width;

  double sum = 0.0;
  for (const double v : values) sum += v;
  const double mean = sum / dn;

  return 0.25 * (q2 + mode_gravity + mean + 0.5 * (q1 + q3));
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
}
