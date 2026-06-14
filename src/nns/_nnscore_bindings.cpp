#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <cstddef>
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
}
