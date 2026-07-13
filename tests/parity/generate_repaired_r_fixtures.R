#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(name, default = NULL) {
  flag <- paste0("--", name)
  idx <- match(flag, args)
  if (is.na(idx) || idx == length(args)) return(default)
  args[[idx + 1]]
}

r_repo <- normalizePath(get_arg("r-repo", "../NNS-r"), mustWork = TRUE)
out_dir <- get_arg("out", "tests/parity/fixtures/repaired_r_13_1_54c98418")
r_commit <- get_arg("commit", NA_character_)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages({
  library(jsonlite)
  library(devtools)
  library(digest)
})

options(
  NNS.native.stack = FALSE,
  NNS.native.mreg = FALSE,
  NNS.native.univariate = FALSE
)

devtools::load_all(r_repo, quiet = TRUE)

pkg_desc <- desc::desc(file.path(r_repo, "DESCRIPTION"))
metadata <- list(
  fixture_schema_version = "repaired_r_13_1_54c98418",
  generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
  r_repository = "OVVO-Financial/NNS",
  r_commit_sha = r_commit,
  nns_version = pkg_desc$get("Version"),
  r_version = paste(R.version$major, R.version$minor, sep = "."),
  platform = R.version$platform,
  os = Sys.info()[["sysname"]],
  native_reference_options = list(
    NNS.native.stack = getOption("NNS.native.stack"),
    NNS.native.mreg = getOption("NNS.native.mreg"),
    NNS.native.univariate = getOption("NNS.native.univariate")
  )
)

write_json(metadata, file.path(out_dir, "metadata.json"), pretty = TRUE, auto_unbox = TRUE)
manifest <- list(
  schema_version = 1,
  r_repository = metadata$r_repository,
  r_commit = metadata$r_commit_sha,
  nns_version = metadata$nns_version,
  reference_backend = list(stack = FALSE, mreg = FALSE, univariate = FALSE)
)
write_json(manifest, file.path(out_dir, "manifest.json"), pretty = TRUE, auto_unbox = TRUE)

as_num <- function(x) as.numeric(x)
case_rows <- list()

with_checksums <- function(case) {
  case$input_checksum <- digest(case$input, algo = "sha256", serialize = TRUE)
  case$output_checksum <- digest(case$output, algo = "sha256", serialize = TRUE)
  case
}

capture_part_case <- function(name, x, y = NULL, order = NULL, type = NULL, noise.reduction = "mean", obs.req = NULL) {
  args <- list(x = x, y = y, order = order, type = type, noise.reduction = noise.reduction, obs.req = obs.req)
  result <- do.call(NNS.part, args[!vapply(args, is.null, logical(1))])
  with_checksums(list(
    name = name,
    kind = "part",
    input = list(x = x, y = y),
    args = list(order = order, type = type, noise.reduction = noise.reduction, obs.req = obs.req),
    output = result
  ))
}

capture_part_error_case <- function(name, x, y = NULL, order = NULL, type = NULL, obs.req = NULL) {
  error <- tryCatch({
    args <- list(x = x, y = y, order = order, type = type, obs.req = obs.req)
    do.call(NNS.part, args[!vapply(args, is.null, logical(1))])
    NULL
  }, error = function(e) conditionMessage(e))
  with_checksums(list(
    name = name,
    kind = "part",
    input = list(x = x, y = y),
    args = list(order = order, type = type, obs.req = obs.req),
    output = list(error = error)
  ))
}

capture_reg_case <- function(name, x, y, point = NULL, type = NULL, order = NULL, n.best = NULL, smooth = FALSE, pred.int = NULL) {
  result <- NNS.reg(
    x = x,
    y = y,
    point.est = point,
    type = type,
    order = order,
    n.best = n.best,
    smooth = smooth,
    pred.int = pred.int,
    plot = FALSE,
    residual.plot = FALSE,
    ncores = 1
  )
  with_checksums(list(
    name = name,
    kind = "reg",
    input = list(x = unclass(x), y = unclass(y), point = if (is.null(point)) NULL else unclass(point)),
    args = list(type = type, order = order, n.best = n.best, smooth = smooth, pred.int = pred.int),
    output = result
  ))
}

capture_mreg_case <- function(name, x, y, point = NULL, type = NULL, order = NULL, n.best = NULL) {
  result <- NNS.M.reg(
    x = x,
    y = y,
    point.est = point,
    type = type,
    order = order,
    n.best = n.best,
    plot = FALSE,
    residual.plot = FALSE,
    ncores = 1
  )
  with_checksums(list(
    name = name,
    kind = "mreg",
    input = list(x = unclass(x), y = unclass(y), point = if (is.null(point)) NULL else unclass(point)),
    args = list(type = type, order = order, n.best = n.best),
    output = result
  ))
}

capture_stack_case <- function(name, x, y, point = NULL, method = c(1, 2), type = NULL, ts.test = NULL, balance = FALSE, pred.int = NULL) {
  result <- NNS.stack(
    IVs.train = x,
    DV.train = y,
    IVs.test = point,
    method = method,
    type = type,
    ts.test = ts.test,
    balance = balance,
    pred.int = pred.int,
    folds = 1,
    status = FALSE,
    ncores = 1
  )
  with_checksums(list(
    name = name,
    kind = "stack",
    input = list(x = unclass(x), y = unclass(y), point = if (is.null(point)) NULL else unclass(point)),
    args = list(method = method, type = type, ts.test = ts.test, balance = balance, pred.int = pred.int),
    output = result
  ))
}


capture_boost_case <- function(name, x, y, point = NULL, type = NULL, ts.test = NULL, pred.int = NULL) {
  result <- NNS.boost(
    IVs.train = x,
    DV.train = y,
    IVs.test = point,
    type = type,
    ts.test = ts.test,
    pred.int = pred.int,
    learner.trials = 10,
    status = FALSE,
    ncores = 1
  )
  with_checksums(list(
    name = name,
    kind = "boost",
    input = list(x = unclass(x), y = unclass(y), point = if (is.null(point)) NULL else unclass(point)),
    args = list(type = type, ts.test = ts.test, pred.int = pred.int, learner.trials = 10),
    output = result
  ))
}

capture_var_case <- function(name, variables, h = 3, tau = 1, dim.red.method = "cor") {
  result <- NNS.VAR(
    variables = variables,
    h = h,
    tau = tau,
    dim.red.method = dim.red.method,
    status = FALSE,
    ncores = 1
  )
  with_checksums(list(
    name = name,
    kind = "var",
    input = list(variables = unclass(variables)),
    args = list(h = h, tau = tau, dim.red.method = dim.red.method),
    output = result
  ))
}


part_x <- c(1, 1, 2, 3, 4, NA, NaN, Inf)
part_y <- c(2, 4, 6, 8, 10, 12, 14, 16)
case_rows[[length(case_rows) + 1]] <- capture_part_case("part_default", part_x, part_y, order = NULL, type = NULL, noise.reduction = "mean")
case_rows[[length(case_rows) + 1]] <- capture_part_case("part_numeric_order", part_x, part_y, order = 2, type = NULL, noise.reduction = "median")
case_rows[[length(case_rows) + 1]] <- capture_part_case("part_order_max", part_x, part_y, order = "max", type = NULL, noise.reduction = "off")
case_rows[[length(case_rows) + 1]] <- capture_part_case("part_order_max_xonly", part_x, part_y, order = "max", type = "XONLY", noise.reduction = "mean")
case_rows[[length(case_rows) + 1]] <- capture_part_case("part_mode", c(1, 1, 2, 2, 3), c(4, 4, 5, 6, 6), order = "max", type = "XONLY", noise.reduction = "mode")
case_rows[[length(case_rows) + 1]] <- capture_part_case("part_mode_class", c(1, 1, 2, 2, 3), c(10, 20, 20, 10, 10), order = "max", type = "XONLY", noise.reduction = "mode.class")
case_rows[[length(case_rows) + 1]] <- capture_part_error_case("part_invalid_type", part_x, part_y, order = 1, type = "INVALID")
case_rows[[length(case_rows) + 1]] <- capture_part_error_case("part_invalid_order", part_x, part_y, order = 0, type = NULL)
case_rows[[length(case_rows) + 1]] <- capture_part_error_case("part_invalid_obs_req", part_x, part_y, order = 1, type = NULL, obs.req = 0)

reg_x <- seq(-3, 3, length.out = 18)
reg_y <- reg_x^3 - reg_x
case_rows[[length(case_rows) + 1]] <- capture_reg_case("reg_default", reg_x, reg_y, point = c(-2.5, 0.25, 3.5), order = NULL)
case_rows[[length(case_rows) + 1]] <- capture_reg_case("reg_integer_order", reg_x, reg_y, point = c(-2.5, 0.25, 3.5), order = 2)
case_rows[[length(case_rows) + 1]] <- capture_reg_case("reg_order_max", c(reg_x, reg_x[5]), c(reg_y, reg_y[5] + 1), point = c(-2.5, 0.25, 3.5), order = "max")
case_rows[[length(case_rows) + 1]] <- capture_reg_case("reg_smooth", reg_x, reg_y + sin(reg_x), point = c(-2.5, 0.25, 3.5), smooth = TRUE)
case_rows[[length(case_rows) + 1]] <- capture_reg_case("reg_class_nonconsecutive", reg_x, ifelse(reg_x < -1, 10, ifelse(reg_x > 1, 30, 20)), point = c(-2, 0, 2), type = "CLASS")
case_rows[[length(case_rows) + 1]] <- capture_reg_case("reg_factor_response", reg_x, factor(ifelse(reg_x < 0, "down", "up"), levels = c("up", "down")), point = c(-2, 2), type = "CLASS")
case_rows[[length(case_rows) + 1]] <- capture_reg_case("reg_pred_int", reg_x, reg_y, point = c(-2.5, 0.25, 3.5), pred.int = 0.95)

x0 <- seq(-2, 2, length.out = 24)
X <- cbind(x0, sin(x0), cos(x0))
y <- x0 + sin(x0)
case_rows[[length(case_rows) + 1]] <- capture_mreg_case("numeric_l2_default", X, y, X[1:4, ], order = NULL, n.best = NULL)
case_rows[[length(case_rows) + 1]] <- capture_mreg_case("numeric_order_max", X, y, X[1:4, ], order = "max", n.best = 1)
case_rows[[length(case_rows) + 1]] <- capture_mreg_case("rightmost_boundary", matrix(c(0, 1, 1.5, 2, 3.5, 4, 5), ncol = 1), c(0, 1, 1, 2, 3, 3, 4), matrix(c(1, 4, 5), ncol = 1), order = "max", n.best = 1)
classes <- ifelse(x0 < -0.5, 1, ifelse(x0 > 0.75, 3, 2))
case_rows[[length(case_rows) + 1]] <- capture_mreg_case("multiclass", X, classes, X[1:4, ], type = "CLASS", order = 1, n.best = 1)
case_rows[[length(case_rows) + 1]] <- capture_stack_case("stack_method1_regression", X, y, X[1:4, ], method = 1)
case_rows[[length(case_rows) + 1]] <- capture_stack_case("stack_method12_ts", X, y, X[1:4, ], method = c(1, 2), ts.test = 5)
case_rows[[length(case_rows) + 1]] <- capture_stack_case("stack_classification", X, classes, X[1:4, ], method = c(1, 2), type = "CLASS")
case_rows[[length(case_rows) + 1]] <- capture_stack_case("stack_pred_int", X, y, X[1:4, ], method = c(1, 2), pred.int = 0.95)
case_rows[[length(case_rows) + 1]] <- capture_boost_case("boost_numeric", X, y, X[1:4, ])
case_rows[[length(case_rows) + 1]] <- capture_boost_case("boost_ts", X, y, X[1:4, ], ts.test = 5)
case_rows[[length(case_rows) + 1]] <- capture_boost_case("boost_class_pred_int", X, classes, X[1:4, ], type = "CLASS", pred.int = 0.95)
variables <- cbind(seq(-2, 17, by = 1), seq(1, 39, by = 2))
case_rows[[length(case_rows) + 1]] <- capture_var_case("var_cor_tau1", variables, h = 3, tau = 1, dim.red.method = "cor")
variables_missing <- variables
variables_missing[5, 1] <- NA
variables_missing[nrow(variables_missing), 2] <- NA
case_rows[[length(case_rows) + 1]] <- capture_var_case("var_cor_missing", variables_missing, h = 3, tau = 2, dim.red.method = "cor")

write_json(list(metadata = metadata, manifest = manifest, cases = case_rows), file.path(out_dir, "fixtures.json"), pretty = TRUE, auto_unbox = TRUE, digits = NA)
cat("Generated repaired R fixtures in ", out_dir, "\n", sep = "")
