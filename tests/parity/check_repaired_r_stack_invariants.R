#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1) args[[1]] else "artifacts/repaired-r-validation"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages({
  library(jsonlite)
  library(devtools)
})

options(
  NNS.native.stack = FALSE,
  NNS.native.mreg = FALSE,
  NNS.native.univariate = FALSE
)

devtools::load_all(getwd(), quiet = TRUE)

write_failure <- function(name, payload) {
  write_json(
    payload,
    file.path(out_dir, paste0("stack-invariant-failure-", name, ".json")),
    pretty = TRUE,
    auto_unbox = TRUE,
    digits = NA
  )
}

assert_equal <- function(name, actual, expected, tolerance = 1e-12, exact = FALSE, context = list()) {
  ok <- if (exact) {
    identical(as.vector(actual), as.vector(expected))
  } else {
    isTRUE(all.equal(as.numeric(actual), as.numeric(expected), tolerance = tolerance, check.attributes = FALSE))
  }
  if (!ok) {
    payload <- c(
      list(
        invariant = name,
        actual = as.vector(actual),
        expected = as.vector(expected),
        tolerance = tolerance,
        exact = exact
      ),
      context
    )
    write_failure(name, payload)
    stop("R repaired stack invariant failed: ", name, call. = FALSE)
  }
}

check_scalar_clean <- function(result, fields) {
  for (field in fields) {
    value <- result[[field]]
    if (is.null(value)) next
    if (length(value) != 1 || !is.null(names(value))) {
      write_failure(
        paste0("scalar-clean-", field),
        list(field = field, value = value, names = names(value), length = length(value))
      )
      stop("R stack scalar field is not an unnamed scalar: ", field, call. = FALSE)
    }
  }
}

find_named_trace <- function(x, names_to_try) {
  if (!is.list(x)) return(NULL)
  for (name in names_to_try) {
    if (!is.null(x[[name]])) return(x[[name]])
  }
  for (item in x) {
    found <- find_named_trace(item, names_to_try)
    if (!is.null(found)) return(found)
  }
  NULL
}

require_method1_trace <- function(result, context) {
  trace <- find_named_trace(
    result,
    c(
      "method1_trace",
      "method.1.trace",
      "Method1.trace",
      "Method.1.trace",
      "NNS.reg.trace",
      "reg_trace",
      "candidate_trace"
    )
  )
  if (is.null(trace)) {
    write_failure(
      "method1-trace-missing",
      c(
        list(
          message = paste(
            "NNS.stack did not expose internal Method 1 fold candidate predictions.",
            "The invariant must compare actual internal stack candidates against direct NNS.reg fits,",
            "so fixture generation is blocked until the R reference exposes this trace."
          ),
          result_names = names(result)
        ),
        context
      )
    )
    stop("R stack invariant cannot inspect actual internal Method 1 candidates", call. = FALSE)
  }
  trace
}

field <- function(x, names_to_try, default = NULL) {
  if (!is.list(x)) return(default)
  for (name in names_to_try) {
    if (!is.null(x[[name]])) return(x[[name]])
  }
  default
}

as_candidate_id <- function(candidate) {
  raw <- field(candidate, c("candidate", "candidate_id", "id", "k", "n.best", "n_best"))
  if (is.null(raw)) return(NA_character_)
  as.character(raw[[1L]])
}

check_method1_internal_candidates_against_direct_reg <- function() {
  set.seed(131)
  n <- 500
  X <- cbind(
    seq(-2, 2, length.out = n),
    sin(seq(-2, 2, length.out = n)),
    cos(seq(-2, 2, length.out = n)),
    rep(c(-1, 0, 1, 0), length.out = n),
    seq(-2, 2, length.out = n)^2
  )
  y <- X[, 1] + 0.5 * X[, 2] - X[, 3] + 0.1 * X[, 4]
  ts_test <- 50
  folds <- 3
  options(NNS.stack.return.method1.trace = TRUE)
  result <- NNS.stack(
    IVs.train = X,
    DV.train = y,
    IVs.test = X[1:5, , drop = FALSE],
    method = 1,
    ts.test = ts_test,
    folds = folds,
    status = FALSE,
    ncores = 1
  )
  trace <- require_method1_trace(result, list(n = n, predictors = 5, ts.test = ts_test, folds = folds))
  ordinary_counts <- list()
  ordinary_scores <- list()
  excluded_candidates <- list()
  all_count <- 0L
  all_score <- 0
  for (fold_idx in seq_along(trace)) {
    fold <- trace[[fold_idx]]
    train_x <- field(fold, c("train_x", "fold_train_x", "encoded_train_x", "x_train"))
    train_y <- field(fold, c("train_y", "fold_train_y", "y_train"))
    validation_x <- field(fold, c("validation_x", "fold_validation_x", "encoded_validation_x", "x_validation", "x_valid"))
    validation_y <- field(fold, c("validation_y", "fold_validation_y", "y_validation", "y_valid"))
    candidates <- field(fold, c("candidates", "candidate_predictions", "method1_candidates"))
    if (is.null(train_x) || is.null(train_y) || is.null(validation_x) || is.null(validation_y) || is.null(candidates)) {
      write_failure("method1-trace-incomplete", list(fold = fold_idx, names = names(fold)))
      stop("Method 1 trace is missing fold inputs or candidate predictions", call. = FALSE)
    }
    for (candidate in candidates) {
      candidate_id <- as_candidate_id(candidate)
      pred <- field(candidate, c("prediction", "predictions", "point_est", "point.est", "Point.est"))
      score <- field(candidate, c("score", "objective", "OBJfn", "sse"), default = NA_real_)
      eligible <- isTRUE(field(candidate, c("eligible", "complete", "scored"), default = TRUE))
      if (is.null(pred)) {
        write_failure("method1-candidate-prediction-missing", list(fold = fold_idx, candidate = candidate_id, names = names(candidate)))
        stop("Method 1 candidate trace is missing predictions", call. = FALSE)
      }
      if (!eligible) {
        if (!is.na(score)) {
          write_failure("method1-excluded-candidate-scored", list(fold = fold_idx, candidate = candidate_id, score = score))
          stop("partial/zero-coverage Method 1 candidate received an objective", call. = FALSE)
        }
        excluded_candidates[[length(excluded_candidates) + 1L]] <- list(fold = fold_idx, candidate = candidate_id)
        next
      }
      if (length(pred) == 0L || length(validation_y) == 0L) {
        write_failure("method1-empty-vector-scored", list(fold = fold_idx, candidate = candidate_id, score = score))
        stop("empty predicted/actual vectors cannot produce an objective", call. = FALSE)
      }
      direct_n_best <- if (tolower(candidate_id) == "all") "all" else as.integer(candidate_id)
      direct <- NNS.reg(
        train_x,
        train_y,
        point.est = validation_x,
        n.best = direct_n_best,
        plot = FALSE,
        residual.plot = FALSE,
        ncores = 1
      )$Point.est
      assert_equal(
        paste0("method1-internal-vs-direct-fold-", fold_idx, "-candidate-", candidate_id),
        pred,
        direct,
        context = list(
          fold = fold_idx,
          candidate = candidate_id,
          train_x = train_x,
          train_y = train_y,
          validation_x = validation_x,
          internal_prediction = pred,
          direct_prediction = direct,
          max_abs_diff = max(abs(as.numeric(pred) - as.numeric(direct)))
        )
      )
      if (tolower(candidate_id) == "all") {
        all_count <- all_count + length(pred)
        all_score <- all_score + sum((as.numeric(pred) - as.numeric(validation_y))^2)
      } else {
        key <- as.character(as.integer(candidate_id))
        ordinary_counts[[key]] <- (ordinary_counts[[key]] %||% 0L) + length(pred)
        ordinary_scores[[key]] <- (ordinary_scores[[key]] %||% 0) + sum((as.numeric(pred) - as.numeric(validation_y))^2)
      }
    }
  }
  if (!length(ordinary_counts) || is.null(ordinary_counts[["1"]])) {
    write_failure("method1-k1-missing", list(counts = ordinary_counts))
    stop("candidate k=1 must define the reference OOF count vector", call. = FALSE)
  }
  reference_count <- ordinary_counts[["1"]]
  mismatched <- ordinary_counts[vapply(ordinary_counts, function(value) value != reference_count, logical(1))]
  if (length(mismatched)) {
    write_failure("method1-ordinary-count-vector", list(counts = ordinary_counts, reference_count = reference_count))
    stop("eligible ordinary Method 1 candidates do not share k=1 OOF coverage", call. = FALSE)
  }
  if (all_count != reference_count) {
    write_failure("method1-all-count", list(all_count = all_count, reference_count = reference_count))
    stop("ALL Method 1 candidate does not have complete OOF coverage", call. = FALSE)
  }
  if (any(!is.finite(unlist(ordinary_scores))) || !is.finite(all_score)) {
    write_failure("method1-pooled-scores", list(ordinary_scores = ordinary_scores, all_score = all_score))
    stop("Method 1 pooled scores must be finite and complete", call. = FALSE)
  }
  write_json(
    list(
      ordinary_counts = ordinary_counts,
      ordinary_scores = ordinary_scores,
      excluded_candidates = excluded_candidates,
      all_count = all_count,
      all_score = all_score,
      selected_candidate = result$NNS.reg.n.best
    ),
    file.path(out_dir, "stack-invariant-method1-coverage-proof.json"),
    pretty = TRUE,
    auto_unbox = TRUE,
    digits = NA
  )
}

`%||%` <- function(x, y) if (is.null(x)) y else x

check_stack_scalar_outputs <- function() {
  x <- seq(-2, 2, length.out = 24)
  X <- cbind(x, sin(x), cos(x))
  y <- x + sin(x)
  result <- NNS.stack(
    IVs.train = X,
    DV.train = y,
    IVs.test = X[1:4, , drop = FALSE],
    method = c(1, 2),
    folds = 1,
    status = FALSE,
    ncores = 1
  )
  check_scalar_clean(
    result,
    c("OBJfn.reg", "NNS.reg.n.best", "OBJfn.dim.red", "NNS.dim.red.threshold", "probability.threshold")
  )
}

check_method1_internal_candidates_against_direct_reg()
check_stack_scalar_outputs()
cat("Repaired R internal Method 1 candidate, complete-OOF coverage, ALL, and scalar invariants passed\n")
