#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1L) args[[1L]] else "artifacts/repaired-r-validation"
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

`%||%` <- function(x, y) if (is.null(x)) y else x

write_failure <- function(name, payload) {
  write_json(
    payload,
    file.path(out_dir, paste0("stack-invariant-failure-", name, ".json")),
    pretty = TRUE,
    auto_unbox = TRUE,
    digits = NA,
    null = "null"
  )
}

fail <- function(name, message, payload = list()) {
  write_failure(name, c(list(message = message), payload))
  stop(message, call. = FALSE)
}

assert_equal <- function(name, actual, expected, tolerance = 1e-12,
                         exact = FALSE, context = list()) {
  ok <- if (exact) {
    identical(as.vector(actual), as.vector(expected))
  } else {
    isTRUE(all.equal(
      as.numeric(actual), as.numeric(expected),
      tolerance = tolerance,
      check.attributes = FALSE
    ))
  }

  if (!ok) {
    fail(
      name,
      paste0("R repaired stack invariant failed: ", name),
      c(
        list(
          actual = as.vector(actual),
          expected = as.vector(expected),
          tolerance = tolerance,
          exact = exact
        ),
        context
      )
    )
  }
}

check_scalar_clean <- function(result, fields) {
  for (field in fields) {
    value <- result[[field]]
    if (is.null(value)) next
    if (length(value) != 1L || !is.null(names(value))) {
      fail(
        paste0("scalar-clean-", field),
        paste0("R stack scalar field is not an unnamed scalar: ", field),
        list(
          field = field,
          value = value,
          names = names(value),
          length = length(value)
        )
      )
    }
  }
}

field <- function(x, names_to_try, default = NULL) {
  if (!is.list(x)) return(default)
  for (name in names_to_try) {
    if (!is.null(x[[name]])) return(x[[name]])
  }
  default
}

candidate_id <- function(candidate) {
  value <- field(candidate, c(
    "candidate", "candidate_id", "id", "k", "n.best", "n_best"
  ))
  if (is.null(value) || !length(value)) return(NA_character_)
  as.character(value[[1L]])
}

with_stack_trace <- function(expr) {
  trace_env <- new.env(parent = emptyenv())
  old <- options(NNS.stack.trace.env = trace_env)
  on.exit(options(old), add = TRUE)

  result <- force(expr)
  trace <- trace_env$method1

  if (is.null(trace)) {
    fail(
      "method1-trace-missing",
      paste(
        "NNS.stack did not write Method 1 internals to",
        "options(NNS.stack.trace.env = <environment>).",
        "Fixture generation is blocked until the R reference exposes",
        "the hidden trace without changing the public return value."
      ),
      list(result_names = names(result))
    )
  }

  list(result = result, trace = trace)
}

normalize_trace_folds <- function(trace) {
  folds <- field(trace, c("folds", "fold", "fold_trace", "fold_traces"))
  if (is.null(folds) && is.list(trace) && length(trace) &&
      all(vapply(trace, is.list, logical(1L)))) {
    folds <- trace
  }
  if (is.null(folds) || !length(folds)) {
    fail(
      "method1-trace-no-folds",
      "Method 1 trace does not contain fold records.",
      list(trace_names = names(trace))
    )
  }
  folds
}

check_internal_candidates <- function() {
  set.seed(131)
  n <- 500L
  grid <- seq(-2, 2, length.out = n)
  X <- cbind(
    x1 = grid,
    x2 = sin(grid),
    x3 = cos(grid),
    x4 = rep(c(-1, 0, 1, 0), length.out = n),
    x5 = grid^2
  )
  y <- X[, 1L] + 0.5 * X[, 2L] - X[, 3L] + 0.1 * X[, 4L]

  run <- with_stack_trace(NNS.stack(
    IVs.train = X,
    DV.train = y,
    IVs.test = X[1:5, , drop = FALSE],
    method = 1,
    ts.test = 50,
    folds = 3,
    status = FALSE,
    ncores = 1
  ))

  result <- run$result
  trace <- run$trace
  folds <- normalize_trace_folds(trace)

  pooled_counts <- list()
  pooled_scores <- list()
  pooled_predictions <- list()
  pooled_actuals <- list()

  for (fold_idx in seq_along(folds)) {
    fold <- folds[[fold_idx]]
    train_x <- field(fold, c(
      "train_design", "train_x", "fold_train_x", "encoded_train_x", "x_train"
    ))
    train_y <- field(fold, c(
      "train_y_fit", "train_y", "fold_train_y", "y_train"
    ))
    validation_x <- field(fold, c(
      "valid_design", "validation_x", "fold_validation_x",
      "encoded_validation_x", "x_validation", "x_valid"
    ))
    validation_y <- field(fold, c(
      "validation_y", "fold_validation_y", "y_validation", "y_valid"
    ))
    response_offset <- as.numeric(field(
      fold, c("response_offset", "offset"), default = 0
    ))
    order_value <- field(fold, c("order"), default = NULL)
    dist_value <- field(fold, c("dist", "distance"), default = "L2")
    candidates <- field(fold, c(
      "candidates", "candidate_predictions", "method1_candidates"
    ))

    if (is.null(validation_y)) {
      valid_idx <- field(fold, c("valid_idx", "validation_idx"))
      if (!is.null(valid_idx)) validation_y <- y[as.integer(valid_idx)]
    }

    required <- list(
      train_x = train_x,
      train_y = train_y,
      validation_x = validation_x,
      validation_y = validation_y,
      candidates = candidates
    )
    missing <- names(required)[vapply(required, is.null, logical(1L))]
    if (length(missing)) {
      fail(
        "method1-trace-incomplete",
        "Method 1 trace is missing fold inputs or candidate predictions.",
        list(fold = fold_idx, missing = missing, fold_names = names(fold))
      )
    }

    for (candidate in candidates) {
      id <- candidate_id(candidate)
      prediction <- field(candidate, c(
        "prediction", "predictions", "point_est", "point.est", "Point.est"
      ))
      eligible <- isTRUE(field(candidate, c(
        "eligible", "complete", "scored"
      ), default = TRUE))
      score <- field(candidate, c("score", "objective", "OBJfn", "sse"),
                     default = NA_real_)

      if (is.na(id) || is.null(prediction)) {
        fail(
          "method1-candidate-trace-invalid",
          "A traced Method 1 candidate lacks an ID or prediction vector.",
          list(fold = fold_idx, candidate_names = names(candidate))
        )
      }

      if (!eligible) {
        if (length(score) && !all(is.na(score))) {
          fail(
            "method1-excluded-candidate-scored",
            "A partial or excluded candidate received an objective.",
            list(fold = fold_idx, candidate = id, score = score)
          )
        }
        next
      }

      if (!length(prediction) || !length(validation_y)) {
        fail(
          "method1-empty-vector-scored",
          "Empty predicted/actual vectors cannot produce an objective.",
          list(fold = fold_idx, candidate = id, score = score)
        )
      }

      direct_n_best <- if (tolower(id) == "all") "all" else as.integer(id)
      direct <- NNS.reg(
        train_x,
        train_y,
        point.est = validation_x,
        n.best = direct_n_best,
        order = order_value,
        dist = dist_value,
        plot = FALSE,
        residual.plot = FALSE,
        factor.2.dummy = FALSE,
        point.only = TRUE,
        ncores = 1
      )$Point.est
      direct <- as.numeric(direct) - response_offset

      assert_equal(
        paste0("method1-internal-vs-direct-fold-", fold_idx,
               "-candidate-", id),
        prediction,
        direct,
        tolerance = 1e-12,
        context = list(
          fold = fold_idx,
          candidate = id,
          internal_prediction = as.numeric(prediction),
          direct_prediction = direct,
          max_abs_diff = max(abs(as.numeric(prediction) - direct))
        )
      )

      pooled_counts[[id]] <- (pooled_counts[[id]] %||% 0L) + length(prediction)
      pooled_predictions[[id]] <- c(
        pooled_predictions[[id]] %||% numeric(), as.numeric(prediction)
      )
      pooled_actuals[[id]] <- c(
        pooled_actuals[[id]] %||% numeric(), as.numeric(validation_y)
      )
      pooled_scores[[id]] <- sum(
        (pooled_predictions[[id]] - pooled_actuals[[id]])^2
      )
    }
  }

  if (is.null(pooled_counts[["1"]])) {
    fail(
      "method1-k1-missing",
      "Candidate k=1 must define the reference OOF coverage.",
      list(counts = pooled_counts)
    )
  }

  reference_count <- pooled_counts[["1"]]
  ordinary_ids <- setdiff(names(pooled_counts), "all")
  mismatched <- ordinary_ids[vapply(
    ordinary_ids,
    function(id) pooled_counts[[id]] != reference_count,
    logical(1L)
  )]
  if (length(mismatched)) {
    fail(
      "method1-ordinary-count-vector",
      "Eligible ordinary candidates do not share k=1 OOF coverage.",
      list(
        counts = pooled_counts,
        reference_count = reference_count,
        mismatched = mismatched
      )
    )
  }

  if (is.null(pooled_counts[["all"]]) ||
      pooled_counts[["all"]] != reference_count) {
    fail(
      "method1-all-count",
      "ALL does not have complete OOF coverage.",
      list(counts = pooled_counts, reference_count = reference_count)
    )
  }

  if (any(!is.finite(unlist(pooled_scores)))) {
    fail(
      "method1-pooled-scores",
      "Pooled Method 1 candidate scores must be finite.",
      list(scores = pooled_scores)
    )
  }

  proof <- list(
    reference_count = reference_count,
    candidate_counts = pooled_counts,
    candidate_scores = pooled_scores,
    eligible_candidates = field(trace, c("eligible_candidates", "eligible_ids"),
                                default = names(pooled_counts)),
    excluded_candidates = field(trace, c("excluded_candidates", "excluded_ids"),
                                default = character()),
    stopping_k = field(trace, c("stopping_k", "stop_k"), default = NULL),
    selected_candidate = field(
      trace,
      c("selected_candidate", "selected_id"),
      default = result$NNS.reg.n.best
    )
  )

  write_json(
    proof,
    file.path(out_dir, "stack-invariant-method1-coverage-proof.json"),
    pretty = TRUE,
    auto_unbox = TRUE,
    digits = NA,
    null = "null"
  )

  check_scalar_clean(
    result,
    c(
      "OBJfn.reg", "NNS.reg.n.best", "OBJfn.dim.red",
      "NNS.dim.red.threshold", "probability.threshold"
    )
  )
}

check_internal_candidates()
cat(paste(
  "Repaired R internal Method 1 candidate, complete-OOF coverage,",
  "ALL, and scalar invariants passed\n"
))
