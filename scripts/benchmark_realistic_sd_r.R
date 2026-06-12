args <- commandArgs(trailingOnly = TRUE)

option_value <- function(name, default) {
  prefix <- paste0("--", name, "=")
  matched <- args[startsWith(args, prefix)]
  if (length(matched) == 0) {
    return(default)
  }
  sub(prefix, "", matched[[1]], fixed = TRUE)
}

time_call <- function(fun, repeats) {
  times <- replicate(repeats, system.time(invisible(fun()))[["elapsed"]])
  c(mean = mean(times), min = min(times), max = max(times))
}

fixture <- option_value(
  "fixture",
  "tests/fixtures/finance/sp500_daily_returns_2019_2023.csv"
)
repeats <- as.integer(option_value("repeats", "3"))
max_repeats <- as.integer(option_value("max-repeats", "1"))
output <- option_value("output", "")

library(NNS)

returns <- read.csv(fixture, check.names = FALSE)
date_values <- as.Date(returns[[1]])
market_col <- if ("GSPC" %in% names(returns)) "GSPC" else "SPY"
tradable_proxy_col <- "SPY"
constituent_cols <- setdiff(names(returns)[-1], c("SPY", "GSPC"))
max_columns <- length(constituent_cols)

constituent_matrix <- function(rows, columns) {
  as.matrix(returns[seq_len(rows), constituent_cols[seq_len(columns)]])
}

period_end_positions <- function(frequency) {
  positions <- integer()
  for (index in seq_along(date_values)) {
    if (index == length(date_values)) {
      positions <- c(positions, index)
      next
    }
    current <- date_values[[index]]
    next_value <- date_values[[index + 1]]
    if (frequency == "monthly") {
      if (format(current, "%Y-%m") != format(next_value, "%Y-%m")) {
        positions <- c(positions, index)
      }
    } else if (frequency == "quarterly") {
      current_quarter <- paste0(format(current, "%Y"), "-", quarters(current))
      next_quarter <- paste0(format(next_value, "%Y"), "-", quarters(next_value))
      if (current_quarter != next_quarter) {
        positions <- c(positions, index)
      }
    }
  }
  positions
}

rolling_windows <- function(lookback, frequency) {
  stops <- period_end_positions(frequency)
  stops <- stops[stops >= lookback]
  lapply(stops, function(stop) c(stop - lookback + 1, stop))
}

average_turnover <- function(sets) {
  if (length(sets) < 2) {
    return(0)
  }
  values <- numeric(length(sets) - 1)
  for (index in seq_len(length(values))) {
    previous <- sets[[index]]
    current <- sets[[index + 1]]
    union_size <- length(union(previous, current))
    values[[index]] <- if (union_size == 0) 0 else 1 - length(intersect(previous, current)) / union_size
  }
  mean(values)
}

rolling_sd_efficient_set_summary <- function(columns, lookback, frequency, degree) {
  mat <- constituent_matrix(nrow(returns), columns)
  windows <- rolling_windows(lookback, frequency)
  sets <- list()
  sizes <- integer(length(windows))
  for (index in seq_along(windows)) {
    span <- windows[[index]]
    result <- NNS::NNS.SD.efficient.set(
      mat[span[[1]]:span[[2]], , drop = FALSE],
      degree = degree,
      type = "discrete",
      status = FALSE
    )
    sets[[index]] <- result
    sizes[[index]] <- length(result)
  }
  list(
    window_count = length(windows),
    average_efficient_set_size = mean(sizes),
    average_turnover = average_turnover(sets),
    result_size = round(mean(sizes))
  )
}

rolling_sd_cluster_summary <- function(columns, lookback, frequency, degree) {
  mat <- constituent_matrix(nrow(returns), columns)
  windows <- rolling_windows(lookback, frequency)
  cluster_counts <- integer(length(windows))
  first_cluster_sizes <- integer(length(windows))
  for (index in seq_along(windows)) {
    span <- windows[[index]]
    result <- NNS::NNS.SD.cluster(
      mat[span[[1]]:span[[2]], , drop = FALSE],
      degree = degree,
      type = "discrete",
      min_cluster = 1,
      dendrogram = FALSE
    )
    cluster_counts[[index]] <- length(result$Clusters)
    first_cluster_sizes[[index]] <- length(result$Clusters[[1]])
  }
  list(
    window_count = length(windows),
    average_cluster_count = mean(cluster_counts),
    average_efficient_set_size = mean(first_cluster_sizes),
    result_size = round(mean(first_cluster_sizes))
  )
}

rolling_sd_degree_comparison_summary <- function(columns, lookback, frequency) {
  mat <- constituent_matrix(nrow(returns), columns)
  windows <- rolling_windows(lookback, frequency)
  degree1_sizes <- integer(length(windows))
  degree2_sizes <- integer(length(windows))
  for (index in seq_along(windows)) {
    span <- windows[[index]]
    window <- mat[span[[1]]:span[[2]], , drop = FALSE]
    degree1_sizes[[index]] <- length(NNS::NNS.SD.efficient.set(
      window,
      degree = 1,
      type = "discrete",
      status = FALSE
    ))
    degree2_sizes[[index]] <- length(NNS::NNS.SD.efficient.set(
      window,
      degree = 2,
      type = "discrete",
      status = FALSE
    ))
  }
  list(
    window_count = length(windows),
    average_degree1_set_size = mean(degree1_sizes),
    average_degree2_set_size = mean(degree2_sizes),
    result_size = round(mean(degree2_sizes))
  )
}

mag7_market_downside_stress_summary <- function() {
  mag7 <- c("AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA")
  cols <- unique(c(mag7, market_col, tradable_proxy_col))
  mat <- as.matrix(returns[, cols])
  assets <- mat[, mag7, drop = FALSE]
  market <- mat[, market_col]
  downside <- market <= -0.01
  stress_assets <- assets[downside, , drop = FALSE]
  stress_market <- market[downside]
  co_lpm_degree1 <- sapply(seq_len(ncol(stress_assets)), function(index) {
    NNS::Co.LPM(1, stress_assets[, index], stress_market, 0, 0)
  })
  co_lpm_degree2 <- sapply(seq_len(ncol(stress_assets)), function(index) {
    NNS::Co.LPM(2, stress_assets[, index], stress_market, 0, 0)
  })
  matrix <- NNS::PM.matrix(
    1,
    1,
    target = rep(0, ncol(stress_assets)),
    variable = stress_assets,
    pop_adj = TRUE,
    norm = TRUE
  )
  stress_points <- matrix(c(rep(-0.05, ncol(stress_assets)), rep(-0.10, ncol(stress_assets))), nrow = 2, byrow = TRUE)
  regression <- NNS::NNS.reg(
    stress_assets,
    stress_market,
    dim.red.method = "cor",
    order = 2,
    point.est = stress_points,
    plot = FALSE,
    residual.plot = FALSE
  )
  list(
    downside_observation_count = nrow(stress_assets),
    stress_regression_r2 = regression$R2,
    result_size = length(co_lpm_degree1) + length(co_lpm_degree2) + nrow(matrix$cov.matrix)
  )
}

partial_moment_covariance_summary <- function(rows, degree, target_kind) {
  mat <- constituent_matrix(rows, max_columns)
  target <- if (target_kind == "mean") NULL else rep(0, ncol(mat))
  matrix <- NNS::PM.matrix(
    degree,
    degree,
    target = target,
    variable = mat,
    pop_adj = TRUE,
    norm = FALSE
  )
  list(
    rows = rows,
    columns = ncol(mat),
    covariance_shape = nrow(matrix$cov.matrix),
    result_size = nrow(matrix$cov.matrix)
  )
}

market_relative_ratio <- function() {
  constituents <- constituent_matrix(nrow(returns), max_columns)
  market <- returns[[market_col]]
  lower <- sqrt(rowMeans(pmax(market - constituents, 0)^2))
  upper <- sqrt(rowMeans(pmax(constituents - market, 0)^2))
  ifelse(lower > 0, upper / lower, 0)
}

dispersion_summary <- function(window = NULL) {
  ratio <- market_relative_ratio()
  signal <- ratio
  market <- returns[[market_col]]
  if (!is.null(window)) {
    signal <- stats::filter(ratio, rep(1 / window, window), sides = 1)
    signal <- as.numeric(signal[window:length(signal)])
    market <- market[window:length(market)]
  }
  finite <- is.finite(signal)
  correlation <- if (length(signal) > 1) stats::cor(signal[-length(signal)], market[-1]) else 0
  list(
    signal_length = length(signal),
    finite_count = sum(finite),
    next_day_market_correlation = correlation,
    result_size = length(signal)
  )
}

sd_cases <- data.frame(
  function_name = c(
    "sd_efficient_set", "nns_sd_cluster",
    "sd_efficient_set", "nns_sd_cluster",
    "sd_efficient_set", "nns_sd_cluster",
    "sd_efficient_set", "nns_sd_cluster",
    "sd_efficient_set", "nns_sd_cluster",
    "sd_efficient_set", "nns_sd_cluster",
    "sd_efficient_set", "nns_sd_cluster",
    "sd_efficient_set", "nns_sd_cluster",
    "sd_efficient_set", "nns_sd_cluster"
  ),
  rows = c(
    252, 252,
    252, 252,
    252, 252,
    252, 252,
    252, 252,
    252, 252,
    1257, 1257,
    1257, 1257,
    1257, 1257
  ),
  columns = c(
    50, 50,
    100, 100,
    250, 250,
    max_columns, max_columns,
    50, 50,
    100, 100,
    100, 100,
    250, 250,
    max_columns, max_columns
  ),
  degree = c(
    1, 1,
    1, 1,
    2, 2,
    2, 2,
    2, 2,
    2, 2,
    2, 2,
    2, 2,
    2, 2
  )
)

results <- data.frame(
  function_name = character(),
  rows = integer(),
  columns = integer(),
  degree = integer(),
  repeats = integer(),
  mean_seconds = numeric(),
  min_seconds = numeric(),
  max_seconds = numeric(),
  result_size = integer()
)

append_result <- function(function_name, rows, columns, degree, case_repeats, timed, result_size) {
  results <<- rbind(results, data.frame(
    function_name = function_name,
    rows = rows,
    columns = columns,
    degree = degree,
    repeats = case_repeats,
    mean_seconds = timed[["mean"]],
    min_seconds = timed[["min"]],
    max_seconds = timed[["max"]],
    result_size = result_size
  ))
}

for (index in seq_len(nrow(sd_cases))) {
  function_name <- sd_cases$function_name[[index]]
  rows <- sd_cases$rows[[index]]
  columns <- sd_cases$columns[[index]]
  degree <- sd_cases$degree[[index]]
  case_repeats <- if (columns == max_columns && rows == nrow(returns)) max_repeats else repeats
  mat <- constituent_matrix(rows, columns)

  if (function_name == "sd_efficient_set") {
    result <- NNS::NNS.SD.efficient.set(
      mat,
      degree = degree,
      type = "discrete",
      status = FALSE
    )
    result_size <- length(result)
    timed <- time_call(function() {
      NNS::NNS.SD.efficient.set(
        mat,
        degree = degree,
        type = "discrete",
        status = FALSE
      )
    }, case_repeats)
  } else {
    result <- NNS::NNS.SD.cluster(
      mat,
      degree = degree,
      type = "discrete",
      min_cluster = 1,
      dendrogram = FALSE
    )
    result_size <- length(unlist(result$Clusters, use.names = FALSE))
    timed <- time_call(function() {
      NNS::NNS.SD.cluster(
        mat,
        degree = degree,
        type = "discrete",
        min_cluster = 1,
        dendrogram = FALSE
      )
    }, case_repeats)
  }

  append_result(function_name, rows, columns, degree, case_repeats, timed, result_size)
}

workflow_cases <- list(
  list("rolling_sd_efficient_set_252d_monthly", 252, 100, 2, repeats, function() {
    rolling_sd_efficient_set_summary(100, 252, "monthly", 2)
  }),
  list("rolling_sd_efficient_set_252d_monthly", 252, max_columns, 2, max_repeats, function() {
    rolling_sd_efficient_set_summary(max_columns, 252, "monthly", 2)
  }),
  list("rolling_sd_cluster_252d_monthly", 252, 100, 2, repeats, function() {
    rolling_sd_cluster_summary(100, 252, "monthly", 2)
  }),
  list("rolling_sd_cluster_252d_monthly", 252, max_columns, 2, max_repeats, function() {
    rolling_sd_cluster_summary(max_columns, 252, "monthly", 2)
  }),
  list("rolling_sd_cluster_756d_quarterly", 756, max_columns, 2, max_repeats, function() {
    rolling_sd_cluster_summary(max_columns, 756, "quarterly", 2)
  }),
  list("rolling_sd_efficient_set_252d_quarterly", 252, max_columns, 1, max_repeats, function() {
    rolling_sd_efficient_set_summary(max_columns, 252, "quarterly", 1)
  }),
  list("rolling_sd_cluster_252d_quarterly", 252, max_columns, 1, max_repeats, function() {
    rolling_sd_cluster_summary(max_columns, 252, "quarterly", 1)
  }),
  list("rolling_sd_efficient_set_degree1_vs_degree2_252d_quarterly", 252, max_columns, 0, max_repeats, function() {
    rolling_sd_degree_comparison_summary(max_columns, 252, "quarterly")
  }),
  list("mag7_market_downside_stress", 1257, 9, 1, repeats, function() {
    mag7_market_downside_stress_summary()
  }),
  list("pm_matrix_degree1_mean", 252, max_columns, 1, repeats, function() {
    partial_moment_covariance_summary(252, 1, "mean")
  }),
  list("pm_matrix_degree1_mean", 1257, max_columns, 1, max_repeats, function() {
    partial_moment_covariance_summary(1257, 1, "mean")
  }),
  list("pm_matrix_degree2_zero", 252, max_columns, 2, repeats, function() {
    partial_moment_covariance_summary(252, 2, "zero")
  }),
  list("market_relative_daily_dispersion", 1257, max_columns, 2, repeats, function() {
    dispersion_summary()
  }),
  list("market_relative_rolling_dispersion_63d", 1257, max_columns, 2, repeats, function() {
    dispersion_summary(63)
  }),
  list("market_relative_rolling_dispersion_252d", 1257, max_columns, 2, repeats, function() {
    dispersion_summary(252)
  })
)

for (case in workflow_cases) {
  function_name <- case[[1]]
  rows <- case[[2]]
  columns <- case[[3]]
  degree <- case[[4]]
  case_repeats <- case[[5]]
  fun <- case[[6]]
  result <- fun()
  timed <- time_call(fun, case_repeats)
  append_result(
    function_name,
    rows,
    columns,
    degree,
    case_repeats,
    timed,
    result$result_size
  )
}

if (output != "") {
  write.csv(results, output, row.names = FALSE, quote = FALSE)
} else {
  write.csv(results, stdout(), row.names = FALSE, quote = FALSE)
}
