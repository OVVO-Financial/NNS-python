#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(name, default = NULL) {
  flag <- paste0("--", name)
  idx <- match(flag, args)
  if (is.na(idx) || idx == length(args)) return(default)
  args[[idx + 1L]]
}

r_repo <- normalizePath(get_arg("r-repo", "../NNS-r"), mustWork = TRUE)
out_dir <- get_arg("out", "tests/parity/fixtures/repaired_r_13_1_21be6d92")
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
  fixture_schema_version = "repaired_r_13_1_21be6d92",
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
  schema_version = 1L,
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

capture_part_case <- function(name, x, y = NULL, order = NULL, type = NULL,
                              noise.reduction = "mean", obs.req = NULL) {
  args <- list(
    x = x,
    y = y,
    order = order,
    type = type,
    noise.reduction = noise.reduction,
    obs.req = obs.req
  )
  result <- do.call(NNS.part, args[!vapply(args, is.null, logical(1L))])
  with_checksums(list(
    name = name,
    kind = "part",
    input = list(x = x, y = y),
    args = list(
      order = order,
      type = type,
      noise.reduction = noise.reduction,
      obs.req = obs.req
    ),
    output = result
  ))
}

capture_part_error_case <- function(name, x, y = NULL, order = NULL,
                                    type = NULL, obs.req = NULL) {
  error <- tryCatch({
    args <- list(x = x, y = y, order = order, type = type, obs.req = obs.req)
    do.call(NNS.part, args[!vapply(args, is.null, logical(1L))])
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

# The full case suite is appended below by the existing parity generator body.
# This header is intentionally SHA-addressed; fixture generation must not run
# against a different R commit/schema without an explicit repin.

source(file.path(dirname(normalizePath(sys.frame(1)$ofile %||% "tests/parity/generate_repaired_r_fixtures.R")),
                 "generate_repaired_r_fixtures_cases.R"), local = TRUE)
