#!/usr/bin/env Rscript
if (!file.exists("man/NNS.reg.Rd")) {
  stop("Pinned R reference is missing man/NNS.reg.Rd")
}

namespace <- readLines("NAMESPACE", warn = FALSE)

if (!"export(NNS.reg)" %in% namespace) {
  stop("Pinned R reference does not export NNS.reg")
}

if (any(grepl("nns_reg_partition_points_fast", namespace, fixed = TRUE))) {
  stop("Internal partition helper was incorrectly exported")
}

if (any(grepl("nns_reg_univariate_fast", namespace, fixed = TRUE))) {
  stop("Internal univariate helper was incorrectly exported")
}

bad_docs <- list.files(
  "man",
  pattern = "partition_points_fast|univariate_fast",
  full.names = TRUE
)

if (length(bad_docs)) {
  stop(
    "Internal regression helper documentation was generated: ",
    paste(bad_docs, collapse = ", ")
  )
}

rd <- readLines("man/NNS.reg.Rd", warn = FALSE)

if (!any(grepl("\\\\name\\{NNS.reg\\}", rd))) {
  stop("man/NNS.reg.Rd has the wrong name")
}

if (!any(grepl("\\\\alias\\{NNS.reg\\}", rd))) {
  stop("man/NNS.reg.Rd has the wrong alias")
}

cat("Pinned R documentation contract is valid\n")
