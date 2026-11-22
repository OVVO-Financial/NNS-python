suppressPackageStartupMessages(library(NNS))
args <- commandArgs(trailingOnly = TRUE)
cases <- jsonlite::fromJSON(args[1], simplifyVector = TRUE, simplifyMatrix = TRUE, simplifyDataFrame = FALSE)
pdf(NULL)
enc <- function(x) {
  if (is.null(x)) return(list(t = "null"))
  if (is.data.frame(x)) {
    return(list(t = "df", names = I(names(x)), v = unname(lapply(as.list(x), enc)), cls = I(class(x))))
  }
  if (is.factor(x)) return(list(t = "fct", v = I(as.character(x)), levels = I(levels(x)), names = if (is.null(names(x))) NULL else I(names(x))))
  if (is.list(x)) {
    nm <- names(x)
    return(list(t = "list", names = if (is.null(nm)) NULL else I(nm), v = unname(lapply(x, enc)), cls = I(class(x))))
  }
  if (is.function(x)) return(list(t = "fn"))
  if (is.environment(x)) return(list(t = "env"))
  if (is.atomic(x)) {
    tp <- if (is.logical(x)) "lgl" else if (is.character(x)) "chr" else if (is.complex(x)) "cplx" else "num"
    v <- if (tp == "chr") as.character(x) else if (tp == "lgl") as.logical(x) else if (tp == "cplx") as.character(x) else as.numeric(x)
    d <- dim(x)
    dn <- dimnames(x)
    out <- list(t = tp, v = I(v), cls = I(class(x)))
    if (!is.null(names(x))) out$names <- I(names(x))
    if (!is.null(d)) out$dim <- I(as.integer(d))
    if (!is.null(dn)) out$dimnames <- lapply(dn, function(z) if (is.null(z)) NULL else I(z))
    return(out)
  }
  list(t = "other", cls = I(class(x)))
}
results <- vector("list", length(cases))
for (i in seq_along(cases)) {
  cs <- cases[[i]]
  env <- new.env(parent = globalenv())
  if (!is.null(cs$data)) for (nm in names(cs$data)) assign(nm, cs$data[[nm]], envir = env)
  warns <- character(0)
  t0 <- proc.time()[["elapsed"]]
  res <- tryCatch(
    withCallingHandlers({
      val <- eval(parse(text = cs$expr), envir = env)
      list(ok = TRUE, value = enc(val))
    }, warning = function(w) { warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") },
       message = function(m) invokeRestart("muffleMessage")),
    error = function(e) list(ok = FALSE, error = conditionMessage(e)))
  res$warnings <- I(warns)
  res$id <- cs$id
  res$secs <- proc.time()[["elapsed"]] - t0
  results[[i]] <- res
}
writeLines(jsonlite::toJSON(results, auto_unbox = TRUE, digits = I(17), na = "string", null = "null"), args[2])
