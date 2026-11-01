suppressPackageStartupMessages({library(NNS);library(jsonlite)})
a <- commandArgs(TRUE); spec <- fromJSON(a[1], simplifyVector=TRUE)
Xtr <- read.csv(spec$xtrain, check.names=FALSE); ytr <- read.csv(spec$ytrain)[[1]]
Xte <- if (!is.null(spec$xtest)) read.csv(spec$xtest, check.names=FALSE) else NULL
if (length(spec$factor_cols)) for (j in spec$factor_cols) { Xtr[[j]] <- factor(Xtr[[j]]); if(!is.null(Xte)) Xte[[j]] <- factor(Xte[[j]], levels=levels(Xtr[[j]])) }
if (isTRUE(spec$yfactor)) ytr <- factor(ytr)
args <- spec$args; if (is.null(args)) args <- list()
if (!is.null(args$obj.fn)) args$obj.fn <- parse(text=args$obj.fn)
t0 <- Sys.time()
r <- do.call(NNS.boost, c(list(IVs.train=Xtr, DV.train=ytr, IVs.test=Xte, status=FALSE, feature.importance=FALSE), args))
r$secs <- as.numeric(Sys.time()-t0, units="secs")
r$feature.frequency <- as.list(r$feature.frequency); r$feature.weights <- as.list(r$feature.weights)
cat(toJSON(r, auto_unbox=TRUE, digits=NA, null="null", na="null"), file=a[2])
