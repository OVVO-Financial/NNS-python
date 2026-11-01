suppressPackageStartupMessages(library(NNS))
e <- new.env(parent=asNamespace("NNS")); sys.source("Stack_dbg.R", envir=e)
DBG <- list()
spec <- jsonlite::fromJSON("cases/boston_m12_pi/spec.json")
Xtr <- read.csv(spec$xtrain); ytr <- read.csv(spec$ytrain)[[1]]; Xte <- read.csv(spec$xtest)
r <- e$NNS.stack(Xtr,ytr,Xte,status=FALSE,ncores=1,obj.fn=expression(sqrt(mean((predicted - actual)^2))))
print(r$OBJfn.reg)
d <- DBG[[1]]
X <- as.matrix(read.csv("dump/f1_x.csv", header=FALSE)); 
cat("train rows", nrow(d$train), nrow(X), " maxdiff", max(abs(as.matrix(d$train)-X)), "\n")
cat("y diff", max(abs(d$y - scan("dump/f1_y.csv",quiet=TRUE))), "\n")
pp <- as.matrix(read.csv("dump/f1_path.csv", header=FALSE)); cat("path diff", max(abs(d$path-pp+0.123456789)), "\n")
print(head(d$valid,10))
for (i in seq_along(DBG)) { pp <- as.matrix(read.csv(sprintf("dump/path%d.csv",i), header=FALSE)); xt <- as.matrix(read.csv(sprintf("dump/xt%d.csv",i), header=FALSE))
 cat(i, dim(DBG[[i]]$path), dim(pp), "xt diff", if(all(dim(xt)==dim(DBG[[i]]$test))) max(abs(xt-as.matrix(DBG[[i]]$test))) else NA, "path diff", if(all(dim(pp)==dim(DBG[[i]]$path))) max(abs(DBG[[i]]$path-pp+0.123456789)) else NA, "\n") }
i <- 3; d <- DBG[[i]]; pp <- as.matrix(read.csv(sprintf("dump/path%d.csv",i), header=FALSE)); xt <- as.matrix(read.csv(sprintf("dump/xt%d.csv",i), header=FALSE))
dd <- abs(d$path - pp + 0.123456789); w <- which(dd > 1e-8, arr.ind=TRUE); print(head(w)); print(table(w[,2]))
r1 <- w[1,1]; cat(sprintf("R test row %.20g %.20g\nPy test row %.20g %.20g\n", d$test[r1,1], d$test[r1,2], xt[r1,1], xt[r1,2]))
s <- NNS:::.nns_mreg_prepare_model(d$train, d$y, order=NULL, noise.reduction="off", is.class=FALSE)
X <- as.matrix(s$RPM[,1:2]); dist <- abs(X[,1]-d$test[r1,1]); o <- order(dist)[1:4]; print(sprintf("%.20g", X[o,1])); print(sprintf("%.20g", dist[o]))
p2 <- NNS:::NNS_mreg_predict_path_v2_cpp(X, s$RPM$y.hat, xt, 18L, 0L, sapply(d$train,min), sapply(d$train,max), FALSE, 1L) - 0.123456789
cat("R kernel on py test rows vs py path:", max(abs(p2-pp+0.123456789)), "\n")
for (i in 1:5) { d <- DBG[[i]]; s <- NNS:::.nns_mreg_prepare_model(d$train, d$y, order=NULL, noise.reduction="off", is.class=FALSE)
 prx <- as.matrix(read.csv(sprintf("dump/prx%d.csv",i), header=FALSE)); ptr <- as.matrix(read.csv(sprintf("dump/ptr%d.csv",i), header=FALSE))
 cat("fold", i, "R rpm", nrow(s$RPM), "py rpm", nrow(prx), "train diff", max(abs(ptr-as.matrix(d$train))), "\n")
 s2 <- NNS:::.nns_mreg_prepare_model(as.data.frame(ptr), d$y, order=NULL, noise.reduction="off", is.class=FALSE); cat("   R rpm on py train", nrow(s2$RPM), "\n") }
for (i in 1:5) write.table(format(DBG[[i]]$train[[1]], digits=17), sprintf("dump/rtr%d.txt",i), row.names=FALSE, col.names=FALSE, quote=FALSE)
cat("BLAS:", extSoftVersion()["BLAS"], La_library(), "\n")
