suppressPackageStartupMessages({library(NNS)})
spec <- jsonlite::fromJSON("cases/boston_m12_pi/spec.json")
Xtr <- read.csv(spec$xtrain); ytr <- read.csv(spec$ytrain)[[1]]; Xte <- read.csv(spec$xtest)
f <- function() NNS.stack(Xtr,ytr,Xte,status=FALSE,ncores=1,obj.fn=expression(sqrt(mean((predicted - actual)^2))))$OBJfn.reg
print(f())
options(NNS.native.stack=FALSE); print(f())
options(NNS.native.stack=TRUE, NNS.native.mreg=FALSE); print(f())
