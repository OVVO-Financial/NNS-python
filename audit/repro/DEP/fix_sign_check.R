suppressMessages(library(NNS)); source("/home/user/audit/work/DEP/dep_pair_R.R")
cop_fixed <- function(xv,yv){ n<-length(xv); if(n<2) return(0); tx<-sum(xv)/n; ty<-sum(yv)/n; dx<-xv-tx; dy<-yv-ty
  d0Co <- (sum(dx>0&dy>0)+sum(dx<=0&dy<=0))/n
  if(d0Co==1) return(1); if(d0Co==0) return(-1)
  cop_signed_exact(xv,yv) }
assignInNamespace("NNS_dep_pair_cpp", make_pair(cop_fixed), "NNS")
set.seed(1); x<-rnorm(1000)
cat("fixed: dep(x,-x)", unlist(NNS.dep(x,-x)), " dep(1:100,-(1:100))", unlist(NNS.dep(1:100,-(1:100))), " dep(x,x)", unlist(NNS.dep(x,x)), " dep(x,-x^3)", unlist(NNS.dep(x,-x^3)), " dep(x,x+e)", unlist(NNS.dep(x,x+rnorm(1000))), "\n")
