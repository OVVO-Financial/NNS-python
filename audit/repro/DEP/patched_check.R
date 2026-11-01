suppressMessages(library(NNS)); source("/home/user/audit/work/DEP/dep_pair_R.R")
args <- commandArgs(TRUE); mode <- args[1]
if(mode=="exact") assignInNamespace("NNS_dep_pair_cpp", make_pair(cop_signed_exact), "NNS")
if(mode=="fp") assignInNamespace("NNS_dep_pair_cpp", make_pair(cop_signed), "NNS")
for(k in args[-1]){ d<-read.csv(paste0("/home/user/audit/work/DEP/cases/",k,".csv")); x<-d$x;y<-d$y
 cat(k, "dep:", unlist(NNS.dep(x,y)), " caus0:", NNS.caus(x,y,tau=0), " caus3:", tryCatch(NNS.caus(x,y,tau=3),error=function(e) NA), "\n")}
