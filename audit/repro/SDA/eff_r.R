suppressMessages(library(NNS)); library(jsonlite)
meta <- fromJSON("eff/meta.json"); out <- list()
for (s in meta) { X <- as.matrix(read.csv(sprintf("eff/m%d.csv",s), header=FALSE)); colnames(X) <- paste0("c",0:(ncol(X)-1))
  r <- list()
  for (d in 1:3) r[[paste0("d",d)]] <- as.integer(sub("c","",NNS.SD.efficient.set(X,d,status=FALSE)))
  r[["d1c"]] <- as.integer(sub("c","",NNS.SD.efficient.set(X,1,type="continuous",status=FALSE)))
  out[[as.character(s)]] <- r }
write_json(out, "eff/r.json")
