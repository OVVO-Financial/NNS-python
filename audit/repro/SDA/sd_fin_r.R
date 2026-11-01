suppressMessages(library(NNS)); library(jsonlite)
R <- read.csv("/home/user/audit/data/returns_daily_equity_etf.csv")
X <- as.matrix(R[,-1]); nm <- colnames(X); k <- ncol(X)
out <- list()
lab <- function(s) if (grepl("^X", s)) 1L else if (grepl("^Y", s)) -1L else 0L
for (nmf in c("FSD","FSDc","SSD","TSD")) {
  M <- matrix(0L, k, k)
  for (i in 1:k) for (j in 1:k) if (i!=j) {
    s <- switch(nmf, FSD=NNS.FSD(X[,i],X[,j],plot=FALSE), FSDc=NNS.FSD(X[,i],X[,j],type="continuous",plot=FALSE),
                SSD=NNS.SSD(X[,i],X[,j],plot=FALSE), TSD=NNS.TSD(X[,i],X[,j],plot=FALSE))
    M[i,j] <- lab(s)
  }
  out[[nmf]] <- M
}
for (nmf in c("FSDu","FSDuc","SSDu","TSDu")) {
  M <- matrix(0L, k, k)
  for (i in 1:k) for (j in 1:k) if (i!=j) {
    M[i,j] <- switch(nmf, FSDu=NNS.FSD.uni(X[,i],X[,j]), FSDuc=NNS.FSD.uni(X[,i],X[,j],type="continuous"),
                SSDu=NNS.SSD.uni(X[,i],X[,j]), TSDu=NNS.TSD.uni(X[,i],X[,j]))
  }
  out[[nmf]] <- M
}
for (d in 1:3) out[[paste0("eff",d)]] <- NNS.SD.efficient.set(X, d, status=FALSE)
out[["eff1c"]] <- NNS.SD.efficient.set(X, 1, type="continuous", status=FALSE)
for (d in 1:3) for (mc in c(1,2,3)) out[[paste0("clu",d,"_",mc)]] <- NNS.SD.cluster(X, degree=d, min_cluster=mc)$Clusters
out[["clu1c"]] <- NNS.SD.cluster(X, degree=1, type="continuous")$Clusters
pdf(NULL); dd <- NNS.SD.cluster(X, degree=2, dendrogram=TRUE)$Dendrogram
out[["dend2"]] <- list(merge=dd$merge, height=dd$height, order=dd$order, labels=dd$labels)
# monthly too
Rm <- read.csv("/home/user/audit/data/returns_monthly.csv"); Xm <- as.matrix(Rm[,-1]); Xm <- Xm[complete.cases(Xm),]
out[["m_names"]] <- colnames(Xm)
for (d in 1:3) out[[paste0("meff",d)]] <- NNS.SD.efficient.set(Xm, d, status=FALSE)
out[["meff1c"]] <- NNS.SD.efficient.set(Xm, 1, type="continuous", status=FALSE)
write_json(out, "sd_fin_r.json", digits=NA, auto_unbox=FALSE)
