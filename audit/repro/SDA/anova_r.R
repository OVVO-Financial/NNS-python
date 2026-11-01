suppressMessages(library(NNS)); library(jsonlite)
ir <- read.csv("/home/user/audit/data/iris.csv"); F <- read.csv("/home/user/audit/data/returns_daily_equity_etf.csv")[,-1]
sp <- split(ir$Sepal.Length, ir$Species); out <- list()
nb <- function(...) { r <- NNS.ANOVA(..., plot=FALSE); r[!names(r) %in% c("Effect_Size_LB","Effect_Size_UB","Confidence_Level")] }
for (mo in c(FALSE,TRUE)) for (md in c(FALSE,TRUE)) {
  tag <- paste0("mo",as.integer(mo),"md",as.integer(md))
  out[[paste0("iris_bin_",tag)]] <- nb(sp$setosa, sp$versicolor, means.only=mo, medians=md, confidence.interval=NULL)
  out[[paste0("iris_multi_",tag)]] <- NNS.ANOVA(unname(sp), means.only=mo, medians=md, plot=FALSE)
  out[[paste0("iris_pw_",tag)]] <- NNS.ANOVA(unname(sp), means.only=mo, medians=md, pairwise=TRUE, plot=FALSE)
  out[[paste0("fin_bin_",tag)]] <- nb(F$SPY, F$TLT, means.only=mo, medians=md, confidence.interval=NULL)
  out[[paste0("fin_multi_",tag)]] <- NNS.ANOVA(as.matrix(F), means.only=mo, medians=md, plot=FALSE)
  out[[paste0("fin_pw_",tag)]] <- unname(NNS.ANOVA(as.matrix(F), means.only=mo, medians=md, pairwise=TRUE, plot=FALSE))
}
out[["same"]] <- nb(sp$setosa, sp$setosa, confidence.interval=NULL)
out[["const"]] <- tryCatch(nb(rep(1,10), rep(1,10), confidence.interval=NULL), error=function(e) conditionMessage(e))
out[["const2"]] <- tryCatch(nb(rep(1,10), 1:10, confidence.interval=NULL), error=function(e) conditionMessage(e))
# CI: many seeds -> distribution
set.seed(1); ci <- t(replicate(50, unlist(NNS.ANOVA(sp$setosa, sp$versicolor, plot=FALSE)[c("Effect_Size_LB","Effect_Size_UB")])))
out[["ci_both"]] <- colMeans(ci)
out[["ci_left"]] <- "R error"
out[["ci_right"]] <- "R error"
set.seed(1); out[["ci_med"]] <- colMeans(t(replicate(30, unlist(NNS.ANOVA(sp$setosa, sp$versicolor, medians=TRUE, plot=FALSE)[c("Effect_Size_LB","Effect_Size_UB")]))))
set.seed(3); rb <- NNS.ANOVA(F$SPY[1:300], F$QQQ[1:300], robust=TRUE, plot=FALSE); out[["robust_names"]] <- names(rb); out[["robust"]] <- rb
out[["robust_left_names"]] <- tryCatch({set.seed(3); names(NNS.ANOVA(F$SPY[1:300], F$QQQ[1:300], robust=TRUE, tails="left", plot=FALSE))}, error=function(e) conditionMessage(e))
out[["robust_seeds"]] <- sapply(1:30, function(s){ set.seed(s); unlist(NNS.ANOVA(F$SPY[1:300], F$QQQ[1:300], robust=TRUE, plot=FALSE)[c("Robust Certainty Estimate","Lower 95% CI","Upper 95% CI")])})
write_json(out, "anova_r.json", digits=NA, auto_unbox=TRUE, na="string")
