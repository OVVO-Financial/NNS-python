suppressMessages(library(NNS)); library(jsonlite)
cases <- fromJSON("sd_cases.json", simplifyVector=TRUE, simplifyDataFrame=FALSE)
lab <- function(s) if (is.na(s)) NA_integer_ else if (grepl("^X", s)) 1L else if (grepl("^Y", s)) -1L else 0L
res <- lapply(cases, function(cs){
  x <- as.numeric(cs$x); y <- as.numeric(cs$y)
  f <- function(e) tryCatch(e, error=function(err) paste("ERR", conditionMessage(err)))
  list(name=cs$name,
    FSD=f(lab(NNS.FSD(x,y,plot=FALSE))), FSDc=f(lab(NNS.FSD(x,y,type="continuous",plot=FALSE))),
    SSD=f(lab(NNS.SSD(x,y,plot=FALSE))), TSD=f(lab(NNS.TSD(x,y,plot=FALSE))),
    FSDu_xy=f(NNS.FSD.uni(x,y)), FSDu_yx=f(NNS.FSD.uni(y,x)),
    FSDuc_xy=f(NNS.FSD.uni(x,y,type="continuous")), FSDuc_yx=f(NNS.FSD.uni(y,x,type="continuous")),
    SSDu_xy=f(NNS.SSD.uni(x,y)), SSDu_yx=f(NNS.SSD.uni(y,x)),
    TSDu_xy=f(NNS.TSD.uni(x,y)), TSDu_yx=f(NNS.TSD.uni(y,x)))
})
write_json(res, "sd_cases_r.json", auto_unbox=TRUE, na="null")
