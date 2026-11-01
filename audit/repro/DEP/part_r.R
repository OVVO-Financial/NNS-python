suppressMessages(library(NNS)); library(jsonlite)
cases <- c("indep_n100","indep_n1000","y_x2","discrete_dep","ties_heavy","step","len5","len3","int_seq","ret_SPY_TLT","circle")
out <- list()
for (k in cases){ d<-read.csv(paste0("/home/user/audit/work/DEP/cases/",k,".csv")); x<-d$x; y<-d$y
 for (typ in list(NULL,"XONLY")) for (order in list(NULL,1,3,"max")) for (obs in c(0,8,50)) for (mos in c(TRUE,FALSE)) for (nr in c("off","mean","median","mode","mode_class")){
  key <- paste(k, if(is.null(typ)) "None" else typ, if(is.null(order)) "None" else order, obs, if(mos) "True" else "False", nr, sep="|")
  r <- tryCatch({p<-NNS.part(x,y,type=typ,order=order,obs.req=obs,min.obs.stop=mos,noise.reduction=nr)
     list(order=p$order, q=as.character(p$dt$quadrant), pq=as.character(p$dt$prior.quadrant), rpq=as.character(p$regression.points$quadrant), rpx=p$regression.points$x, rpy=p$regression.points$y)}, error=function(e) paste("ERR", conditionMessage(e)))
  out[[key]] <- r }}
write(toJSON(out, auto_unbox=FALSE, digits=NA), "/home/user/audit/work/DEP/part_r.json")
