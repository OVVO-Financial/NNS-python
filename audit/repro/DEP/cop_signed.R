cop_signed <- function(xv, yv, verbose=FALSE){
  n <- length(xv); if(n<2) return(0)
  tx <- sum(xv)/n; ty <- sum(yv)/n; dx <- xv-tx; dy <- yv-ty
  d0c <- sum(dx>0&dy>0); d0l <- sum(dx<=0&dy<=0); dpm0 <- sum(!((dx<0&dy<0)|(dx>0&dy>0)))
  cu <- sum((dx*dy)[dx>=0&dy>=0]); cl <- sum((dx*dy)[!(dx>=0&dy>=0) & dx<=0&dy<=0]); cd <- sum((abs(dx)*abs(dy))[!(dx>=0&dy>=0) & !(dx<=0&dy<=0)])
  inv <- 1/n; d0Co <- (d0c+d0l)*inv
  if(verbose) print(c(n=n,tx=tx,d0c=d0c,d0l=d0l,dpm0=dpm0,d0Co=d0Co,cu=cu,cl=cl,cd=cd))
  if(d0Co==1||d0Co==0) return(1)
  tot <- cu+cl+cd; cod1 <- if(tot>0)(cu+cl)/tot else 0; dpmd0<-dpm0*inv; dpmd1<- if(tot>0) cd/tot else 0
  v <- sqrt((min(1,abs(d0Co-.5)/.5)+min(1,abs(cod1-.5)/.5)+abs(dpmd0-.5)/.5+abs(dpmd1-.5)/.5)/4)
  cv <- sum(dx*dy); vx <- sum(dx*dx); s <- if(vx==0) 0 else sign(cv); v*s }
