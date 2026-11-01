Rscript -e 'suppressMessages(library(NNS)); x<-c(1:20, Inf, -Inf); y<-c(20:1, 5, 7); p<-NNS.part(x,y, order=2); print(table(p$dt$quadrant)); print(p$dt[22,])'
/home/user/NNS-python/.venv/bin/python -c "
import numpy as np, nns, collections
x=np.r_[np.arange(1,21.), np.inf, -np.inf]; y=np.r_[np.arange(20,0,-1.), 5, 7]
p=nns.nns_part(x,y,order=2); print(collections.Counter(p['dt']['quadrant'])); print(p['dt']['quadrant'][21])"
