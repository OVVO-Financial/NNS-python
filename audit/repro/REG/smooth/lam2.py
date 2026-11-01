import numpy as np
from scipy.interpolate import BSpline
def sigma_r(knot, nk, third=0.333):
    sig=np.zeros((nk,nk)); basis=[BSpline(knot,np.eye(nk)[j],3) for j in range(nk)]
    for i in range(len(knot)-1):
        a,b=knot[i],knot[i+1]
        if b<=a: continue
        y1=np.array([bb(a,nu=2) for bb in basis]); y2=np.array([bb(b-1e-15*0,nu=2) if b<knot[-1] else bb(b,nu=2) for bb in basis])
        # second derivative on the left-piece: evaluate right end as limit from left
        y2=np.array([BSpline(knot,np.eye(nk)[j],3)(b-(b-a)*1e-12,nu=2) for j in range(nk)])-y1
        w=b-a
        sig+= w*(np.outer(y1,y1)+ (np.outer(y2,y1)+np.outer(y1,y2))*.5 + np.outer(y2,y2)*third)
    return sig
x = np.array([0, 0.7, 1.3, 2.0, 3.1, 4.0, 5.5, 6.2]); 
xbar=(x-x[0])/(x[-1]-x[0]); knot=np.concatenate(([xbar[0]]*3,xbar,[xbar[-1]]*3)); nk=len(x)+2
D=BSpline.design_matrix(xbar,knot,3).toarray(); xtwx=D.T@D
for third in (1/3, .333):
  sig=sigma_r(knot,nk,third)
  r=np.diag(xtwx)[2:nk-3].sum()/np.diag(sig)[2:nk-3].sum(); print(third, r*256**(3*.3-1), "R: 0.000177990279204337")
