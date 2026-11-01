import numpy as np
from itertools import pairwise
from scipy.interpolate import BSpline
x = np.array([0, 0.7, 1.3, 2.0, 3.1, 4.0, 5.5, 6.2]); 
ux=x; xbar=(ux-ux[0])/(ux[-1]-ux[0]); knot=np.concatenate(([xbar[0]]*3,xbar,[xbar[-1]]*3)); nk=len(ux)+2
D=BSpline.design_matrix(xbar,knot,3).toarray(); xtwx=D.T@D
sig=np.zeros((nk,nk)); gx,gw=np.polynomial.legendre.leggauss(4)
basis=[BSpline(knot,np.eye(nk)[j],3) for j in range(nk)]
for a,b in pairwise(np.unique(knot)):
    pts=(a+b)/2+(b-a)/2*gx; w=(b-a)/2*gw
    d2=np.column_stack([bb(pts,nu=2) for bb in basis]); sig+=(d2*w[:,None]).T@d2
for s in (0.3,0.6,0.75):
  r=np.diag(xtwx)[2:nk-3].sum()/np.diag(sig)[2:nk-3].sum(); print(s, r*256**(3*s-1))
print(np.diag(sig)[:4])
