import numpy as np, nns
x=np.array([-1,np.nan,1,2.])
print("LPM1 t0:", nns.lpm(1,0,x), nns.upm(1,0,x), nns.lpm(0,0,x))
print("vec", nns.lpm(1, np.linspace(-1,1,40), x)[:3])
print("NaN target", nns.lpm(1,np.nan,np.array([1,2,3.])), nns.lpm(1,np.inf,np.array([1,2,3.])), nns.upm(1,-np.inf,np.array([1,2,3.])))
print("ratio all eq", nns.lpm_ratio(1,2,np.array([2,2,2.])), nns.upm_ratio(1,2,np.array([2,2,2.])))
print("co_lpm NA", nns.co_lpm(1, np.array([1,np.nan,3,0]), np.array([1,2,np.nan,0]), 1, 1))
print("inf x", nns.lpm(1,0,np.array([-np.inf,1])), nns.upm(1,0,np.array([np.inf,1.])))
print("inf x 40", nns.lpm(1,np.linspace(-1,1,40),np.array([-np.inf,1.]))[:2])
