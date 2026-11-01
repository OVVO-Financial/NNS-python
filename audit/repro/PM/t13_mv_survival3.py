"""Multivariate NNS.CDF type='survival' uses 1 - sum(marginals) + joint CDF, which is the
inclusion-exclusion formula for d = 2 only; for d >= 3 it is wrong. Also out-of-bounds 3rd target accepted."""
import sys, warnings
sys.path.insert(0, "/home/user/audit/work/PM")
import numpy as np, nns
from rrun import rcall, arr
rng = np.random.default_rng(4)
A = rng.normal(size=(400, 3)); A[:, 1] += A[:, 0]; A[:, 2] += 0.5 * A[:, 1]
true_s = np.array([np.mean(np.all(A > row, axis=1)) for row in A])
py = nns.nns_cdf(A, 0, type="survival")["Function"]["CDF"]
r = arr(rcall("NNS.CDF(A, 0, type='survival', plot=FALSE)$Function$CDF", A=A))
print("d=3 survival: max|py - true P(X>x,Y>y,Z>z)| =", np.max(np.abs(py - true_s)), " max|R - true| =", np.max(np.abs(r - true_s)), " py==R:", np.allclose(py, r))
i = int(np.argmax(np.abs(py - true_s))); print("  worst row", A[i].round(3), "true", true_s[i], "nns", py[i])
# target bounds: third coordinate far out of support accepted
print("py target 3rd coord out of support:", nns.nns_cdf(A, 0, target=np.array([0.0, 0.0, 1e6]))["target.value"])
print("R  target 3rd coord out of support:", rcall("NNS.CDF(A, 0, target=c(0,0,1e6), plot=FALSE)$target.value", A=A))
