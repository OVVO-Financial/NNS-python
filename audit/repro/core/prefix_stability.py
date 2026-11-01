import numpy as np, pandas as pd, nns
from nns import _native
z = pd.read_csv("/home/user/audit/data/prices_daily.csv")["SPY"].dropna().to_numpy()
tz = np.linspace(z.min()-50, z.max()+50, 100)
upm_v = _native.native_fn("upm_v")
for d in (2,4,8):
    v = upm_v(float(d), tz, z)
    direct = np.array([np.mean(np.maximum(z-t,0)**d) for t in tz])
    small = np.array([_native.native_fn("upm")(float(d), float(t), z) for t in tz])
    rel = np.abs(np.array(v)-direct)/np.maximum(np.abs(direct),1e-300)
    print(f"deg{d} NNS-core upm_v(100 targets) max rel err {rel.max():.3g}, negatives {(np.array(v)<0).sum()}; scalar path rel err {np.max(np.abs(small-direct)/np.maximum(direct,1e-300)):.3g}")
    print("   nns.UPM public:", np.min(nns.upm(d, tz, z)))
i = int(np.argmax(np.abs(np.array(upm_v(8.0, tz, z))-np.array([np.mean(np.maximum(z-t,0)**8) for t in tz]))/np.maximum([np.mean(np.maximum(z-t,0)**8) for t in tz],1e-300)))
print("worst target", tz[i], "upm_v", upm_v(8.0, tz, z)[i], "direct", np.mean(np.maximum(z-tz[i],0)**8), "first-32-targets call", upm_v(8.0, tz[:32] if i<32 else tz[i:i+1], z)[min(i,31) if i<32 else 0])
x = np.random.default_rng(1).normal(100,1,10000)
t = np.sort(x)[-100:]-1e-9
print("deg2 near max min:", np.min(nns.upm(2, t, x)))
