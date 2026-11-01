import sys, json, numpy as np
import stack_patched as S
from rcor_emul import r_spearman
def sp(x,y):
    out=np.array([r_spearman(x[:,j],y) for j in range(x.shape[1])]); out[~np.isfinite(out)]=0; return out
import nns; nns.nns_stack=S.nns_stack; py_stack_mod=None
import py_stack, compare; py_stack.nns_stack=S.nns_stack
from compare import report
for name in sys.argv[1:]:
    spec=json.load(open(f"cases/{name}/spec.json")); R=json.load(open(f"cases/{name}/r.json"))
    pa = {}
    m={"method":"method","stack":"stack","pred.int":"pred_int","dim.red.method":"dim_red_method","folds":"folds","seed":"seed","order":"order","dist":"dist","type":"type","balance":"balance","optimize.threshold":"optimize_threshold","CV.size":"cv_size","ts.test":"ts_test","objective":"objective"}
    for k,v in spec["args"].items():
        if k=="obj.fn": pa["obj_fn"]=[f for (rs,f) in compare.OBJ.values() if rs==v][0]
        else: pa[m[k]]=v
    spec["pyargs"]=pa
    P=py_stack.run(spec); report(name+" [PATCHED cor]", R, P, keys=("OBJfn.reg","NNS.reg.n.best","OBJfn.dim.red","probability.threshold","weights","stack"))
