import json, math, sys
P = json.load(open(sys.argv[1])); R = json.load(open("/home/user/audit/work/DEP/r_out.json"))
def flat(v, pre=""):
    if isinstance(v, dict):
        o = {}
        for k, w in v.items(): o.update(flat(w, pre + "." + k))
        return o
    if isinstance(v, list) and len(v) == 1: v = v[0]
    return {pre: v}
for case in P:
    p = flat(P[case]); r = flat(R[case])
    for k in sorted(set(p) | set(r)):
        a, b = p.get(k), r.get(k)
        bad = False
        try:
            fa, fb = float(a), float(b)
            bad = not (abs(fa - fb) <= 1e-8 * max(1, abs(fb)) or (math.isnan(fa) and math.isnan(fb)))
        except Exception:
            bad = str(a) != str(b)
        if bad: print(f"{case}{k}: py={a} r={b}")
