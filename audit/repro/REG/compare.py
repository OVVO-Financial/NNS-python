import json
import os
import sys

import numpy as np

R = "/home/user/audit/work/REG/out_r"
P = __import__("os").environ.get("PYOUT", "/home/user/audit/work/REG/out_py")
RTOL = 1e-7
ATOL = 1e-9


def num(v):
    try:
        a = np.asarray([np.nan if z is None else z for z in v] if isinstance(v, list) else [np.nan if v is None else v], dtype=float)
        return a
    except (TypeError, ValueError):
        return None


def cmp(path, r, p, diffs):
    if isinstance(r, dict) and isinstance(p, dict):
        for k in sorted(set(r) | set(p)):
            if k == "elapsed":
                continue
            if k not in p:
                if r[k] not in (None, [], {}):
                    diffs.append(f"{path}.{k}: missing in py")
                continue
            if k not in r:
                if p[k] not in (None, [], {}):
                    diffs.append(f"{path}.{k}: missing in R (py has {str(p[k])[:60]})")
                continue
            cmp(f"{path}.{k}", r[k], p[k], diffs)
        return
    if isinstance(r, dict) or isinstance(p, dict):
        diffs.append(f"{path}: type mismatch R={type(r).__name__} py={type(p).__name__}")
        return
    if r is None and p is None:
        return
    if r is None or p is None:
        if (r in ([], None)) and (p in ([], None)):
            return
        diffs.append(f"{path}: R={str(r)[:60]} py={str(p)[:60]}")
        return
    ra, pa = num(r), num(p)
    if ra is not None and pa is not None:
        if ra.size != pa.size:
            diffs.append(f"{path}: length R={ra.size} py={pa.size}")
            return
        ok = np.isclose(ra, pa, rtol=RTOL, atol=ATOL, equal_nan=True)
        if not ok.all():
            i = np.flatnonzero(~ok)
            rel = np.nanmax(np.abs(ra[i] - pa[i]) / np.maximum(np.abs(ra[i]), 1e-12))
            diffs.append(f"{path}: {i.size}/{ra.size} differ, max rel {rel:.3g}; first idx {i[0]} R={ra[i[0]]:.10g} py={pa[i[0]]:.10g}")
        return
    rl = r if isinstance(r, list) else [r]
    pl = p if isinstance(p, list) else [p]
    if [str(z) for z in rl] != [str(z) for z in pl]:
        diffs.append(f"{path}: R={str(rl)[:80]} py={str(pl)[:80]}")


def main(cases):
    for cs in cases:
        r = json.load(open(os.path.join(R, cs + ".json")))
        p = json.load(open(os.path.join(P, cs + ".json")))
        diffs = []
        cmp("", r, p, diffs)
        tag = "OK " if not diffs else "DIFF"
        print(f"{tag} {cs}  (R {r.get('elapsed', 0):.3f}s, py {p.get('elapsed', 0):.3f}s)")
        for d in diffs[:12]:
            print("     ", d)


if __name__ == "__main__":
    main(sys.argv[1:] or sorted(f[:-5] for f in os.listdir(R)))
