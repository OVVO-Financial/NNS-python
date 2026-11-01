import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

import nns

ROOT = "/home/user/audit/work/REG/cases"
OUT = "/home/user/audit/work/REG/out_py"
os.makedirs(OUT, exist_ok=True)


def to_jsonable(v):
    if v is None:
        return None
    if isinstance(v, dict):
        return {k: to_jsonable(z) for k, z in v.items()}
    if isinstance(v, (list, tuple)):
        return [to_jsonable(z) for z in v]
    if isinstance(v, np.ndarray):
        if v.dtype.kind in "fc":
            return [None if not np.isfinite(z) else float(z) for z in v.ravel()]
        return [z.item() if hasattr(z, "item") else z for z in v.ravel()]
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    return v


def load_case(cs):
    d = os.path.join(ROOT, cs)
    meta = json.load(open(os.path.join(d, "args.json")))
    x = pd.read_csv(os.path.join(d, "x.csv"))
    y = pd.read_csv(os.path.join(d, "y.csv"))["y"].to_numpy()
    pe = pd.read_csv(os.path.join(d, "pe.csv")) if meta["has_pe"] else None
    for fc in meta["x_factor_cols"]:
        x[fc] = x[fc].astype(str)
        if pe is not None:
            pe[fc] = pe[fc].astype(str)
    if meta["univariate"]:
        x = x.iloc[:, 0].to_numpy()
        if pe is not None:
            pe = pe.iloc[:, 0].to_numpy()
    kw = {k.replace(".", "_"): v for k, v in meta["args"].items()}
    return x, y, pe, kw


def main(cases):
    for cs in cases:
        x, y, pe, kw = load_case(cs)
        t0 = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = nns.nns_reg(x, y, point_est=pe, plot=False, ncores=1, **kw)
            res = dict(res)
        except Exception as e:  # noqa: BLE001
            res = {"error": f"{type(e).__name__}: {e}"}
        el = time.perf_counter() - t0
        res["elapsed"] = el
        json.dump(to_jsonable(res), open(os.path.join(OUT, cs + ".json"), "w"))
        print(cs, res.get("error", "ok"), f"{el:.3f}s")


if __name__ == "__main__":
    main(sys.argv[1:] or sorted(os.listdir(ROOT)))
