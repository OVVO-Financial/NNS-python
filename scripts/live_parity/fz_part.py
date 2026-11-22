"""Differential sweep: NNS.part at exact equality."""

import json
import sys
import warnings

import numpy as np

sys.path.insert(0, ".")
from harness import fmt_res, out_dir, run_r, tree_close

import nns

rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 91)
cases, pycalls = [], {}
for kind in ("cont", "ties", "disc", "heavy", "lin"):
    for n in (5, 20, 100, 400):
        x = rng.normal(size=n)
        if kind == "ties":
            x = np.round(x, 1)
            y = np.round(x * 2 + rng.normal(size=n) * 0.5, 1)
        elif kind == "disc":
            x = rng.integers(0, 5, n).astype(float)
            y = rng.integers(0, 3, n).astype(float)
        elif kind == "heavy":
            x = rng.standard_t(2, n)
            y = x**3 + rng.standard_t(2, n)
        elif kind == "lin":
            y = 2 * x + 1
        else:
            y = np.sin(2 * x) + rng.normal(size=n) * 0.3
        for typ, rtyp in ((None, "NULL"), ("XONLY", "'XONLY'")):
            for order, rord in ((None, "NULL"), (1, "1"), (2, "2"), (4, "4"), ("max", "'max'")):
                for obs, mos in ((8, True), (1, False), (3, True)):
                    for noise in ("off", "mean", "median", "mode", "mode_class"):
                        if noise != "off" and (order not in (None, 2) or obs != 8):
                            continue
                        cid = f"part|{kind}|{n}|{typ}|{order}|{obs}|{mos}|{noise}"
                        cases.append(
                            {
                                "id": cid,
                                "expr": f"r <- NNS.part(x, y, type = {rtyp}, order = {rord}, obs.req = {obs}, min.obs.stop = {'TRUE' if mos else 'FALSE'}, noise.reduction = '{noise}'); list(order = r$order, dt = as.list(r$dt), rp = as.list(r$regression.points))",
                                "data": {"x": x.tolist(), "y": y.tolist()},
                            }
                        )
                        pycalls[cid] = (
                            lambda x=x, y=y, typ=typ, order=order, obs=obs, mos=mos, noise=noise: (
                                lambda r: {
                                    "order": r["order"],
                                    "dt": dict(r["dt"]),
                                    "rp": dict(r["regression.points"]),
                                }
                            )(
                                nns.nns_part(
                                    x,
                                    y,
                                    type=typ,
                                    order=order,
                                    obs_req=obs,
                                    min_obs_stop=mos,
                                    noise_reduction=noise,
                                )
                            )
                        )
print("cases", len(cases), flush=True)
R = run_r(cases)
problems = {}
for c in cases:
    cid = c["id"]
    r = R[cid]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            py = pycalls[cid]()
            perr = None
    except Exception as exc:
        py, perr = None, f"{type(exc).__name__}: {exc}"
    msgs = []
    if not r["ok"]:
        if perr is None:
            msgs.append(f"R error but py ok: R={r['error'][:150]}")
    elif perr is not None:
        msgs.append(f"py error but R ok: {perr[:200]}")
    else:
        msgs += tree_close(py, r["value"], rtol=0, atol=0)
    problems[cid] = msgs

json.dump(
    {k: v for k, v in problems.items() if v}, open(f"{out_dir()}/part_problems.json", "w"), indent=1
)
print(fmt_res("NNS.part", problems, len(cases)))
