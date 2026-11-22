"""Input validation against R 14.2's input layer (R/inputs.R).

Each exported function's data arguments get the same invalid or edge inputs in R
and in Python: missing and infinite values, character, factor and logical
vectors, a one-column data frame, a two-column matrix, empty and length-one
vectors and (for two-variable functions) unequal lengths. A case matches when
both accept it or both stop with the same message. Differences that NNS Python
documents (docs/conventions.md) are listed separately by ``documented``; any
other difference fails.
"""

import json
import warnings
from itertools import groupby

import numpy as np
import pandas as pd
from harness import out_dir, run_r

import nns

rng = np.random.default_rng(7)
x0 = np.round(rng.normal(size=30), 6)
y0 = np.round(rng.normal(size=30), 6)


def put(v, i, val):
    v = np.asarray(v, dtype=float).copy()
    v[i] = val
    return v


PERTURB = {
    "na": ("replace(v, 3, NA)", lambda v: put(v, 2, np.nan)),
    "inf": ("replace(v, 3, Inf)", lambda v: put(v, 2, np.inf)),
    "ninf": ("replace(v, 3, -Inf)", lambda v: put(v, 2, -np.inf)),
    "chr": ("as.character(v)", lambda v: np.array([f"{u:.15g}" for u in v], dtype=object)),
    "fct": ("factor(v)", lambda v: pd.Categorical(v)),
    "lgl": ("v > 0", lambda v: v > 0),
    "df1": ("data.frame(a = v)", lambda v: pd.DataFrame({"a": v})),
    "mat2": ("cbind(v, v)", lambda v: np.column_stack([v, v])),
    "empty": ("numeric(0)", lambda v: np.array([], dtype=float)),
    "len1": ("v[1]", lambda v: v[:1]),
    "short": ("v[-1]", lambda v: v[1:]),
}

UNI = {
    "LPM": ("LPM(1, 0, X)", lambda X: nns.lpm(1, 0, X)),
    "UPM": ("UPM(1, 0, X)", lambda X: nns.upm(1, 0, X)),
    "LPM.ratio": ("LPM.ratio(1, 0, X)", lambda X: nns.lpm_ratio(1, 0, X)),
    "UPM.ratio": ("UPM.ratio(1, 0, X)", lambda X: nns.upm_ratio(1, 0, X)),
    "LPM.VaR": ("LPM.VaR(0.05, 1, X)", lambda X: nns.lpm_var(0.05, 1, X)),
    "UPM.VaR": ("UPM.VaR(0.05, 1, X)", lambda X: nns.upm_var(0.05, 1, X)),
    "NNS.moments": ("NNS.moments(X)", lambda X: nns.nns_moments(X)),
    "NNS.CDF": ("NNS.CDF(X, plot = FALSE)", lambda X: nns.nns_cdf(X)),
    "NNS.gravity": ("NNS.gravity(X)", lambda X: nns.nns_gravity(X)),
    "NNS.mode": ("NNS.mode(X)", lambda X: nns.nns_mode(X)),
    "NNS.rescale": ("NNS.rescale(X, 0, 1)", lambda X: nns.nns_rescale(X, 0, 1)),
    "NNS.seas": ("NNS.seas(X, plot = FALSE)", lambda X: nns.nns_seas(X)),
    "NNS.ARMA": ("NNS.ARMA(X, h = 2, plot = FALSE)", lambda X: nns.nns_arma(X, h=2)),
    "NNS.meboot": ("NNS.meboot(X, reps = 3)", lambda X: nns.nns_meboot(X, reps=3)),
    "NNS.MC": ("NNS.MC(X, reps = 3, by = 0.5)", lambda X: nns.nns_mc(X, reps=3, by=0.5)),
}
BI = {
    "Co.LPM": ("Co.LPM(1, X, Y, 0, 0)", lambda X, Y: nns.co_lpm(1, X, Y, 0, 0)),
    "Co.UPM": ("Co.UPM(1, X, Y, 0, 0)", lambda X, Y: nns.co_upm(1, X, Y, 0, 0)),
    "D.LPM": ("D.LPM(1, 1, X, Y, 0, 0)", lambda X, Y: nns.d_lpm(1, 1, X, Y, 0, 0)),
    "D.UPM": ("D.UPM(1, 1, X, Y, 0, 0)", lambda X, Y: nns.d_upm(1, 1, X, Y, 0, 0)),
    "NNS.dep": ("NNS.dep(X, Y)", lambda X, Y: nns.nns_dep(X, Y)),
    "NNS.caus": ("NNS.caus(X, Y, tau = 0)", lambda X, Y: nns.nns_causation(X, Y, tau=0)),
    "NNS.part": ("NNS.part(X, Y)", lambda X, Y: nns.nns_part(X, Y)),
    "NNS.reg": ("NNS.reg(X, Y, plot = FALSE)", lambda X, Y: nns.nns_reg(X, Y)),
    "dy.dx": (
        "dy.dx(X, Y, eval.point = 'overall')",
        lambda X, Y: nns.dy_dx(X, Y, eval_point="overall"),
    ),
    "NNS.ANOVA": (
        "NNS.ANOVA(control = X, treatment = Y, plot = FALSE)",
        lambda X, Y: nns.nns_anova(control=X, treatment=Y),
    ),
    "NNS.FSD": ("NNS.FSD(X, Y, plot = FALSE)", lambda X, Y: nns.fsd(X, Y)),
    "NNS.FSD.uni": ("NNS.FSD.uni(X, Y)", lambda X, Y: nns.fsd_uni(X, Y)),
    "NNS.SSD": ("NNS.SSD(X, Y, plot = FALSE)", lambda X, Y: nns.ssd(X, Y)),
    "NNS.SSD.uni": ("NNS.SSD.uni(X, Y)", lambda X, Y: nns.ssd_uni(X, Y)),
    "NNS.TSD": ("NNS.TSD(X, Y, plot = FALSE)", lambda X, Y: nns.tsd(X, Y)),
    "NNS.TSD.uni": ("NNS.TSD.uni(X, Y)", lambda X, Y: nns.tsd_uni(X, Y)),
    "NNS.SS": ("NNS.SS(X, Y)", lambda X, Y: nns.nns_ss(X, Y)),
    "NNS.copula": ("NNS.copula(cbind(X, Y))", lambda X, Y: nns.nns_copula(X, Y)),
}


def documented(cid: str, r: str, py: str) -> str | None:
    """Why a difference is one NNS Python documents, or None."""
    fn, _, kind = cid.split("|")
    if r == "RERR:R CRASHED":
        return "R crashes (NNS.MC on an infinite value); Python raises"
    if "missing value where TRUE/FALSE needed" in r:
        return "R stops with an internal error on infinite values; Python computes or raises its own message"
    if kind == "df1" and (
        "'list' object cannot be coerced" in r or "cannot xtfrm data frames" in r
    ):
        return "a one-column DataFrame is a vector in Python; R fails coercing the data frame"
    if kind == "mat2" and py.startswith("ERR:") and r != py:
        return "Python rejects a matrix with several columns; R flattens it"
    if kind == "empty" and r == "OK":
        return "Python rejects empty input; R returns NaN or an empty result"
    if fn == "NNS.copula" and kind in ("len1", "short", "chr"):
        return "Python's nns_copula takes x and y; R's NNS.copula takes one matrix (cbind recycles, base R rejects text)"
    return None


cases, py = [], {}


def add(cid, rexpr, fn):
    cases.append(
        {
            "id": cid,
            "expr": "v <- x0; "
            + rexpr.split("|||")[0]
            + "; tryCatch({ "
            + rexpr.split("|||")[1]
            + "; 'OK' }, error = function(e) paste0('ERR:', conditionMessage(e)))",
            "data": {"x0": x0.tolist(), "y0": y0.tolist()},
        }
    )
    py[cid] = fn


for name, (rt, pf) in UNI.items():
    for pn, (rp, pp) in PERTURB.items():
        if pn == "short":
            continue
        add(f"{name}|x|{pn}", f"X <- {rp}|||{rt}", lambda pf=pf, pp=pp: pf(pp(x0)))
for name, (rt, pf) in BI.items():
    for pn, (rp, pp) in PERTURB.items():
        add(f"{name}|x|{pn}", f"X <- {rp}; Y <- y0|||{rt}", lambda pf=pf, pp=pp: pf(pp(x0), y0))
        add(
            f"{name}|y|{pn}",
            f"X <- x0; v <- y0; Y <- {rp}|||{rt}",
            lambda pf=pf, pp=pp: pf(x0, pp(y0)),
        )
R = {}
crashed = []
for _fname, group in groupby(cases, key=lambda c: c["id"].split("|")[0]):
    group = list(group)
    try:
        R.update(run_r(group))
    except RuntimeError:
        # find the crashing case(s) one by one
        for c in group:
            try:
                R.update(run_r([c]))
            except RuntimeError:
                crashed.append(c["id"])
                R[c["id"]] = {"ok": False, "error": "R CRASHED"}
if crashed:
    print("R crashed on:", ", ".join(crashed))
out = {}
bad = 0
for c in cases:
    cid = c["id"]
    r = R[cid]
    rres = r["value"]["v"][0] if r["ok"] else "RERR:" + r["error"]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            py[cid]()
            pres = "OK"
    except Exception as e:
        pres = "ERR:" + str(e)
    if rres != pres:
        bad += 1
        out[cid] = {"r": rres, "py": pres}
known = {cid: why for cid, v in out.items() if (why := documented(cid, v["r"], v["py"]))}
failing = {cid: v for cid, v in out.items() if cid not in known}
print(f"== input validation: {len(cases) - len(failing)}/{len(cases)} cases match")
for cid, v in failing.items():
    print(f"  [{cid}] R: {v['r'][:120]} | py: {v['py'][:120]}")
reasons = {}
for why in known.values():
    reasons[why] = reasons.get(why, 0) + 1
print(f"documented differences: {len(known)}")
for why, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
    print(f"  {n:3d}  {why}")
json.dump(
    {"failing": failing, "documented": {k: {**out[k], "why": known[k]} for k in known}},
    open(f"{out_dir()}/input_validation.json", "w"),
    indent=1,
)
