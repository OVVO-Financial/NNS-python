"""Live differential harness: evaluate R NNS expressions in one Rscript batch
and compare their results with the Python port's.

R runs ``rbatch.R`` over a JSON list of cases ``{"id", "expr", "data"}`` and
returns every value with all 17 significant digits, so comparisons can be exact.
``NNS_LIVE_R_LIBS`` selects the R library holding the NNS oracle (default: R's
own library path). ``RTOL``/``ATOL`` default to 0: every number must equal R's.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
# The R library holding the NNS oracle; empty uses R's default library path.
R_LIBS = os.environ.get("NNS_LIVE_R_LIBS", "")


def out_dir() -> Path:
    """Directory for the sweeps' mismatch reports (``NNS_LIVE_OUT``)."""
    path = Path(os.environ.get("NNS_LIVE_OUT") or tempfile.gettempdir())
    path.mkdir(parents=True, exist_ok=True)
    return path


def _num(v: Any) -> float:
    if isinstance(v, str):
        return {"NA": math.nan, "NaN": math.nan, "Inf": math.inf, "-Inf": -math.inf}[v]
    if v is None:
        return math.nan
    return float(v)


def decode(node: Any) -> Any:
    t = node["t"]
    if t == "null":
        return None
    if t in ("list",):
        vals = [decode(v) for v in node["v"]]
        names = node.get("names")
        if names:
            return {"__rlist__": True, "names": list(names), "values": vals, "cls": node.get("cls")}
        return vals
    if t == "df":
        return {
            "__df__": True,
            "names": list(node["names"]),
            "values": [decode(v) for v in node["v"]],
        }
    if t == "fct":
        return {"__fct__": True, "v": list(node["v"]), "levels": list(node["levels"])}
    if t in ("num",):
        arr = np.array([_num(v) for v in node["v"]], dtype=float)
    elif t == "lgl":
        arr = np.array(
            [math.nan if v in (None, "NA") else float(bool(v)) for v in node["v"]], dtype=float
        )
    elif t in ("chr", "cplx"):
        arr = np.array(["NA" if v is None else str(v) for v in node["v"]], dtype=object)
    else:
        return {"__other__": node.get("cls")}
    dim = node.get("dim")
    if dim:
        arr = (
            arr.reshape(tuple(dim)[::-1]).T if len(dim) == 2 else arr.reshape(tuple(dim), order="F")
        )
    out: dict[str, Any] = {"__vec__": True, "t": t, "v": arr}
    if node.get("names"):
        out["names"] = list(node["names"])
    if node.get("dimnames") is not None:
        out["dimnames"] = node.get("dimnames")
    return out


def run_r(cases: list[dict[str, Any]], timeout: int = 3600) -> dict[str, dict[str, Any]]:
    with tempfile.TemporaryDirectory() as tmp:
        inp = Path(tmp) / "in.json"
        out = Path(tmp) / "out.json"
        inp.write_text(json.dumps(cases, allow_nan=False))
        env = dict(os.environ, R_LIBS=R_LIBS) if R_LIBS else dict(os.environ)
        proc = subprocess.run(
            ["Rscript", str(HERE / "rbatch.R"), str(inp), str(out)],
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr[-4000:])
        raw = json.loads(out.read_text())
    res = {}
    for item in raw:
        if item["ok"]:
            item["value"] = decode(item["value"])
        res[item["id"]] = item
    return res


def rvec(x: Any) -> np.ndarray:
    """R atomic -> numpy array (numeric)."""
    if x is None:
        return np.array([], dtype=float)
    if isinstance(x, dict) and x.get("__vec__"):
        return x["v"]
    if isinstance(x, dict) and x.get("__fct__"):
        return np.array(x["v"], dtype=object)
    raise TypeError(f"not a vector: {type(x)} {str(x)[:200]}")


def rget(x: Any, key: str) -> Any:
    if isinstance(x, dict) and (x.get("__rlist__") or x.get("__df__")):
        names = x["names"]
        if key not in names:
            raise KeyError(f"{key} not in {names}")
        return x["values"][names.index(key)]
    if isinstance(x, dict) and x.get("__vec__") and x.get("names"):
        names = x["names"]
        if key not in names:
            raise KeyError(f"{key} not in {names}")
        return {
            "__vec__": True,
            "t": x["t"],
            "v": x["v"].ravel()[names.index(key) : names.index(key) + 1],
        }
    raise TypeError(f"not a list: {type(x)}")


def rnames(x: Any) -> list[str]:
    if isinstance(x, dict) and (x.get("__rlist__") or x.get("__df__")):
        return list(x["names"])
    if isinstance(x, dict) and x.get("__vec__"):
        return list(x.get("names") or [])
    return []


def close(
    a: Any, b: Any, rtol: float | None = None, atol: float | None = None, path: str = ""
) -> list[str]:
    """Compare python value a with R-decoded value b. Return list of mismatch messages."""
    rtol = float(os.environ.get("RTOL", "0")) if rtol is None else rtol
    atol = float(os.environ.get("ATOL", "0")) if atol is None else atol
    msgs: list[str] = []
    try:
        A = np.asarray(a, dtype=float) if not isinstance(a, np.ndarray) or a.dtype != object else a
        B = rvec(b) if not isinstance(b, np.ndarray) else b
    except Exception as exc:
        return [f"{path}: cannot coerce ({exc})"]
    if B.dtype == object or (isinstance(A, np.ndarray) and A.dtype == object):
        As = np.asarray(A).astype(str).ravel()
        Bs = np.asarray(B).astype(str).ravel()
        if As.shape != Bs.shape or not np.all(As == Bs):
            msgs.append(f"{path}: strings differ py={As[:8]} r={Bs[:8]}")
        return msgs
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.size != B.size:
        return [f"{path}: size py={A.shape} r={B.shape}"]
    A = A.ravel() if A.ndim != B.ndim or A.shape != B.shape else A
    B = B.ravel() if A.ndim != B.ndim or A.shape != B.shape else B
    if A.shape != B.shape:
        A, B = A.ravel(), B.ravel()
    nanA, nanB = np.isnan(A), np.isnan(B)
    if np.any(nanA != nanB):
        idx = np.where(nanA != nanB)[0][:5] if A.ndim == 1 else np.argwhere(nanA != nanB)[:5]
        msgs.append(
            f"{path}: NaN pattern differs at {idx.tolist()} py={A.ravel()[:6]} r={B.ravel()[:6]}"
        )
        return msgs
    m = ~nanA
    inf_bad = np.any((np.isinf(A) | np.isinf(B)) & m & (A != B))
    fin = m & np.isfinite(A) & np.isfinite(B)
    diff = np.abs(A[fin] - B[fin])
    tol = atol + rtol * np.abs(B[fin])
    if inf_bad or np.any(diff > tol):
        worst = int(np.argmax(diff - tol)) if diff.size else 0
        msgs.append(
            f"{path}: max|d|={diff.max() if diff.size else 'inf'} at flat {worst}; "
            f"py={A[fin].ravel()[worst] if diff.size else A.ravel()[:4]} r={B[fin].ravel()[worst] if diff.size else B.ravel()[:4]}"
        )
    return msgs


def fmt_res(name: str, problems: dict[str, list[str]], total: int) -> str:
    bad = {k: v for k, v in problems.items() if v}
    lines = [f"== {name}: {total - len(bad)}/{total} cases match"]
    for k, v in list(bad.items())[:40]:
        lines.append(f"  [{k}] " + " | ".join(v[:4]))
    return "\n".join(lines)


def _is_rlist(x: Any) -> bool:
    return isinstance(x, dict) and (x.get("__rlist__") or x.get("__df__"))


def tree_close(
    py: Any,
    r: Any,
    path: str = "",
    rtol: float = 0.0,
    atol: float = 0.0,
    skip: tuple[str, ...] = (),
    key_map: dict[str, str] | None = None,
) -> list[str]:
    """Recursively compare a Python result tree with a decoded R value."""
    msgs: list[str] = []
    if any(path.endswith(s) or ("." + s + ".") in path for s in skip):
        return msgs
    if r is None:
        if py is None:
            return msgs
        if isinstance(py, (list, tuple, np.ndarray)) and len(py) == 0:
            return msgs
        if isinstance(py, dict) and not py:
            return msgs
        return [f"{path}: R NULL, py={type(py).__name__} {str(py)[:80]}"]
    if _is_rlist(r):
        names = r["names"]
        if hasattr(py, "to_dict") and hasattr(py, "columns"):
            py = {str(c): py[c].to_numpy() for c in py.columns}
        if not isinstance(py, dict):
            return [f"{path}: R list/df {names[:6]} but py {type(py).__name__}"]
        pk = set(map(str, py.keys()))
        missing = [n for n in names if (key_map or {}).get(n, n) not in pk]
        extra = [k for k in pk if k not in {(key_map or {}).get(n, n) for n in names}]
        if missing or extra:
            msgs.append(f"{path}: keys missing_in_py={missing[:8]} extra_in_py={extra[:8]}")
        for n, v in zip(names, r["values"], strict=True):
            k = (key_map or {}).get(n, n)
            if k in py:
                msgs += tree_close(py[k], v, f"{path}.{n}", rtol, atol, skip, key_map)
        return msgs
    if isinstance(r, list):
        if not isinstance(py, (list, tuple)):
            return [f"{path}: R unnamed list len {len(r)} vs py {type(py).__name__}"]
        if len(py) != len(r):
            return [f"{path}: list len py={len(py)} r={len(r)}"]
        for i, (a, b) in enumerate(zip(py, r, strict=True)):
            msgs += tree_close(a, b, f"{path}[{i}]", rtol, atol, skip, key_map)
        return msgs
    if isinstance(r, dict) and r.get("__fct__"):
        a = np.asarray(py).astype(str).ravel()
        b = np.asarray(r["v"]).astype(str).ravel()
        if a.shape != b.shape or not np.all(a == b):
            # allow numeric codes
            try:
                lv = r["levels"]
                codes = np.array([lv.index(v) + 1 for v in r["v"]], dtype=float)
                if np.allclose(np.asarray(py, dtype=float).ravel(), codes):
                    return msgs
            except Exception:
                pass
            msgs.append(f"{path}: factor differs py={a[:6]} r={b[:6]}")
        return msgs
    if isinstance(r, dict) and r.get("__vec__"):
        if py is None:
            if r["v"].size == 0:
                return msgs
            if (
                r["v"].size == 1
                and r["t"] in ("num", "lgl")
                and np.isnan(np.asarray(r["v"], dtype=float)).all()
            ):
                return msgs  # R scalar NA <-> Python None convention
            return [f"{path}: py None but R vec {r['v'].ravel()[:4]}"]
        if isinstance(py, dict):
            dn = r.get("dimnames")
            if r["v"].ndim == 2 and dn and len(dn) == 2 and dn[1] is not None:
                cols = [str(c) for c in dn[1]]
                if set(map(str, py.keys())) != set(cols):
                    return [f"{path}: matrix cols {cols} vs py keys {list(py)}"]
                out = []
                for j, c in enumerate(cols):
                    key = c if c in py else next(k for k in py if str(k) == c)
                    out += close(
                        np.asarray(py[key], dtype=float), r["v"][:, j], rtol, atol, f"{path}[{c}]"
                    )
                return out
            if r.get("names") and set(map(str, py.keys())) == set(r["names"]):
                out = []
                for k, nm in enumerate(r["names"]):
                    key = nm if nm in py else next(q for q in py if str(q) == nm)
                    out += close(
                        np.asarray(py[key], dtype=float),
                        r["v"].ravel()[k : k + 1],
                        rtol,
                        atol,
                        f"{path}[{nm}]",
                    )
                return out
            return [
                f"{path}: py dict {list(py)[:5]} but R vector names={r.get('names')} dimnames={dn}"
            ]
        if r["t"] in ("chr",):
            a = np.asarray(py).astype(str).ravel()
            b = np.asarray(r["v"]).astype(str).ravel()
            if a.shape != b.shape or not np.all(a == b):
                return [f"{path}: chr differs py={a[:6]} r={b[:6]}"]
            return msgs
        try:
            return close(np.asarray(py, dtype=float), r["v"], rtol, atol, path)
        except (TypeError, ValueError) as exc:
            return [f"{path}: coerce error {exc} py={str(py)[:60]}"]
    return [f"{path}: unhandled R node {str(r)[:80]}"]
