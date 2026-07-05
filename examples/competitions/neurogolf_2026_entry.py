#!/usr/bin/env python3
"""NNS.part-aware NeuroGolf 2026 starter submission generator."""
from __future__ import annotations

import argparse, json, math, zipfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

try:
    import nns
except Exception as exc:
    raise RuntimeError("Install with: pip install ovvo-nns") from exc

C, H, W = 10, 30, 30
OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]


def pad_grid(g: list[list[int]], fill: int = -1) -> np.ndarray:
    a = np.full((H, W), fill, dtype=np.int16)
    x = np.asarray(g, dtype=np.int16)
    if x.ndim != 2:
        raise ValueError("grid must be rectangular")
    h, w = x.shape
    if h > H or w > W:
        raise ValueError(f"grid shape {(h, w)} exceeds {(H, W)}")
    a[:h, :w] = x
    return a


def onehot(g: np.ndarray) -> np.ndarray:
    z = np.zeros((C, H, W), dtype=np.float32)
    for c in range(C):
        z[c] = (g == c).astype(np.float32)
    return z


def targets(g: np.ndarray) -> np.ndarray:
    return np.moveaxis(onehot(g), 0, -1).reshape(-1, C)


def shift(g: np.ndarray, dr: int, dc: int) -> np.ndarray:
    z = np.full((H, W), -1, dtype=np.int16)
    sr0, sr1 = max(0, -dr), min(H, H - dr)
    sc0, sc1 = max(0, -dc), min(W, W - dc)
    rr0, rr1 = max(0, dr), min(H, H + dr)
    rc0, rc1 = max(0, dc), min(W, W + dc)
    z[rr0:rr1, rc0:rc1] = g[sr0:sr1, sc0:sc1]
    return z.reshape(-1)


def design(grids: list[np.ndarray], offs: list[tuple[int, int]]) -> np.ndarray:
    rows = []
    for g in grids:
        cols = []
        for dr, dc in offs:
            s = shift(g, dr, dc)
            o = np.zeros((H * W, C), dtype=np.float32)
            m = (s >= 0) & (s < C)
            r = np.where(m)[0]
            o[r, s[r]] = 1.0
            cols.append(o)
        rows.append(np.concatenate(cols, axis=1))
    return np.concatenate(rows, axis=0)


def dep(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return 0.0
    try:
        v = nns.nns_dep(x, y)
    except Exception:
        c = np.corrcoef(x, y)[0, 1]
        return float(abs(c)) if np.isfinite(c) else 0.0
    if isinstance(v, dict):
        for k in ("Dependence", "dependence", "NNS.dep", "nns_dep"):
            if k in v:
                return float(np.asarray(v[k]).ravel()[0])
        return float(np.asarray(next(iter(v.values()))).ravel()[0])
    return float(np.asarray(v).ravel()[0])


def part_cells(ins: list[np.ndarray], outs: list[np.ndarray], order: int, obs: int) -> tuple[np.ndarray, np.ndarray]:
    ids = np.arange(H * W, dtype=float)
    xs, ys = [], []
    for i, o in zip(ins, outs):
        ii, oo = i.reshape(-1), o.reshape(-1)
        valid = oo >= 0
        xs.append(ids[valid])
        ys.append(((ii != oo) & valid).astype(float)[valid])
    x, y = np.concatenate(xs), np.concatenate(ys)
    if y.size == 0 or np.nanstd(y) == 0:
        return np.ones((H, W)), np.ones((H, W), dtype=bool)
    p = nns.nns_part(x, y, type="XONLY", order=order, obs_req=obs, noise_reduction="mean")
    q = np.asarray(p["dt"]["quadrant"], dtype=str)
    idv = np.asarray(p["dt"]["x"], dtype=np.int64)
    imp = np.zeros(H * W, dtype=float)
    for u in np.unique(q):
        m = q == u
        np.maximum.at(imp, idv[m], float(np.mean(y[m])))
    active = imp >= max(0.01, float(np.mean(y)))
    if not np.any(active):
        active[:] = True
    return imp.reshape(H, W), active.reshape(H, W)


def choose_offsets(ins: list[np.ndarray], outs: list[np.ndarray], mn: float, mx: int, order: int, obs: int):
    imp, act = part_cells(ins, outs, order, obs)
    a = act.reshape(-1)
    y = np.concatenate([o.reshape(-1)[a] for o in outs]).astype(float)
    scored = []
    for off in OFFS:
        x = np.concatenate([shift(g, off[0], off[1])[a] for g in ins]).astype(float)
        scored.append((dep(x, y), off))
    scored.sort(reverse=True, key=lambda z: z[0])
    sel = [off for s, off in scored if s >= mn][:mx] or [(0, 0)]
    sel = sorted(sel, key=lambda off: 0 if off == (0, 0) else 1)
    log = [{"offset": list(off), "score": float(s)} for s, off in scored]
    return sel, imp, act, log


def fit(ins: list[np.ndarray], outs: list[np.ndarray], offs: list[tuple[int, int]]):
    x = design(ins, offs)
    y = np.concatenate([targets(o) for o in outs], axis=0)
    b, *_ = np.linalg.lstsq(x, y, rcond=None)
    b = b.astype(np.float32)
    return b, float(np.mean((x @ b - y) ** 2))


def predict(g: np.ndarray, offs: list[tuple[int, int]], b: np.ndarray) -> np.ndarray:
    return (design([g], offs) @ b).reshape(H, W, C).transpose(2, 0, 1)


def exact(pred: np.ndarray, tgt: np.ndarray, tol: float) -> bool:
    return bool(np.max(np.abs(pred - onehot(tgt))) <= tol)


def weights(offs: list[tuple[int, int]], b: np.ndarray):
    k = 1 if offs == [(0, 0)] else 3
    w = np.zeros((C, C, k, k), dtype=np.float32)
    for oi, (dr, dc) in enumerate(offs):
        kr, kc = (0, 0) if k == 1 else (dr + 1, dc + 1)
        if 0 <= kr < k and 0 <= kc < k:
            for ci in range(C):
                for co in range(C):
                    w[co, ci, kr, kc] = b[oi * C + ci, co]
    w[np.abs(w) < 1e-7] = 0.0
    return w, np.zeros(C, dtype=np.float32), k


def identity():
    w = np.zeros((C, C, 1, 1), dtype=np.float32)
    for c in range(C):
        w[c, c, 0, 0] = 1.0
    return w, np.zeros(C, dtype=np.float32), 1


def zero():
    return np.zeros((C, C, 1, 1), dtype=np.float32), np.zeros(C, dtype=np.float32), 1


def save_onnx(path: Path, w: np.ndarray, b: np.ndarray, k: int, doc: str) -> None:
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, C, H, W])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, C, H, W])
    node = helper.make_node("Conv", ["input", "W", "B"], ["output"], kernel_shape=[k, k], pads=([0, 0, 0, 0] if k == 1 else [1, 1, 1, 1]), strides=[1, 1])
    graph = helper.make_graph([node], "nns_neurogolf_local_conv", [inp], [out], [numpy_helper.from_array(w, "W"), numpy_helper.from_array(b, "B")])
    model = helper.make_model(graph, producer_name="ovvo-nns-neurogolf-starter", doc_string=doc, opset_imports=[helper.make_operatorsetid("", 17)])
    onnx.checker.check_model(model)
    onnx.save(model, path)


def pairs(task: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    out = []
    for split in ("train", "arc-gen", "test"):
        for e in task.get(split, []):
            if "input" in e and "output" in e:
                out.append((pad_grid(e["input"]), pad_grid(e["output"])))
    return out


def same_size(task: dict[str, Any]) -> bool:
    for split in ("train", "arc-gen", "test"):
        for e in task.get(split, []):
            if "input" not in e or "output" not in e:
                continue
            ih, oh = len(e["input"]), len(e["output"])
            iw, ow = len(e["input"][0]) if ih else 0, len(e["output"][0]) if oh else 0
            if (ih, iw) != (oh, ow):
                return False
    return True


def solve(path: Path, out: Path, args: argparse.Namespace) -> dict[str, Any]:
    task = json.loads(path.read_text(encoding="utf-8"))
    ps = pairs(task)
    log = {"task": path.stem, "status": "fallback", "offsets": [], "offset_scores": [], "train_mse": math.nan, "exact_train_match": False, "part_importance_max": math.nan, "part_importance_mean": math.nan, "part_active_cells": 0}
    if ps and same_size(task):
        ins, outs = [p[0] for p in ps], [p[1] for p in ps]
        offs, imp, act, scores = choose_offsets(ins, outs, args.min_dep, args.max_offsets, args.part_order, args.part_obs_req)
        b, mse = fit(ins, outs, offs)
        ok = all(exact(predict(i, offs, b), o, args.exact_tol) for i, o in zip(ins, outs))
        log.update({"offsets": [list(o) for o in offs], "offset_scores": scores, "train_mse": mse, "exact_train_match": ok, "part_importance_max": float(np.max(imp)), "part_importance_mean": float(np.mean(imp)), "part_active_cells": int(np.sum(act))})
        if ok:
            w, bb, k = weights(offs, b)
            save_onnx(out, w, bb, k, f"{path.stem}: NNS.part-aware offsets {offs}")
            log["status"] = "nns_part_local_conv"
            return log
    w, bb, k = identity() if args.fallback == "identity" else zero()
    save_onnx(out, w, bb, k, f"{path.stem}: fallback")
    return log


def demo_grid(g: list[list[int]], mp: dict[int, int]) -> list[list[int]]:
    return [[mp.get(v, v) for v in r] for r in g]


def make_demo(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    a = [[1, 2, 3, 4], [4, 3, 2, 1], [1, 1, 2, 2]]
    b = [[2, 1, 4, 3], [3, 4, 1, 2], [2, 2, 1, 1]]
    mp = {1: 6, 2: 7, 3: 8, 4: 9}
    t1 = {"train": [{"input": a, "output": demo_grid(a, mp)}, {"input": b, "output": demo_grid(b, mp)}], "arc-gen": [], "test": []}
    c = [[0, 1, 0, 1], [2, 2, 3, 3], [4, 5, 4, 5]]
    t2 = {"train": [{"input": c, "output": c}], "arc-gen": [], "test": []}
    x = [[1, 1, 1, 1], [1, 2, 2, 1], [1, 2, 2, 1], [1, 1, 1, 1]]
    y = [[1, 1, 1, 1], [1, 9, 9, 1], [1, 9, 9, 1], [1, 1, 1, 1]]
    t3 = {"train": [{"input": x, "output": y}], "arc-gen": [], "test": []}
    for name, obj in {"task001": t1, "task002": t2, "task003": t3}.items():
        (d / f"{name}.json").write_text(json.dumps(obj, indent=2), encoding="utf-8")


def task_files(d: Path) -> list[Path]:
    return sorted(p for p in d.rglob("task*.json") if p.is_file())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=Path)
    p.add_argument("--make_demo_data", type=Path)
    p.add_argument("--work_dir", type=Path, default=Path("nns_neurogolf_models"))
    p.add_argument("--out", type=Path, default=Path("submission.zip"))
    p.add_argument("--report", type=Path, default=Path("nns_neurogolf_report.json"))
    p.add_argument("--min_dep", type=float, default=0.02)
    p.add_argument("--max_offsets", type=int, default=5)
    p.add_argument("--part_order", type=int, default=8)
    p.add_argument("--part_obs_req", type=int, default=4)
    p.add_argument("--exact_tol", type=float, default=1e-4)
    p.add_argument("--fallback", choices=["identity", "zero"], default="identity")
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    if args.make_demo_data is not None:
        make_demo(args.make_demo_data)
        args.data_dir = args.data_dir or args.make_demo_data
    if args.data_dir is None:
        raise SystemExit("Provide --data_dir or --make_demo_data.")
    files = task_files(args.data_dir)
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError(f"No task*.json files found under {args.data_dir}")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    print(f"nns version: {getattr(nns, '__version__', 'unknown')}")
    print(f"data_dir: {args.data_dir}")
    logs = []
    for i, f in enumerate(files, 1):
        log = solve(f, args.work_dir / f"{f.stem}.onnx", args)
        logs.append(log)
        print(f"[{i:03d}/{len(files):03d}] {log['task']} {log['status']} exact={log['exact_train_match']} mse={log['train_mse']} active_cells={log['part_active_cells']} offsets={log['offsets']}")
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(args.work_dir.glob("task*.onnx")):
            z.write(f, f.name)
    summary = {"n_tasks": len(logs), "n_solved": sum(x["status"] == "nns_part_local_conv" for x in logs), "n_fallback": sum(x["status"] != "nns_part_local_conv" for x in logs), "nns_version": getattr(nns, "__version__", "unknown"), "tasks": logs}
    args.report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote models: {args.work_dir}\nWrote zip:    {args.out}\nWrote report: {args.report}")
    print(f"Tasks emitted: {len(logs)}")
    print(f"NNS.part local-conv exact train matches: {summary['n_solved']}")
    print(f"Fallbacks: {summary['n_fallback']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
