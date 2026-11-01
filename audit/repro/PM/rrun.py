"""Tiny R bridge for the PM audit: rcall(code, **data) -> python object.

Data are passed as JSON (full precision); `code` is R code evaluated with each
data item bound as a variable; the value of the last expression is returned.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile

import numpy as np

WORK = os.path.dirname(os.path.abspath(__file__))


def _enc(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def rcall(code: str, **data):
    with tempfile.TemporaryDirectory(dir=WORK) as td:
        din = os.path.join(td, "in.json")
        dout = os.path.join(td, "out.json")
        dcode = os.path.join(td, "code.R")
        with open(din, "w") as f:
            json.dump({k: _enc(v) for k, v in data.items()}, f)
        with open(dcode, "w") as f:
            f.write(code)
        script = f"""
suppressPackageStartupMessages({{library(NNS); library(jsonlite)}})
RcppParallel::setThreadOptions(numThreads = 1)
.d <- fromJSON('{din}', simplifyVector = TRUE, simplifyMatrix = TRUE)
for (.k in names(.d)) assign(.k, .d[[.k]])
.res <- eval(parse('{dcode}'))
writeLines(toJSON(.res, digits = NA, auto_unbox = TRUE, na = 'string', matrix = 'rowmajor', force = TRUE), '{dout}')
"""
        p = subprocess.run(["Rscript", "-e", script], capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(p.stderr[-3000:])
        with open(dout) as f:
            return json.load(f)


def arr(v):
    def conv(z):
        if z == "NaN" or z == "NA":
            return np.nan
        if z == "Inf":
            return np.inf
        if z == "-Inf":
            return -np.inf
        return z
    if isinstance(v, list):
        return np.array([arr(z) if isinstance(z, list) else conv(z) for z in v], dtype=float)
    return np.array(conv(v), dtype=float)
