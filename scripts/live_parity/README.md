# Live R sweeps

Differential sweeps that run the Python port and the installed R NNS on the same
inputs and require every number to be equal to R's (`RTOL = ATOL = 0`). They
complement the committed cache in `tests/_r_cache/`: the cache pins chosen cases
for offline tests, while the sweeps generate thousands of cases per run across
every exported area, including the edge cases (ties, signed zeros, infinities,
tiny scales, pooled cross-validation folds) where a port drifts first.

## Requirements

- R 4.5.0 with the reference BLAS/LAPACK, as in [docs/parity.md](../../docs/parity.md),
  with `jsonlite` and the pinned NNS oracle installed.
- The `nns` package importable (an editable install of this repository).

## Running

```bash
python scripts/live_parity/run.py --mode both --jobs 4 --summary live-parity.md
```

`--mode native` checks the compiled extension, `--mode fallback` the NumPy
implementation (`NNS_DISABLE_NATIVE=1`), and `--mode both` runs each sweep both
ways. `--only fz_reg fz_stack` runs a subset. The exit status is 1 when any case
differs, a sweep fails to run or a sweep reports no result.

A single sweep can also run on its own, for example
`RTOL=0 ATOL=0 python scripts/live_parity/fz_pm.py`. It prints
`== <area>: <matched>/<total> cases match` and the first mismatches.

## Environment variables

| Variable | Meaning |
| --- | --- |
| `NNS_LIVE_R_LIBS` | R library holding the NNS oracle (default: R's own library path) |
| `NNS_LIVE_OUT` | Directory for the sweeps' JSON mismatch reports (default: the system temp directory) |
| `RTOL`, `ATOL` | Comparison tolerances; 0 (exact) unless set |
| `NNS_DISABLE_NATIVE` | `1` runs the NumPy implementation; set by `run.py --mode fallback` |

## Files

- `harness.py` - runs one `Rscript rbatch.R` batch per sweep and compares
  decoded R values with Python results (`tree_close`, `close`).
- `rbatch.R` - evaluates each case's R expression with its data and returns
  every value with 17 significant digits, errors and warnings included.
- `fz_*.py`, `stack_ties.py`, `dep_perm.py`, `big_mreg.py`, `edge_new.py` - the
  sweeps; `run.py` lists the area each one covers.

The cache builder (`.github/workflows/inspect-r-api-update.yml`) runs every
sweep in both modes after it regenerates the cache for a new R release, and
reports the table in its pull request.
