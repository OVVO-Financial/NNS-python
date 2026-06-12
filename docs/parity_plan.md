# Parity Plan

This branch completes the pre-migration parity suite for `pyNNS-core-backed` while keeping the `NNS-python` migration out of scope.

## Closed gap workstream (branch `close-all-parity-gaps`)

The gaps previously tracked in `docs/parity_results.md` are now closed or
formally resolved:

1. **`nns_boost` cache-parity failure** — triaged as seed-sensitivity on the
   CV-split path for an unseeded call. The boosted result is empirically
   seed-invariant and matches the committed R cache to ~3.5e-15; the parity test
   now pins a seed and a `test_nns_boost_ivs_test_none_is_seed_invariant`
   regression guard was added.
2. **`NNS.copula` discrete mode** — implemented (`continuous=False`) and adopted.
3. **`NNS.copula` multivariate / three-column** — implemented (matrix input) and
   adopted for both continuous and discrete.
4. **`PM.matrix` data-frame naming** — optional NumPy-first `names` echo added
   with a parity test; numeric behavior unchanged.
5. **Plot / graphics policy** — formalized in `docs/plot_parity_policy.md`.
6. **Skips** — the only remaining skips are intentional live-R-only practical
   examples (not cache-backed parity gaps).

## Scope

- Preserve public-behavior parity tests against R NNS 12.1 through `tests/parity/`.
- Keep R calls isolated in the test harness and cache tooling.
- Allow CI to run parity checks without `Rscript` by using committed cache fixtures with `PYNNS_R_CACHE_ONLY=1`.
- Preserve native-vs-Python fallback coverage for partial moments and related helpers.
- Preserve the merged PR #6 fix that blocks non-finite partial-moment inputs from native dispatch.

## Cache workflow

- `tests/_r_cache.json` is the committed R-compatible cache used by CI.
- `PYNNS_R_CACHE_ONLY=1` forces cache-only parity and must be used in CI.
- To refresh cache entries on a workstation with R and NNS installed, run:

```bash
python scripts/regenerate_r_cache.py
```

Pass pytest selectors after `--` to refresh a narrower subset, for example:

```bash
python scripts/regenerate_r_cache.py -- tests/parity/test_core.py
```

## Guardrails

- Do not require `Rscript` in CI.
- Do not reintroduce stale native expectations for partial moments.
- Do not import `pynns.pm_matrix` through the package-level public function when module access is required; use `importlib.import_module("pynns.pm_matrix")`.
- Do not route `NaN` or infinite partial-moment inputs through native `lpm`, `upm`, `lpm_ratio`, or `upm_ratio` dispatch.
