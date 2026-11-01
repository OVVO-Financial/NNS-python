# NNS-python sync contract

`OVVO-Financial/NNS-python` is the final downstream integration layer in the NNS
supply chain.

```text
OVVO-Financial/NNS
  -> OVVO-Financial/NNS-core
  -> OVVO-Financial/NNS-python
```

## Authority

`OVVO-Financial/NNS` is the statistical source of truth.

`OVVO-Financial/NNS-core` is the accepted portable C++ extraction of the R
`src/**` layer.

`OVVO-Financial/NNS-python` consumes accepted `NNS-core` snapshots and verifies
public Python API behavior against live or cached R NNS.

## Native code ingress rule

Native C++ source changes enter Python only through accepted public
`OVVO-Financial/NNS-core` commits.

Python must not auto-port native C++ directly from `OVVO-Financial/NNS`.

Native snapshots are vendored under:

```text
extern/NNS-core
```

`.github/workflows/sync-from-nns-core.yml` receives NNS-core's
`nns-core-updated` repository dispatch (payload
`core_commit, r_repo, r_commit, r_version, r_src_tree_hash`). It can also be run
by hand with a `core_commit`. It re-vendors the commit with
`scripts/sync_nns_core_snapshot.py`, which copies the `CMakeLists.txt`,
`include/` and `src/` of that commit and records `core_commit`. It then builds the
extension, runs the invariant and cached parity gates with and without
`NNS_DISABLE_NATIVE=1`, and opens a pull request, as a draft when a gate fails.
The receiver is in `workflow-baseline.yml`'s protected list; NNS-core's
`notify-python.yml` checks that it exists before dispatching.

## R behavior fidelity rule

Python API behavior is tested directly against live `OVVO-Financial/NNS` at the
recorded R commit.

The following upstream changes require Python parity review:

* `R/**`
* `NAMESPACE`
* `DESCRIPTION`
* exported return behavior
* tests or examples that expose changed public behavior

A `DESCRIPTION` version change requires fresh live-R parity-cache regeneration.

## Required gates

A Python sync PR is not complete until these pass:

```bash
python -m pip install -e . --force-reinstall
python -m pytest -q tests/invariants
NNS_R_CACHE_ONLY=1 python -m pytest -q tests/parity
NNS_DISABLE_NATIVE=1 NNS_R_CACHE_ONLY=1 python -m pytest -q tests/invariants tests/parity
python -m pytest -q tests/parity/test_r13_smoke.py
python -m pytest -q tests/docs/test_vignette_examples.py
ruff check .
mypy
python -m build
```

If `tests/docs/test_vignette_examples.py` is not present yet, the workflow may
skip that gate with an explicit message.

## Fresh R cache rule

When R `DESCRIPTION` changes, or when mapped live-R parity proves changed public
behavior, regenerate from empty:

```bash
python scripts/install_local_r_nns.py
python scripts/regenerate_r_cache.py --fresh -- -n 0 tests/parity
NNS_R_CACHE_ONLY=1 python -m pytest -q -n 0 tests/parity
```

## Traceability

Every Python release must be traceable to:

* one R NNS commit for behavioral truth
* one NNS-core commit for native truth
* one parity cache generated from the R truth commit
