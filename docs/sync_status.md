# NNS sync status

The authoritative machine-readable file is:

```text
sync/nns_source.json
```

| Layer    | Repository                  |       Commit | Role                                    |
| -------- | --------------------------- | -----------: | --------------------------------------- |
| R source | `OVVO-Financial/NNS`        | See manifest | Statistical and R API source of truth   |
| C++ core | `OVVO-Financial/NNS-core`   | See manifest | Accepted portable native core           |
| Python   | `OVVO-Financial/NNS-python` | Current repo | Python API, parity, packaging, examples |

Native code enters Python only through the accepted public `NNS-core` snapshot
vendored under:

```text
extern/NNS-core
```

R behavior is checked directly against live R NNS when required, and against the
committed parity cache in ordinary CI.

## Notes on manifest fields

`r_src_tree_hash` records the git tree object hash of the vendored R `src/**`
tree (`tools/NNS/src`). The `r_commit` and `core_commit` fields are populated by
the sync automation when an upstream `NNS` or `NNS-core` event fires; until a
sync event records them they may read `unknown`. The `r_version` is taken from
the vendored R `DESCRIPTION` (`13.0`).
