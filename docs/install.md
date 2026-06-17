# Install

## From PyPI

```bash
pip install ovvo-nns
```

The distribution package is **`ovvo-nns`**; the import package is **`nns`**:

```python
import nns

print(nns.__version__)
```

Installing `ovvo-nns` includes the Matplotlib plotting API (`nns.plotting`).
Matplotlib is a regular dependency and is imported lazily, so `import nns` stays
light. See the [plot parity policy](plot_parity_policy.md) for details.

## Requirements

| Requirement | Value |
|---|---|
| Python | `>=3.11` (CPython 3.11, 3.12, 3.13, 3.14) |
| Runtime dependencies | NumPy, SciPy, Matplotlib |
| R at runtime | Not required |

R is used only for parity tests and local cache regeneration, never at normal
runtime.

## Wheels vs. source builds

Published wheels should be preferred when available. They ship the optional
private native extension (`nns._nnscore`) prebuilt for supported platforms.

Source builds compile the optional native extension with
[`scikit-build-core`](https://scikit-build-core.readthedocs.io/) and
[`nanobind`](https://nanobind.readthedocs.io/), which require a C++17 toolchain.
Public APIs keep Python implementations and explicit fallback behavior, so the
native extension remains a private, benchmark-backed implementation detail rather
than a public API.

## Development install

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
uv run mypy
```

The default parity suite is cache-backed and does not require `Rscript`.
`Rscript` and the R `NNS` package are needed only when regenerating parity caches
or running live R comparison scripts.
