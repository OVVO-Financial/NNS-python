# Releasing NNS-python to PyPI

NNS-python ships a native (C++17 / nanobind) extension, so a release builds
multi-platform wheels with [`cibuildwheel`](https://cibuildwheel.pypa.io) plus an
sdist, and publishes via **PyPI Trusted Publishing (OIDC)** — no API token is
stored in the repo.

The pipeline is `.github/workflows/release.yml`.

The official distribution name is **`ovvo-nns`** (`pip install ovvo-nns`, then
`import nns`). The import package stays `nns`.

## One-time setup

### 1. Claim the PyPI project name
The first Trusted-Publishing upload registers the `ovvo-nns` name to your
account/organization — no takeover of the legacy `NNS` project is required. Use
your own PyPI account (enable 2FA), and consider owning it under a PyPI
**Organization** (e.g. `OVVO-Financial`). For the very first publish, add a
**pending** Trusted Publisher (PyPI → *Your projects* → *Publishing* → *Add a
pending publisher*) for the not-yet-existing `ovvo-nns` project.

(Optional: if you later also obtain the legacy `NNS` name, you can publish it as
an alias pointing at the same `import nns` package.)

### 2. Configure Trusted Publishers
On PyPI (and TestPyPI) for project **`ovvo-nns`** → **Publishing** → add a GitHub
Actions trusted publisher:

| Field | Value |
| --- | --- |
| PyPI Project Name | `ovvo-nns` |
| Owner | `OVVO-Financial` |
| Repository | `NNS-python` |
| Workflow name | `release.yml` |
| Environment | `pypi` (and `testpypi` on TestPyPI) |

### 3. Create GitHub environments
Repo **Settings → Environments** → create `pypi` and `testpypi`. Optionally add
required reviewers on `pypi` for a manual approval gate before publish.

No `ANTHROPIC_API_KEY` / PyPI token secret is needed — Trusted Publishing uses
the workflow's OIDC identity, and PEP 740 provenance attestations are generated
automatically.

## Wheels built

`[tool.cibuildwheel]` in `pyproject.toml` builds CPython **3.11–3.13** for:

* Linux manylinux + musllinux (`x86_64`)
* macOS `x86_64` and `arm64`
* Windows `AMD64`

Each wheel is smoke-tested (`import nns._nnscore` + a numeric call). PyPy and
32-bit targets are skipped.

## Cutting a release

1. Finalize the version in `pyproject.toml` (drop the pre-release suffix when
   ready, e.g. `1.0.0a0` → `1.0.0`).
2. **Make the release traceable**: set `r_commit` and `core_commit` in
   `sync/nns_source.json` to the R and NNS-core commits this build corresponds
   to. The provenance gate **fails a tagged release** while these are `unknown`.
3. Dry run end to end against TestPyPI:
   * Actions → **Release** → *Run workflow* → `publish: testpypi`.
   * Verify the artifacts and `pip install -i https://test.pypi.org/simple/ ovvo-nns`.
4. Tag and push to publish to PyPI:
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```
   The tag triggers `release.yml`, which gates on provenance, builds all
   wheels + sdist, and publishes to PyPI.

`workflow_dispatch` with `publish: none` builds and uploads the artifacts to the
Actions run without publishing — useful for inspecting wheels.

## Provenance gate

`scripts/check_release_provenance.py` enforces, for a tagged release:

* the tag matches the `pyproject.toml` version (`v<version>`), and
* `sync/nns_source.json` records non-placeholder `r_commit`, `core_commit`, and
  `r_version`.

Dry runs pass `--allow-unknown` so placeholder provenance does not block testing.
