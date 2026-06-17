# Release notes

The authoritative, per-version release history is published to:

- **PyPI release history:** <https://pypi.org/project/ovvo-nns/#history>
- **GitHub Releases:** <https://github.com/OVVO-Financial/NNS-python/releases>

The currently published version is shown by the badge on the
[home page](index.md) and by:

```python
import nns

print(nns.__version__)
```

## Versioning

`ovvo-nns` uses standard semantic-style version numbers (`MAJOR.MINOR.PATCH`).
The public API is **stable and parity-focused**: documented public behavior is
not expected to break across minor releases. Known partial, guarded, and
known-gap paths are tracked on the [API status](api_status.md) page, and
intentional divergences from R `NNS` are recorded in the
[conventions](conventions.md).

Version numbers are kept consistent across `pyproject.toml`, the package
`__version__`, and the README. This is enforced in CI by
`scripts/check_version_consistency.py`.

For how releases are built, signed, and published, see the
[release process](releasing.md).
