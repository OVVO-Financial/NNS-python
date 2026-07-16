# Release Checklist

Use this checklist for every `ovvo-nns` version bump and PyPI/TestPyPI release.

## 1. Update every version reference

Update all authoritative and user-facing version locations:

- `pyproject.toml`: `[project].version`
- `src/nns/__init__.py`: `__version__`
- `README.md`: current-version table entry
- `uv.lock`: regenerate with `uv lock`; do not hand-edit generated lockfile contents

Confirm they all agree:

```bash
grep -n 'version = ' pyproject.toml
grep -n '__version__' src/nns/__init__.py
grep -n 'Current version' README.md
grep -A3 'name = "ovvo-nns"' uv.lock
```

## 2. Preserve complete files when editing

Never replace a file with a partial snippet when using the GitHub Contents API or another full-file update tool.

Before writing:

1. Fetch the complete current file.
2. Change only the intended lines.
3. Write the complete file back.
4. Inspect the resulting diff for unintended deletions.

This is especially important for:

- `pyproject.toml`
- `src/nns/__init__.py`
- `README.md`
- generated workflow and lock files

## 3. Make README links PyPI-safe

The README is embedded into the wheel and source distribution as package metadata. PyPI and TestPyPI render it outside the GitHub repository context.

Therefore, links in `README.md` must not rely on repository-relative targets such as:

```markdown
[Forecasting example](examples/vignettes/09_forecasting.py)
```

Use absolute URLs instead:

```markdown
[Forecasting example](https://github.com/OVVO-Financial/NNS-python/blob/main/examples/vignettes/09_forecasting.py)
```

Before every release, audit all Markdown links in `README.md` and convert repository-file links to absolute GitHub URLs. This includes:

- example scripts
- vignette files
- manifests
- repository documentation files
- images not already using absolute URLs

External documentation URLs may remain absolute links to the published documentation site.

## 4. Remember that uploaded metadata is immutable

PyPI and TestPyPI do not update an existing release page when the repository README changes.

Once a distribution version is uploaded, its long description and links are fixed for that uploaded artifact.

If a TestPyPI upload contains broken README links:

- fix the README in the repository;
- bump to a new test version such as `1.5.0.post1` or the next patch version;
- rebuild from a clean `dist/` directory;
- upload the new version.

Do not expect the old TestPyPI page to change retroactively.

A production PyPI version cannot be replaced with different files under the same version number.

## 5. Regenerate the lockfile

After changing `pyproject.toml`:

```bash
uv lock
```

Verify the local package entry shows the new version:

```bash
grep -A3 'name = "ovvo-nns"' uv.lock
```

Commit the regenerated `uv.lock` with the version bump.

## 6. Run the full validation suite

Before building:

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
uv run mypy
```

Confirm all GitHub Actions jobs are green for the exact commit being released.

## 7. Build from a clean tree

```bash
git pull origin main
rm -rf dist build *.egg-info
python -m build
python -m twine check dist/*
```

Confirm the generated filenames contain the intended version.

Inspect the wheel metadata if needed:

```bash
python -m zipfile -l dist/ovvo_nns-<VERSION>-*.whl
```

## 8. TestPyPI verification

Upload a unique test version to TestPyPI when validating packaging or README rendering:

```bash
python -m twine upload --repository testpypi dist/*
```

On the TestPyPI project page, manually verify:

- package version
- README rendering
- every README link
- project URLs
- installation instructions
- code blocks and tables

Do not reuse a TestPyPI version that has already been uploaded.

## 9. Production upload

Only after CI, build checks, and TestPyPI validation pass:

```bash
python -m twine upload dist/*
```

Then verify installation from PyPI in a clean environment:

```bash
python -m pip install --upgrade ovvo-nns==<VERSION>
python -c "import nns; print(nns.__version__)"
```

## 10. Tag the exact released commit

After confirming the release contents:

```bash
git tag -a v<VERSION> -m "ovvo-nns <VERSION>"
git push origin v<VERSION>
```

The tag must point to the exact commit used to build the uploaded distributions.
