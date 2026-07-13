from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

EXPECTED_REPOSITORY = "OVVO-Financial/NNS"
EXPECTED_SCHEMA = "repaired_r_13_1_54c98418"
EXPECTED_REFERENCE_OPTIONS = {
    "NNS.native.stack": False,
    "NNS.native.mreg": False,
    "NNS.native.univariate": False,
}
REQUIRED_FAMILIES = {"part", "reg", "mreg", "stack", "boost", "var"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_metadata(source: Path) -> dict[str, object]:
    metadata_path = source / "metadata.json"
    if not metadata_path.exists():
        raise SystemExit(f"missing fixture metadata: {metadata_path}")
    return json.loads(metadata_path.read_text())


def verify_metadata(metadata: dict[str, object], expected_commit: str) -> None:
    if metadata.get("r_repository") != EXPECTED_REPOSITORY:
        raise SystemExit(f"unexpected R repository: {metadata.get('r_repository')!r}")
    if metadata.get("r_commit_sha") != expected_commit:
        raise SystemExit(
            f"unexpected R commit: {metadata.get('r_commit_sha')!r}; expected {expected_commit}"
        )
    if metadata.get("fixture_schema_version") != EXPECTED_SCHEMA:
        raise SystemExit(f"unexpected fixture schema: {metadata.get('fixture_schema_version')!r}")
    options = metadata.get("native_reference_options")
    if options != EXPECTED_REFERENCE_OPTIONS:
        raise SystemExit(f"unexpected native/reference options: {options!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Import verified repaired R fixture artifacts.")
    parser.add_argument("artifact_dir", nargs="?", type=Path)
    parser.add_argument("--artifact", dest="artifact", type=Path)
    parser.add_argument("--expected-commit", dest="expected_commit")
    parser.add_argument("--expected-r-sha", dest="expected_commit")
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("tests/parity/fixtures/repaired_r_13_1_54c98418"),
    )
    args = parser.parse_args()
    artifact_dir = args.artifact or args.artifact_dir
    if artifact_dir is None:
        raise SystemExit("provide an artifact directory as a positional argument or --artifact")
    if args.expected_commit is None:
        raise SystemExit("--expected-r-sha is required")
    valid_commit = len(args.expected_commit) == 40 and all(
        c in "0123456789abcdef" for c in args.expected_commit
    )
    if not valid_commit:
        raise SystemExit("--expected-r-sha must be a 40-character lowercase git SHA")
    metadata = load_metadata(artifact_dir)
    verify_metadata(metadata, args.expected_commit)
    fixtures_path = artifact_dir / "fixtures.json"
    if not fixtures_path.exists():
        raise SystemExit(f"missing fixtures file: {fixtures_path}")
    fixtures = json.loads(fixtures_path.read_text())
    cases = fixtures.get("cases", [])
    names = [case.get("name") for case in cases]
    if len(names) != len(set(names)):
        raise SystemExit("fixture case names must be unique")
    families = {str(case.get("kind")) for case in cases}
    missing_families = REQUIRED_FAMILIES - families
    if missing_families:
        raise SystemExit(
            "missing required fixture families: " + ", ".join(sorted(missing_families))
        )
    fixture_hashes = {path.name: sha256_file(path) for path in sorted(artifact_dir.glob("*.json"))}
    args.dest.mkdir(parents=True, exist_ok=True)
    for source in artifact_dir.iterdir():
        if source.is_file():
            shutil.copy2(source, args.dest / source.name)
    (args.dest / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "r_repository": EXPECTED_REPOSITORY,
                "r_commit": args.expected_commit,
                "nns_version": metadata.get("nns_version"),
                "reference_backend": {"stack": False, "mreg": False, "univariate": False},
                "fixture_hashes": fixture_hashes,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"Imported repaired R fixtures into {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
