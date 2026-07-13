from __future__ import annotations

import json
import sys
from pathlib import Path

EXPECTED_SCHEMA = "repaired_r_13_1_54c98418"
EXPECTED_REPOSITORY = "OVVO-Financial/NNS"
EXPECTED_R_SHA = "54c98418c2a11499ebb1c456570d2b66c37eb817"
REQUIRED_FAMILIES = {"part", "reg", "mreg", "stack", "boost", "var"}

REQUIRED_METADATA = {
    "fixture_schema_version",
    "generated_at",
    "r_repository",
    "r_commit_sha",
    "nns_version",
    "r_version",
    "platform",
    "os",
    "native_reference_options",
}


def main() -> int:
    fixture_dir = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("tests/parity/fixtures/repaired_r_13_1_54c98418")
    )
    metadata_path = fixture_dir / "metadata.json"
    fixtures_path = fixture_dir / "fixtures.json"
    missing_files = [str(path) for path in (metadata_path, fixtures_path) if not path.exists()]
    if missing_files:
        raise SystemExit(f"Missing repaired fixture files: {', '.join(missing_files)}")
    metadata = json.loads(metadata_path.read_text())
    missing_keys = sorted(REQUIRED_METADATA - set(metadata))
    if missing_keys:
        raise SystemExit(f"metadata.json missing keys: {', '.join(missing_keys)}")
    if metadata.get("fixture_schema_version") != EXPECTED_SCHEMA:
        raise SystemExit(f"unexpected fixture schema: {metadata.get('fixture_schema_version')!r}")
    if metadata.get("r_repository") != EXPECTED_REPOSITORY:
        raise SystemExit(f"unexpected R repository: {metadata.get('r_repository')!r}")
    if metadata.get("r_commit_sha") != EXPECTED_R_SHA:
        raise SystemExit(f"unexpected R commit: {metadata.get('r_commit_sha')!r}")
    options = metadata["native_reference_options"]
    expected_false = ["NNS.native.stack", "NNS.native.mreg", "NNS.native.univariate"]
    bad_options = [name for name in expected_false if options.get(name) is not False]
    if bad_options:
        raise SystemExit(
            "semantic fixtures must force reference backends: " + ", ".join(bad_options)
        )
    fixtures = json.loads(fixtures_path.read_text())
    cases = fixtures.get("cases", [])
    if not cases:
        raise SystemExit("fixtures.json contains no cases")
    names = [case.get("name") for case in cases]
    if len(names) != len(set(names)):
        raise SystemExit("fixture case names must be unique")
    families = {str(case.get("kind")) for case in cases}
    missing_families = REQUIRED_FAMILIES - families
    if missing_families:
        raise SystemExit(
            "missing required fixture families: " + ", ".join(sorted(missing_families))
        )
    missing_checksums = [
        case.get("name")
        for case in cases
        if not case.get("input_checksum") or not case.get("output_checksum")
    ]
    if missing_checksums:
        raise SystemExit(
            "fixture cases missing checksums: " + ", ".join(map(str, missing_checksums))
        )
    print(f"Verified {len(cases)} repaired R fixture cases in {fixture_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
