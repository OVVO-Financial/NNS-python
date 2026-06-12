from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / "original_tests" / "testthat"
EXPECTED_PATH = ROOT / "tests" / "fixtures" / "original_tests_expected.json"


def expected(file_name: str) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(EXPECTED_PATH.read_text())[file_name])


def r_vector(file_name: str, name: str) -> NDArray[np.float64]:
    text = (ORIGINAL / file_name).read_text()
    match = re.search(rf"^{re.escape(name)}\s*<-\s*c\((.*?)\)", text, re.M | re.S)
    if match is None:
        raise AssertionError(f"Could not find vector {name!r} in {file_name}.")
    return np.fromstring(match.group(1), sep=",")


def r_string_vector_assignment(file_name: str, assignment: str) -> NDArray[np.str_]:
    text = (ORIGINAL / file_name).read_text()
    match = re.search(rf"{re.escape(assignment)}\s*<-\s*c\((.*?)\)", text, re.S)
    if match is None:
        raise AssertionError(f"Could not find string vector assignment {assignment!r}.")
    return np.asarray(re.findall(r'"([^"]+)"', match.group(1)), dtype=str)
