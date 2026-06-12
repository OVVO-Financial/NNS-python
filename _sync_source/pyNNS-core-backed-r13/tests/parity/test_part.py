from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pytest
from _r import nns
from _tolerances import EXACT

from pynns import nns_part
from pynns.part import NoiseReduction

SIZES = [50, 200, 1000]
RELATIONSHIPS = ["linear", "quadratic", "sin", "random"]
CASES: list[tuple[str | None, str, int | None, int, bool]] = [
    (None, "off", None, 8, True),
    ("XONLY", "off", None, 8, False),
    (None, "mean", 1, 3, True),
    ("XONLY", "median", 2, 3, False),
    (None, "mode", 3, 16, True),
    ("XONLY", "mode_class", 5, 8, False),
]


@pytest.mark.parity
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("relationship", RELATIONSHIPS)
@pytest.mark.parametrize(("part_type", "noise", "order", "obs_req", "min_obs_stop"), CASES)
def test_nns_part_matches_r(
    rng: np.random.Generator,
    size: int,
    relationship: str,
    part_type: str | None,
    noise: str,
    order: int | None,
    obs_req: int,
    min_obs_stop: bool,
) -> None:
    x, y = _relationship(relationship, size, rng)

    expected = nns(
        "NNS.part",
        x.tolist(),
        y.tolist(),
        False,
        part_type,
        order,
        obs_req,
        min_obs_stop,
        noise,
    )
    actual = nns_part(
        x,
        y,
        type=part_type,
        order=order,
        obs_req=obs_req,
        min_obs_stop=min_obs_stop,
        noise_reduction=cast(NoiseReduction, noise),
    )

    _assert_part_matches(actual, expected)


@pytest.mark.parity
def test_nns_part_installed_r_collapses_any_non_null_type_to_xonly() -> None:
    x = np.arange(1.0, 9.0)
    y = x[::-1]

    expected = nns("NNS.part", x.tolist(), y.tolist(), False, "Y", 2, 0, False, "off")
    actual = nns_part(x, y, type="Y", order=2, obs_req=0, min_obs_stop=False)

    _assert_part_matches(actual, expected)


def _assert_part_matches(actual: Mapping[str, object], expected: object) -> None:
    assert isinstance(expected, dict)
    assert actual["order"] == int(_array(expected["order"]).item())

    actual_dt = actual["dt"]
    expected_dt = expected["dt"]
    assert isinstance(actual_dt, dict)
    assert isinstance(expected_dt, dict)
    np.testing.assert_allclose(_float_column(actual_dt, "x"), _array(expected_dt["x"]), atol=EXACT)
    np.testing.assert_allclose(_float_column(actual_dt, "y"), _array(expected_dt["y"]), atol=EXACT)
    np.testing.assert_array_equal(
        _str_column(actual_dt, "quadrant"),
        _strings(expected_dt["quadrant"]),
    )
    np.testing.assert_array_equal(
        _str_column(actual_dt, "prior.quadrant"),
        _strings(expected_dt["prior.quadrant"]),
    )

    actual_rp = actual["regression.points"]
    expected_rp = expected["regression.points"]
    assert isinstance(actual_rp, dict)
    assert isinstance(expected_rp, dict)
    np.testing.assert_array_equal(
        _str_column(actual_rp, "quadrant"),
        _strings(expected_rp["quadrant"]),
    )
    np.testing.assert_allclose(_float_column(actual_rp, "x"), _array(expected_rp["x"]), atol=EXACT)
    np.testing.assert_allclose(_float_column(actual_rp, "y"), _array(expected_rp["y"]), atol=EXACT)


def _array(value: object) -> np.ndarray:
    assert isinstance(value, np.ndarray)
    return value.astype(np.float64)


def _strings(value: object) -> np.ndarray:
    if isinstance(value, list):
        return np.asarray(value, dtype=str)
    assert isinstance(value, str)
    return np.asarray([value], dtype=str)


def _float_column(values: dict[str, Any], key: str) -> np.ndarray:
    column = values[key]
    assert isinstance(column, np.ndarray)
    return column.astype(np.float64)


def _str_column(values: dict[str, Any], key: str) -> np.ndarray:
    column = values[key]
    assert isinstance(column, np.ndarray)
    return column.astype(str)


def _relationship(
    relationship: str,
    size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    x = rng.normal(size=size)
    if relationship == "linear":
        return x, 0.8 * x + 0.2 * rng.normal(size=size)
    if relationship == "quadratic":
        return x, x * x + 0.1 * rng.normal(size=size)
    if relationship == "sin":
        return x, np.sin(x) + 0.05 * rng.normal(size=size)
    return x, rng.normal(size=size)
