from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

import numpy as np
import pytest
from _r import nns
from _tolerances import EXACT

from pynns import pm_matrix

PMTarget: TypeAlias = float | Literal["mean"] | np.ndarray | None


@dataclass(frozen=True)
class PMMatrixCase:
    n_variables: int
    t_obs: int
    lpm_degree: int
    upm_degree: int
    target_kind: str
    pop_adj: bool


def _pm_matrix_cases() -> Iterator[PMMatrixCase]:
    for n_variables in (2, 3, 7):
        for t_obs in (50, 200, 1000):
            for lpm_degree in (1, 2, 3):
                for upm_degree in (1, 2, 3):
                    for target_kind in ("zero", "mean", "vector"):
                        for pop_adj in (False, True):
                            yield PMMatrixCase(
                                n_variables,
                                t_obs,
                                lpm_degree,
                                upm_degree,
                                target_kind,
                                pop_adj,
                            )


@pytest.mark.parity
@pytest.mark.parametrize("case", list(_pm_matrix_cases()))
def test_pm_matrix_matches_r(case: PMMatrixCase) -> None:
    variable = _variable(case.t_obs, case.n_variables)
    target, r_target = _target(case.target_kind, variable)

    actual = pm_matrix(case.lpm_degree, case.upm_degree, target, variable, case.pop_adj)
    expected = cast(
        dict[str, np.ndarray],
        nns(
            "PM.matrix",
            case.lpm_degree,
            case.upm_degree,
            r_target,
            variable.tolist(),
            case.pop_adj,
        ),
    )

    assert actual.keys() == expected.keys()
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], atol=EXACT)


def _variable(t_obs: int, n_variables: int) -> np.ndarray:
    row = np.arange(t_obs, dtype=np.float64)[:, np.newaxis]
    col = np.arange(n_variables, dtype=np.float64)[np.newaxis, :]
    return np.sin((row + 1.0) * (col + 1.0) / 11.0) + np.cos((row + 2.0) / (col + 3.0))


def _target(target_kind: str, variable: np.ndarray) -> tuple[PMTarget, object]:
    if target_kind == "zero":
        return 0.0, [0.0] * variable.shape[1]
    if target_kind == "mean":
        return "mean", None
    return np.linspace(-0.25, 0.25, variable.shape[1]), np.linspace(
        -0.25,
        0.25,
        variable.shape[1],
    ).tolist()
