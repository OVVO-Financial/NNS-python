from __future__ import annotations

import numpy as np
import pytest

from nns import fsd, fsd_uni, sd_efficient_set, ssd, ssd_uni, tsd, tsd_uni

from ._original import expected, r_vector


def _sd_label(value: int, degree: str) -> str:
    return {1: f"X {degree} Y", -1: f"Y {degree} X", 0: f"NO {degree} EXISTS"}[value]


@pytest.mark.parity
def test_original_fsd_ssd_tsd_labels_match_r_fixtures() -> None:
    x = r_vector("test_FSD_SSD_TSD.R", "x")
    y = r_vector("test_FSD_SSD_TSD.R", "y")
    y_squared = y**2
    exp = expected("test_FSD_SSD_TSD.R")

    assert _sd_label(fsd(x, y), "FSD") == exp["fsd_xy"]
    assert _sd_label(fsd(x, y_squared), "FSD") == exp["fsd_x_y_squared"]
    assert _sd_label(fsd(y_squared, x), "FSD") == exp["fsd_y_squared_x"]
    assert _sd_label(ssd(x, y), "SSD") == exp["ssd_xy"]
    assert _sd_label(ssd(x, y_squared), "SSD") == exp["ssd_x_y_squared"]
    assert _sd_label(ssd(y_squared, x), "SSD") == exp["ssd_y_squared_x"]
    assert _sd_label(tsd(x, y), "TSD") == exp["tsd_xy"]
    assert _sd_label(tsd(x, y_squared), "TSD") == exp["tsd_x_y_squared"]
    assert _sd_label(tsd(y_squared, x), "TSD") == exp["tsd_y_squared_x"]


@pytest.mark.parity
def test_original_unidirectional_sd_routines_match_r_fixtures() -> None:
    x = r_vector("test_Uni_SD_Routines.R", "x")
    y = r_vector("test_Uni_SD_Routines.R", "y")
    y_squared = y**2
    exp = expected("test_Uni_SD_Routines.R")

    assert fsd_uni(x, y, "discrete") == exp["fsd_xy_discrete"]
    assert fsd_uni(x, y_squared, "discrete") == exp["fsd_x_y_squared_discrete"]
    assert fsd_uni(x, y_squared, "continuous") == exp["fsd_x_y_squared_continuous"]
    assert ssd_uni(x, y) == exp["ssd_xy"]
    assert ssd_uni(x, y_squared) == exp["ssd_x_y_squared"]
    assert tsd_uni(x, y) == exp["tsd_xy"]
    assert tsd_uni(x, y_squared) == exp["tsd_x_y_squared"]


@pytest.mark.parity
def test_original_sd_efficient_set_preserves_r_names_and_order() -> None:
    x = r_vector("test_SD_efficient_Set.R", "x")
    y = r_vector("test_SD_efficient_Set.R", "y")
    z = r_vector("test_SD_efficient_Set.R", "z")
    names = ["x", "y", "z", "xx", "yy", "zz"]
    values = np.column_stack((x, y, z, x + 10, y + 10, z + 10))
    exp = expected("test_SD_efficient_Set.R")

    def selected(degree: int, type_value: str = "discrete") -> list[str]:
        return [names[index] for index in sd_efficient_set(values, degree, type=type_value)]

    assert selected(1, "discrete") == exp["degree_1_discrete"]
    assert selected(1, "continuous") == exp["degree_1_continuous"]
    assert selected(2) == exp["degree_2"]
    assert selected(3) == exp["degree_3"]
