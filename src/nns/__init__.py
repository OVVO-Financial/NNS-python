from __future__ import annotations

from typing import Any

from nns.pm_matrix import pm_matrix as pm_matrix

__version__ = "1.5.0"

_EXPORTS = {
    "BoostResult": ("nns.boost", "BoostResult"),
    "FactorDesign": ("nns.regression", "FactorDesign"),
    "MRegResult": ("nns.multivariate_regression", "MRegResult"),
    "MebootResult": ("nns.meboot", "MebootResult"),
    "PartResult": ("nns.part", "PartResult"),
    "Reg