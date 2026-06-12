from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pynns.core import _as_1d_values, lpm, upm


def mean_pm(x: NDArray[np.float64]) -> float:
    """mean(x) = UPM(1, 0, x) - LPM(1, 0, x)."""
    values = _as_1d_values(x)
    return float(upm(1, 0, values) - lpm(1, 0, values))


def var_pm(x: NDArray[np.float64], ddof: int = 0) -> float:
    """var(x) = UPM(2, mu, x) + LPM(2, mu, x), with optional ddof scaling."""
    values = _as_1d_values(x)
    if ddof < 0 or ddof >= values.size:
        raise ValueError("ddof must satisfy 0 <= ddof < len(x).")

    mean = float(np.mean(values))
    variance = float(upm(2, mean, values) + lpm(2, mean, values))
    if ddof == 0:
        return variance
    return float(np.var(values, ddof=ddof))


def skew_pm(x: NDArray[np.float64]) -> float:
    """Skew via degree-3 partial moments around mean, normalized by var^1.5."""
    values = _as_1d_values(x)
    mean = float(np.mean(values))
    variance = var_pm(values)
    skew_base = float(upm(3, mean, values) - lpm(3, mean, values))
    return float(skew_base / variance**1.5)


def kurt_pm(x: NDArray[np.float64], excess: bool = True) -> float:
    """Kurt via degree-4 partial moments around mean, normalized by var^2."""
    values = _as_1d_values(x)
    mean = float(np.mean(values))
    variance = var_pm(values)
    kurtosis = float(upm(4, mean, values) + lpm(4, mean, values)) / variance**2
    if excess:
        return kurtosis - 3.0
    return kurtosis


def nns_moments(x: NDArray[np.float64], population: bool = True) -> dict[str, float]:
    """Return R NNS.moments' first four partial-moment moments."""
    values = _as_1d_values(x)
    n = values.size
    center = float(np.mean(values))
    mean = float(upm(1, 0.0, values) - lpm(1, 0.0, values))
    variance = float(upm(2, center, values) + lpm(2, center, values))
    skew_base = float(upm(3, center, values) - lpm(3, center, values))
    kurt_base = float(upm(4, center, values) + lpm(4, center, values))

    if population:
        skewness = float(skew_base / variance**1.5)
        kurtosis = float(kurt_base / variance**2 - 3.0)
    else:
        skewness = float((n / ((n - 1) * (n - 2))) * ((n * skew_base) / variance**1.5))
        kurtosis = float(
            ((n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3)))
            * ((n * kurt_base) / (variance * (n / (n - 1))) ** 2)
            - ((3 * ((n - 1) ** 2)) / ((n - 2) * (n - 3)))
        )
        variance = float(variance * (n / (n - 1)))

    return {
        "mean": mean,
        "variance": variance,
        "skewness": skewness,
        "kurtosis": kurtosis,
    }


def ecdf_pm(
    x: NDArray[np.float64],
    points: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Empirical CDF computed as lpm(0, points, x). If points is None, use sorted x."""
    values = _as_1d_values(x)
    targets = np.sort(values) if points is None else np.asarray(points, dtype=np.float64)
    if targets.ndim != 1:
        raise ValueError("points must be 1D.")
    return np.asarray(lpm(0, targets, values), dtype=np.float64)
