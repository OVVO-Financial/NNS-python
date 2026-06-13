"""Vignette 07 — Time series forecasting.

Python translation of the R NNS "Forecasting" vignette
(``tools/NNS/vignettes/NNSvignette_09_Forecasting.Rmd``): ``nns_seas``,
``nns_arma``, ``nns_arma_optim``, and ``nns_var``.

The nonseasonal nonlinear ARMA forecast on the AirPassengers-style series is
the deterministic value verified against live R NNS 13.0 (PR #3):
``[128.5, 113.5, 155.5, 213.6667]``.

Run with::

    python examples/vignettes/time_series_forecasting.py
"""

from __future__ import annotations

import numpy as np

from nns import nns_arma, nns_arma_optim, nns_seas, nns_var


def main() -> None:
    # AirPassengers-style 24-point monthly series (two years).
    series = np.array(
        [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
         115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140],
        dtype=float,
    )

    # Deterministic forecasts (match live R NNS 13.0).
    nonseasonal = nns_arma(series, h=4, seasonal_factor=False, method="nonlin")
    seasonal = nns_arma(series, h=6, seasonal_factor=12, method="lin")
    np.testing.assert_allclose(
        nonseasonal, [128.5, 113.5, 155.5, 213.66666666666666], atol=1e-9
    )
    np.testing.assert_allclose(seasonal, [118.0, 134.0, 150.0, 141.0, 129.0, 163.0], atol=1e-9)

    # Seasonality detection on a clean deterministic sine series.
    z = np.sin(np.arange(1, 121) / 8.0)
    seas = nns_seas(z, plot=False)
    assert set(seas) == {"all.periods", "best.period", "periods"}

    # Deterministic NNS.ARMA.optim: validates a set of candidate seasonal
    # factors and returns the selected periods/method plus prediction bands.
    optim = nns_arma_optim(z, h=12, seasonal_factor=[10, 20, 30], plot=False, print_trace=False)
    for key in ("periods", "obj.fn", "method", "results", "lower.pred.int", "upper.pred.int"):
        assert key in optim
    results = np.asarray(optim["results"], dtype=float)
    lower = np.asarray(optim["lower.pred.int"], dtype=float)
    upper = np.asarray(optim["upper.pred.int"], dtype=float)
    assert results.shape == (12,)
    assert np.all(lower <= upper)

    # Multivariate forecasting with NNS.VAR on a small 3-series panel.
    t = np.arange(1, 61)
    panel = np.column_stack((np.sin(t / 6.0), np.cos(t / 5.0), np.sin(t / 4.0) + 0.5))
    var = nns_var(panel, h=4, tau=3, ncores=1, status=False)
    assert np.asarray(var["ensemble"]).shape == (4, 3)

    print("nonseasonal nonlinear ARMA:", np.round(nonseasonal, 4))
    print("seasonal linear ARMA:", np.round(seasonal, 4))
    print("detected periods:", np.asarray(optim["periods"]))
    print("optim method:", optim["method"], " obj.fn:", round(float(optim["obj.fn"]), 6))
    print("optim results[:4]:", np.round(results[:4], 4))
    print("VAR ensemble shape:", np.asarray(var["ensemble"]).shape)


if __name__ == "__main__":
    main()
