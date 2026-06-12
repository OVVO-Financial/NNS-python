from __future__ import annotations

import numpy as np

from pynns import nns_arma, nns_arma_optim, nns_seas, nns_var


def main() -> None:
    t = np.arange(1, 60, dtype=np.float64)
    series = 10.0 + np.sin(t / 3.0) + 0.05 * t

    seasonality = nns_seas(series, modulo=[3, 4, 6], mod_only=True)
    arma = nns_arma(series, h=3, seasonal_factor=4, method="lin")
    arma_both = nns_arma(series, h=3, seasonal_factor=4, method="both")
    optim = nns_arma_optim(
        series,
        h=3,
        seasonal_factor=[3, 4, 5],
        lin_only=True,
        print_trace=False,
    )

    panel = np.column_stack(
        (
            series,
            0.8 * series + np.cos(t / 5.0),
            4.0 + 0.03 * t + np.sin(t / 4.0),
        )
    )
    var = nns_var(panel, h=2, tau=[1, 2, 3], dim_red_method="cor", naive_weights=False)
    interpolated = nns_var(panel, h=0, tau=2)

    assert seasonality["periods"].ndim == 1
    assert seasonality["best.period"] in set(seasonality["periods"])
    assert arma.shape == (3,)
    assert arma_both.shape == (3,)
    assert optim["results"].shape == (3,)
    assert var["ensemble"].shape == (2, panel.shape[1])
    assert interpolated["interpolated_and_extrapolated"].shape == panel.shape

    print("best seasonal period:", seasonality["best.period"])
    print("candidate seasonal periods:", seasonality["periods"])
    print("ARMA forecast:", arma)
    print("ARMA both-method forecast:", arma_both)
    print("optimized ARMA forecast:", optim["results"])
    print("VAR ensemble forecast:")
    print(var["ensemble"])
    print("interpolated panel head:")
    print(interpolated["interpolated_and_extrapolated"][:3])


if __name__ == "__main__":
    main()
