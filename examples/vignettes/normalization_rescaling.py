"""Vignette 04 — Normalization and rescaling.

Python translation of the R NNS "Normalization and Rescaling" vignette
(``tools/NNS/vignettes/NNSvignette_04_Normalization_and_Rescaling.Rmd``):
``nns_norm`` (linear and nonlinear) and ``nns_rescale`` (min-max and
risk-neutral).

Run with::

    python examples/vignettes/normalization_rescaling.py
"""

from __future__ import annotations

import numpy as np

from nns import nns_norm, nns_rescale


def main() -> None:
    rng = np.random.default_rng(123)
    a = rng.normal(0.0, 1.0, size=100)
    b = rng.normal(0.0, 5.0, size=100)
    c = rng.normal(10.0, 1.0, size=100)
    d = rng.normal(10.0, 10.0, size=100)
    data = np.column_stack((a, b, c, d))

    # Linear normalization aligns the columns onto a common mean scale.
    linear = nns_norm(data, linear=True)
    nonlinear = nns_norm(data, linear=False)
    assert linear.shape == data.shape
    assert nonlinear.shape == data.shape
    linear_means = linear.mean(axis=0)
    # All linear-normalized columns share a common mean.
    np.testing.assert_allclose(linear_means, linear_means[0], atol=1e-6)

    # Min-max rescale onto an explicit [a, b] interval.
    raw = np.array([-2.5, 0.2, 1.1, 3.7, 5.0])
    scaled = nns_rescale(raw, a=5.0, b=10.0, method="minmax")
    assert np.isclose(scaled.min(), 5.0)
    assert np.isclose(scaled.max(), 10.0)

    # Risk-neutral rescale: the rescaled mean matches the forward S0*exp(r*T).
    s0, r, t = 100.0, 0.03, 1.0
    prices = s0 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, size=250)))
    terminal = nns_rescale(
        prices, a=s0, b=r, method="riskneutral", time_to_maturity=t, type="Terminal"
    )
    assert np.isclose(float(terminal.mean()), s0 * np.exp(r * t))

    discounted = nns_rescale(
        prices, a=s0, b=r, method="riskneutral", time_to_maturity=t, type="Discounted"
    )
    assert np.isclose(float(discounted.mean()), s0)

    print("linear-normalized column means:", np.round(linear_means, 6))
    print("min-max range:", round(float(scaled.min()), 4), round(float(scaled.max()), 4))
    print("risk-neutral terminal mean vs forward:",
          round(float(terminal.mean()), 4), round(s0 * np.exp(r * t), 4))
    print("risk-neutral discounted mean vs S0:", round(float(discounted.mean()), 4), s0)


if __name__ == "__main__":
    main()
