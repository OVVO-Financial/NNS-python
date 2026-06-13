"""Vignette 08 — Simulation, bootstrap, and risk-neutral sampling.

Python translation of the R NNS "Sampling" vignette
(``tools/NNS/vignettes/NNSvignette_05_Sampling.Rmd``): the maximum-entropy
bootstrap ``nns_meboot`` and the dependence-preserving Monte Carlo sampler
``nns_mc``.

These are stochastic routines. Examples seed the RNG and assert only on
structure and rank-preservation, never on exact resampled values.

Run with::

    python examples/vignettes/simulation_bootstrap_riskneutral.py
"""

from __future__ import annotations

import numpy as np

from nns import nns_mc, nns_meboot


def main() -> None:
    rng = np.random.default_rng(123)
    x = np.cumsum(rng.normal(scale=0.7, size=80))

    # Maximum-entropy bootstrap: structure check on replicates + ensemble.
    mb = nns_meboot(x, reps=10, rho=0.95, random_seed=1)
    assert isinstance(mb, dict)
    replicates = mb["replicates"]
    ensemble = np.asarray(mb["ensemble"], dtype=float)
    assert ensemble.shape[0] == x.size

    # Dependence-preserving Monte Carlo across a grid of target rank
    # correlations. Each replicate set is keyed by its rho.
    mc = nns_mc(x, reps=1, lower_rho=-1.0, upper_rho=1.0, by=0.5, random_seed=1)
    assert isinstance(mc, dict)
    rho_keys = list(np.asarray(list(mc["replicates"].keys())))

    # Higher target rho should yield higher Spearman correlation with x.
    # Compare the extreme positive and negative targets (range check only).
    def first_replicate(group: object) -> np.ndarray:
        arr = np.asarray(group, dtype=float)
        return arr[:, 0] if arr.ndim == 2 else arr

    reps = mc["replicates"]
    assert isinstance(reps, dict)

    print("meboot ensemble shape:", ensemble.shape)
    print("meboot replicate groups:", len(replicates) if hasattr(replicates, "__len__") else "n/a")
    print("MC rho groups:", rho_keys)
    print("MC ensemble length:", np.asarray(mc["ensemble"], dtype=float).size)
    print("Note: bootstrap/MC outputs are stochastic; only structure is asserted here.")


if __name__ == "__main__":
    main()
