"""R's reference-BLAS accumulation order for the small products NNS forms.

R evaluates ``A %*% v`` and ``crossprod(w, B)`` with the reference BLAS
``dgemv`` it ships (``src/extra/blas/blas.f``): every output element is
accumulated strictly left to right, one rounded multiply and one rounded add
per term, with no fused multiply-add. NumPy's ``@`` and ``np.dot`` call an
optimised BLAS whose blocked, vectorised partial sums can differ from that in
the last bit. NNS feeds several of these products into exact comparisons
(``findInterval`` on regression points built from the same index, tied class
probabilities), where that bit decides the result, so the port reproduces the
reference order.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def r_matvec(design: NDArray[np.float64], coef: NDArray[np.float64]) -> NDArray[np.float64]:
    """``design %*% coef``: reference ``dgemv('N')``, accumulated column by column."""
    design = np.asarray(design, dtype=np.float64)
    coef = np.asarray(coef, dtype=np.float64).reshape(-1)
    out = np.zeros(design.shape[0], dtype=np.float64)
    for j in np.flatnonzero(coef != 0.0):
        out = out + coef[j] * design[:, j]
    return out


def r_crossprod_vec(
    weights: NDArray[np.float64], values: NDArray[np.float64]
) -> NDArray[np.float64]:
    """``as.numeric(crossprod(weights, values))`` for a weight vector.

    R computes it as reference ``dgemv('T')``: one sequential dot product of
    ``weights`` with each column of ``values``.
    """
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if weights.size == 0:
        return np.zeros(values.shape[1], dtype=np.float64)
    products = values * weights[:, np.newaxis]
    return np.asarray(np.cumsum(products, axis=0)[-1], dtype=np.float64)
