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


class RSingularError(ValueError):
    """R's ``solve()``: "Lapack routine dgesv: system is exactly singular"."""


def _dgetrf2(a: NDArray[np.float64], r0: int, c0: int, m: int, n: int, ipiv: list[int]) -> None:
    """Reference LAPACK ``DGETRF2`` (recursive LU, partial pivoting) on a block.

    ``ipiv`` holds global 0-based pivot rows. The column case scales the
    multipliers by the pivot's reciprocal (``DSCAL``), as LAPACK does.
    """
    if m == 1:
        ipiv[r0] = r0
        if a[r0, c0] == 0.0:
            raise RSingularError("Lapack routine dgesv: system is exactly singular")
        return
    if n == 1:
        column = np.abs(a[r0 : r0 + m, c0])
        i = int(np.argmax(column))  # IDAMAX: first maximal |value|
        ipiv[r0] = r0 + i
        if a[r0 + i, c0] == 0.0:
            raise RSingularError("Lapack routine dgesv: system is exactly singular")
        if i != 0:
            a[r0, c0], a[r0 + i, c0] = a[r0 + i, c0], a[r0, c0]
        pivot = a[r0, c0]
        if abs(pivot) >= np.finfo(np.float64).tiny:
            reciprocal = 1.0 / pivot
            for k in range(1, m):
                a[r0 + k, c0] = reciprocal * a[r0 + k, c0]
        else:
            for k in range(1, m):
                a[r0 + k, c0] = a[r0 + k, c0] / pivot
        return
    n1 = min(m, n) // 2
    n2 = n - n1
    _dgetrf2(a, r0, c0, m, n1, ipiv)
    right = slice(c0 + n1, c0 + n)
    for k in range(n1):  # DLASWP on [A12; A22]
        p = ipiv[r0 + k]
        if p != r0 + k:
            a[[r0 + k, p], right] = a[[p, r0 + k], right]
    for j in range(c0 + n1, c0 + n):  # DTRSM L L N U: A12 <- L11^-1 A12
        for k in range(n1):
            if a[r0 + k, j] != 0.0:
                for i in range(k + 1, n1):
                    a[r0 + i, j] = a[r0 + i, j] - a[r0 + k, j] * a[r0 + i, c0 + k]
    for j in range(c0 + n1, c0 + n):  # DGEMM: A22 <- A22 - A21 A12
        for col in range(n1):
            temp = -a[r0 + col, j]
            for i in range(r0 + n1, r0 + m):
                a[i, j] = a[i, j] + temp * a[i, c0 + col]
    _dgetrf2(a, r0 + n1, c0 + n1, m - n1, n2, ipiv)
    left = slice(c0, c0 + n1)
    for k in range(n1, min(m, n)):  # DLASWP on A21
        p = ipiv[r0 + k]
        if p != r0 + k:
            a[[r0 + k, p], left] = a[[p, r0 + k], left]


def r_solve(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """R ``solve(a, b)`` for a small square system and one right-hand side.

    Reference LAPACK ``DGESV`` as R calls it: ``DGETRF`` (which uses the
    recursive ``DGETRF2`` below its block size) and ``DGETRS``. R's
    reciprocal-condition check is not reproduced; an exactly singular
    system raises :class:`RSingularError` as R's does.
    """
    lu = np.array(a, dtype=np.float64, copy=True)
    x = np.array(b, dtype=np.float64, copy=True).reshape(-1)
    n = lu.shape[0]
    if lu.shape != (n, n) or x.size != n:
        raise ValueError("solve() needs a square matrix and a matching right-hand side.")
    ipiv = list(range(n))
    _dgetrf2(lu, 0, 0, n, n, ipiv)
    for k in range(n):  # DLASWP on b
        p = ipiv[k]
        if p != k:
            x[k], x[p] = x[p], x[k]
    for k in range(n):  # DTRSM L L N U
        if x[k] != 0.0:
            for i in range(k + 1, n):
                x[i] = x[i] - x[k] * lu[i, k]
    for k in range(n - 1, -1, -1):  # DTRSM L U N N
        if x[k] != 0.0:
            x[k] = x[k] / lu[k, k]
            for i in range(k):
                x[i] = x[i] - x[k] * lu[i, k]
    return x
