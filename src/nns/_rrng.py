"""Faithful reproduction of R's default random-number generator.

NNS.stack and NNS.boost draw their cross-validation splits and stochastic
sampling from R's Mersenne-Twister stream after ``set.seed(seed)``. To match R
bit-for-bit the port reproduces R's ``set.seed``, ``unif_rand`` (Mersenne
Twister), ``R_unif_index`` (the R >= 3.6 "Rejection" sampler), ``runif``, and
``sample``/``sample.int`` (with and without replacement). Validated against live
R for ``sample.int``, permutations, and ``runif`` sequences.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

_N = 624
_MATRIX_A = 0x9908B0DF
_UPPER = 0x80000000
_LOWER = 0x7FFFFFFF
_I2_32M1 = 2.3283064365386963e-10


class RRNG:
    """R's default Mersenne-Twister RNG seeded through ``set.seed``."""

    def __init__(self, seed: int) -> None:
        self._mt = [0] * _N
        self._mti = _N + 1
        self.set_seed(seed)

    def set_seed(self, seed: int) -> None:
        s = int(seed) & 0xFFFFFFFF
        # Initial scrambling, then fill the 625-entry seed vector, exactly as
        # R's RNG_Init does for the Mersenne-Twister.
        for _ in range(50):
            s = (69069 * s + 1) & 0xFFFFFFFF
        i_seed = [0] * 625
        for j in range(625):
            s = (69069 * s + 1) & 0xFFFFFFFF
            i_seed[j] = s
        self._mt = i_seed[1:625]
        self._mti = _N  # FixupSeeds forces regeneration on first draw.

    def _genrand(self) -> int:
        mag01 = (0, _MATRIX_A)
        mt = self._mt
        if self._mti >= _N:
            for kk in range(_N - 397):
                y = (mt[kk] & _UPPER) | (mt[kk + 1] & _LOWER)
                mt[kk] = mt[kk + 397] ^ (y >> 1) ^ mag01[y & 1]
            for kk in range(_N - 397, _N - 1):
                y = (mt[kk] & _UPPER) | (mt[kk + 1] & _LOWER)
                mt[kk] = mt[kk + (397 - _N)] ^ (y >> 1) ^ mag01[y & 1]
            y = (mt[_N - 1] & _UPPER) | (mt[0] & _LOWER)
            mt[_N - 1] = mt[396] ^ (y >> 1) ^ mag01[y & 1]
            self._mti = 0
        y = mt[self._mti]
        self._mti += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y & 0xFFFFFFFF

    def unif_rand(self) -> float:
        v = self._genrand() * _I2_32M1
        # R's fixup keeps the draw strictly inside (0, 1).
        if v <= 0.0:
            return 0.5 * _I2_32M1
        if 1.0 - v <= 0.0:
            return 1.0 - 0.5 * _I2_32M1
        return v

    def runif(self, low: float = 0.0, high: float = 1.0) -> float:
        return low + (high - low) * self.unif_rand()

    def _rbits(self, bits: int) -> float:
        v = 0
        n = 0
        while n <= bits:
            v1 = int(math.floor(self.unif_rand() * 65536))
            v = 65536 * v + v1
            n += 16
        return float(v & ((1 << bits) - 1))

    def unif_index(self, dn: float) -> float:
        """R's R_unif_index with the default "Rejection" sampler (R >= 3.6)."""
        if dn <= 0:
            return 0.0
        bits = int(math.ceil(math.log2(dn)))
        while True:
            dv = self._rbits(bits)
            if dn > dv:
                return dv

    def sample_int(self, n: int, size: int | None = None, replace: bool = False) -> NDArray[np.int64]:
        """R's ``sample.int(n, size, replace)`` returning 1-based draws."""
        if size is None:
            size = n
        if replace:
            return np.array(
                [int(self.unif_index(n)) + 1 for _ in range(size)], dtype=np.int64
            )
        x = list(range(n))
        nn = n
        out = np.empty(size, dtype=np.int64)
        for i in range(size):
            j = int(self.unif_index(nn))
            out[i] = x[j] + 1
            x[j] = x[nn - 1]
            nn -= 1
        return out

    def sample(
        self,
        values: NDArray[np.int64],
        size: int | None = None,
        replace: bool = False,
    ) -> NDArray[np.int64]:
        """R's ``sample(x, size, replace)`` for an explicit vector ``x``."""
        values = np.asarray(values)
        idx = self.sample_int(values.size, size, replace)
        return values[idx - 1]
