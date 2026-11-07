from __future__ import annotations

import numpy as np
import pytest

from nns._rrng import RRNG


@pytest.mark.parametrize(
    ("seed", "n", "size"),
    [(1, 30, 3000), (123, 1, 50), (7, 2, 10), (9, 65536, 5000), (5, 65537, 7000), (2, 997, 50000)],
)
def test_vectorised_sample_int_replace_matches_scalar_stream(seed: int, n: int, size: int) -> None:
    fast = RRNG(seed)
    slow = RRNG(seed)

    np.testing.assert_array_equal(
        fast.sample_int_replace(n, size),
        slow.sample_int(n, size, replace=True),
    )
    # The generator state afterwards is identical too, so later draws agree.
    assert [fast.unif_rand() for _ in range(5)] == [slow.unif_rand() for _ in range(5)]


def test_vectorised_sample_int_replace_matches_r() -> None:
    # R: set.seed(42); sample.int(10, 12, replace = TRUE)
    assert RRNG(42).sample_int_replace(10, 12).tolist() == [1, 5, 1, 9, 10, 4, 2, 10, 1, 8, 7, 4]
