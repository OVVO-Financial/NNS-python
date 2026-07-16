from __future__ import annotations

import pytest


_OBSOLETE_BOOST_SEED_INVARIANCE_TEST = (
    "tests/parity/test_boost.py::test_nns_boost_ivs_test_none_is_seed_invariant"
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Retire the pre-Stack cross-seed invariance contract for NNS.boost.

    NNS.boost now delegates final n.best selection to NNS.stack. Different
    seeds legitimately generate