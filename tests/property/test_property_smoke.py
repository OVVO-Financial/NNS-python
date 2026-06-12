import pytest
from hypothesis import given
from hypothesis import strategies as st


@pytest.mark.property
@given(st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=20))
def test_reversing_preserves_length(values: list[float]) -> None:
    assert len(values) == len(list(reversed(values)))
