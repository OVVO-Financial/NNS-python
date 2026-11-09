"""libstdc++'s ``std::sort`` and ``std::partial_sort``, comparison for comparison.

R's C++ kernels order candidates with ``std::sort`` / ``std::partial_sort``,
which are not stable: among equal keys the order is whatever libstdc++'s
introsort and heap select leave behind. Where that order feeds a result (rank
weights, a nearest-neighbour pick), the port replays the same algorithms on
the same comparisons (``key[a] < key[b]``) so ties land where R puts them.

Each function sorts ``order``, a list of indices into ``key``, in place.
"""

from __future__ import annotations

from collections.abc import Sequence

_S_THRESHOLD = 16


def _adjust_heap(
    order: list[int], key: Sequence[float], base: int, hole: int, length: int, value: int
) -> None:
    top = hole
    second = hole
    while second < (length - 1) // 2:
        second = 2 * (second + 1)
        if key[order[base + second]] < key[order[base + second - 1]]:
            second -= 1
        order[base + hole] = order[base + second]
        hole = second
    if (length & 1) == 0 and second == (length - 2) // 2:
        second = 2 * (second + 1)
        order[base + hole] = order[base + second - 1]
        hole = second - 1
    # __push_heap
    parent = (hole - 1) // 2
    while hole > top and key[order[base + parent]] < key[value]:
        order[base + hole] = order[base + parent]
        hole = parent
        parent = (hole - 1) // 2
    order[base + hole] = value


def _make_heap(order: list[int], key: Sequence[float], first: int, last: int) -> None:
    length = last - first
    if length < 2:
        return
    parent = (length - 2) // 2
    while True:
        _adjust_heap(order, key, first, parent, length, order[first + parent])
        if parent == 0:
            return
        parent -= 1


def _pop_heap(order: list[int], key: Sequence[float], first: int, last: int, result: int) -> None:
    value = order[result]
    order[result] = order[first]
    _adjust_heap(order, key, first, 0, last - first, value)


def heap_select(order: list[int], key: Sequence[float], middle: int) -> None:
    """``std::__heap_select(begin, begin + middle, end)``: the first ``middle``
    slots end up holding the selected (smallest) elements, as a heap."""
    _heap_select(order, key, 0, middle, len(order))


def _heap_select(
    order: list[int], key: Sequence[float], first: int, middle: int, last: int
) -> None:
    _make_heap(order, key, first, middle)
    for i in range(middle, last):
        if key[order[i]] < key[order[first]]:
            _pop_heap(order, key, first, middle, i)


def _sort_heap(order: list[int], key: Sequence[float], first: int, last: int) -> None:
    while last - first > 1:
        last -= 1
        _pop_heap(order, key, first, last, last)


def partial_sort(order: list[int], key: Sequence[float], middle: int) -> None:
    """``std::partial_sort(begin, begin + middle, end, comp)``."""
    _heap_select(order, key, 0, middle, len(order))
    _sort_heap(order, key, 0, middle)


def _unguarded_linear_insert(order: list[int], key: Sequence[float], last: int) -> None:
    value = order[last]
    previous = last - 1
    while key[value] < key[order[previous]]:
        order[last] = order[previous]
        last = previous
        previous -= 1
    order[last] = value


def _insertion_sort(order: list[int], key: Sequence[float], first: int, last: int) -> None:
    if first == last:
        return
    for i in range(first + 1, last):
        if key[order[i]] < key[order[first]]:
            value = order[i]
            order[first + 1 : i + 1] = order[first:i]
            order[first] = value
        else:
            _unguarded_linear_insert(order, key, i)


def _final_insertion_sort(order: list[int], key: Sequence[float], first: int, last: int) -> None:
    if last - first > _S_THRESHOLD:
        _insertion_sort(order, key, first, first + _S_THRESHOLD)
        for i in range(first + _S_THRESHOLD, last):
            _unguarded_linear_insert(order, key, i)
    else:
        _insertion_sort(order, key, first, last)


def _move_median_to_first(
    order: list[int], key: Sequence[float], result: int, a: int, b: int, c: int
) -> None:
    ka, kb, kc = key[order[a]], key[order[b]], key[order[c]]
    if ka < kb:
        if kb < kc:
            pick = b
        elif ka < kc:
            pick = c
        else:
            pick = a
    elif ka < kc:
        pick = a
    elif kb < kc:
        pick = c
    else:
        pick = b
    order[result], order[pick] = order[pick], order[result]


def _unguarded_partition(
    order: list[int], key: Sequence[float], first: int, last: int, pivot: int
) -> int:
    pivot_key = key[order[pivot]]
    while True:
        while key[order[first]] < pivot_key:
            first += 1
        last -= 1
        while pivot_key < key[order[last]]:
            last -= 1
        if not first < last:
            return first
        order[first], order[last] = order[last], order[first]
        first += 1


def _introsort_loop(
    order: list[int], key: Sequence[float], first: int, last: int, depth_limit: int
) -> None:
    while last - first > _S_THRESHOLD:
        if depth_limit == 0:
            _heap_select(order, key, first, last, last)
            _sort_heap(order, key, first, last)
            return
        depth_limit -= 1
        middle = first + (last - first) // 2
        _move_median_to_first(order, key, first, first + 1, middle, last - 1)
        cut = _unguarded_partition(order, key, first + 1, last, first)
        _introsort_loop(order, key, cut, last, depth_limit)
        last = cut


def std_sort(order: list[int], key: Sequence[float]) -> None:
    """``std::sort(begin, end, comp)`` (introsort, then a final insertion sort)."""
    n = len(order)
    if n:
        _introsort_loop(order, key, 0, n, 2 * (n.bit_length() - 1))
        _final_insertion_sort(order, key, 0, n)
