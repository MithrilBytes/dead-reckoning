# SPDX-License-Identifier: Apache-2.0
"""Properties a hybrid logical clock must hold whatever the wall clock does."""

from __future__ import annotations

import itertools

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from deadreckoning.clock import HLC, MAX_LOGICAL, HybridLogicalClock

wall_readings = st.lists(st.integers(min_value=0, max_value=2**41), min_size=1, max_size=60)


def _clock(readings: list[int], node_id: str = "n") -> HybridLogicalClock:
    state = {"i": 0}

    def now() -> int:
        value = readings[min(state["i"], len(readings) - 1)]
        state["i"] += 1
        return value

    return HybridLogicalClock(node_id, now)


@given(wall_readings)
def test_stamps_strictly_increase_however_the_wall_clock_moves(readings: list[int]) -> None:
    clock = _clock(readings)
    stamps = [clock.now() for _ in readings]
    assert all(a < b for a, b in itertools.pairwise(stamps))


@given(wall_readings)
def test_the_clock_never_regresses_across_a_restart(readings: list[int]) -> None:
    clock = _clock(readings)
    before = [clock.now() for _ in readings][-1]
    # A restart restores the persisted stamp and hands it back, even if the
    # machine came up believing it is 1970.
    resumed = HybridLogicalClock("n", lambda: 0, last=before)
    assert resumed.now() > before


@given(
    st.integers(min_value=0, max_value=2**41),
    st.integers(min_value=0, max_value=2**41),
    st.integers(min_value=0, max_value=MAX_LOGICAL),
)
def test_observing_a_peer_puts_us_ahead_of_both(
    local_now: int, remote_ms: int, remote_l: int
) -> None:
    clock = HybridLogicalClock("a", lambda: local_now)
    mine = clock.now()
    remote = HLC(physical_ms=remote_ms, logical=remote_l, node_id="b")
    merged = clock.observe(remote)
    assert merged > mine
    assert (merged.physical_ms, merged.logical) > (remote.physical_ms, remote.logical)


@given(st.lists(st.tuples(st.integers(0, 2**20), st.integers(0, 255)), min_size=1, max_size=20))
def test_merging_is_idempotent_for_a_stamp_already_seen(pairs: list[tuple[int, int]]) -> None:
    clock = HybridLogicalClock("a", lambda: 0)
    for physical, logical in pairs:
        clock.observe(HLC(physical_ms=physical, logical=logical, node_id="b"))
    settled = clock.last
    again = clock.observe(HLC(physical_ms=pairs[0][0], logical=pairs[0][1], node_id="b"))
    assert again > settled


def test_ordering_is_total_and_breaks_ties_on_node_id() -> None:
    assert HLC(5, 1, "a") < HLC(5, 1, "b")
    assert HLC(5, 1, "z") < HLC(5, 2, "a")
    assert HLC(4, 99, "z") < HLC(5, 0, "a")


@settings(max_examples=25)
@given(st.integers(min_value=1, max_value=200))
def test_a_burst_inside_one_millisecond_stays_inside_the_wire_format(count: int) -> None:
    clock = HybridLogicalClock("n", lambda: 1000, last=HLC(1000, MAX_LOGICAL - 1, "n"))
    for _ in range(count):
        stamp = clock.now()
        assert 0 <= stamp.logical <= MAX_LOGICAL


def test_logical_overflow_carries_into_the_physical_component() -> None:
    clock = HybridLogicalClock("n", lambda: 1000, last=HLC(1000, MAX_LOGICAL, "n"))
    stamp = clock.now()
    assert stamp.physical_ms == 1001
    assert stamp.logical == 0


def test_a_restored_clock_must_belong_to_this_node() -> None:
    with pytest.raises(ValueError, match="belongs to node"):
        HybridLogicalClock("truck-7", lambda: 0, last=HLC(1, 0, "truck-12"))
