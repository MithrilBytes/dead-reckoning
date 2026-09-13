# SPDX-License-Identifier: Apache-2.0
"""Merging logs: lossless, order independent, and refusing what does not join up."""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from deadreckoning.canonical import GENESIS_PREV_HASH
from deadreckoning.clock import TimeTrust
from deadreckoning.node import Node
from deadreckoning.records import Mode, Record, RecordKind
from deadreckoning.sync.merge import (
    ChainBreakError,
    merge,
    sync_vector,
    unsynced,
    verify_continuity,
)


def chain_for(node_id: str, count: int, start: int = 100) -> list[Record]:
    """A short, genuinely linked chain, hashed the way the store hashes one."""
    records: list[Record] = []
    prev = GENESIS_PREV_HASH
    for i in range(count):
        record = Record(
            id=f"{node_id}-{i}",
            node_id=node_id,
            hlc={"physical_ms": start + i, "logical": 0, "node_id": node_id},
            wall_time="2026-01-01T00:00:00+00:00",
            time_trust=TimeTrust.UNTRUSTED,
            kind=RecordKind.CHECKPOINT,
            mode=Mode.ISLANDED,
            prev_hash=prev,
            body={"phase": "START"},
        )
        record = record.model_copy(update={"hash": record.compute_hash()})
        prev = record.hash
        records.append(record)
    return records


def test_merging_is_union_and_loses_nothing() -> None:
    mine, theirs = chain_for("truck-7", 3), chain_for("truck-12", 2)
    result = merge(mine, theirs)
    assert result.count == 2
    assert {r.id for r in mine + result.accepted} == {r.id for r in mine + theirs}


def test_merging_the_same_batch_twice_changes_nothing() -> None:
    mine, theirs = chain_for("truck-7", 2), chain_for("truck-12", 2)
    first = merge(mine, theirs)
    second = merge(mine + first.accepted, theirs)
    assert second.accepted == []
    assert sorted(second.already_held) == sorted(r.id for r in theirs)


def test_order_does_not_change_the_result() -> None:
    """Merging two peers either way round leaves the same set held."""
    mine = chain_for("truck-7", 2)
    a, b = chain_for("truck-12", 2), chain_for("hq", 2)

    held = list(mine)
    for batch in (a, b):
        held += merge(held, batch).accepted
    forwards = {r.id for r in held}

    held = list(mine)
    for batch in (b, a):
        held += merge(held, batch).accepted
    assert forwards == {r.id for r in held}


def test_a_batch_that_does_not_join_up_is_refused_whole() -> None:
    """Accepting half would leave a hole that every later verification reports
    and nobody can explain."""
    theirs = chain_for("truck-12", 3)
    broken = [theirs[0], theirs[2]]
    with pytest.raises(ChainBreakError) as caught:
        verify_continuity([], broken)
    assert caught.value.node_id == "truck-12"
    assert caught.value.record_id == "truck-12-2"


def test_a_tampered_record_is_caught_even_though_its_links_still_match() -> None:
    """The subtle one. Editing a body without touching the stored hash leaves the
    prev_hash links intact, so only recomputing the digest catches it."""
    theirs = chain_for("truck-12", 3)
    tampered = theirs[1].model_copy(update={"body": {"phase": "COMPLETE"}})
    assert tampered.prev_hash == theirs[1].prev_hash
    assert tampered.hash == theirs[1].hash, "the stored digest is unchanged, which is the trap"
    with pytest.raises(ChainBreakError):
        verify_continuity([], [theirs[0], tampered, theirs[2]])


def test_the_sync_vector_is_the_latest_stamp_per_node() -> None:
    records = chain_for("truck-7", 3) + chain_for("truck-12", 2, start=500)
    vector = sync_vector(records)
    assert vector["truck-7"]["physical_ms"] == 102
    assert vector["truck-12"]["physical_ms"] == 501


def test_only_what_a_peer_has_not_seen_is_sent() -> None:
    mine = chain_for("truck-7", 4)
    peer_has = {"truck-7": {"physical_ms": 101, "logical": 0, "node_id": "truck-7"}}
    to_send = unsynced(mine, "truck-7", peer_has)
    assert [r.id for r in to_send] == ["truck-7-2", "truck-7-3"]


def test_a_peer_that_has_nothing_gets_everything() -> None:
    mine = chain_for("truck-7", 3)
    assert len(unsynced(mine, "truck-7", {})) == 3


def test_what_is_sent_is_in_chain_order() -> None:
    """So the receiver can follow the links rather than buffer and sort."""
    mine = chain_for("truck-7", 5)
    sent = unsynced(list(reversed(mine)), "truck-7", {})
    assert [r.id for r in sent] == [r.id for r in mine]


def test_a_merged_record_verifies_in_the_receiving_store(node: Node) -> None:
    """The real check: a peer's records land in a real store and the chain holds."""
    theirs = chain_for("truck-12", 3)
    for record in theirs:
        node.store.append(record)
    assert node.store.verify_chain("truck-12") is None
    assert sorted(node.store.node_ids()) == ["truck-12"]


@settings(max_examples=100)
@given(st.integers(0, 5), st.integers(0, 5))
def test_union_is_order_independent_for_any_two_chains(mine_n: int, theirs_n: int) -> None:
    mine, theirs = chain_for("truck-7", mine_n), chain_for("truck-12", theirs_n, start=900)
    forwards = {r.id for r in merge(mine, theirs).accepted}
    backwards = {r.id for r in merge(theirs, mine).accepted}
    assert forwards == {r.id for r in theirs}
    assert backwards == {r.id for r in mine}


@settings(max_examples=50)
@given(st.integers(1, 6))
def test_a_whole_chain_always_joins_up(count: int) -> None:
    verify_continuity([], chain_for("truck-12", count))


@settings(max_examples=50)
@given(st.integers(3, 6))
def test_dropping_any_middle_record_is_caught(count: int) -> None:
    """A middle record only. Dropping the last leaves a shorter valid chain, which
    is what a partial sync legitimately looks like."""
    chain = chain_for("truck-12", count)
    for drop in range(1, count - 1):
        with pytest.raises(ChainBreakError):
            verify_continuity([], chain[:drop] + chain[drop + 1 :])


@settings(max_examples=50)
@given(st.integers(1, 6))
def test_a_prefix_of_a_chain_is_valid(count: int) -> None:
    """Partial sync is normal: a peer that has sent the first n records has not
    sent a broken chain, it has sent part of one."""
    chain = chain_for("truck-12", count)
    verify_continuity([], chain[: max(1, count - 1)])


def test_superseded_by_peer_gives_execution_rights_to_the_lower_stamp() -> None:
    """Two nodes queued the same action while isolated. Only one may perform it,
    and both must compute the same answer without agreeing on anything first."""
    from deadreckoning.clock import HLC
    from deadreckoning.sync.merge import superseded_by_peer

    class E:
        def __init__(self, entry_id: str, node: str, physical: int) -> None:
            self.id = entry_id
            self.node_id = node
            self.idempotency_key = "dispatch:T-101"
            self.created_hlc = HLC(physical, 0, node)

    mine = [E("mine-1", "truck-7", 200)]
    theirs: list[Any] = [E("theirs-1", "truck-12", 100)]
    assert superseded_by_peer(mine, theirs) == [("mine-1", "theirs-1")]
    assert superseded_by_peer(theirs, mine) == [], "the earlier one keeps its rights"
