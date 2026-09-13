# SPDX-License-Identifier: Apache-2.0
"""Combining two nodes' logs without either losing anything.

Union on id, and that is the whole merge. It works because records are immutable
and hash chained: there is no field to reconcile, no later version to prefer, and
no order in which the result comes out different. What two nodes disagree about
is never the records, only the conclusions inside them, and those are conflicts
rather than merge failures.

The one thing checked before accepting anything is chain continuity. A batch whose
`prev_hash` links do not join up with what is already held is refused whole,
because accepting half of it would leave a chain with a hole that every later
verification would report and nobody could explain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from deadreckoning.canonical import GENESIS_PREV_HASH
from deadreckoning.records import Record

Vector = dict[str, dict[str, Any]]


class ChainBreakError(ValueError):
    """An incoming batch does not join up with what is held."""

    def __init__(self, node_id: str, record_id: str, expected: str, found: str) -> None:
        super().__init__(
            f"chain break on {node_id} at {record_id}: expected prev_hash {expected}, found {found}"
        )
        self.node_id = node_id
        self.record_id = record_id
        self.expected = expected
        self.found = found


@dataclass(slots=True)
class MergeResult:
    accepted: list[Record] = field(default_factory=list[Record])
    already_held: list[str] = field(default_factory=list[str])
    vector: Vector = field(default_factory=dict[str, dict[str, Any]])

    @property
    def count(self) -> int:
        return len(self.accepted)


def sync_vector(records: list[Record]) -> Vector:
    """The latest stamp held from each node.

    This is what makes concurrency detectable later: a record carries the vector
    its author held when deciding, and two records neither of which appears in the
    other's vector were decided in ignorance of each other.
    """
    latest: dict[str, tuple[int, int]] = {}
    vector: Vector = {}
    for record in records:
        stamp = (int(record.hlc["physical_ms"]), int(record.hlc["logical"]))
        if stamp > latest.get(record.node_id, (-1, -1)):
            latest[record.node_id] = stamp
            vector[record.node_id] = dict(record.hlc)
    return vector


def chain_of(records: list[Record], node_id: str) -> list[Record]:
    return sorted(
        (r for r in records if r.node_id == node_id),
        key=lambda r: (int(r.hlc["physical_ms"]), int(r.hlc["logical"])),
    )


def verify_continuity(held: list[Record], incoming: list[Record]) -> None:
    """Refuse a batch that does not join up. All or nothing.

    Accepting part of a broken batch would leave a hole that `dr verify` reports
    forever and nobody can explain, which is worse than refusing the sync and
    saying so.
    """
    for node_id in sorted({r.node_id for r in incoming}):
        combined = chain_of(held, node_id) + [
            r for r in chain_of(incoming, node_id) if r.id not in {h.id for h in held}
        ]
        expected = GENESIS_PREV_HASH
        for record in combined:
            recomputed = record.compute_hash()
            if recomputed != record.hash:
                # Checked before the links, because a record whose content no
                # longer matches its own digest would otherwise pass: its stored
                # hash still chains correctly to the next record, and only
                # recomputing catches that the content beneath it changed.
                raise ChainBreakError(node_id, record.id, record.hash, recomputed)
            if record.prev_hash != expected:
                raise ChainBreakError(node_id, record.id, expected, record.prev_hash)
            expected = record.hash


def merge(held: list[Record], incoming: list[Record]) -> MergeResult:
    """Union by id, after checking the batch joins up.

    Order independent and idempotent, because union is. Merging the same batch
    twice changes nothing, and merging two batches in either order gives the same
    set, which is what lets nodes sync in whatever order they reconnect.
    """
    verify_continuity(held, incoming)
    held_ids = {record.id for record in held}
    accepted = [record for record in incoming if record.id not in held_ids]
    already = [record.id for record in incoming if record.id in held_ids]
    return MergeResult(
        accepted=accepted,
        already_held=already,
        vector=sync_vector(held + accepted),
    )


def unsynced(records: list[Record], node_id: str, peer_vector: Vector) -> list[Record]:
    """This node's records a peer has not seen yet.

    Sent oldest first so the receiver's continuity check can follow the chain
    rather than having to buffer and sort.
    """
    seen = peer_vector.get(node_id)
    floor = (int(seen["physical_ms"]), int(seen["logical"])) if seen else (-1, -1)
    return [
        record
        for record in chain_of(records, node_id)
        if (int(record.hlc["physical_ms"]), int(record.hlc["logical"])) > floor
    ]


def superseded_by_peer(mine: list[Any], theirs: list[Any]) -> list[tuple[str, str]]:
    """Which of my outbox entries a peer's entry has execution rights over.

    Two nodes that queued the same action while isolated must not both perform it.
    The lower stamp wins, which is arbitrary but stable: both nodes compute the
    same answer from the same data without having to agree on anything first.
    """
    by_key: dict[str, Any] = {entry.idempotency_key: entry for entry in theirs}
    losers: list[tuple[str, str]] = []
    for entry in mine:
        peer = by_key.get(entry.idempotency_key)
        if peer is None:
            continue
        mine_stamp = (entry.created_hlc.physical_ms, entry.created_hlc.logical, entry.node_id)
        peer_stamp = (peer.created_hlc.physical_ms, peer.created_hlc.logical, peer.node_id)
        if peer_stamp < mine_stamp:
            losers.append((entry.id, peer.id))
    return losers
