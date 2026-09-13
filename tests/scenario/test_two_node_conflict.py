# SPDX-License-Identifier: Apache-2.0
"""Two trucks, one ticket, opposite calls, and a person deciding.

This is the scenario the project exists for. Both nodes are dark. truck-7 has the
ticket text and says P2. truck-12 has a field report of a live conductor on the
ground and says P1. Neither has seen the other. Both queue actions on the strength
of their own decision.

When the link returns, the logs merge and the disagreement surfaces as a conflict
rather than as a winner. A supervisor picks one, and the actions that traced to
the losing decision are cancelled while the ones that traced to the chosen
decision are released. No last write wins, anywhere in that sentence.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from deadreckoning.canonical import GENESIS_PREV_HASH
from deadreckoning.clock import HLC, TimeTrust
from deadreckoning.config import Config
from deadreckoning.node import Node
from deadreckoning.outbox import Outbox, OutboxState
from deadreckoning.records import Mode, Record, RecordKind
from deadreckoning.sync.conflicts import (
    dedupe_key,
    detect,
    plan_from,
    undetected,
)
from deadreckoning.sync.merge import merge, sync_vector, unsynced
from deadreckoning.tools.contract import (
    Consequence,
    OfflinePolicy,
    SideEffect,
    ToolContract,
)

PRIORITY = ToolContract(
    name="set_ticket_priority",
    backend="ticket-api",
    offline_policy=OfflinePolicy.QUEUE,
    side_effect=SideEffect.IDEMPOTENT,
    consequence=Consequence.MEDIUM,
    idempotency_key="prio:{ticket_id}:{priority}",
    expiry_s=14400,
)


@pytest.fixture
def truck7(node: Node) -> Node:
    return node


@pytest.fixture
def truck12(tmp_path: Path, config: Config) -> Iterator[Node]:
    """A second node with its own database, as a second truck would have."""
    other = config.model_copy(
        update={"node": config.node.model_copy(update={"node_id": "truck-12"})}
    )
    with Node.open(other, tmp_path / "peer") as opened:
        yield opened


def decide(node: Node, value: str, observed: dict[str, Any] | None = None) -> Record:
    return node.emit(
        RecordKind.FINAL,
        body={
            "outcome": {"key": "priority", "value": value},
            "rationale": (
                "ticket text only" if value == "P2" else "field report: live conductor on ground"
            ),
            "evidence": [],
            "depends_on": ["get_field_reports"],
            "confidence": 0.6,
            "tool_call_ids": [],
        },
        subject="ticket:T-104",
        tier={"name": "local-q4", "rank": 2, "kind": "local", "model": "q4"},
        review_required=True,
        review_reason="TIER_BELOW_REVIEW_THRESHOLD",
        manifest_hash="a" * 64,
        **({"observed": observed} if observed else {}),
    )


def queue(node: Node, decision: Record, priority: str, entry_id: str) -> Any:
    box = Outbox(node.database, node.node_id)
    return box.defer(
        contract=PRIORITY,
        args={"ticket_id": "T-104", "priority": priority},
        subject="ticket:T-104",
        decision_id=decision.id,
        deferral_record_id=decision.id,
        observations=[],
        created_hlc=HLC(node.clock.last.physical_ms, 0, node.node_id),
        now_ms=node.clock.last.physical_ms,
        approval_required=False,
        approval_reason=None,
        entry_id=entry_id,
    )


def test_two_isolated_trucks_reach_opposite_conclusions(truck7: Node, truck12: Node) -> None:
    p2 = decide(truck7, "P2")
    p1 = decide(truck12, "P1")

    assert p2.observed.keys() == {"truck-7"}
    assert p1.observed.keys() == {"truck-12"}, "neither has heard of the other"

    conflicts = detect([p2, p1])
    assert len(conflicts) == 1
    assert conflicts[0].outcome_key == "priority"
    assert conflicts[0].values[p2.id] == "P2"
    assert conflicts[0].values[p1.id] == "P1"


def test_merging_the_peer_log_keeps_both_and_verifies(truck7: Node, truck12: Node) -> None:
    decide(truck7, "P2")
    decide(truck12, "P1")

    theirs = list(truck12.store.iter_records(node_id="truck-12"))
    mine = list(truck7.store.iter_records())
    result = merge(mine, theirs)

    for record in result.accepted:
        truck7.store.append(record)

    assert truck7.store.verify_chain("truck-12") is None
    assert truck7.store.verify_chain("truck-7") is None
    assert sorted(truck7.store.node_ids()) == ["truck-12", "truck-7"]
    assert set(result.vector) == {"truck-7", "truck-12"}


def test_only_what_the_peer_has_not_seen_is_sent(truck7: Node, truck12: Node) -> None:
    decide(truck7, "P2")
    first_round = list(truck7.store.iter_records(node_id="truck-7"))
    peer_vector = sync_vector(first_round)

    decide(truck7, "P3")
    to_send = unsynced(list(truck7.store.iter_records(node_id="truck-7")), "truck-7", peer_vector)
    assert len(to_send) == 1, "only the new decision goes over the wire"


def test_each_node_writes_its_own_conflict_sharing_one_identity(
    truck7: Node, truck12: Node
) -> None:
    """Detection is local. What makes it one conflict is the dedupe key."""
    p2 = decide(truck7, "P2")
    p1 = decide(truck12, "P1")
    for record in merge(list(truck7.store.iter_records()), [p1]).accepted:
        truck7.store.append(record)
    for record in merge(list(truck12.store.iter_records()), [p2]).accepted:
        truck12.store.append(record)

    seven = undetected(list(truck7.store.iter_records()), "truck-7")
    twelve = undetected(list(truck12.store.iter_records()), "truck-12")
    assert len(seven) == 1
    assert len(twelve) == 1
    assert seven[0].dedupe_key == twelve[0].dedupe_key

    conflict = truck7.emit(
        RecordKind.CONFLICT,
        body=seven[0].as_body(detected_by="truck-7", held=["ob-7"], already_executed=[]),
        subject="ticket:T-104",
    )
    assert conflict.body["dedupe_key"] == dedupe_key([p2.id, p1.id])
    assert undetected(list(truck7.store.iter_records()), "truck-7") == [], (
        "recorded once, not every merge"
    )


def test_the_resolution_releases_one_side_and_cancels_the_other(
    truck7: Node, truck12: Node
) -> None:
    """The supervisor picks P1. Both nodes reach the same verdict on their own
    entries from the same record, without coordinating."""
    p2 = decide(truck7, "P2")
    p1 = decide(truck12, "P1")
    seven_entry = queue(truck7, p2, "P2", "ob-7")
    twelve_entry = queue(truck12, p1, "P1", "ob-12")

    seven_box, twelve_box = Outbox(truck7.database, "truck-7"), Outbox(truck12.database, "truck-12")
    for box, entry in ((seven_box, seven_entry), (twelve_box, twelve_entry)):
        entry.hold = {"conflict_id": "c-1", "previous_state": str(entry.state)}
        box.transition(entry, OutboxState.ON_HOLD, reason="CONCURRENT_DECISION")

    resolution = Record(
        id="res-1",
        node_id="truck-7",
        hlc={"physical_ms": 9000, "logical": 0, "node_id": "truck-7"},
        wall_time="2026-01-01T00:00:00+00:00",
        time_trust=TimeTrust.TRUSTED,
        kind=RecordKind.RESOLUTION,
        mode=Mode.RECONNECTING,
        subject="ticket:T-104",
        supersedes=[p2.id, p1.id],
        prev_hash=GENESIS_PREV_HASH,
        body={
            "conflict_ids": ["c-1"],
            "dedupe_keys": [dedupe_key([p2.id, p1.id])],
            "chosen": p1.id,
            "effective_outcome": {"key": "priority", "value": "P1"},
            "by": "supervisor-mchen",
            "note": "field report confirms energized conductor",
            "released_outbox_ids": [],
            "cancelled_outbox_ids": [],
        },
    )

    plan = plan_from(resolution)
    assert plan.verdict(p1.id) == "RELEASE"
    assert plan.verdict(p2.id) == "CANCEL"
    assert plan.verdict("some-other-decision") == "LEAVE"

    for box, entry_id in ((seven_box, "ob-7"), (twelve_box, "ob-12")):
        entry = box.get(entry_id)
        assert entry is not None
        verdict = plan.verdict(entry.decision_id)
        if verdict == "RELEASE":
            box.transition(entry, OutboxState.READY, reason="RELEASED_BY_RESOLUTION")
        elif verdict == "CANCEL":
            box.transition(entry, OutboxState.CANCELLED, reason="CANCELLED_BY_RESOLUTION")

    assert seven_box.get("ob-7").state is OutboxState.CANCELLED  # pyright: ignore[reportOptionalMemberAccess]
    assert twelve_box.get("ob-12").state is OutboxState.READY  # pyright: ignore[reportOptionalMemberAccess]


def test_a_resolved_conflict_stays_resolved_after_further_merges(
    truck7: Node, truck12: Node
) -> None:
    """Otherwise a human answers the same question every time the nodes sync."""
    p2 = decide(truck7, "P2")
    p1 = decide(truck12, "P1")
    resolution = Record(
        id="res-1",
        node_id="truck-7",
        hlc={"physical_ms": 9000, "logical": 0, "node_id": "truck-7"},
        wall_time="2026-01-01T00:00:00+00:00",
        time_trust=TimeTrust.TRUSTED,
        kind=RecordKind.RESOLUTION,
        mode=Mode.RECONNECTING,
        subject="ticket:T-104",
        supersedes=[p2.id, p1.id],
        prev_hash=GENESIS_PREV_HASH,
        body={
            "conflict_ids": ["c-1"],
            "dedupe_keys": [dedupe_key([p2.id, p1.id])],
            "chosen": p1.id,
            "effective_outcome": {"key": "priority", "value": "P1"},
            "by": "supervisor-mchen",
            "note": None,
            "released_outbox_ids": [],
            "cancelled_outbox_ids": [],
        },
    )
    assert detect([p2, p1]) != []
    assert detect([p2, p1, resolution]) == []
    assert undetected([p2, p1, resolution], "truck-12") == []


def test_the_whole_story_is_readable_from_either_node(truck7: Node, truck12: Node) -> None:
    """After sync, `dr log --subject ticket:T-104` tells the same story on both."""
    p2 = decide(truck7, "P2")
    p1 = decide(truck12, "P1")
    for record in merge(list(truck7.store.iter_records()), [p1]).accepted:
        truck7.store.append(record)
    for record in merge(list(truck12.store.iter_records()), [p2]).accepted:
        truck12.store.append(record)

    seven = {r.id for r in truck7.store.iter_records(subject="ticket:T-104")}
    twelve = {r.id for r in truck12.store.iter_records(subject="ticket:T-104")}
    assert seven == twelve == {p2.id, p1.id}
    assert truck7.store.verify_chain("truck-12") is None
    assert truck12.store.verify_chain("truck-7") is None
