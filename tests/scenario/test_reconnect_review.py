# SPDX-License-Identifier: Apache-2.0
"""Phase 4's re-review: the frontier model reads what the truck could not.

truck-7 saw a ticket that said "lines down near the school" and called it P2. It
was right on what it had. truck-12, also dark, had a field report of an energized
conductor on the ground and called the same ticket P1.

When the link returns and the logs merge, the frontier model reviews truck-7's
decision with truck-12's evidence in front of it, and disagrees. That disagreement
is not an error: it is two correct decisions made on different information, and
the runtime's job is to surface it for a person rather than pick a winner.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from deadreckoning.budget import PowerState
from deadreckoning.clock import TimeTrust
from deadreckoning.config import TierConfig, TierKind
from deadreckoning.health import HealthState
from deadreckoning.node import Node
from deadreckoning.outbox import Outbox, OutboxState
from deadreckoning.records import Mode, Record, RecordKind
from deadreckoning.review import (
    merged_evidence,
    review_preamble,
    reviewable,
    verdict_for,
)
from deadreckoning.router import Router

FRONTIER = TierConfig(name="frontier", rank=0, kind=TierKind.REMOTE, model="big", base_url="u")
LOCAL_Q8 = TierConfig(name="local-q8", rank=1, kind=TierKind.LOCAL, model="q8", base_url="u")
LOCAL_Q4 = TierConfig(name="local-q4", rank=2, kind=TierKind.LOCAL, model="q4", base_url="u")
TIERS = [FRONTIER, LOCAL_Q8, LOCAL_Q4]


def islanded_health() -> dict[str, HealthState]:
    return {
        "frontier": HealthState.UNREACHABLE,
        "local-q8": HealthState.HEALTHY,
        "local-q4": HealthState.HEALTHY,
    }


def reconnected_health() -> dict[str, HealthState]:
    return dict.fromkeys(("frontier", "local-q8", "local-q4"), HealthState.HEALTHY)


def peer_record(kind: RecordKind, body: dict[str, Any], record_id: str) -> Record:
    """A record merged in from truck-12, as sync would deliver it."""
    return Record(
        id=record_id,
        node_id="truck-12",
        hlc={"physical_ms": 1, "logical": 0, "node_id": "truck-12"},
        wall_time="2026-01-01T00:00:00+00:00",
        time_trust=TimeTrust.DRIFTING,
        kind=kind,
        mode=Mode.ISLANDED,
        subject="ticket:T-104",
        prev_hash="0" * 64,
        body=body,
        tier={"name": "local-q4", "rank": 2, "kind": "local", "model": "q4"},
    )


@pytest.fixture
def islanded_decision(node: Node) -> Record:
    """truck-7 decides T-104 in the dark, on a small model, with no field report."""
    selection = Router(TIERS).select(
        task_class="triage",
        min_rank=2,
        review_above_rank=0,
        health=islanded_health(),
        power=PowerState.BATTERY_LOW,
    )
    assert selection.tier.name == "local-q4"
    assert selection.review_required
    return node.emit(
        RecordKind.FINAL,
        body={
            "outcome": {"key": "priority", "value": "P2"},
            "rationale": "ticket text only; no field report on this truck",
            "evidence": [],
            "depends_on": ["lookup_asset", "get_field_reports"],
            "confidence": 0.6,
            "tool_call_ids": [],
        },
        subject="ticket:T-104",
        tier=selection.stamp(),
        tiers_available=selection.tiers_available,
        review_required=selection.review_required,
        review_reason=selection.review_reason,
        manifest_hash="b" * 64,
    )


def test_the_islanded_decision_is_stamped_with_what_made_it(
    islanded_decision: Record,
) -> None:
    assert islanded_decision.tier is not None
    assert islanded_decision.tier["rank"] == 2
    assert islanded_decision.review_required is True
    assert islanded_decision.tiers_available == ["local-q8", "local-q4"]
    assert "frontier" not in (islanded_decision.tiers_available or [])


def test_the_frontier_model_reviews_it_once_the_link_returns(
    node: Node, islanded_decision: Record
) -> None:
    router = Router(TIERS)
    reviewer = router.select(
        task_class="triage",
        min_rank=2,
        review_above_rank=0,
        health=reconnected_health(),
    ).tier
    assert reviewer.name == "frontier"

    items = reviewable([islanded_decision], reviewer, review_above_rank=0)
    assert [i.record.id for i in items] == [islanded_decision.id]


def test_the_review_sees_the_field_report_the_truck_never_had(
    node: Node, islanded_decision: Record
) -> None:
    """Sync runs before review precisely so this is true."""
    peer = peer_record(
        RecordKind.TOOL_RESULT,
        {
            "tool": "get_field_reports",
            "call_id": "c-1",
            "availability": "LOCAL",
            "result": [{"observation": "energized conductor on ground, 40m from school entrance"}],
        },
        "tr-12",
    )
    evidence = merged_evidence([islanded_decision, peer], "ticket:T-104", exclude_node=node.node_id)
    assert len(evidence) == 1
    preamble = review_preamble(reviewable([islanded_decision], FRONTIER, 0)[0], evidence)
    assert "energized conductor" in preamble
    assert "rank 2" in preamble


def test_the_review_disagrees_and_that_becomes_a_conflict(
    node: Node, islanded_decision: Record
) -> None:
    """The milestone's gate. One REVIEW_DISAGREEMENT on ticket:T-104."""
    item = reviewable([islanded_decision], FRONTIER, review_above_rank=0)[0]
    verdict, diff = verdict_for(
        item,
        FRONTIER,
        {"key": "priority", "value": "P1"},
        "a downed energized conductor beside a school is P1",
    )
    assert verdict.agrees is False

    review = node.emit(
        RecordKind.REVIEW, body={**verdict.as_body(), "diff": diff}, subject="ticket:T-104"
    )
    conflict = node.emit(
        RecordKind.CONFLICT,
        body={
            "subtype": "REVIEW_DISAGREEMENT",
            "subject": "ticket:T-104",
            "outcome_key": "priority",
            "record_ids": sorted([islanded_decision.id, review.id]),
            "dedupe_key": "c" * 64,
            "held_outbox_ids": [],
            "detected_by": node.node_id,
        },
        subject="ticket:T-104",
    )

    conflicts = [
        r
        for r in node.store.iter_records(kind=RecordKind.CONFLICT)
        if r.body["subtype"] == "REVIEW_DISAGREEMENT"
    ]
    assert len(conflicts) == 1
    assert conflicts[0].id == conflict.id
    assert set(conflicts[0].body["record_ids"]) == {islanded_decision.id, review.id}

    stored = node.store.get(islanded_decision.id)
    assert stored is not None
    assert stored.body["outcome"]["value"] == "P2", "the original is never rewritten"
    assert node.store.verify_chain(node.node_id) is None


def test_a_disagreement_holds_the_actions_that_traced_to_the_decision(
    node: Node, islanded_decision: Record
) -> None:
    """The priority change queued on the strength of the P2 must not fire while a
    person is still deciding whether P2 was right."""
    from deadreckoning.clock import HLC
    from deadreckoning.tools.contract import (
        Consequence,
        OfflinePolicy,
        SideEffect,
        ToolContract,
    )

    box = Outbox(node.database, node.node_id)
    contract = ToolContract(
        name="set_ticket_priority",
        backend="ticket-api",
        offline_policy=OfflinePolicy.QUEUE,
        side_effect=SideEffect.IDEMPOTENT,
        consequence=Consequence.MEDIUM,
        idempotency_key="prio:{ticket_id}:{priority}",
        expiry_s=14400,
    )
    entry = box.defer(
        contract=contract,
        args={"ticket_id": "T-104", "priority": "P2"},
        subject="ticket:T-104",
        decision_id=islanded_decision.id,
        deferral_record_id=islanded_decision.id,
        observations=[],
        created_hlc=HLC(1, 0, node.node_id),
        now_ms=0,
        approval_required=False,
        approval_reason=None,
        entry_id="ob-1",
    )
    assert entry.state is OutboxState.READY

    traced = box.tracing_to({islanded_decision.id})
    assert [e.id for e in traced] == ["ob-1"]
    for held in traced:
        held.hold = {"conflict_id": "conf-1", "previous_state": str(held.state)}
        box.transition(held, OutboxState.ON_HOLD, reason="REVIEW_DISAGREEMENT")

    after = box.get("ob-1")
    assert after is not None
    assert after.state is OutboxState.ON_HOLD
    assert after.hold is not None
    assert after.hold["previous_state"] == "READY", "so a resolution can put it back"


def test_on_mains_the_same_decision_would_have_been_reviewed_anyway(node: Node) -> None:
    """The re-review story does not depend on a battery setting."""
    on_mains = Router(TIERS).select(
        task_class="triage",
        min_rank=2,
        review_above_rank=0,
        health=islanded_health(),
        power=PowerState.MAINS,
    )
    assert on_mains.tier.name == "local-q8"
    assert on_mains.review_required
    assert on_mains.review_reason == "TIER_BELOW_REVIEW_THRESHOLD"
