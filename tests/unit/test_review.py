# SPDX-License-Identifier: Apache-2.0
"""Re-reading a decision made in the dark, and what happens when the answer differs."""

from __future__ import annotations

from typing import Any

from deadreckoning.clock import TimeTrust
from deadreckoning.config import TierConfig, TierKind
from deadreckoning.node import Node
from deadreckoning.records import Mode, Record, RecordKind
from deadreckoning.review import (
    compare,
    merged_evidence,
    review_preamble,
    reviewable,
    verdict_for,
)

FRONTIER = TierConfig(name="frontier", rank=0, kind=TierKind.REMOTE, model="big", base_url="u")
LOCAL_Q8 = TierConfig(name="local-q8", rank=1, kind=TierKind.LOCAL, model="q8", base_url="u")
LOCAL_Q4 = TierConfig(name="local-q4", rank=2, kind=TierKind.LOCAL, model="q4", base_url="u")


def decision(
    node: Node,
    subject: str,
    value: str,
    rank: int,
    review_required: bool = True,
    kind: RecordKind = RecordKind.FINAL,
) -> Record:
    return node.emit(
        kind,
        body={
            "outcome": {"key": "priority", "value": value},
            "rationale": f"decided at rank {rank}",
            "evidence": [],
            "depends_on": [],
            "confidence": 0.6,
            "tool_call_ids": [],
        },
        subject=subject,
        tier={"name": f"tier-{rank}", "rank": rank, "kind": "local", "model": "m"},
        review_required=review_required,
        review_reason="TIER_BELOW_REVIEW_THRESHOLD" if review_required else None,
        manifest_hash="a" * 64,
    )


def test_a_flagged_decision_is_reviewable_by_a_better_tier(node: Node) -> None:
    records = [decision(node, "ticket:T-104", "P2", rank=2)]
    assert [i.record.id for i in reviewable(records, FRONTIER, review_above_rank=0)] == [
        records[0].id
    ]


def test_a_tier_may_not_review_its_own_fidelity(node: Node) -> None:
    """Without the strict inequality the queue fills with reviews that agree by
    construction, which is worse than no review because it looks like assurance."""
    records = [decision(node, "ticket:T-104", "P2", rank=1)]
    assert reviewable(records, LOCAL_Q8, review_above_rank=1) == []


def test_a_worse_tier_may_not_review_a_better_one(node: Node) -> None:
    records = [decision(node, "ticket:T-104", "P2", rank=0)]
    assert reviewable(records, LOCAL_Q4, review_above_rank=2) == []


def test_a_decision_that_was_never_flagged_is_left_alone(node: Node) -> None:
    records = [decision(node, "ticket:T-104", "P2", rank=2, review_required=False)]
    assert reviewable(records, FRONTIER, review_above_rank=0) == []


def test_a_decision_already_reviewed_is_not_reviewed_again(node: Node) -> None:
    original = decision(node, "ticket:T-104", "P2", rank=2)
    already = node.emit(
        RecordKind.REVIEW,
        body={
            "reviewed_id": original.id,
            "agrees": True,
            "original_tier": {"name": "t", "rank": 2, "kind": "local", "model": "m"},
            "reviewer_tier": {"name": "frontier", "rank": 0, "kind": "remote", "model": "b"},
            "original_mode": "ISLANDED",
            "diff": None,
        },
        subject="ticket:T-104",
    )
    assert reviewable([original, already], FRONTIER, review_above_rank=0) == []


def test_abstentions_are_reviewable_too(node: Node) -> None:
    """An abstention made on a small model may be an abstention a bigger one
    would not have needed."""
    record = node.emit(
        RecordKind.ABSTENTION,
        body={
            "outcome": {"key": "priority"},
            "reason": "no live load",
            "depends_on_unavailable": ["get_live_load"],
            "source": "MODEL",
        },
        subject="ticket:T-106",
        tier={"name": "local-q4", "rank": 2, "kind": "local", "model": "m"},
        review_required=True,
    )
    assert len(reviewable([record], FRONTIER, review_above_rank=0)) == 1


def test_agreement_is_the_same_key_and_the_same_value(node: Node) -> None:
    item = reviewable([decision(node, "t:1", "P2", rank=2)], FRONTIER, 0)[0]
    assert compare(item, {"key": "priority", "value": "P2"})
    assert not compare(item, {"key": "priority", "value": "P1"})
    assert not compare(item, {"key": "urgency", "value": "P2"})


def test_a_disagreement_carries_both_values_and_the_reviewer_reasoning(node: Node) -> None:
    item = reviewable([decision(node, "ticket:T-104", "P2", rank=2)], FRONTIER, 0)[0]
    verdict, diff = verdict_for(
        item, FRONTIER, {"key": "priority", "value": "P1"}, "the field report changes this"
    )
    assert verdict.agrees is False
    assert diff is not None
    assert diff["original_value"] == "P2"
    assert diff["reviewer_value"] == "P1"
    assert diff["reviewer_rationale"]
    assert verdict.as_body()["reviewer_tier"]["rank"] == 0


def test_agreement_produces_no_diff(node: Node) -> None:
    item = reviewable([decision(node, "ticket:T-104", "P2", rank=2)], FRONTIER, 0)[0]
    verdict, diff = verdict_for(item, FRONTIER, {"key": "priority", "value": "P2"})
    assert verdict.agrees
    assert diff is None


def test_the_original_record_is_never_touched(node: Node) -> None:
    """A review adds a record. It does not edit one, and could not: the chain
    would break and the store has no update path."""
    original = decision(node, "ticket:T-104", "P2", rank=2)
    before = original.hash
    item = reviewable([original], FRONTIER, 0)[0]
    verdict_for(item, FRONTIER, {"key": "priority", "value": "P1"})
    stored = node.store.get(original.id)
    assert stored is not None
    assert stored.hash == before
    assert stored.body["outcome"]["value"] == "P2"


def test_the_reviewer_sees_what_other_nodes_knew(node: Node) -> None:
    """This is why sync runs before review. A reviewer re-reading only what the
    original node saw is confirming a partial view, not checking it."""
    mine = decision(node, "ticket:T-104", "P2", rank=2)
    theirs = Record(
        id="x-1",
        node_id="truck-12",
        hlc={"physical_ms": 1, "logical": 0, "node_id": "truck-12"},
        wall_time="2026-01-01T00:00:00+00:00",
        time_trust=TimeTrust.DRIFTING,
        kind=RecordKind.TOOL_RESULT,
        mode=Mode.ISLANDED,
        subject="ticket:T-104",
        prev_hash="0" * 64,
        body={
            "tool": "get_field_reports",
            "call_id": "c",
            "availability": "LOCAL",
            "result": [{"observation": "energized conductor on ground, 40m from the school"}],
        },
    )
    evidence = merged_evidence([mine, theirs], "ticket:T-104", exclude_node=node.node_id)
    assert len(evidence) == 1
    assert evidence[0]["node"] == "truck-12"
    assert "energized conductor" in str(evidence[0]["result"])


def test_the_preamble_tells_the_reviewer_the_conditions_not_just_the_answer(
    node: Node,
) -> None:
    item = reviewable([decision(node, "ticket:T-104", "P2", rank=2)], FRONTIER, 0)[0]
    text = review_preamble(item, [{"node": "truck-12", "result": "live wire"}])
    assert "rank 2" in text
    assert "ISLANDED" in text
    assert "live wire" in text
    assert "Agreeing is a useful answer; so is disagreeing" in text
    assert "Do not" in text and "defer to the original" in text


def test_evidence_from_this_node_is_not_fed_back_to_it(node: Node) -> None:
    mine = decision(node, "ticket:T-104", "P2", rank=2)
    assert merged_evidence([mine], "ticket:T-104", exclude_node=node.node_id) == []


def test_a_review_of_a_different_subject_is_not_offered_as_evidence(node: Node) -> None:
    other: list[Any] = [decision(node, "ticket:T-999", "P3", rank=2)]
    assert merged_evidence(other, "ticket:T-104", exclude_node="elsewhere") == []
