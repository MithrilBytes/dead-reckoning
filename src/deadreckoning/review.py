# SPDX-License-Identifier: Apache-2.0
"""Looking again at decisions made in the dark, once there is light.

A decision taken on a small local model with a stale cache is not wrong by
definition, and treating it as suspect would make the whole offline path
pointless. But it was taken with less, and when more becomes available the
cheapest useful thing a system can do is check.

Two rules make a review mean something rather than being theatre. The reviewer
must be strictly better than the tier under review, or the queue fills with
reviews that agree by construction. And the review sees evidence merged from every
node, not only what the original node had, which is why sync runs before review:
the reviewer should have the best picture anyone had, or it is re-reading the same
partial view and calling that confirmation.

The original record is never touched. A disagreement produces a conflict for a
human, and the answer that stands is decided by a person, not by whichever model
spoke last.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

from deadreckoning.canonical import content_hash
from deadreckoning.config import TierConfig
from deadreckoning.records import Record, RecordKind

REVIEWABLE = frozenset({RecordKind.FINAL, RecordKind.ABSTENTION})


@dataclass(frozen=True, slots=True)
class ReviewItem:
    """One decision waiting to be looked at again."""

    record: Record
    original_tier: dict[str, Any]

    @property
    def subject(self) -> str | None:
        return self.record.subject

    @property
    def _outcome(self) -> dict[str, Any]:
        outcome: Any = self.record.body.get("outcome")
        return cast("dict[str, Any]", outcome) if isinstance(outcome, dict) else {}

    @property
    def outcome_key(self) -> str | None:
        key: Any = self._outcome.get("key")
        return str(key) if key is not None else None

    @property
    def outcome_value(self) -> Any:
        return self._outcome.get("value")


def reviewable(
    records: list[Record], reviewer: TierConfig, review_above_rank: int
) -> list[ReviewItem]:
    """Which decisions this tier is entitled to review.

    Three conditions, and the strict inequality is the one that matters. Without
    it a tier that survived the island would qualify as its own reviewer, and the
    queue would fill with reviews that re-run the same fidelity on the same
    evidence and agree by construction.
    """
    already_reviewed = {r.body.get("reviewed_id") for r in records if r.kind is RecordKind.REVIEW}
    items: list[ReviewItem] = []
    for record in records:
        if record.kind not in REVIEWABLE or not record.review_required:
            continue
        if record.id in already_reviewed:
            continue
        tier = record.tier or {}
        original_rank = tier.get("rank")
        if not isinstance(original_rank, int):
            continue
        if reviewer.rank > review_above_rank or reviewer.rank >= original_rank:
            continue
        items.append(ReviewItem(record=record, original_tier=tier))
    return items


def merged_evidence(
    records: list[Record], subject: str | None, exclude_node: str
) -> list[dict[str, Any]]:
    """What other nodes know about this subject.

    This is why sync runs before review. A reviewer re-reading only what the
    original node saw is confirming a partial view, not checking it.
    """
    if subject is None:
        return []
    relevant: list[dict[str, Any]] = []
    for record in records:
        if record.node_id == exclude_node or record.subject != subject:
            continue
        if record.kind in (RecordKind.FINAL, RecordKind.ABSTENTION):
            relevant.append(
                {
                    "node": record.node_id,
                    "kind": str(record.kind),
                    "tier": record.tier,
                    "mode": str(record.mode),
                    "outcome": record.body.get("outcome"),
                    "rationale": record.body.get("rationale") or record.body.get("reason"),
                }
            )
        elif record.kind is RecordKind.TOOL_RESULT:
            relevant.append(
                {
                    "node": record.node_id,
                    "kind": str(record.kind),
                    "tool": record.body.get("tool"),
                    "availability": record.body.get("availability"),
                    "result": record.body.get("result"),
                }
            )
    return relevant


def review_preamble(item: ReviewItem, evidence: list[dict[str, Any]]) -> str:
    """What the reviewing model is told before it is asked.

    It is shown the original's fidelity and circumstances rather than its
    conclusion alone, because "a small model decided this in the dark" is the
    context that makes a second look worth taking.
    """
    tier = item.original_tier
    lines = [
        "You are reviewing a decision another model made earlier, under worse conditions.",
        "",
        f"Original decision: {item.outcome_key} = {item.outcome_value!r} on {item.subject}",
        f"Made by: {tier.get('name')} (rank {tier.get('rank')}), in mode {item.record.mode}",
        f"Its rationale: {item.record.body.get('rationale') or item.record.body.get('reason')}",
    ]
    if evidence:
        lines += [
            "",
            "Evidence from other nodes about the same subject, which the original"
            " decision did not have:",
        ]
        lines.extend(f"  {entry}" for entry in evidence)
    lines += [
        "",
        "Decide for yourself. Agreeing is a useful answer; so is disagreeing. Do not",
        "defer to the original because it exists.",
    ]
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class ReviewVerdict:
    """The outcome of one review, in the shape the record body takes."""

    reviewed_id: str
    agrees: bool
    original_tier: dict[str, Any]
    reviewer_tier: dict[str, Any]
    original_mode: str
    reviewer_outcome: dict[str, Any]
    original_manifest_hash: str | None = None
    conflict_id: str | None = None

    def as_body(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "reviewed_id": self.reviewed_id,
            "agrees": self.agrees,
            "original_tier": self.original_tier,
            "reviewer_tier": self.reviewer_tier,
            "original_mode": self.original_mode,
            "reviewer_outcome": self.reviewer_outcome,
            "diff": None,
        }
        if self.original_manifest_hash:
            body["original_manifest_hash"] = self.original_manifest_hash
        if self.conflict_id:
            body["conflict_id"] = self.conflict_id
        return body


def compare(item: ReviewItem, reviewer_outcome: dict[str, Any]) -> bool:
    """Agreement is the same key and the same value, canonicalised.

    String equality after canonicalisation rather than anything cleverer: a
    comparator that decided P1 and P2 were close enough would be deciding the
    thing a human is supposed to adjudicate.
    """
    if item.outcome_key != reviewer_outcome.get("key"):
        return False
    return content_hash(item.outcome_value) == content_hash(reviewer_outcome.get("value"))


def verdict_for(
    item: ReviewItem,
    reviewer: TierConfig,
    reviewer_outcome: dict[str, Any],
    reviewer_rationale: str = "",
) -> tuple[ReviewVerdict, dict[str, Any] | None]:
    """Build the verdict, and the diff when the two disagree."""
    agrees = compare(item, reviewer_outcome)
    verdict = ReviewVerdict(
        reviewed_id=item.record.id,
        agrees=agrees,
        original_tier=item.original_tier,
        reviewer_tier={
            "name": reviewer.name,
            "rank": reviewer.rank,
            "kind": str(reviewer.kind),
            "model": reviewer.model,
        },
        original_mode=str(item.record.mode),
        reviewer_outcome=reviewer_outcome,
        original_manifest_hash=item.record.manifest_hash,
    )
    if agrees:
        return verdict, None
    return verdict, {
        "original_value": item.outcome_value,
        "reviewer_value": reviewer_outcome.get("value"),
        "reviewer_rationale": reviewer_rationale,
    }


@dataclass(slots=True)
class ReviewRun:
    """What one pass over the queue produced."""

    reviewed: list[str] = field(default_factory=list[str])
    disagreements: list[str] = field(default_factory=list[str])
    held_outbox_ids: list[str] = field(default_factory=list[str])
