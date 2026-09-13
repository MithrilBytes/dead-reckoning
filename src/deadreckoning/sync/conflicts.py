# SPDX-License-Identifier: Apache-2.0
"""Noticing that two nodes decided the same question differently.

This is the part of the system that does not exist elsewhere. Sync engines merge
data, and when two copies disagree they pick one: last write wins, or a field
merge, or a CRDT that guarantees convergence. All of those are correct for data
and wrong for decisions, because the thing that makes two decisions differ is
usually that the deciders knew different things, and silently keeping the later
one throws away the evidence along with the answer.

So detection here produces a question rather than an answer. Two decisions about
the same subject and the same outcome key, with different values, made without
either node having seen the other, are a conflict, and a conflict is for a human.

Detection is a pure function over a set of records. That matters more than it
looks: emitting a conflict record changes the record set, so if detection read its
own output it would not be idempotent, and two nodes merging in different orders
could reach different conclusions about what happened. The caller filters what has
already been recorded; the function itself only ever reads decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, cast

from deadreckoning.canonical import content_hash, sha256_hex
from deadreckoning.records import Record, RecordKind

DECIDING = frozenset({RecordKind.FINAL, RecordKind.RESOLUTION})
COMPARABLE = frozenset({RecordKind.FINAL, RecordKind.ABSTENTION, RecordKind.RESOLUTION})


def outcome_of(record: Record) -> dict[str, Any]:
    """The assertion a record makes, wherever that record keeps it.

    A resolution carries `effective_outcome` rather than `outcome`, because what
    it asserts is the value that now stands, which on the override path is not
    any existing record's value at all.
    """
    key = "effective_outcome" if record.kind is RecordKind.RESOLUTION else "outcome"
    value: Any = record.body.get(key)
    return cast("dict[str, Any]", value) if isinstance(value, dict) else {}


def hlc_tuple(hlc: dict[str, Any]) -> tuple[int, int]:
    return (int(hlc["physical_ms"]), int(hlc["logical"]))


def observed_by(record: Record, node_id: str) -> tuple[int, int] | None:
    """The latest stamp `record`'s author had seen from `node_id` when deciding."""
    seen = record.observed.get(node_id)
    return hlc_tuple(seen) if isinstance(seen, dict) else None


def concurrent(a: Record, b: Record) -> bool:
    """Neither decider had seen the other.

    If one had, it is not a conflict: whoever decided second decided knowing what
    the first said, and disagreeing on purpose with more information is called
    superseding.
    """
    if a.node_id == b.node_id:
        return False
    a_seen_by_b = observed_by(b, a.node_id)
    b_seen_by_a = observed_by(a, b.node_id)
    a_was_visible = a_seen_by_b is not None and hlc_tuple(a.hlc) <= a_seen_by_b
    b_was_visible = b_seen_by_a is not None and hlc_tuple(b.hlc) <= b_seen_by_a
    return not a_was_visible and not b_was_visible


def dedupe_key(record_ids: list[str]) -> str:
    """A conflict's identity, portable across nodes.

    Two nodes each detect the same disagreement and each write their own record
    with their own id. What makes them one conflict rather than two is this: a
    hash of the pair they are about.
    """
    return sha256_hex("\n".join(sorted(record_ids)).encode("utf-8"))


@dataclass(frozen=True, slots=True)
class Conflict:
    """Two decisions that cannot both stand."""

    subject: str
    outcome_key: str
    record_ids: tuple[str, str]
    values: dict[str, Any]

    @property
    def dedupe_key(self) -> str:
        return dedupe_key(list(self.record_ids))

    def as_body(
        self, detected_by: str, held: list[str], already_executed: list[str]
    ) -> dict[str, Any]:
        return {
            "subtype": "CONCURRENT_DECISION",
            "subject": self.subject,
            "outcome_key": self.outcome_key,
            "record_ids": sorted(self.record_ids),
            "dedupe_key": self.dedupe_key,
            "values": self.values,
            "held_outbox_ids": sorted(held),
            "already_executed_outbox_ids": sorted(already_executed),
            "detected_by": detected_by,
        }


def superseded_ids(records: list[Record]) -> set[str]:
    """Everything a resolution has already settled.

    A conflict that has been adjudicated is not a conflict any more, and
    re-detecting it every merge would mean a human answering the same question
    forever.
    """
    settled: set[str] = set()
    for record in records:
        if record.kind is RecordKind.RESOLUTION:
            settled.update(record.supersedes)
    return settled


def detect(records: list[Record]) -> list[Conflict]:
    """Every conflict present in this set of records.

    Pure: same input, same output, whatever order the records arrived in and
    however many times it is called. It reads decisions only, never the conflict
    records it leads to, which is what keeps it idempotent.
    """
    settled = superseded_ids(records)
    candidates = [
        record
        for record in records
        if record.kind in DECIDING
        and record.subject
        and outcome_of(record).get("key")
        and record.id not in settled
    ]

    conflicts: list[Conflict] = []
    for a, b in combinations(candidates, 2):
        outcome_a, outcome_b = outcome_of(a), outcome_of(b)
        if a.subject != b.subject or outcome_a["key"] != outcome_b["key"]:
            continue
        if content_hash(outcome_a.get("value")) == content_hash(outcome_b.get("value")):
            continue
        if not concurrent(a, b):
            continue
        pair = tuple(sorted((a.id, b.id)))
        conflicts.append(
            Conflict(
                subject=str(a.subject),
                outcome_key=str(outcome_a["key"]),
                record_ids=(pair[0], pair[1]),
                values={a.id: outcome_a.get("value"), b.id: outcome_b.get("value")},
            )
        )
    return sorted(conflicts, key=lambda c: c.dedupe_key)


def undetected(records: list[Record], node_id: str) -> list[Conflict]:
    """Conflicts this node has not yet written a record for.

    The filter that keeps `detect` pure. One conflict record per node per pair:
    detection is local, so two nodes both seeing the pair will each write their
    own, and the `dedupe_key` is what makes them one conflict rather than two.
    """
    mine = {
        record.body.get("dedupe_key")
        for record in records
        if record.kind is RecordKind.CONFLICT and record.node_id == node_id
    }
    return [conflict for conflict in detect(records) if conflict.dedupe_key not in mine]


# --- resolution --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResolutionPlan:
    """What a resolution does to this node's held actions.

    Expressed as a predicate over one entry rather than as a list, so that the
    resolving node and every node that later merges the record reach the same
    answer without coordinating. A list computed on one node would have to travel
    and would be wrong the moment another node held a different entry.
    """

    chosen: str | None
    superseded: frozenset[str]

    def verdict(self, decision_id: str) -> str:
        """RELEASE, CANCEL, or LEAVE, for an entry tracing to this decision."""
        if self.chosen is not None and decision_id == self.chosen:
            return "RELEASE"
        if decision_id in self.superseded:
            return "CANCEL"
        return "LEAVE"


def plan_from(resolution: Record) -> ResolutionPlan:
    body = resolution.body
    chosen = body.get("chosen")
    return ResolutionPlan(
        chosen=str(chosen) if chosen else None,
        superseded=frozenset(resolution.supersedes),
    )
