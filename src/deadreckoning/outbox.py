# SPDX-License-Identifier: Apache-2.0
"""Side effects that could not happen yet, and the conditions that justified them.

A queue of pending actions is an ordinary thing. What makes this one different is
that every entry carries the reasons it was created: the crew was free, the ticket
was unassigned. When the link returns those are checked again, and an entry whose
justification has evaporated does not fire. It becomes a question for the agent
instead.

That is the difference between deferring work and deferring a decision. Most
systems replay the action. This one re-examines whether the action is still the
right one, which is the only safe thing to do when the gap might have been eight
hours long and somebody else might have been working the same problem.

Exactly once is enforced at two levels: an idempotency key rejects a duplicate
intent before it is ever stored, and an entry found mid-execution after a crash is
never blindly retried.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from deadreckoning.clock import HLC
from deadreckoning.runtime import Database
from deadreckoning.tools.contract import Consequence, SideEffect, ToolContract
from deadreckoning.tools.preconditions import Observation

SCHEMA = """
CREATE TABLE IF NOT EXISTS outbox (
    id                 TEXT PRIMARY KEY,
    node_id            TEXT NOT NULL,
    physical_ms        INTEGER NOT NULL,
    logical            INTEGER NOT NULL,
    idempotency_key    TEXT NOT NULL,
    tool               TEXT NOT NULL,
    args               TEXT NOT NULL,
    subject            TEXT,
    consequence        TEXT NOT NULL,
    side_effect        TEXT NOT NULL,
    decision_id        TEXT NOT NULL,
    deferral_record_id TEXT NOT NULL,
    preconditions      TEXT NOT NULL,
    state              TEXT NOT NULL,
    approval           TEXT,
    hold               TEXT,
    expires_at_ms      INTEGER NOT NULL,
    attempts           INTEGER NOT NULL DEFAULT 0,
    last_error         TEXT,
    result_record_id   TEXT,
    re_decide_task_id  TEXT,
    superseded_by      TEXT,
    supersedes         TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS outbox_key ON outbox (node_id, idempotency_key);
CREATE INDEX IF NOT EXISTS outbox_by_state ON outbox (state);
CREATE INDEX IF NOT EXISTS outbox_by_subject ON outbox (subject) WHERE subject IS NOT NULL;
"""


class OutboxState(StrEnum):
    PENDING = "PENDING"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    READY = "READY"
    EXECUTING = "EXECUTING"
    DONE = "DONE"
    FAILED = "FAILED"
    PRECONDITION_FAILED = "PRECONDITION_FAILED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"
    ON_HOLD = "ON_HOLD"
    SUPERSEDED_BY_PEER = "SUPERSEDED_BY_PEER"


TERMINAL = frozenset(
    {
        OutboxState.DONE,
        OutboxState.EXPIRED,
        OutboxState.CANCELLED,
        OutboxState.SUPERSEDED_BY_PEER,
    }
)


class DuplicateIntentError(ValueError):
    """An entry with this idempotency key already exists on this node."""

    def __init__(self, key: str, existing_id: str) -> None:
        super().__init__(f"an entry with key {key!r} already exists as {existing_id}")
        self.key = key
        self.existing_id = existing_id


@dataclass(slots=True)
class Approval:
    required: bool
    reason: str | None = None
    decision: str | None = None
    by: str | None = None
    note: str | None = None

    @property
    def granted(self) -> bool:
        return self.decision == "APPROVED"

    def as_dict(self) -> dict[str, Any]:
        return {
            "required": self.required,
            "reason": self.reason,
            "decision": self.decision,
            "by": self.by,
            "note": self.note,
        }


@dataclass(slots=True)
class Entry:
    id: str
    node_id: str
    created_hlc: HLC
    idempotency_key: str
    tool: str
    args: dict[str, Any]
    subject: str | None
    consequence: Consequence
    side_effect: SideEffect
    decision_id: str
    deferral_record_id: str
    preconditions: list[dict[str, Any]]
    state: OutboxState
    expires_at_ms: int
    approval: Approval | None = None
    hold: dict[str, Any] | None = None
    attempts: int = 0
    last_error: str | None = None
    result_record_id: str | None = None
    re_decide_task_id: str | None = None
    superseded_by: str | None = None
    supersedes: list[str] = field(default_factory=list[str])

    def expired(self, now_ms: int) -> bool:
        return now_ms >= self.expires_at_ms

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL


@dataclass(frozen=True, slots=True)
class Transition:
    entry_id: str
    before: OutboxState
    after: OutboxState
    reason: str | None = None
    detail: dict[str, Any] | None = None

    def as_body(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "outbox_id": self.entry_id,
            "from_state": str(self.before),
            "to_state": str(self.after),
        }
        if self.reason:
            body["reason"] = self.reason
        if self.detail:
            body.update(self.detail)
        return body


class Outbox:
    """Stores deferred intents and drains them when their justification still holds."""

    def __init__(self, database: Database, node_id: str) -> None:
        self._db = database
        self.node_id = node_id
        self._db.connection.executescript(SCHEMA)

    # --- writing -------------------------------------------------------------

    def defer(
        self,
        *,
        contract: ToolContract,
        args: dict[str, Any],
        subject: str | None,
        decision_id: str,
        deferral_record_id: str,
        observations: list[Observation],
        created_hlc: HLC,
        now_ms: int,
        approval_required: bool,
        approval_reason: str | None,
        entry_id: str,
    ) -> Entry:
        """Record an intent, with the conditions that justified it.

        Rejects a duplicate by key before storing anything, which is the first of
        the two exactly-once guards: the second is the refusal to blindly retry an
        entry found mid-execution.
        """
        key = contract.key_for(args)
        existing = self.by_key(key)
        if existing is not None and not existing.terminal:
            raise DuplicateIntentError(key, existing.id)

        expires_at = now_ms + (contract.expiry_s or 0) * 1000
        entry = Entry(
            id=entry_id,
            node_id=self.node_id,
            created_hlc=created_hlc,
            idempotency_key=key,
            tool=contract.name,
            args=args,
            subject=subject,
            consequence=contract.consequence,
            side_effect=contract.side_effect,
            decision_id=decision_id,
            deferral_record_id=deferral_record_id,
            preconditions=[o.as_dict() for o in observations],
            state=OutboxState.AWAITING_APPROVAL if approval_required else OutboxState.READY,
            expires_at_ms=expires_at,
            approval=Approval(required=approval_required, reason=approval_reason),
        )
        self._insert(entry)
        return entry

    def _insert(self, entry: Entry) -> None:
        self._db.connection.execute(
            "INSERT INTO outbox (id, node_id, physical_ms, logical, idempotency_key, tool, args,"
            " subject, consequence, side_effect, decision_id, deferral_record_id, preconditions,"
            " state, approval, hold, expires_at_ms, attempts, last_error, result_record_id,"
            " re_decide_task_id, superseded_by, supersedes)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                entry.id,
                entry.node_id,
                entry.created_hlc.physical_ms,
                entry.created_hlc.logical,
                entry.idempotency_key,
                entry.tool,
                json.dumps(entry.args, sort_keys=True),
                entry.subject,
                str(entry.consequence),
                str(entry.side_effect),
                entry.decision_id,
                entry.deferral_record_id,
                json.dumps(entry.preconditions),
                str(entry.state),
                json.dumps(entry.approval.as_dict()) if entry.approval else None,
                json.dumps(entry.hold) if entry.hold else None,
                entry.expires_at_ms,
                entry.attempts,
                entry.last_error,
                entry.result_record_id,
                entry.re_decide_task_id,
                entry.superseded_by,
                json.dumps(entry.supersedes),
            ),
        )

    def _update(self, entry: Entry) -> None:
        self._db.connection.execute(
            "UPDATE outbox SET state = ?, approval = ?, hold = ?, attempts = ?, last_error = ?,"
            " result_record_id = ?, re_decide_task_id = ?, superseded_by = ?, supersedes = ?,"
            " preconditions = ? WHERE id = ?",
            (
                str(entry.state),
                json.dumps(entry.approval.as_dict()) if entry.approval else None,
                json.dumps(entry.hold) if entry.hold else None,
                entry.attempts,
                entry.last_error,
                entry.result_record_id,
                entry.re_decide_task_id,
                entry.superseded_by,
                json.dumps(entry.supersedes),
                json.dumps(entry.preconditions),
                entry.id,
            ),
        )

    def transition(
        self,
        entry: Entry,
        to: OutboxState,
        reason: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> Transition:
        before = entry.state
        entry.state = to
        self._update(entry)
        return Transition(entry.id, before, to, reason, detail)

    # --- reading -------------------------------------------------------------

    def get(self, entry_id: str) -> Entry | None:
        row = self._db.connection.execute(
            "SELECT * FROM outbox WHERE id = ?", (entry_id,)
        ).fetchone()
        return None if row is None else _row_to_entry(row)

    def by_key(self, key: str) -> Entry | None:
        row = self._db.connection.execute(
            "SELECT * FROM outbox WHERE node_id = ? AND idempotency_key = ?",
            (self.node_id, key),
        ).fetchone()
        return None if row is None else _row_to_entry(row)

    def all(self, state: OutboxState | None = None) -> list[Entry]:
        sql = "SELECT * FROM outbox"
        params: tuple[Any, ...] = ()
        if state is not None:
            sql += " WHERE state = ?"
            params = (str(state),)
        sql += " ORDER BY physical_ms ASC, logical ASC"
        return [_row_to_entry(row) for row in self._db.connection.execute(sql, params)]

    def drainable(self) -> list[Entry]:
        """Entries ready to fire, oldest first, so the order matches what was decided."""
        return self.all(OutboxState.READY)

    def tracing_to(self, decision_ids: set[str]) -> list[Entry]:
        """Entries whose reasoning came from one of these decisions."""
        return [e for e in self.all() if e.decision_id in decision_ids]

    # --- approval ------------------------------------------------------------

    def approve(self, entry: Entry, by: str, note: str | None = None) -> Transition:
        approval = entry.approval or Approval(required=True)
        approval.decision = "APPROVED"
        approval.by = by
        approval.note = note
        entry.approval = approval
        return self.transition(entry, OutboxState.READY, reason="APPROVED", detail={"by": by})

    def reject(self, entry: Entry, by: str, note: str | None = None) -> Transition:
        approval = entry.approval or Approval(required=True)
        approval.decision = "REJECTED"
        approval.by = by
        approval.note = note
        entry.approval = approval
        return self.transition(entry, OutboxState.CANCELLED, reason="REJECTED", detail={"by": by})


def _row_to_entry(row: Any) -> Entry:
    approval_raw = row["approval"]
    return Entry(
        id=row["id"],
        node_id=row["node_id"],
        created_hlc=HLC(int(row["physical_ms"]), int(row["logical"]), row["node_id"]),
        idempotency_key=row["idempotency_key"],
        tool=row["tool"],
        args=json.loads(row["args"]),
        subject=row["subject"],
        consequence=Consequence(row["consequence"]),
        side_effect=SideEffect(row["side_effect"]),
        decision_id=row["decision_id"],
        deferral_record_id=row["deferral_record_id"],
        preconditions=json.loads(row["preconditions"]),
        state=OutboxState(row["state"]),
        expires_at_ms=int(row["expires_at_ms"]),
        approval=Approval(**json.loads(approval_raw)) if approval_raw else None,
        hold=json.loads(row["hold"]) if row["hold"] else None,
        attempts=int(row["attempts"]),
        last_error=row["last_error"],
        result_record_id=row["result_record_id"],
        re_decide_task_id=row["re_decide_task_id"],
        superseded_by=row["superseded_by"],
        supersedes=json.loads(row["supersedes"]) if row["supersedes"] else [],
    )
