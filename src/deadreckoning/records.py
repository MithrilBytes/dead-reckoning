# SPDX-License-Identifier: Apache-2.0
"""The append only, hash chained log.

A record is the atom of what a node did. Records are never edited and never
deleted: a later record supersedes an earlier one, and both stay readable. That is
what makes a decision contestable months afterwards, and it is why the store below
exposes no update path and the database refuses one.

Each record carries the hash of the previous record from the same node, so a chain
per node runs from a genesis record to the present. Altering any record in the
middle changes its digest and orphans everything after it, which `verify_chain`
reports.
"""

from __future__ import annotations

import secrets
from collections.abc import Iterator, Mapping
from enum import StrEnum
from typing import Any, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from deadreckoning.canonical import (
    DEFAULT_REDACT_KEYS,
    GENESIS_PREV_HASH,
    canonical_json,
    redact,
    sha256_hex,
)
from deadreckoning.clock import HLC
from deadreckoning.runtime import Database


class RecordKind(StrEnum):
    MODE_CHANGE = "MODE_CHANGE"
    HEALTH_CHANGE = "HEALTH_CHANGE"
    MODEL_TURN = "MODEL_TURN"
    TOOL_CALL = "TOOL_CALL"
    TOOL_RESULT = "TOOL_RESULT"
    DEFERRAL = "DEFERRAL"
    UNAVAILABLE = "UNAVAILABLE"
    FINAL = "FINAL"
    ABSTENTION = "ABSTENTION"
    ESCALATION = "ESCALATION"
    OUTBOX_STATE = "OUTBOX_STATE"
    REVIEW = "REVIEW"
    CONFLICT = "CONFLICT"
    RESOLUTION = "RESOLUTION"
    IDENTITY_CHANGE = "IDENTITY_CHANGE"
    TIME_TRUST_CHANGE = "TIME_TRUST_CHANGE"
    SYNC = "SYNC"
    CHECKPOINT = "CHECKPOINT"
    PROVISION = "PROVISION"
    RESOURCE_CHANGE = "RESOURCE_CHANGE"
    NODE_INIT = "NODE_INIT"


class Mode(StrEnum):
    CONNECTED = "CONNECTED"
    DEGRADED = "DEGRADED"
    ISLANDED = "ISLANDED"
    RECONNECTING = "RECONNECTING"


class TimeTrust(StrEnum):
    TRUSTED = "TRUSTED"
    DRIFTING = "DRIFTING"
    UNTRUSTED = "UNTRUSTED"


class IdentityState(StrEnum):
    FRESH = "FRESH"
    CACHED = "CACHED"
    STALE = "STALE"
    NONE = "NONE"


SUBJECT_BEARING_KINDS = frozenset({RecordKind.FINAL, RecordKind.ABSTENTION})
"""Kinds that must name a subject and an outcome key to be eligible for conflict
detection later. A record without them cannot be compared against a peer's, so it
would silently never conflict."""


class Record(BaseModel):
    """One entry in a node's chain.

    Field presence is load bearing. A field left unset is omitted from the
    canonical form entirely rather than written as null, because the wire format
    types several of them without a null variant, and because two implementations
    that disagree about absent versus null would produce different digests for the
    same record.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    node_id: str
    hlc: dict[str, int | str]
    wall_time: str
    time_trust: TimeTrust
    kind: RecordKind
    mode: Mode
    observed: dict[str, dict[str, int | str]] = Field(default_factory=dict)
    supersedes: list[str] = Field(default_factory=list)
    prev_hash: str
    hash: str = ""
    body: dict[str, Any] = Field(default_factory=dict)

    task_id: str | None = None
    step: int | None = None
    subject: str | None = None
    signature: str | None = None

    tier: dict[str, Any] | None = None
    tiers_available: list[str] | None = None
    fallback_from: str | None = None
    params: dict[str, Any] | None = None
    manifest_hash: str | None = None
    prompt_template_hash: str | None = None
    inputs_hash: str | None = None
    identity_state: IdentityState | None = None
    review_required: bool | None = None
    review_reason: str | None = None

    @model_validator(mode="after")
    def _check_prev_hash(self) -> Self:
        if len(self.prev_hash) != 64:
            raise ValueError(f"prev_hash must be 64 hex characters, got {self.prev_hash!r}")
        return self

    def unhashed_payload(self) -> dict[str, Any]:
        """The record as it is hashed: every set field except hash and signature."""
        payload = self.model_dump(mode="json", exclude_none=True)
        payload.pop("hash", None)
        payload.pop("signature", None)
        return payload

    def compute_hash(self) -> str:
        return sha256_hex(canonical_json(self.unhashed_payload()))

    def stored_payload(self) -> dict[str, Any]:
        """The full record as persisted and synced, hash included."""
        payload = self.model_dump(mode="json", exclude_none=True)
        payload["hash"] = self.hash
        return payload

    def missing_conflict_fields(self) -> list[str]:
        """Which fields this record would need to take part in conflict detection.

        Empty for every kind that is not compared across nodes.
        """
        if self.kind not in SUBJECT_BEARING_KINDS:
            return []
        missing: list[str] = []
        if not self.subject:
            missing.append("subject")
        outcome: object = self.body.get("outcome")
        if not isinstance(outcome, dict) or not cast(Mapping[str, Any], outcome).get("key"):
            missing.append("body.outcome.key")
        return missing


def new_record_id(hlc: HLC) -> str:
    """Globally unique and sorted by construction.

    The random suffix is what makes two records stamped in the same millisecond on
    the same node distinguishable if the logical counter is ever reset by a bug.
    """
    return f"{hlc.physical_ms}-{hlc.logical}-{hlc.node_id}-{secrets.token_hex(4)}"


class ChainBreak(BaseModel):
    """Where a node's chain stops verifying, and why."""

    node_id: str
    record_id: str
    position: int
    reason: str
    expected: str
    found: str


class RecordStore:
    """Append and read. There is deliberately no update or delete method.

    Removing the capability from the type is the first line of defence; the
    database triggers are the second, for anything reaching past this class.
    """

    def __init__(
        self, database: Database, redact_keys: frozenset[str] = DEFAULT_REDACT_KEYS
    ) -> None:
        self._db = database
        self._redact_keys = redact_keys

    def append(self, record: Record) -> Record:
        """Redact, hash, and persist. Returns the record as it was stored.

        Redaction runs before hashing so the digest covers what is kept, not what
        was collected. Records travel to peers and to the hub; a secret that
        reached this point must not leave with them.
        """
        safe_body = redact(record.body, self._redact_keys)
        sealed = record.model_copy(update={"body": safe_body, "hash": ""})
        sealed = sealed.model_copy(update={"hash": sealed.compute_hash()})

        payload = canonical_json(sealed.stored_payload()).decode("utf-8")
        self._db.connection.execute(
            "INSERT INTO records"
            " (id, node_id, physical_ms, logical, kind, subject, task_id, prev_hash, hash, payload)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                sealed.id,
                sealed.node_id,
                int(sealed.hlc["physical_ms"]),
                int(sealed.hlc["logical"]),
                str(sealed.kind),
                sealed.subject,
                sealed.task_id,
                sealed.prev_hash,
                sealed.hash,
                payload,
            ),
        )
        return sealed

    def head(self, node_id: str) -> Record | None:
        """The latest record on a node's chain, or None if it has none."""
        row = self._db.connection.execute(
            "SELECT payload FROM records WHERE node_id = ?"
            " ORDER BY physical_ms DESC, logical DESC LIMIT 1",
            (node_id,),
        ).fetchone()
        return Record.model_validate_json(row["payload"]) if row else None

    def prev_hash_for(self, node_id: str) -> str:
        head = self.head(node_id)
        return head.hash if head else GENESIS_PREV_HASH

    def get(self, record_id: str) -> Record | None:
        row = self._db.connection.execute(
            "SELECT payload FROM records WHERE id = ?", (record_id,)
        ).fetchone()
        return Record.model_validate_json(row["payload"]) if row else None

    def node_ids(self) -> list[str]:
        rows = self._db.connection.execute(
            "SELECT DISTINCT node_id FROM records ORDER BY node_id"
        ).fetchall()
        return [row["node_id"] for row in rows]

    def count(self) -> int:
        return int(self._db.connection.execute("SELECT COUNT(*) AS n FROM records").fetchone()["n"])

    def iter_records(
        self,
        node_id: str | None = None,
        kind: RecordKind | None = None,
        subject: str | None = None,
        task_id: str | None = None,
        limit: int | None = None,
    ) -> Iterator[Record]:
        """Chain order within a node, then across nodes by stamp."""
        clauses: list[str] = []
        params: list[object] = []
        for column, value in (
            ("node_id", node_id),
            ("kind", str(kind) if kind else None),
            ("subject", subject),
            ("task_id", task_id),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        sql = "SELECT payload FROM records"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY physical_ms ASC, logical ASC, node_id ASC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        for row in self._db.connection.execute(sql, params):
            yield Record.model_validate_json(row["payload"])

    def verify_chain(self, node_id: str) -> ChainBreak | None:
        """Walk one node's chain and report the first break, or None if intact.

        Two things can be wrong: a record's stored digest no longer matches its
        content, which means it was altered, or its prev_hash does not match the
        digest of the record before it, which means one was removed or inserted.
        """
        expected_prev = GENESIS_PREV_HASH
        for position, record in enumerate(self.iter_records(node_id=node_id)):
            recomputed = record.compute_hash()
            if recomputed != record.hash:
                return ChainBreak(
                    node_id=node_id,
                    record_id=record.id,
                    position=position,
                    reason="content does not match its hash",
                    expected=record.hash,
                    found=recomputed,
                )
            if record.prev_hash != expected_prev:
                return ChainBreak(
                    node_id=node_id,
                    record_id=record.id,
                    position=position,
                    reason="prev_hash does not match the preceding record",
                    expected=expected_prev,
                    found=record.prev_hash,
                )
            expected_prev = record.hash
        return None

    def verify_all(self) -> dict[str, ChainBreak | None]:
        return {node_id: self.verify_chain(node_id) for node_id in self.node_ids()}
