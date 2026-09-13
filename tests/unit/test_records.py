# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import sqlite3

import pytest

from deadreckoning.canonical import GENESIS_PREV_HASH
from deadreckoning.node import Node
from deadreckoning.records import Mode, Record, RecordKind, RecordStore, TimeTrust


def test_genesis_starts_the_chain_at_sixty_four_zeros(node: Node) -> None:
    first = node.emit(RecordKind.CHECKPOINT, body={"event": "genesis"})
    assert first.prev_hash == GENESIS_PREV_HASH


def test_each_record_links_to_the_one_before_it(node: Node) -> None:
    a = node.emit(RecordKind.CHECKPOINT, body={"event": "genesis"})
    b = node.emit(RecordKind.MODE_CHANGE, body={"to_mode": "ISLANDED"})
    c = node.emit(RecordKind.SYNC, body={"peer": "hq-hub"})
    assert b.prev_hash == a.hash
    assert c.prev_hash == b.hash
    assert node.store.verify_chain(node.node_id) is None


def test_hash_covers_content_and_recomputes(node: Node) -> None:
    record = node.emit(RecordKind.CHECKPOINT, body={"event": "genesis"})
    assert record.compute_hash() == record.hash
    assert node.store.get(record.id) == record


def test_the_store_offers_no_way_to_change_a_record() -> None:
    forbidden = {"update", "delete", "remove", "edit", "set", "replace"}
    assert not forbidden & set(dir(RecordStore))


def test_the_database_refuses_updates_and_deletes(node: Node) -> None:
    node.emit(RecordKind.CHECKPOINT, body={"event": "genesis"})
    for statement in ("UPDATE records SET kind = 'FINAL'", "DELETE FROM records"):
        with pytest.raises(sqlite3.IntegrityError, match="append only"):
            node.database.connection.execute(statement)


def test_verify_detects_a_tampered_record(node: Node) -> None:
    node.emit(RecordKind.CHECKPOINT, body={"event": "genesis"})
    target = node.emit(RecordKind.MODE_CHANGE, body={"to_mode": "ISLANDED"})
    node.emit(RecordKind.SYNC, body={"peer": "hq-hub"})

    _drop_immutability_trigger(node)
    row = node.database.connection.execute(
        "SELECT payload FROM records WHERE id = ?", (target.id,)
    ).fetchone()
    payload = json.loads(row["payload"])
    payload["body"]["to_mode"] = "CONNECTED"
    node.database.connection.execute(
        "UPDATE records SET payload = ? WHERE id = ?",
        (json.dumps(payload, sort_keys=True, separators=(",", ":")), target.id),
    )

    break_ = node.store.verify_chain(node.node_id)
    assert break_ is not None
    assert break_.record_id == target.id
    assert break_.position == 1
    assert "hash" in break_.reason


def test_verify_detects_a_removed_record(node: Node) -> None:
    node.emit(RecordKind.CHECKPOINT, body={"event": "genesis"})
    middle = node.emit(RecordKind.MODE_CHANGE, body={"to_mode": "ISLANDED"})
    node.emit(RecordKind.SYNC, body={"peer": "hq-hub"})

    _drop_permanence_trigger(node)
    node.database.connection.execute("DELETE FROM records WHERE id = ?", (middle.id,))

    break_ = node.store.verify_chain(node.node_id)
    assert break_ is not None
    assert "prev_hash" in break_.reason


def test_secrets_never_reach_the_record_or_its_hash(node: Node) -> None:
    record = node.emit(
        RecordKind.SYNC,
        body={"authorization": "Bearer super-secret", "api_key_env": "DR_HUB_TOKEN"},
    )
    assert record.body["authorization"] == "[REDACTED]"
    assert record.body["api_key_env"] == "DR_HUB_TOKEN"
    stored = json.dumps(node.store.get(record.id).stored_payload())  # pyright: ignore[reportOptionalMemberAccess]
    assert "super-secret" not in stored


def test_unset_fields_are_absent_rather_than_null(node: Node) -> None:
    record = node.emit(RecordKind.CHECKPOINT, body={"event": "genesis"})
    payload = record.unhashed_payload()
    for absent in ("tier", "params", "subject", "task_id", "review_reason"):
        assert absent not in payload
    assert "hash" not in payload
    assert "signature" not in payload


def test_a_final_without_a_subject_reports_what_it_is_missing() -> None:
    record = Record(
        id="1-0-n-aa",
        node_id="n",
        hlc={"physical_ms": 1, "logical": 0, "node_id": "n"},
        wall_time="2026-01-01T00:00:00+00:00",
        time_trust=TimeTrust.UNTRUSTED,
        kind=RecordKind.FINAL,
        mode=Mode.ISLANDED,
        prev_hash=GENESIS_PREV_HASH,
        body={"rationale": "no subject, no outcome key"},
    )
    assert record.missing_conflict_fields() == ["subject", "body.outcome.key"]


def test_a_checkpoint_needs_no_subject() -> None:
    record = Record(
        id="1-0-n-aa",
        node_id="n",
        hlc={"physical_ms": 1, "logical": 0, "node_id": "n"},
        wall_time="2026-01-01T00:00:00+00:00",
        time_trust=TimeTrust.UNTRUSTED,
        kind=RecordKind.CHECKPOINT,
        mode=Mode.ISLANDED,
        prev_hash=GENESIS_PREV_HASH,
        body={},
    )
    assert record.missing_conflict_fields() == []


def test_the_observed_vector_carries_this_node(node: Node) -> None:
    record = node.emit(RecordKind.CHECKPOINT, body={"event": "genesis"})
    assert node.node_id in record.observed


def _drop_immutability_trigger(node: Node) -> None:
    node.database.connection.execute("DROP TRIGGER records_are_immutable")


def _drop_permanence_trigger(node: Node) -> None:
    node.database.connection.execute("DROP TRIGGER records_are_permanent")
