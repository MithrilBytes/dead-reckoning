# SPDX-License-Identifier: Apache-2.0
"""Conflict detection, as properties rather than examples.

Detection has to be a pure function over a set of records: idempotent, symmetric,
and independent of the order records arrived in. Those are not stylistic
preferences. Two nodes reconnect in whatever order the weather allows, merge in
whatever order the hub serves, and must still reach the same conclusion about what
happened, or the adjudication a human is asked for depends on which truck got
signal first.
"""

from __future__ import annotations

import itertools
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from deadreckoning.canonical import GENESIS_PREV_HASH
from deadreckoning.clock import TimeTrust
from deadreckoning.records import Mode, Record, RecordKind
from deadreckoning.sync.conflicts import concurrent, dedupe_key, detect, undetected


def decision(
    record_id: str,
    node: str,
    physical: int,
    subject: str = "ticket:T-104",
    key: str = "priority",
    value: str = "P2",
    observed: dict[str, Any] | None = None,
    kind: RecordKind = RecordKind.FINAL,
) -> Record:
    body: dict[str, Any] = (
        {"effective_outcome": {"key": key, "value": value}, "conflict_ids": [], "dedupe_keys": []}
        if kind is RecordKind.RESOLUTION
        else {
            "outcome": {"key": key, "value": value},
            "rationale": "",
            "evidence": [],
            "depends_on": [],
            "confidence": 0.5,
            "tool_call_ids": [],
        }
    )
    return Record(
        id=record_id,
        node_id=node,
        hlc={"physical_ms": physical, "logical": 0, "node_id": node},
        wall_time="2026-01-01T00:00:00+00:00",
        time_trust=TimeTrust.UNTRUSTED,
        kind=kind,
        mode=Mode.ISLANDED,
        subject=subject,
        observed=observed or {node: {"physical_ms": physical, "logical": 0, "node_id": node}},
        prev_hash=GENESIS_PREV_HASH,
        body=body,
    )


def a_and_b() -> list[Record]:
    """The scenario: two trucks, same ticket, opposite calls, neither aware."""
    return [
        decision("a", "truck-7", 100, value="P2"),
        decision("b", "truck-12", 110, value="P1"),
    ]


def test_two_isolated_nodes_disagreeing_is_a_conflict() -> None:
    conflicts = detect(a_and_b())
    assert len(conflicts) == 1
    assert conflicts[0].subject == "ticket:T-104"
    assert conflicts[0].outcome_key == "priority"
    assert set(conflicts[0].record_ids) == {"a", "b"}
    assert set(conflicts[0].values.values()) == {"P1", "P2"}


def test_agreeing_is_not_a_conflict() -> None:
    records = [
        decision("a", "truck-7", 100, value="P1"),
        decision("b", "truck-12", 110, value="P1"),
    ]
    assert detect(records) == []


def test_a_different_subject_is_not_a_conflict() -> None:
    records = [
        decision("a", "truck-7", 100, subject="ticket:T-104", value="P2"),
        decision("b", "truck-12", 110, subject="ticket:T-999", value="P1"),
    ]
    assert detect(records) == []


def test_a_different_outcome_key_is_not_a_conflict() -> None:
    records = [
        decision("a", "truck-7", 100, key="priority", value="P2"),
        decision("b", "truck-12", 110, key="dispatch", value="P1"),
    ]
    assert detect(records) == []


def test_deciding_after_seeing_the_other_supersedes_rather_than_conflicts() -> None:
    """Disagreeing on purpose with more information is not a conflict.

    truck-12 had already merged truck-7's decision when it decided. It knew, and
    chose otherwise, which is a later decision rather than a contradiction.
    """
    records = [
        decision("a", "truck-7", 100, value="P2"),
        decision(
            "b",
            "truck-12",
            200,
            value="P1",
            observed={
                "truck-12": {"physical_ms": 200, "logical": 0, "node_id": "truck-12"},
                "truck-7": {"physical_ms": 100, "logical": 0, "node_id": "truck-7"},
            },
        ),
    ]
    assert detect(records) == []
    assert not concurrent(records[0], records[1])


def test_seeing_an_earlier_record_from_a_node_is_not_seeing_this_one() -> None:
    """The vector has to cover the specific record, not merely the node."""
    records = [
        decision("a", "truck-7", 300, value="P2"),
        decision(
            "b",
            "truck-12",
            310,
            value="P1",
            observed={
                "truck-12": {"physical_ms": 310, "logical": 0, "node_id": "truck-12"},
                "truck-7": {"physical_ms": 100, "logical": 0, "node_id": "truck-7"},
            },
        ),
    ]
    assert len(detect(records)) == 1


def test_one_node_cannot_conflict_with_itself() -> None:
    """Steps on one node are serialised per subject; a later one supersedes."""
    records = [
        decision("a", "truck-7", 100, value="P2"),
        decision("b", "truck-7", 200, value="P1"),
    ]
    assert detect(records) == []


def test_a_resolution_settles_the_conflict_for_good() -> None:
    """Re-detecting an adjudicated conflict would mean asking a human the same
    question forever."""
    records = a_and_b()
    resolution = decision("r", "truck-7", 300, value="P1", kind=RecordKind.RESOLUTION).model_copy(
        update={"supersedes": ["a", "b"]}
    )
    assert detect([*records, resolution]) == []


def test_a_conflict_has_one_identity_across_nodes() -> None:
    assert dedupe_key(["a", "b"]) == dedupe_key(["b", "a"])
    assert dedupe_key(["a", "b"]) != dedupe_key(["a", "c"])


def test_each_node_records_the_conflict_once() -> None:
    """Detection is local, so both nodes write their own record. The dedupe key is
    what makes them one conflict rather than two."""
    records = a_and_b()
    assert len(undetected(records, "truck-7")) == 1

    mine = Record(
        id="conf-7",
        node_id="truck-7",
        hlc={"physical_ms": 400, "logical": 0, "node_id": "truck-7"},
        wall_time="2026-01-01T00:00:00+00:00",
        time_trust=TimeTrust.UNTRUSTED,
        kind=RecordKind.CONFLICT,
        mode=Mode.RECONNECTING,
        subject="ticket:T-104",
        prev_hash=GENESIS_PREV_HASH,
        body={
            "subtype": "CONCURRENT_DECISION",
            "subject": "ticket:T-104",
            "outcome_key": "priority",
            "record_ids": ["a", "b"],
            "dedupe_key": dedupe_key(["a", "b"]),
            "held_outbox_ids": [],
        },
    )
    assert undetected([*records, mine], "truck-7") == []
    assert len(undetected([*records, mine], "truck-12")) == 1, (
        "the peer has not recorded it yet and still should"
    )


# --- the properties S6 actually requires --------------------------------------

values = st.sampled_from(["P1", "P2", "P3"])
nodes = st.sampled_from(["truck-7", "truck-12", "hq"])
subjects = st.sampled_from(["ticket:T-104", "ticket:T-101"])

record_sets = st.lists(
    st.tuples(nodes, st.integers(1, 500), subjects, values),
    min_size=0,
    max_size=7,
).map(
    lambda rows: [
        decision(f"r{i}", node, physical, subject, "priority", value)
        for i, (node, physical, subject, value) in enumerate(rows)
    ]
)


@settings(max_examples=200)
@given(record_sets)
def test_detection_is_idempotent(records: list[Record]) -> None:
    assert detect(records) == detect(records)


@settings(max_examples=200)
@given(record_sets)
def test_detection_does_not_depend_on_merge_order(records: list[Record]) -> None:
    """Two nodes merge in whatever order they reconnect and must agree on what
    happened."""
    forwards = detect(records)
    backwards = detect(list(reversed(records)))
    assert {c.dedupe_key for c in forwards} == {c.dedupe_key for c in backwards}


@settings(max_examples=100)
@given(record_sets)
def test_detection_is_symmetric_in_the_pair(records: list[Record]) -> None:
    for conflict in detect(records):
        assert conflict.record_ids == tuple(sorted(conflict.record_ids))


@settings(max_examples=100)
@given(record_sets)
def test_every_permutation_gives_the_same_conflicts(records: list[Record]) -> None:
    if len(records) > 5:
        return
    keys = {
        frozenset(c.dedupe_key for c in detect(list(order)))
        for order in itertools.permutations(records)
    }
    assert len(keys) <= 1


@settings(max_examples=100)
@given(record_sets)
def test_concurrency_is_symmetric(records: list[Record]) -> None:
    for a, b in itertools.combinations(records, 2):
        assert concurrent(a, b) == concurrent(b, a)


@settings(max_examples=100)
@given(record_sets)
def test_a_record_is_never_in_conflict_with_itself(records: list[Record]) -> None:
    for conflict in detect(records):
        assert conflict.record_ids[0] != conflict.record_ids[1]
