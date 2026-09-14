# SPDX-License-Identifier: Apache-2.0
"""Two nodes and a hub, over a real socket on loopback.

The hub is the dullest thing in the system and that is the design. It stores every
record it is given and decides nothing, so an attacker who takes it gets a copy of
the logs and no ability to make a truck do anything.

These run against a real HTTP server rather than an in-process fake, because the
things most likely to be wrong in a sync layer are the ones a fake would paper
over: a batch refused with the wrong status, a chain check that only ran locally,
a token that was never actually required.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from hub.server import Hub, serve

from deadreckoning.canonical import GENESIS_PREV_HASH
from deadreckoning.clock import TimeTrust
from deadreckoning.config import Config
from deadreckoning.node import Node
from deadreckoning.records import Mode, Record, RecordKind
from deadreckoning.sync.client import SyncClient
from deadreckoning.sync.conflicts import undetected


@pytest.fixture
def hub(tmp_path: Path) -> Iterator[tuple[str, Hub]]:
    server, instance = serve(tmp_path / "hub", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        yield f"http://127.0.0.1:{port}", instance
    finally:
        server.shutdown()
        instance.close()


@pytest.fixture
def truck12(tmp_path: Path, config: Config) -> Iterator[Node]:
    other = config.model_copy(
        update={"node": config.node.model_copy(update={"node_id": "truck-12"})}
    )
    with Node.open(other, tmp_path / "peer") as opened:
        yield opened


def decide(node: Node, value: str) -> Record:
    return node.emit(
        RecordKind.FINAL,
        body={
            "outcome": {"key": "priority", "value": value},
            "rationale": "ticket text only" if value == "P2" else "field report: live conductor",
            "evidence": [],
            "depends_on": [],
            "confidence": 0.6,
            "tool_call_ids": [],
        },
        subject="ticket:T-104",
        tier={"name": "local-q4", "rank": 2, "kind": "local", "model": "q4"},
        review_required=True,
        manifest_hash="a" * 64,
    )


def test_the_hub_starts_empty_and_says_so(hub: tuple[str, Hub]) -> None:
    url, _ = hub
    assert httpx.get(f"{url}/sync/head", timeout=5).json() == {"heads": {}}


def test_a_node_pushes_and_the_hub_holds_it(hub: tuple[str, Hub], node: Node) -> None:
    url, instance = hub
    decide(node, "P2")
    client = SyncClient(url, peer_name="hq-hub")
    outcome = client.push(list(node.store.iter_records()), client.head())
    assert outcome.pushed == 1
    assert outcome.error is None

    held = list(instance.store.iter_records())
    assert {r.node_id for r in held} == {"truck-7"}
    assert instance.store.verify_chain("truck-7") is None


def test_pushing_twice_adds_nothing_the_second_time(hub: tuple[str, Hub], node: Node) -> None:
    url, instance = hub
    decide(node, "P2")
    client = SyncClient(url, peer_name="hq-hub")
    client.push(list(node.store.iter_records()), client.head())
    before = instance.store.count()
    second = client.push(list(node.store.iter_records()), client.head())
    assert second.pushed == 0
    assert instance.store.count() == before


def test_a_broken_chain_is_refused_with_409(hub: tuple[str, Hub]) -> None:
    """Half a batch would leave a hole every later verification reports."""
    url, instance = hub
    orphan = Record(
        id="orphan",
        node_id="truck-9",
        hlc={"physical_ms": 5, "logical": 0, "node_id": "truck-9"},
        wall_time="2026-01-01T00:00:00+00:00",
        time_trust=TimeTrust.UNTRUSTED,
        kind=RecordKind.CHECKPOINT,
        mode=Mode.ISLANDED,
        prev_hash="f" * 64,
        body={"phase": "START"},
    )
    orphan = orphan.model_copy(update={"hash": orphan.compute_hash()})
    response = httpx.post(f"{url}/sync/records", json=[orphan.stored_payload()], timeout=5)
    assert response.status_code == 409
    assert response.json()["error"] == "CHAIN_BREAK"
    assert instance.store.count() == 0, "nothing was kept"


def test_a_tampered_record_is_refused_by_the_hub(hub: tuple[str, Hub], node: Node) -> None:
    """The receiver recomputes rather than trusting the digest it was handed."""
    url, instance = hub
    decide(node, "P2")
    records = [r.stored_payload() for r in node.store.iter_records()]
    records[-1]["body"]["rationale"] = "edited in transit"
    response = httpx.post(f"{url}/sync/records", json=records, timeout=5)
    assert response.status_code == 409
    assert instance.store.count() == 0


def test_two_trucks_meet_through_the_hub_and_the_conflict_surfaces(
    hub: tuple[str, Hub], node: Node, truck12: Node
) -> None:
    """Two trucks disagree, sync through a real hub, and find one conflict."""
    url, _ = hub
    decide(node, "P2")
    decide(truck12, "P1")

    for truck in (node, truck12):
        client = SyncClient(url, peer_name="hq-hub")
        client.push(list(truck.store.iter_records()), client.head())

    for truck in (node, truck12):
        client = SyncClient(url, peer_name="hq-hub")
        held = list(truck.store.iter_records())
        pulled = client.pull(held, sorted(set(client.head()) | {truck.node_id}))
        for record in pulled.accepted:
            truck.store.append(record)

    for truck in (node, truck12):
        assert truck.store.verify_chain("truck-7") is None
        assert truck.store.verify_chain("truck-12") is None
        found = undetected(list(truck.store.iter_records()), truck.node_id)
        assert len(found) == 1, f"{truck.node_id} should see exactly one conflict"

    seven = undetected(list(node.store.iter_records()), "truck-7")[0]
    twelve = undetected(list(truck12.store.iter_records()), "truck-12")[0]
    assert seven.dedupe_key == twelve.dedupe_key, "one conflict, two detectors"


def test_the_hub_holds_the_union_and_nothing_else(
    hub: tuple[str, Hub], node: Node, truck12: Node
) -> None:
    url, instance = hub
    decide(node, "P2")
    decide(truck12, "P1")
    for truck in (node, truck12):
        client = SyncClient(url, peer_name="hq-hub")
        client.push(list(truck.store.iter_records()), client.head())

    held = {r.id for r in instance.store.iter_records()}
    union = {r.id for r in node.store.iter_records()} | {r.id for r in truck12.store.iter_records()}
    assert held == union


def test_a_token_is_actually_required_when_one_is_set(tmp_path: Path) -> None:
    server, instance = serve(tmp_path / "guarded", port=0, token="secret")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        url = f"http://127.0.0.1:{port}"
        assert httpx.get(f"{url}/sync/head", timeout=5).status_code == 401
        ok = httpx.get(f"{url}/sync/head", headers={"Authorization": "Bearer secret"}, timeout=5)
        assert ok.status_code == 200
    finally:
        server.shutdown()
        instance.close()


def test_the_hub_never_writes_a_record_of_its_own(hub: tuple[str, Hub], node: Node) -> None:
    """It stores and forwards. It does not decide, and it does not sign anything."""
    url, instance = hub
    decide(node, "P2")
    client = SyncClient(url, peer_name="hq-hub")
    client.push(list(node.store.iter_records()), client.head())
    held = list(instance.store.iter_records())
    assert all(r.node_id != "hub" for r in held)
    assert {r.node_id for r in held} == {"truck-7"}


def test_an_unknown_route_is_a_404_not_a_silence(hub: tuple[str, Hub]) -> None:
    url, _ = hub
    assert httpx.get(f"{url}/sync/everything", timeout=5).status_code == 404


def test_a_chain_arriving_at_the_hub_starts_where_it_started_at_home(
    hub: tuple[str, Hub], node: Node
) -> None:
    """The chain is the record, not a local artefact of the store it came from."""
    url, instance = hub
    decide(node, "P2")
    client = SyncClient(url, peer_name="hq-hub")
    client.push(list(node.store.iter_records()), client.head())
    first = next(iter(instance.store.iter_records(node_id="truck-7")))
    assert first.prev_hash == GENESIS_PREV_HASH
    assert instance.store.verify_chain("truck-7") is None
