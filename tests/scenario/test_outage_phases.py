# SPDX-License-Identifier: Apache-2.0
"""Phases 1 and 2 of the reference scenario, headless and deterministic.

Phase 1 is a normal day: everything reachable, decisions at full fidelity. Phase 2
is the interesting one. SCADA goes, and two tickets arrive. The first cannot be
decided without live load, so the agent abstains and says what it would have
concluded either way. The second can be decided without it, and is.

That pair is the whole argument. A system that answered both would be lying about
one; a system that abstained on both would be useless. Asserting on record kinds
and fields rather than on prose, because prose varies and structure does not.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from demo.tools_outage import Backends, build_registry

from deadreckoning.agent import AgentLoop, Task
from deadreckoning.canonical import content_hash
from deadreckoning.clock import TimeTrust
from deadreckoning.health import FailureClass, HealthState, ObservationSource
from deadreckoning.llm.scripted import ScriptedModel, abstain, final, tool_call
from deadreckoning.local_store import LocalStore
from deadreckoning.manifest import TierView, build_manifest
from deadreckoning.node import Node
from deadreckoning.records import IdentityState, Mode, Record, RecordKind
from deadreckoning.tools.enforcer import ToolEnforcer

NOW = 1_700_000_000_000
PROMPT = Path("docs/prompts/planner_v1.md").read_text()
BACKEND_NAMES: tuple[str, ...] = (
    "gis-api",
    "ticket-api",
    "scada-api",
    "dispatch-api",
    "notify-api",
    "weather-api",
)


def all_healthy(**overrides: HealthState) -> dict[str, HealthState]:
    return {**dict.fromkeys(BACKEND_NAMES, HealthState.HEALTHY), **overrides}


@pytest.fixture
def world(node: Node) -> tuple[Node, Backends, LocalStore, Any]:
    backends = Backends()
    return node, backends, LocalStore(node.database), build_registry(backends)


def run_task(
    node: Node,
    registry: Any,
    store: LocalStore,
    script: dict[tuple[str, int], str],
    health: dict[str, HealthState],
    task: Task,
    mode: Mode,
) -> tuple[Any, list[Record]]:
    emitted: list[Record] = []

    def emit(kind: RecordKind, body: dict[str, Any], **fields: Any) -> Record:
        record = node.emit(kind, body=body, **fields)
        emitted.append(record)
        return record

    manifest = build_manifest(
        mode=mode,
        tier=TierView("frontier", 0, "scripted-frontier"),
        identity=IdentityState.FRESH,
        identity_ttl_s=3600,
        time_trust=TimeTrust.TRUSTED,
        registry=registry,
        store=store,
        health=health,
        now_ms=NOW,
    )
    loop = AgentLoop(
        client=ScriptedModel(script),
        enforcer=ToolEnforcer(registry, store, health, NOW),
        emit=emit,
        prompt_template=PROMPT,
        prompt_template_hash=content_hash(PROMPT),
    )
    return loop.run(task, manifest), emitted


def body_of(records: list[Record], kind: RecordKind) -> dict[str, Any]:
    return next(r for r in records if r.kind is kind).body


# --- Phase 1: connected, full fidelity ---------------------------------------


def test_phase_one_decides_at_full_fidelity_with_live_tools(
    world: tuple[Node, Backends, LocalStore, Any],
) -> None:
    node, _backends, store, registry = world
    health = all_healthy()
    script = {
        ("T-101", 1): tool_call("lookup_asset", asset_id="F-31"),
        ("T-101", 2): tool_call("get_live_load", feeder_id="F-31"),
        ("T-101", 3): final(
            "ticket:T-101",
            "priority",
            "P1",
            "hospital on 8h backup, feeder loaded",
            depends_on=["lookup_asset", "get_live_load"],
            evidence=[{"kind": "tool_result", "ref": "lookup_asset"}],
        ),
    }
    outcome, emitted = run_task(
        node,
        registry,
        store,
        script,
        health,
        Task("T-101", "triage", "Triage T-101.", subject="ticket:T-101"),
        Mode.CONNECTED,
    )

    assert outcome.kind is RecordKind.FINAL
    assert body_of(emitted, RecordKind.FINAL)["outcome"] == {"key": "priority", "value": "P1"}
    results = [r for r in emitted if r.kind is RecordKind.TOOL_RESULT]
    assert [r.body["tool"] for r in results] == ["lookup_asset", "get_live_load"]
    assert {r.body["availability"] for r in results} == {"LIVE"}
    assert not [r for r in emitted if r.kind is RecordKind.DEFERRAL], (
        "nothing is queued while connected"
    )


def test_phase_one_executes_a_side_effect_live(
    world: tuple[Node, Backends, LocalStore, Any],
) -> None:
    node, backends, store, registry = world
    health = all_healthy()
    script = {
        ("T-105", 1): tool_call(
            "notify_customer", customer_id="C-RES-105", ticket_id="T-105", template="outage_ack"
        ),
        ("T-105", 2): final("ticket:T-105", "priority", "P3", "single residence"),
    }
    _outcome, emitted = run_task(
        node,
        registry,
        store,
        script,
        health,
        Task("T-105", "triage", "Triage T-105.", subject="ticket:T-105"),
        Mode.CONNECTED,
    )
    assert body_of(emitted, RecordKind.TOOL_RESULT)["availability"] == "LIVE"
    assert backends.executed == [
        (
            "notify_customer",
            {"customer_id": "C-RES-105", "ticket_id": "T-105", "template": "outage_ack"},
        )
    ]


# --- Phase 2: SCADA is gone ---------------------------------------------------


def test_losing_scada_is_recorded_as_injected_and_degrades_the_mode(node: Node) -> None:
    node.arm_fault("scada-api", failure_class=FailureClass.CONNECT_TIMEOUT)
    health_change = next(
        r
        for r in node.store.iter_records(kind=RecordKind.HEALTH_CHANGE)
        if r.body["dependency"] == "scada-api"
    )
    assert health_change.body["to_state"] == "UNREACHABLE"
    assert health_change.body["failure_class"] == "CONNECT_TIMEOUT"
    assert health_change.body["source"] == str(ObservationSource.FAULT_INJECTION)
    assert health_change.body["remediation"]


def test_phase_two_abstains_when_the_answer_needs_the_dead_sensor(
    world: tuple[Node, Backends, LocalStore, Any],
) -> None:
    """T-106 is a load anomaly. Without live load there is no honest answer, so the
    agent gives the conditional one instead of inventing a number."""
    node, _backends, store, registry = world
    health = {
        **dict.fromkeys(BACKEND_NAMES, HealthState.HEALTHY),
        "scada-api": HealthState.UNREACHABLE,
    }
    script = {
        ("T-106", 1): tool_call("get_live_load", feeder_id="F-32"),
        ("T-106", 2): abstain(
            "ticket:T-106",
            "priority",
            "priority depends on live feeder load; get_live_load is UNAVAILABLE",
            ["get_live_load"],
            partial={"value_if_load_high": "P1", "value_if_load_low": "P3"},
        ),
    }
    outcome, emitted = run_task(
        node,
        registry,
        store,
        script,
        health,
        Task("T-106", "triage", "Triage T-106.", subject="ticket:T-106"),
        Mode.DEGRADED,
    )

    assert outcome.kind is RecordKind.ABSTENTION
    assert not [r for r in emitted if r.kind is RecordKind.FINAL], "no FINAL for T-106"

    unavailable = body_of(emitted, RecordKind.UNAVAILABLE)
    assert unavailable["tool"] == "get_live_load"
    assert "scada-api" in unavailable["reason"]

    body = body_of(emitted, RecordKind.ABSTENTION)
    assert body["depends_on_unavailable"] == ["get_live_load"]
    assert body["source"] == "MODEL"
    assert body["partial"] == {"value_if_load_high": "P1", "value_if_load_low": "P3"}


def test_phase_two_still_decides_what_it_can_decide(
    world: tuple[Node, Backends, LocalStore, Any],
) -> None:
    """The counterweight. A water pumping station on four hours of backup is P1
    whatever the feeder load is, and a system that abstained here would be as
    useless as one that invented a number for T-106."""
    node, _backends, store, registry = world
    health = {
        **dict.fromkeys(BACKEND_NAMES, HealthState.HEALTHY),
        "scada-api": HealthState.UNREACHABLE,
    }
    script = {
        ("T-103", 1): tool_call("lookup_asset", asset_id="F-33"),
        ("T-103", 2): final(
            "ticket:T-103",
            "priority",
            "P1",
            "water pumping on 4 h backup is P1 regardless of load",
            depends_on=["lookup_asset"],
        ),
    }
    outcome, emitted = run_task(
        node,
        registry,
        store,
        script,
        health,
        Task("T-103", "triage", "Triage T-103.", subject="ticket:T-103"),
        Mode.DEGRADED,
    )
    assert outcome.kind is RecordKind.FINAL
    body = body_of(emitted, RecordKind.FINAL)
    assert body["outcome"] == {"key": "priority", "value": "P1"}
    assert "get_live_load" not in body["depends_on"]


def test_a_priority_change_still_executes_live_while_its_own_backend_is_up(
    world: tuple[Node, Backends, LocalStore, Any],
) -> None:
    """Only SCADA is down. The ticket system is fine, so the write is not queued."""
    node, backends, store, registry = world
    health = {
        **dict.fromkeys(BACKEND_NAMES, HealthState.HEALTHY),
        "scada-api": HealthState.UNREACHABLE,
    }
    script = {
        ("T-103b", 1): tool_call(
            "set_ticket_priority", ticket_id="T-103", priority="P1", rationale="backup running out"
        ),
        ("T-103b", 2): final("ticket:T-103", "priority", "P1", "set"),
    }
    _outcome, emitted = run_task(
        node,
        registry,
        store,
        script,
        health,
        Task("T-103b", "triage", "Set T-103 priority.", subject="ticket:T-103"),
        Mode.DEGRADED,
    )
    assert body_of(emitted, RecordKind.TOOL_RESULT)["availability"] == "LIVE"
    assert (
        "set_ticket_priority",
        {"ticket_id": "T-103", "priority": "P1", "rationale": "backup running out"},
    ) in backends.executed


def test_the_manifest_the_model_saw_is_recoverable_from_the_record(
    world: tuple[Node, Backends, LocalStore, Any],
) -> None:
    """Answering "why did it decide that" starts with what it was told it could do."""
    node, _backends, store, registry = world
    health = {
        **dict.fromkeys(BACKEND_NAMES, HealthState.HEALTHY),
        "scada-api": HealthState.UNREACHABLE,
    }
    script = {("T-1", 1): final("ticket:T-1", "priority", "P2", "fine")}
    _outcome, emitted = run_task(
        node,
        registry,
        store,
        script,
        health,
        Task("T-1", "triage", "Triage.", subject="ticket:T-1"),
        Mode.DEGRADED,
    )
    turn = next(r for r in emitted if r.kind is RecordKind.MODEL_TURN)
    tools = {t["name"]: t["availability"] for t in turn.body["manifest"]["tools"]}
    assert tools["get_live_load"] == "UNAVAILABLE"
    assert tools["lookup_asset"] == "LIVE"
    assert "get_live_load" in turn.body["system"], "the prompt named the tool and its state"
