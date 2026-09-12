# SPDX-License-Identifier: Apache-2.0
"""Phase 3 and the deferral it sets up, headless.

Everything is gone: the frontier model, the ticket system, dispatch, notify, GIS.
The truck keeps triaging on cached data and a local model, and the actions it
decides on are recorded as intents rather than fired into a void.

The last test is the one the whole outbox exists for. A crew is dispatched to the
hospital, the link dies, headquarters sends someone else in the meantime, and on
reconnect the action does not happen. Not because it failed, but because the
reason for it stopped being true, and the runtime noticed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from demo.tools_outage import CHECKERS, Backends, build_registry

from deadreckoning.agent import AgentLoop, Task
from deadreckoning.canonical import content_hash
from deadreckoning.clock import HLC, TimeTrust
from deadreckoning.deferral import DeferralContext, OutboxDeferrer
from deadreckoning.drain import Drainer, redecide_tasks
from deadreckoning.health import HealthState
from deadreckoning.llm.scripted import ScriptedModel, escalate, final, tool_call
from deadreckoning.local_store import LocalStore
from deadreckoning.manifest import TierView, build_manifest
from deadreckoning.node import Node
from deadreckoning.outbox import Outbox, OutboxState
from deadreckoning.records import IdentityState, Mode, Record, RecordKind
from deadreckoning.tools.enforcer import ToolEnforcer
from deadreckoning.tools.preconditions import CheckerRegistry

NOW = 1_700_000_000_000
PROMPT = Path("docs/prompts/planner_v1.md").read_text()
BACKENDS = ("gis-api", "ticket-api", "scada-api", "dispatch-api", "notify-api", "weather-api")


def islanded() -> dict[str, HealthState]:
    """Everything off the truck is gone. Only what is on it still answers."""
    return dict.fromkeys(BACKENDS, HealthState.UNREACHABLE)


@pytest.fixture
def world(node: Node) -> dict[str, Any]:
    backends = Backends()
    store = LocalStore(node.database)
    registry = build_registry(backends)
    checkers = CheckerRegistry()
    for name, fn in CHECKERS.items():
        checkers.register(name, lambda _fn=fn, **kwargs: _fn(backends, **kwargs))  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    outbox = Outbox(node.database, node.node_id)
    counter = {"n": 0}

    def next_id() -> str:
        counter["n"] += 1
        return f"ob-{counter['n']}"

    deferrer = OutboxDeferrer(
        outbox, checkers, next_id, lambda: NOW, lambda: HLC(NOW, counter["n"], node.node_id)
    )
    return {
        "node": node,
        "backends": backends,
        "store": store,
        "registry": registry,
        "checkers": checkers,
        "outbox": outbox,
        "deferrer": deferrer,
    }


def run(world: dict[str, Any], script: dict[tuple[str, int], str], task: Task) -> list[Record]:
    node: Node = world["node"]
    emitted: list[Record] = []

    def emit(kind: RecordKind, body: dict[str, Any], **fields: Any) -> Record:
        record = node.emit(kind, body=body, **fields)
        emitted.append(record)
        if kind is RecordKind.MODEL_TURN:
            world["deferrer"].context = DeferralContext(
                decision_id=record.id,
                subject=task.subject,
                identity=IdentityState.CACHED,
                connected=False,
                observed_source="LOCAL",
            )
        return record

    health = islanded()
    manifest = build_manifest(
        mode=Mode.ISLANDED,
        tier=TierView("local-q4", 2, "scripted-local"),
        identity=IdentityState.CACHED,
        identity_ttl_s=5400,
        time_trust=TimeTrust.DRIFTING,
        registry=world["registry"],
        store=world["store"],
        health=health,
        now_ms=NOW,
    )
    loop = AgentLoop(
        client=ScriptedModel(script),
        enforcer=ToolEnforcer(world["registry"], world["store"], health, NOW, world["deferrer"]),
        emit=emit,
        prompt_template=PROMPT,
        prompt_template_hash=content_hash(PROMPT),
    )
    loop.run(task, manifest)
    return emitted


def test_a_priority_change_while_islanded_becomes_an_intent_not_a_write(
    world: dict[str, Any],
) -> None:
    world["store"].put("lookup_asset", {"asset_id": "F-33"}, {"customers": 2210}, NOW - 3_600_000)
    emitted = run(
        world,
        {
            ("T-104", 1): tool_call("lookup_asset", asset_id="F-33"),
            ("T-104", 2): tool_call(
                "set_ticket_priority", ticket_id="T-104", priority="P2", rationale="lines down"
            ),
            ("T-104", 3): final(
                "ticket:T-104",
                "priority",
                "P2",
                "no field report here",
                depends_on=["lookup_asset"],
            ),
        },
        Task("T-104", "triage", "Triage T-104.", subject="ticket:T-104"),
    )

    deferral = next(r for r in emitted if r.kind is RecordKind.DEFERRAL)
    assert deferral.body["tool"] == "set_ticket_priority"
    assert deferral.body["availability"] == "QUEUED"

    entry = world["outbox"].get(deferral.body["outbox_id"])
    assert entry is not None
    assert entry.state is OutboxState.READY, "medium consequence, cached identity, no approval"
    assert entry.idempotency_key == "prio:T-104:P2"
    assert world["backends"].executed == [], "nothing reached the ticket system"

    local = next(r for r in emitted if r.kind is RecordKind.TOOL_RESULT)
    assert local.body["availability"] == "LOCAL"
    assert local.body["data_age_s"] == 3600


def test_a_high_consequence_dispatch_waits_for_a_person(world: dict[str, Any]) -> None:
    """HIGH consequence and not connected, so it is held rather than queued to run."""
    emitted = run(
        world,
        {
            ("D-101", 1): tool_call("dispatch_crew", crew_id="C-1", ticket_id="T-101"),
            ("D-101", 2): final("ticket:T-101", "dispatch", "C-1", "hospital first"),
        },
        Task("D-101", "dispatch", "Dispatch to T-101.", subject="ticket:T-101"),
    )
    entry = world["outbox"].get(
        next(r for r in emitted if r.kind is RecordKind.DEFERRAL).body["outbox_id"]
    )
    assert entry is not None
    assert entry.state is OutboxState.AWAITING_APPROVAL
    assert entry.approval is not None and entry.approval.required
    assert [p["check"] for p in entry.preconditions] == ["crew_available", "ticket_unassigned"]
    assert all(p["observed_value"] is True for p in entry.preconditions)
    assert all(p["observed_source"] == "LOCAL" for p in entry.preconditions), (
        "captured from the snapshot, because the backend is gone, and the log says so"
    )


def test_a_duplicate_intent_produces_one_entry(world: dict[str, Any]) -> None:
    """A second caller reporting the same hazard is normal. Two actions are not."""
    run(
        world,
        {
            ("T-104", 1): tool_call(
                "set_ticket_priority", ticket_id="T-104", priority="P2", rationale="first"
            ),
            ("T-104", 2): final("ticket:T-104", "priority", "P2", "first"),
        },
        Task("T-104", "triage", "Triage T-104.", subject="ticket:T-104"),
    )
    run(
        world,
        {
            ("T-107", 1): tool_call(
                "set_ticket_priority", ticket_id="T-104", priority="P2", rationale="duplicate"
            ),
            ("T-107", 2): final("ticket:T-107", "priority", "P2", "same hazard"),
        },
        Task("T-107", "triage", "Triage T-107.", subject="ticket:T-107"),
    )
    keyed = [e for e in world["outbox"].all() if e.idempotency_key == "prio:T-104:P2"]
    assert len(keyed) == 1
    assert world["deferrer"].duplicates == ["prio:T-104:P2"]


def test_something_outside_the_remit_escalates_and_queues_nothing(world: dict[str, Any]) -> None:
    emitted = run(
        world,
        {("T-108", 1): escalate("ticket:T-108", "burning smell near a transformer")},
        Task("T-108", "triage", "Triage T-108.", subject="ticket:T-108"),
    )
    assert any(r.kind is RecordKind.ESCALATION for r in emitted)
    assert world["outbox"].all() == []


def test_the_action_does_not_fire_when_headquarters_got_there_first(
    world: dict[str, Any],
) -> None:
    """The whole reason the outbox re-checks rather than replays.

    The crew is dispatched while the link is down. Headquarters sends its own crew
    in the meantime. On reconnect the intent is not executed, and the delta says
    exactly why.
    """
    emitted = run(
        world,
        {
            ("D-101", 1): tool_call("dispatch_crew", crew_id="C-1", ticket_id="T-101"),
            ("D-101", 2): final("ticket:T-101", "dispatch", "C-1", "hospital on backup"),
        },
        Task("D-101", "dispatch", "Dispatch to T-101.", subject="ticket:T-101"),
    )
    entry_id = next(r for r in emitted if r.kind is RecordKind.DEFERRAL).body["outbox_id"]
    outbox: Outbox = world["outbox"]
    entry = outbox.get(entry_id)
    assert entry is not None

    # A human approves it while the truck is still dark.
    outbox.approve(entry, by="dispatcher-jlee", note="confirmed by radio")

    # Meanwhile, at headquarters.
    backends: Backends = world["backends"]
    backends.ticket("T-101")["assigned"] = True  # pyright: ignore[reportOptionalSubscript]

    result = Drainer(
        outbox,
        world["registry"],
        world["checkers"],
        identity=IdentityState.FRESH,
        connected=True,
        now_ms=NOW + 60_000,
    ).run()

    assert result.executed == []
    assert result.precondition_failed == [entry_id]
    assert backends.executed == [], "no second crew was sent"

    delta = next(d for d in result.deltas[entry_id] if d.check == "ticket_unassigned")
    assert delta.observed_value is True
    assert delta.current_value is False

    tasks = redecide_tasks(outbox, result)
    assert len(tasks) == 1
    assert tasks[0].outbox_id == entry_id
    assert "was True when you decided, is False now" in tasks[0].instructions()
    assert outbox.get(entry_id).re_decide_task_id == tasks[0].task_id  # pyright: ignore[reportOptionalMemberAccess]


def test_an_intent_whose_reasons_still_hold_does_fire(world: dict[str, Any]) -> None:
    """The counterweight: re-checking is not an excuse to never act."""
    emitted = run(
        world,
        {
            ("T-103", 1): tool_call(
                "set_ticket_priority", ticket_id="T-103", priority="P1", rationale="water pumping"
            ),
            ("T-103", 2): final("ticket:T-103", "priority", "P1", "backup running out"),
        },
        Task("T-103", "triage", "Triage T-103.", subject="ticket:T-103"),
    )
    entry_id = next(r for r in emitted if r.kind is RecordKind.DEFERRAL).body["outbox_id"]
    result = Drainer(
        world["outbox"],
        world["registry"],
        world["checkers"],
        identity=IdentityState.FRESH,
        connected=True,
        now_ms=NOW + 60_000,
    ).run()
    assert result.executed == [entry_id]
    assert world["backends"].executed == [
        (
            "set_ticket_priority",
            {"ticket_id": "T-103", "priority": "P1", "rationale": "water pumping"},
        )
    ]
