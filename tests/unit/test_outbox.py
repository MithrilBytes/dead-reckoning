# SPDX-License-Identifier: Apache-2.0
"""Deferred side effects: exactly once, and only if still justified."""

from __future__ import annotations

from typing import Any

import pytest

from deadreckoning.clock import HLC
from deadreckoning.drain import Drainer, reconcile_interrupted
from deadreckoning.node import Node
from deadreckoning.outbox import DuplicateIntentError, Outbox, OutboxState
from deadreckoning.records import IdentityState
from deadreckoning.tools.contract import (
    Approval,
    Consequence,
    OfflinePolicy,
    PreconditionSpec,
    SideEffect,
    ToolContract,
)
from deadreckoning.tools.preconditions import CheckerRegistry, Observation
from deadreckoning.tools.registry import ToolRegistry

NOW = 1_700_000_000_000
HLC0 = HLC(NOW, 0, "truck-7")

DISPATCH = ToolContract(
    name="dispatch_crew",
    backend="dispatch-api",
    offline_policy=OfflinePolicy.QUEUE,
    side_effect=SideEffect.NON_IDEMPOTENT,
    consequence=Consequence.HIGH,
    approval=Approval.WHEN_NOT_CONNECTED,
    preconditions=[
        PreconditionSpec(check="crew_available", args_from={"crew_id": "crew_id"}),
        PreconditionSpec(check="ticket_unassigned", args_from={"ticket_id": "ticket_id"}),
    ],
    idempotency_key="dispatch:{ticket_id}",
    expiry_s=7200,
)

PRIORITY = ToolContract(
    name="set_ticket_priority",
    backend="ticket-api",
    offline_policy=OfflinePolicy.QUEUE,
    side_effect=SideEffect.IDEMPOTENT,
    consequence=Consequence.MEDIUM,
    preconditions=[PreconditionSpec(check="ticket_open", args_from={"ticket_id": "ticket_id"})],
    idempotency_key="prio:{ticket_id}:{priority}",
    expiry_s=14400,
)


class World:
    """The backends as they are at drain time, which may not be as they were."""

    def __init__(self) -> None:
        self.crew_free = True
        self.ticket_unassigned = True
        self.ticket_open = True
        self.executions: list[tuple[str, dict[str, Any]]] = []
        self.explode = False

    def run(self, tool: str, args: dict[str, Any]) -> Any:
        if self.explode:
            raise TimeoutError("dispatch-api went away mid-call")
        self.executions.append((tool, dict(args)))
        return {"ok": True}


@pytest.fixture
def world() -> World:
    return World()


@pytest.fixture
def checkers(world: World) -> CheckerRegistry:
    registry = CheckerRegistry()

    def crew_available(crew_id: str) -> tuple[bool, Any]:
        return (world.crew_free, world.crew_free)

    def ticket_unassigned(ticket_id: str) -> tuple[bool, Any]:
        return (world.ticket_unassigned, world.ticket_unassigned)

    def ticket_open(ticket_id: str) -> tuple[bool, Any]:
        return (world.ticket_open, world.ticket_open)

    registry.register("crew_available", crew_available)
    registry.register("ticket_unassigned", ticket_unassigned)
    registry.register("ticket_open", ticket_open)
    return registry


@pytest.fixture
def tools(world: World) -> ToolRegistry:
    registry = ToolRegistry()
    for contract in (DISPATCH, PRIORITY):
        registry.register(
            contract,
            live=lambda args, _c=contract: world.run(_c.name, args),  # pyright: ignore[reportUnknownLambdaType]
        )
    return registry


@pytest.fixture
def outbox(node: Node) -> Outbox:
    return Outbox(node.database, node.node_id)


def defer(
    outbox: Outbox,
    contract: ToolContract,
    args: dict[str, Any],
    *,
    observations: list[Observation],
    entry_id: str = "e1",
    approval_required: bool = False,
    now_ms: int = NOW,
) -> Any:
    return outbox.defer(
        contract=contract,
        args=args,
        subject=f"ticket:{args.get('ticket_id')}",
        decision_id="d1",
        deferral_record_id="r1",
        observations=observations,
        created_hlc=HLC0,
        now_ms=now_ms,
        approval_required=approval_required,
        approval_reason="tool policy" if approval_required else None,
        entry_id=entry_id,
    )


def obs(check: str, args: dict[str, Any], value: Any) -> Observation:
    return Observation(check=check, args=args, holds=True, value=value, source="LOCAL")


def drainer(outbox: Outbox, tools: ToolRegistry, checkers: CheckerRegistry, **kwargs: Any):
    return Drainer(
        outbox,
        tools,
        checkers,
        identity=kwargs.pop("identity", IdentityState.FRESH),
        connected=kwargs.pop("connected", True),
        now_ms=kwargs.pop("now_ms", NOW),
        **kwargs,
    )


def test_an_intent_carries_the_reasons_it_was_created(outbox: Outbox) -> None:
    entry = defer(
        outbox,
        DISPATCH,
        {"crew_id": "C-1", "ticket_id": "T-101"},
        observations=[
            obs("crew_available", {"crew_id": "C-1"}, True),
            obs("ticket_unassigned", {"ticket_id": "T-101"}, True),
        ],
    )
    assert entry.idempotency_key == "dispatch:T-101"
    assert [p["check"] for p in entry.preconditions] == ["crew_available", "ticket_unassigned"]
    assert entry.preconditions[0]["observed_value"] is True
    assert entry.expires_at_ms == NOW + 7200 * 1000


def test_a_duplicate_intent_is_rejected_before_it_is_stored(outbox: Outbox) -> None:
    """Two callers, one action. The key says which arguments make them the same."""
    defer(outbox, DISPATCH, {"crew_id": "C-1", "ticket_id": "T-104"}, observations=[])
    with pytest.raises(DuplicateIntentError) as caught:
        defer(
            outbox,
            DISPATCH,
            {"crew_id": "C-2", "ticket_id": "T-104"},
            observations=[],
            entry_id="e2",
        )
    assert caught.value.existing_id == "e1"
    assert len(outbox.all()) == 1


def test_draining_twice_executes_once(
    outbox: Outbox, tools: ToolRegistry, checkers: CheckerRegistry, world: World
) -> None:
    """The guarantee the whole outbox exists to make."""
    defer(
        outbox,
        PRIORITY,
        {"ticket_id": "T-103", "priority": "P1"},
        observations=[obs("ticket_open", {"ticket_id": "T-103"}, True)],
    )
    first = drainer(outbox, tools, checkers).run()
    second = drainer(outbox, tools, checkers).run()

    assert first.executed == ["e1"]
    assert second.executed == []
    assert world.executions == [("set_ticket_priority", {"ticket_id": "T-103", "priority": "P1"})]
    assert outbox.get("e1").state is OutboxState.DONE  # pyright: ignore[reportOptionalMemberAccess]


def test_an_intent_whose_justification_evaporated_does_not_fire(
    outbox: Outbox, tools: ToolRegistry, checkers: CheckerRegistry, world: World
) -> None:
    """Headquarters got there first. The crew is not sent twice."""
    defer(
        outbox,
        DISPATCH,
        {"crew_id": "C-1", "ticket_id": "T-101"},
        observations=[
            obs("crew_available", {"crew_id": "C-1"}, True),
            obs("ticket_unassigned", {"ticket_id": "T-101"}, True),
        ],
    )
    world.ticket_unassigned = False

    result = drainer(outbox, tools, checkers).run()

    assert result.executed == []
    assert result.precondition_failed == ["e1"]
    assert world.executions == []
    assert outbox.get("e1").state is OutboxState.PRECONDITION_FAILED  # pyright: ignore[reportOptionalMemberAccess]

    delta = next(d for d in result.deltas["e1"] if d.check == "ticket_unassigned")
    assert delta.observed_value is True
    assert delta.current_value is False
    assert delta.changed, "the adjudicator has to see exactly what changed"


def test_an_expired_intent_is_never_executed(
    outbox: Outbox, tools: ToolRegistry, checkers: CheckerRegistry, world: World
) -> None:
    """However healthy the world looks now, the reasoning is too old to act on."""
    defer(
        outbox,
        DISPATCH,
        {"crew_id": "C-1", "ticket_id": "T-101"},
        observations=[obs("crew_available", {"crew_id": "C-1"}, True)],
    )
    result = drainer(outbox, tools, checkers, now_ms=NOW + 7201 * 1000).run()
    assert result.expired == ["e1"]
    assert world.executions == []
    assert outbox.get("e1").state is OutboxState.EXPIRED  # pyright: ignore[reportOptionalMemberAccess]


def test_a_high_consequence_action_waits_for_a_human(
    outbox: Outbox, tools: ToolRegistry, checkers: CheckerRegistry, world: World
) -> None:
    defer(
        outbox,
        DISPATCH,
        {"crew_id": "C-1", "ticket_id": "T-101"},
        observations=[
            obs("crew_available", {"crew_id": "C-1"}, True),
            obs("ticket_unassigned", {"ticket_id": "T-101"}, True),
        ],
        approval_required=True,
    )
    result = drainer(outbox, tools, checkers, connected=False).run()
    assert result.attempted == 0, "an entry awaiting approval is not drainable"
    assert world.executions == []

    entry = outbox.get("e1")
    assert entry is not None
    outbox.approve(entry, by="dispatcher-jlee", note="confirmed by radio")
    assert outbox.get("e1").state is OutboxState.READY  # pyright: ignore[reportOptionalMemberAccess]

    after = drainer(outbox, tools, checkers, connected=False).run()
    assert after.executed == ["e1"]


def test_approval_is_re_evaluated_at_drain_not_only_at_creation(
    outbox: Outbox, tools: ToolRegistry, checkers: CheckerRegistry, world: World
) -> None:
    """Created while connected, drained while not. The rule that fires is the one
    in force when the action would actually happen."""
    defer(
        outbox,
        DISPATCH,
        {"crew_id": "C-1", "ticket_id": "T-101"},
        observations=[
            obs("crew_available", {"crew_id": "C-1"}, True),
            obs("ticket_unassigned", {"ticket_id": "T-101"}, True),
        ],
        approval_required=False,
    )
    result = drainer(outbox, tools, checkers, connected=False).run()
    assert result.awaiting_approval == ["e1"]
    assert world.executions == []


def test_a_stale_identity_holds_an_approved_entry_rather_than_running_it(
    outbox: Outbox, tools: ToolRegistry, checkers: CheckerRegistry, world: World
) -> None:
    """Approved hours ago under a fresh grant. The grant has since expired, and the
    approval does not carry the authority with it."""
    entry = defer(
        outbox,
        PRIORITY,
        {"ticket_id": "T-103", "priority": "P1"},
        observations=[obs("ticket_open", {"ticket_id": "T-103"}, True)],
    )
    outbox.approve(entry, by="someone")
    result = drainer(outbox, tools, checkers, identity=IdentityState.STALE).run()
    assert result.identity_blocked == ["e1"]
    assert world.executions == []
    assert outbox.get("e1").state is OutboxState.READY, "held, not lost"  # pyright: ignore[reportOptionalMemberAccess]


def test_no_identity_cancels_the_entry_outright(
    outbox: Outbox, tools: ToolRegistry, checkers: CheckerRegistry, world: World
) -> None:
    defer(
        outbox,
        PRIORITY,
        {"ticket_id": "T-103", "priority": "P1"},
        observations=[obs("ticket_open", {"ticket_id": "T-103"}, True)],
    )
    result = drainer(outbox, tools, checkers, identity=IdentityState.NONE).run()
    assert result.identity_blocked == ["e1"]
    assert outbox.get("e1").state is OutboxState.CANCELLED  # pyright: ignore[reportOptionalMemberAccess]


def test_rejecting_cancels_and_records_who(
    outbox: Outbox, tools: ToolRegistry, checkers: CheckerRegistry, world: World
) -> None:
    entry = defer(
        outbox,
        DISPATCH,
        {"crew_id": "C-1", "ticket_id": "T-101"},
        observations=[],
        approval_required=True,
    )
    outbox.reject(entry, by="supervisor-mchen", note="crew reassigned")
    stored = outbox.get("e1")
    assert stored is not None
    assert stored.state is OutboxState.CANCELLED
    assert stored.approval is not None
    assert stored.approval.by == "supervisor-mchen"
    assert drainer(outbox, tools, checkers).run().executed == []


def test_a_failing_execution_is_recorded_not_swallowed(
    outbox: Outbox, tools: ToolRegistry, checkers: CheckerRegistry, world: World
) -> None:
    defer(
        outbox,
        PRIORITY,
        {"ticket_id": "T-103", "priority": "P1"},
        observations=[obs("ticket_open", {"ticket_id": "T-103"}, True)],
    )
    world.explode = True
    result = drainer(outbox, tools, checkers).run()
    assert result.failed == ["e1"]
    stored = outbox.get("e1")
    assert stored is not None
    assert stored.state is OutboxState.FAILED
    assert "TimeoutError" in (stored.last_error or "")


def test_an_entry_found_mid_execution_is_never_blindly_retried(
    outbox: Outbox, tools: ToolRegistry, world: World
) -> None:
    """Somewhere between the call going out and the record coming back, a crew may
    already have been dispatched. The one thing that must not happen is a second."""
    entry = defer(outbox, DISPATCH, {"crew_id": "C-1", "ticket_id": "T-101"}, observations=[])
    outbox.transition(entry, OutboxState.EXECUTING)

    transitions = reconcile_interrupted(outbox, tools)

    assert [t.after for t in transitions] == [OutboxState.FAILED]
    assert transitions[0].reason == "UNKNOWN_EXECUTION_STATE"
    assert world.executions == [], "nothing was re-run"


def test_a_tool_that_can_verify_is_asked_instead(outbox: Outbox, world: World) -> None:
    registry = ToolRegistry()
    registry.register(
        DISPATCH,
        live=lambda args: world.run("dispatch_crew", args),
        verify=lambda _key: "executed",
    )
    entry = defer(outbox, DISPATCH, {"crew_id": "C-1", "ticket_id": "T-101"}, observations=[])
    outbox.transition(entry, OutboxState.EXECUTING)
    transitions = reconcile_interrupted(outbox, registry)
    assert transitions[0].after is OutboxState.DONE
    assert world.executions == []


def test_a_verified_unexecuted_entry_goes_back_in_the_queue(outbox: Outbox, world: World) -> None:
    registry = ToolRegistry()
    registry.register(
        DISPATCH,
        live=lambda args: world.run("dispatch_crew", args),
        verify=lambda _key: "not_executed",
    )
    entry = defer(outbox, DISPATCH, {"crew_id": "C-1", "ticket_id": "T-101"}, observations=[])
    outbox.transition(entry, OutboxState.EXECUTING)
    assert reconcile_interrupted(outbox, registry)[0].after is OutboxState.READY


def test_drain_serialises_per_subject(
    outbox: Outbox, tools: ToolRegistry, checkers: CheckerRegistry, world: World
) -> None:
    """Two actions about one ticket must not interleave; the second may depend on
    what the first did."""
    defer(
        outbox,
        PRIORITY,
        {"ticket_id": "T-104", "priority": "P1"},
        observations=[obs("ticket_open", {"ticket_id": "T-104"}, True)],
        entry_id="a",
    )
    defer(
        outbox,
        PRIORITY,
        {"ticket_id": "T-104", "priority": "P2"},
        observations=[obs("ticket_open", {"ticket_id": "T-104"}, True)],
        entry_id="b",
    )
    result = drainer(outbox, tools, checkers).run()
    assert len(result.executed) == 1
    assert len(world.executions) == 1


def test_a_checker_nothing_registered_fails_at_startup_not_at_drain() -> None:
    registry = CheckerRegistry()

    def ticket_open(ticket_id: str) -> tuple[bool, Any]:
        return (True, True)

    registry.register("ticket_open", ticket_open)
    with pytest.raises(KeyError, match="crew_available"):
        registry.validate(["ticket_open", "crew_available"])
