# SPDX-License-Identifier: Apache-2.0
"""The dispatch matrix, every cell, and the refusal of a decision that ignored it."""

from __future__ import annotations

from typing import Any

import pytest

from deadreckoning.health import HealthState
from deadreckoning.local_store import LocalStore
from deadreckoning.node import Node
from deadreckoning.tools.contract import (
    Approval,
    Availability,
    Consequence,
    HydrateSpec,
    OfflinePolicy,
    SideEffect,
    ToolContract,
)
from deadreckoning.tools.enforcer import Dispatch, ToolEnforcer, validate_decision
from deadreckoning.tools.registry import ToolRegistry

NOW = 1_700_000_000_000


class FakeOutbox:
    def __init__(self) -> None:
        self.deferred: list[tuple[str, dict[str, Any]]] = []

    def defer(self, contract: ToolContract, args: dict[str, Any]) -> str:
        self.deferred.append((contract.name, args))
        return f"outbox-{len(self.deferred)}"


def local_tool(**kwargs: Any) -> ToolContract:
    return ToolContract(
        name=kwargs.pop("name", "lookup_asset"),
        backend="gis-api",
        offline_policy=OfflinePolicy.LOCAL,
        side_effect=SideEffect.NONE,
        consequence=Consequence.LOW,
        staleness_budget_s=kwargs.pop("staleness_budget_s", 86400),
        hydrate=HydrateSpec(subject_types=["asset"], arg_from_subject="asset_id"),
        **kwargs,
    )


def queue_tool(**kwargs: Any) -> ToolContract:
    return ToolContract(
        name=kwargs.pop("name", "dispatch_crew"),
        backend="dispatch-api",
        offline_policy=OfflinePolicy.QUEUE,
        side_effect=SideEffect.NON_IDEMPOTENT,
        consequence=Consequence.HIGH,
        approval=Approval.WHEN_NOT_CONNECTED,
        idempotency_key="dispatch:{ticket_id}",
        expiry_s=7200,
        **kwargs,
    )


def fail_tool(**kwargs: Any) -> ToolContract:
    return ToolContract(
        name=kwargs.pop("name", "get_live_load"),
        backend="scada-api",
        offline_policy=OfflinePolicy.FAIL,
        side_effect=SideEffect.NONE,
        consequence=Consequence.LOW,
        **kwargs,
    )


@pytest.fixture
def store(node: Node) -> LocalStore:
    return LocalStore(node.database)


def build(
    store: LocalStore, contracts: list[ToolContract], state: HealthState, outbox: Any = None
) -> ToolEnforcer:
    registry = ToolRegistry()
    for contract in contracts:
        registry.register(
            contract,
            live=lambda args: {"live": True, "args": args},
            local=None,
        )
    health = {"gis-api": state, "dispatch-api": state, "scada-api": state}
    return ToolEnforcer(registry, store, health, NOW, outbox)


# --- the matrix, one test per cell -------------------------------------------


@pytest.mark.parametrize("state", [HealthState.HEALTHY, HealthState.SLOW])
def test_a_reachable_backend_runs_live_whatever_the_policy(
    store: LocalStore, state: HealthState
) -> None:
    expected = Availability.LIVE if state is HealthState.HEALTHY else Availability.LIVE_SLOW
    for contract in (local_tool(), queue_tool(), fail_tool()):
        enforcer = build(store, [contract], state, FakeOutbox())
        assert enforcer.dispatch(contract.name, {"ticket_id": "T-1"}).availability is expected


@pytest.mark.parametrize("state", [HealthState.UNREACHABLE, HealthState.AUTH_BROKEN])
def test_fail_policy_returns_unavailable_with_a_reason(
    store: LocalStore, state: HealthState
) -> None:
    enforcer = build(store, [fail_tool()], state)
    result = enforcer.dispatch("get_live_load", {"feeder_id": "F-31"})
    assert result.availability is Availability.UNAVAILABLE
    assert result.reason and "scada-api" in result.reason


@pytest.mark.parametrize("state", [HealthState.UNREACHABLE, HealthState.AUTH_BROKEN])
def test_queue_policy_defers_and_returns_the_outbox_id(
    store: LocalStore, state: HealthState
) -> None:
    outbox = FakeOutbox()
    enforcer = build(store, [queue_tool()], state, outbox)
    result = enforcer.dispatch("dispatch_crew", {"ticket_id": "T-101", "crew_id": "C-1"})
    assert result.availability is Availability.QUEUED
    assert result.outbox_id == "outbox-1"
    assert outbox.deferred == [("dispatch_crew", {"ticket_id": "T-101", "crew_id": "C-1"})]


def test_local_policy_answers_from_the_store_and_reports_age(store: LocalStore) -> None:
    store.put(
        "lookup_asset", {"asset_id": "F-31"}, {"customers": 1840}, captured_ms=NOW - 3_600_000
    )
    enforcer = build(store, [local_tool()], HealthState.UNREACHABLE)
    result = enforcer.dispatch("lookup_asset", {"asset_id": "F-31"})
    assert result.availability is Availability.LOCAL
    assert result.result == {"customers": 1840}
    assert result.data_age_s == 3600
    assert result.content_hash


def test_local_data_past_its_budget_is_reported_stale_not_hidden(store: LocalStore) -> None:
    store.put(
        "lookup_asset", {"asset_id": "F-31"}, {"customers": 1840}, captured_ms=NOW - 90_000_000
    )
    enforcer = build(store, [local_tool(staleness_budget_s=86400)], HealthState.UNREACHABLE)
    result = enforcer.dispatch("lookup_asset", {"asset_id": "F-31"})
    assert result.availability is Availability.LOCAL_STALE
    assert result.result == {"customers": 1840}, "stale data is still offered, with a warning"


def test_fail_when_stale_turns_old_data_into_unavailable(store: LocalStore) -> None:
    """Some data is worse than none. A weather forecast six hours old is not a forecast."""
    store.put("weather", {"area": "SB-3"}, {"wind": 40}, captured_ms=NOW - 90_000_000)
    contract = local_tool(name="weather", staleness_budget_s=21600, fail_when_stale=True)
    enforcer = build(store, [contract], HealthState.UNREACHABLE)
    result = enforcer.dispatch("weather", {"area": "SB-3"})
    assert result.availability is Availability.UNAVAILABLE
    assert result.reason and "budget" in result.reason
    assert result.result is None


def test_prefer_local_when_slow_takes_the_cached_answer(store: LocalStore) -> None:
    store.put("lookup_asset", {"asset_id": "F-31"}, {"customers": 1840}, captured_ms=NOW)
    contract = local_tool(prefer_local_when_slow=True)
    enforcer = build(store, [contract], HealthState.SLOW)
    assert (
        enforcer.dispatch("lookup_asset", {"asset_id": "F-31"}).availability is Availability.LOCAL
    )


def test_an_unprobed_backend_is_treated_as_unreachable(store: LocalStore) -> None:
    """UNKNOWN means nobody has checked. Acting on an unchecked dependency is the
    assumption this runtime exists to avoid."""
    enforcer = build(store, [fail_tool()], HealthState.UNKNOWN)
    assert enforcer.dispatch("get_live_load", {}).availability is Availability.UNAVAILABLE


def test_a_tool_with_no_backend_is_always_reachable(store: LocalStore) -> None:
    contract = ToolContract(
        name="get_field_reports",
        backend=None,
        offline_policy=OfflinePolicy.LOCAL,
        side_effect=SideEffect.NONE,
        consequence=Consequence.LOW,
        staleness_budget_s=None,
        local_source="field_input",
    )
    registry = ToolRegistry()
    registry.register(contract, live=lambda args: {"reports": []})
    enforcer = ToolEnforcer(registry, store, {}, NOW)
    assert enforcer.dispatch("get_field_reports", {"ticket_id": "T-104"}).availability is (
        Availability.LIVE
    )


def test_queueing_with_no_outbox_attached_is_unavailable_not_silent(store: LocalStore) -> None:
    enforcer = build(store, [queue_tool()], HealthState.UNREACHABLE, outbox=None)
    result = enforcer.dispatch("dispatch_crew", {"ticket_id": "T-1"})
    assert result.availability is Availability.UNAVAILABLE


def test_local_with_nothing_stored_is_unavailable(store: LocalStore) -> None:
    """Nothing to be stale about. Saying LOCAL would promise an answer that does not exist."""
    enforcer = build(store, [local_tool()], HealthState.UNREACHABLE)
    assert enforcer.dispatch("lookup_asset", {"asset_id": "F-99"}).availability is (
        Availability.UNAVAILABLE
    )


# --- the enforcement of abstention -------------------------------------------


def test_a_final_resting_on_an_unavailable_tool_is_refused() -> None:
    """The rule the whole system exists for.

    A model told a tool is unavailable, which then produces a confident answer
    depending on it, has done the one thing that must not happen.
    """
    dispatches = [
        Dispatch("lookup_asset", Availability.LOCAL, result={"x": 1}),
        Dispatch("get_live_load", Availability.UNAVAILABLE, reason="scada-api is UNREACHABLE"),
    ]
    result = validate_decision(["lookup_asset", "get_live_load"], dispatches, abstained=False)
    assert result.allowed is False
    assert result.unavailable_dependencies == ["get_live_load"]
    assert "get_live_load" in result.reason


def test_abstaining_is_always_allowed() -> None:
    dispatches = [Dispatch("get_live_load", Availability.UNAVAILABLE, reason="down")]
    assert validate_decision(["get_live_load"], dispatches, abstained=True).allowed


def test_a_final_that_does_not_depend_on_the_dead_tool_stands() -> None:
    """Triaging around a missing input is correct behaviour, not a violation."""
    dispatches = [
        Dispatch("lookup_asset", Availability.LOCAL, result={"x": 1}),
        Dispatch("get_live_load", Availability.UNAVAILABLE, reason="down"),
    ]
    assert validate_decision(["lookup_asset"], dispatches, abstained=False).allowed


def test_the_rule_is_about_the_result_not_the_policy() -> None:
    """A LOCAL tool refused for staleness is as unavailable as a FAIL tool down.

    Scoping this to FAIL-policy tools would let the staleness path through, which
    is the single thing the requirement exists to prevent.
    """
    dispatches = [Dispatch("weather", Availability.UNAVAILABLE, reason="past its staleness budget")]
    result = validate_decision(["weather"], dispatches, abstained=False)
    assert result.allowed is False
    assert result.unavailable_dependencies == ["weather"]


def test_stale_local_data_does_not_block_a_decision() -> None:
    """LOCAL_STALE is usable with a caveat. Refusing it would abstain on everything."""
    dispatches = [Dispatch("lookup_asset", Availability.LOCAL_STALE, result={"x": 1})]
    assert validate_decision(["lookup_asset"], dispatches, abstained=False).allowed


def test_a_queued_tool_does_not_block_a_decision() -> None:
    """The intent was recorded. That is a successful outcome, not a missing input."""
    dispatches = [Dispatch("set_ticket_priority", Availability.QUEUED, outbox_id="o-1")]
    assert validate_decision(["set_ticket_priority"], dispatches, abstained=False).allowed
