# SPDX-License-Identifier: Apache-2.0
"""Hydration: the write path the local store never had."""

from __future__ import annotations

from typing import Any

import pytest

from deadreckoning.health import HealthState
from deadreckoning.local_store import LocalStore
from deadreckoning.node import Node
from deadreckoning.provisioning import (
    Provisioner,
    SkipReason,
    Trigger,
    planned_calls,
    subject_id,
    subject_type,
)
from deadreckoning.tools.contract import (
    Consequence,
    HydrateSpec,
    OfflinePolicy,
    SideEffect,
    ToolContract,
)
from deadreckoning.tools.registry import ToolRegistry

NOW = 1_700_000_000_000
WORKING_SET = ["ticket:T-101", "feeder:F-31", "feeder:F-33", "crew:C-1"]


@pytest.fixture
def store(node: Node) -> LocalStore:
    return LocalStore(node.database)


def asset_tool(calls: list[dict[str, Any]] | None = None) -> tuple[ToolContract, Any]:
    contract = ToolContract(
        name="lookup_asset",
        backend="gis-api",
        offline_policy=OfflinePolicy.LOCAL,
        side_effect=SideEffect.NONE,
        consequence=Consequence.LOW,
        staleness_budget_s=86400,
        hydrate=HydrateSpec(subject_types=["feeder"], arg_from_subject="asset_id"),
    )

    def live(args: dict[str, Any]) -> dict[str, Any]:
        if calls is not None:
            calls.append(args)
        return {"id": args["asset_id"], "customers": 1840}

    return contract, live


def tickets_tool() -> tuple[ToolContract, Any]:
    contract = ToolContract(
        name="list_open_tickets",
        backend="ticket-api",
        offline_policy=OfflinePolicy.LOCAL,
        side_effect=SideEffect.NONE,
        consequence=Consequence.LOW,
        staleness_budget_s=3600,
        hydrate=HydrateSpec(static_args=[{}]),
    )

    def live(_args: dict[str, Any]) -> dict[str, Any]:
        return {"open": ["T-101", "T-103"]}

    return contract, live


def provisioner(store: LocalStore, tools: list[tuple[ToolContract, Any]], state: HealthState):
    registry = ToolRegistry()
    for contract, live in tools:
        registry.register(contract, live=live)
    health = {"gis-api": state, "ticket-api": state}
    return Provisioner(registry, store, health)


def test_subject_identifiers_split_on_the_first_colon() -> None:
    assert subject_type("ticket:T-104") == "ticket"
    assert subject_id("ticket:T-104") == "T-104"
    assert subject_id("customer:C-HOSP-1") == "C-HOSP-1"


def test_the_runtime_expands_nothing(store: LocalStore) -> None:
    """A working set is flat and explicit. Computing a closure would need a
    declared domain model, which is a stated non-goal."""
    contract, _ = asset_tool()
    calls = planned_calls(contract, WORKING_SET)
    assert [args["asset_id"] for args, _ in calls] == ["F-31", "F-33"]
    assert all(subject is not None for _, subject in calls)


def test_a_tool_keyed_on_nothing_hydrates_through_static_args() -> None:
    contract, _ = tickets_tool()
    assert planned_calls(contract, WORKING_SET) == [({}, None)]


def test_hydration_writes_rows_a_local_substitute_can_find(store: LocalStore) -> None:
    seen: list[dict[str, Any]] = []
    p = provisioner(store, [asset_tool(seen), tickets_tool()], HealthState.HEALTHY)
    summary = p.run(WORKING_SET, NOW)

    assert seen == [{"asset_id": "F-31"}, {"asset_id": "F-33"}]
    assert summary.tools["lookup_asset"].written == 2
    assert summary.tools["list_open_tickets"].written == 1

    row = store.get("lookup_asset", {"asset_id": "F-31"})
    assert row is not None
    assert row.content == {"id": "F-31", "customers": 1840}
    assert row.subject == "feeder:F-31"
    assert row.source == "gis-api"
    assert row.age_s(NOW) == 0


def test_refetching_unchanged_content_still_refreshes_the_age(store: LocalStore) -> None:
    """Age is time since the source last confirmed the value, not since it changed.

    Reading this backwards makes a node report data as stale that it has just
    confirmed is current, and abstain when it did not need to.
    """
    p = provisioner(store, [asset_tool()], HealthState.HEALTHY)
    p.run(["feeder:F-31"], NOW)
    first = store.get("lookup_asset", {"asset_id": "F-31"})
    assert first is not None

    later = NOW + 3_600_000
    summary = p.run(["feeder:F-31"], later)
    second = store.get("lookup_asset", {"asset_id": "F-31"})

    assert second is not None
    assert summary.tools["lookup_asset"].revalidated_unchanged == 1
    assert summary.tools["lookup_asset"].written == 0
    assert second.content_hash == first.content_hash, "unchanged data stays visibly unchanged"
    assert second.age_s(later) == 0, "but it is current, because the source just said so"


def test_changed_content_counts_as_a_write(store: LocalStore) -> None:
    answers = [{"customers": 1840}, {"customers": 1200}]
    contract = ToolContract(
        name="lookup_asset",
        backend="gis-api",
        offline_policy=OfflinePolicy.LOCAL,
        side_effect=SideEffect.NONE,
        consequence=Consequence.LOW,
        staleness_budget_s=86400,
        hydrate=HydrateSpec(subject_types=["feeder"], arg_from_subject="asset_id"),
    )

    def live(_args: dict[str, Any]) -> dict[str, Any]:
        return answers.pop(0)

    p = provisioner(store, [(contract, live)], HealthState.HEALTHY)
    p.run(["feeder:F-31"], NOW)
    summary = p.run(["feeder:F-31"], NOW + 1000)
    assert summary.tools["lookup_asset"].written == 1
    assert summary.tools["lookup_asset"].revalidated_unchanged == 0


def test_a_pass_runs_only_against_reachable_backends(store: LocalStore) -> None:
    p = provisioner(store, [asset_tool()], HealthState.UNREACHABLE)
    summary = p.run(WORKING_SET, NOW)
    assert summary.skipped_reason is SkipReason.BACKEND_UNAVAILABLE
    assert store.count() == 0


def test_a_pass_is_skipped_outright_on_a_low_battery(store: LocalStore) -> None:
    """Unconditional and not configurable: speculative work must never compete
    with the task in front of it."""
    p = provisioner(store, [asset_tool()], HealthState.HEALTHY)
    summary = p.run(WORKING_SET, NOW, power="BATTERY_LOW")
    assert summary.skipped_reason is SkipReason.BATTERY_LOW
    assert store.count() == 0


def test_a_pass_stops_cleanly_at_its_call_bound(store: LocalStore) -> None:
    p = provisioner(store, [asset_tool()], HealthState.HEALTHY)
    p.max_calls_per_pass = 1
    summary = p.run(WORKING_SET, NOW)
    assert summary.skipped_reason is SkipReason.BUDGET_EXHAUSTED
    assert summary.rows_written == 1, "the counts are what it achieved before stopping"


def test_a_failing_backend_is_counted_not_swallowed(store: LocalStore) -> None:
    def angry(args: dict[str, Any]) -> Any:
        raise TimeoutError("no route to host")

    contract, _ = asset_tool()
    p = provisioner(store, [(contract, angry)], HealthState.HEALTHY)
    summary = p.run(["feeder:F-31"], NOW)
    assert summary.tools["lookup_asset"].failures == {"TimeoutError": 1}
    assert summary.rows_written == 0


def test_eviction_spares_the_current_working_set(store: LocalStore) -> None:
    contract = ToolContract(
        name="lookup_asset",
        backend="gis-api",
        offline_policy=OfflinePolicy.LOCAL,
        side_effect=SideEffect.NONE,
        consequence=Consequence.LOW,
        staleness_budget_s=86400,
        hydrate=HydrateSpec(subject_types=["feeder"], arg_from_subject="asset_id", max_rows=2),
    )

    def live(args: dict[str, Any]) -> dict[str, Any]:
        return {"id": args["asset_id"]}

    p = provisioner(store, [(contract, live)], HealthState.HEALTHY)
    p.run(["feeder:F-01", "feeder:F-02", "feeder:F-03"], NOW)
    p.run(["feeder:F-03"], NOW + 1000)
    assert store.count("lookup_asset") <= 2
    assert store.get("lookup_asset", {"asset_id": "F-03"}) is not None, (
        "what the node is working on now survives"
    )


def test_rows_that_cannot_be_refetched_are_never_evicted(store: LocalStore) -> None:
    """A field report entered by hand has no source to fetch it from again."""
    store.put("get_field_reports", {"ticket_id": "T-104"}, {"obs": "live wire"}, NOW, source=None)
    assert store.evict_oldest("get_field_reports", 5, keep_subjects=set()) == 0
    assert store.count("get_field_reports") == 1


def test_a_dry_run_fetches_nothing(store: LocalStore) -> None:
    seen: list[dict[str, Any]] = []
    p = provisioner(store, [asset_tool(seen)], HealthState.HEALTHY)
    summary = p.run(WORKING_SET, NOW, dry_run=True)
    assert seen == []
    assert store.count() == 0
    assert summary.rows_written == 2, "it still says what it would have fetched"


def test_the_summary_is_per_tool_not_per_row(store: LocalStore) -> None:
    p = provisioner(store, [asset_tool(), tickets_tool()], HealthState.HEALTHY)
    body = p.run(WORKING_SET, NOW, trigger=Trigger.MANUAL).as_body()
    assert body["trigger"] == "MANUAL"
    assert body["working_set_size"] == 4
    assert set(body["tools"]) == {"lookup_asset", "list_open_tickets"}
    assert body["skipped_reason"] is None
