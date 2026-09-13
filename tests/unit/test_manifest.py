# SPDX-License-Identifier: Apache-2.0
"""What the model is told, and the guarantee that it is told everything."""

from __future__ import annotations

from typing import Any

import pytest

from deadreckoning.clock import TimeTrust
from deadreckoning.health import HealthState
from deadreckoning.local_store import LocalStore
from deadreckoning.manifest import (
    BudgetView,
    TierView,
    build_manifest,
    manifest_hash,
    render_table,
)
from deadreckoning.node import Node
from deadreckoning.records import IdentityState, Mode
from deadreckoning.tools.contract import (
    Approval,
    Consequence,
    HydrateSpec,
    OfflinePolicy,
    SideEffect,
    ToolContract,
)
from deadreckoning.tools.registry import ToolRegistry

NOW = 1_700_000_000_000


@pytest.fixture
def store(node: Node) -> LocalStore:
    return LocalStore(node.database)


@pytest.fixture
def registry() -> ToolRegistry:
    r = ToolRegistry()
    r.register(
        ToolContract(
            name="lookup_asset",
            description="Look up a feeder or customer.",
            backend="gis-api",
            offline_policy=OfflinePolicy.LOCAL,
            side_effect=SideEffect.NONE,
            consequence=Consequence.LOW,
            staleness_budget_s=86400,
            hydrate=HydrateSpec(subject_types=["asset"], arg_from_subject="asset_id"),
        )
    )
    r.register(
        ToolContract(
            name="get_live_load",
            backend="scada-api",
            offline_policy=OfflinePolicy.FAIL,
            side_effect=SideEffect.NONE,
            consequence=Consequence.LOW,
        )
    )
    r.register(
        ToolContract(
            name="dispatch_crew",
            backend="dispatch-api",
            offline_policy=OfflinePolicy.QUEUE,
            side_effect=SideEffect.NON_IDEMPOTENT,
            consequence=Consequence.HIGH,
            approval=Approval.WHEN_NOT_CONNECTED,
            idempotency_key="dispatch:{ticket_id}",
            expiry_s=7200,
        )
    )
    return r


BACKENDS = ("gis-api", "scada-api", "dispatch-api")


def all_backends(state: HealthState) -> dict[str, HealthState]:
    return {name: state for name in BACKENDS}


def manifest(
    registry: ToolRegistry, store: LocalStore, health: dict[str, HealthState], **kwargs: Any
) -> dict[str, Any]:
    return build_manifest(
        mode=kwargs.pop("mode", Mode.ISLANDED),
        tier=TierView("local-q4", 2, "qwen2.5:7b"),
        identity=IdentityState.CACHED,
        identity_ttl_s=5400,
        time_trust=TimeTrust.DRIFTING,
        registry=registry,
        store=store,
        health=health,
        now_ms=NOW,
        budget=BudgetView(18000, 540, "BATTERY_LOW"),
        **kwargs,
    )


def test_every_registered_tool_appears_even_when_it_is_down(
    registry: ToolRegistry, store: LocalStore
) -> None:
    """Omitting an unavailable tool is the obvious economy and it is wrong.

    A model that cannot see a capability exists routes around its absence
    silently. One told it exists and is unreachable can abstain and say why.
    """
    health = all_backends(HealthState.UNREACHABLE)
    result = manifest(registry, store, health)
    assert {t["name"] for t in result["tools"]} == set(registry.names())


def test_an_unavailable_tool_carries_the_reason(registry: ToolRegistry, store: LocalStore) -> None:
    health = all_backends(HealthState.UNREACHABLE)
    live_load = next(
        t for t in manifest(registry, store, health)["tools"] if t["name"] == "get_live_load"
    )
    assert live_load["availability"] == "UNAVAILABLE"
    assert "scada-api" in live_load["reason"]


def test_a_high_consequence_queued_tool_says_approval_is_required(
    registry: ToolRegistry, store: LocalStore
) -> None:
    health = all_backends(HealthState.UNREACHABLE)
    crew = next(
        t for t in manifest(registry, store, health)["tools"] if t["name"] == "dispatch_crew"
    )
    assert crew["availability"] == "QUEUED"
    assert crew["approval"] == "REQUIRED"
    assert crew["consequence"] == "HIGH"


def test_local_tools_report_how_old_their_data_is(
    registry: ToolRegistry, store: LocalStore
) -> None:
    store.put("lookup_asset", {"asset_id": "F-31"}, {"n": 1}, captured_ms=NOW - 3_720_000)
    health = all_backends(HealthState.UNREACHABLE)
    asset = next(
        t for t in manifest(registry, store, health)["tools"] if t["name"] == "lookup_asset"
    )
    assert asset["availability"] == "LOCAL"
    assert asset["data_age_s"] == 3720
    assert asset["stale"] is False


def test_a_healthy_world_reports_everything_live(registry: ToolRegistry, store: LocalStore) -> None:
    health = all_backends(HealthState.HEALTHY)
    result = manifest(registry, store, health, mode=Mode.CONNECTED)
    assert {t["availability"] for t in result["tools"]} == {"LIVE"}


def test_the_hash_changes_when_what_the_model_can_do_changes(
    registry: ToolRegistry, store: LocalStore
) -> None:
    """The hash is what ties a decision to what the model was actually told."""
    healthy = manifest(
        registry,
        store,
        all_backends(HealthState.HEALTHY),
    )
    degraded = manifest(
        registry,
        store,
        {
            "gis-api": HealthState.HEALTHY,
            "scada-api": HealthState.UNREACHABLE,
            "dispatch-api": HealthState.HEALTHY,
        },
    )
    assert manifest_hash(healthy) != manifest_hash(degraded)


def test_the_hash_is_stable_for_the_same_inputs(registry: ToolRegistry, store: LocalStore) -> None:
    health = all_backends(HealthState.HEALTHY)
    assert manifest_hash(manifest(registry, store, health)) == manifest_hash(
        manifest(registry, store, health)
    )


def test_the_builder_answers_a_hypothetical_with_the_same_code(
    registry: ToolRegistry, store: LocalStore
) -> None:
    """The forecast and the live manifest are one implementation, so they cannot drift."""
    now = manifest(
        registry,
        store,
        all_backends(HealthState.HEALTHY),
    )
    if_the_link_dropped = manifest(
        registry,
        store,
        all_backends(HealthState.UNREACHABLE),
    )
    assert [t["availability"] for t in now["tools"]] == ["LIVE", "LIVE", "LIVE"]
    assert sorted(t["availability"] for t in if_the_link_dropped["tools"]) == [
        "QUEUED",
        "UNAVAILABLE",
        "UNAVAILABLE",
    ]


def test_the_rendered_table_leads_with_the_reason(
    registry: ToolRegistry, store: LocalStore
) -> None:
    health = all_backends(HealthState.UNREACHABLE)
    table = render_table(manifest(registry, store, health))
    assert "ISLANDED" in table
    assert "get_live_load" in table
    assert "scada-api" in table
    assert "approval required" in table
