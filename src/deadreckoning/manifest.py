# SPDX-License-Identifier: Apache-2.0
"""What the model is told it can do, right now.

This is the runtime's answer to a model that would otherwise assume. It is rebuilt
before every call from live health, and it lists every registered tool, including
the ones that are down, with the reason. Omitting an unavailable tool would be
the obvious economy and it would be wrong: a model that cannot see a capability
exists routes around its absence silently, while one told the capability exists
and is unreachable can say so and stop.

The builder takes the health vector and the evaluation instant as parameters
rather than reading them, so the same code renders the live manifest and answers
"what could this node still do if the link dropped now", instead of two
implementations drifting apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from deadreckoning.canonical import canonical_json, content_hash
from deadreckoning.clock import TimeTrust
from deadreckoning.health import HealthState
from deadreckoning.local_store import LocalStore
from deadreckoning.records import IdentityState, Mode
from deadreckoning.tools.contract import Availability, Consequence
from deadreckoning.tools.enforcer import ToolEnforcer
from deadreckoning.tools.registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class TierView:
    name: str
    rank: int
    model: str


@dataclass(frozen=True, slots=True)
class BudgetView:
    tokens_remaining: int | None = None
    seconds_remaining: int | None = None
    power: str = "UNKNOWN"


def build_manifest(
    *,
    mode: Mode,
    tier: TierView | None,
    identity: IdentityState,
    identity_ttl_s: int | None,
    time_trust: TimeTrust,
    registry: ToolRegistry,
    store: LocalStore,
    health: dict[str, HealthState],
    now_ms: int,
    budget: BudgetView | None = None,
) -> dict[str, Any]:
    """Render the manifest for a stated health vector and instant.

    Nothing here reads ambient state. Pass a hypothetical health vector and a
    future instant and it answers the forecast question with the same code that
    answers the live one.
    """
    enforcer = ToolEnforcer(registry, store, health, now_ms)
    tools: list[dict[str, Any]] = []
    for contract in sorted(registry.contracts(), key=lambda c: c.name):
        availability, reason = enforcer.availability(contract)
        entry: dict[str, Any] = {"name": contract.name, "availability": str(availability)}
        if contract.description:
            entry["description"] = contract.description
        if reason:
            entry["reason"] = reason
        if availability in (Availability.LOCAL, Availability.LOCAL_STALE):
            age = store.freshest_age_s(contract.name, now_ms)
            if age is not None:
                entry["data_age_s"] = age
            entry["stale"] = availability is Availability.LOCAL_STALE
        if availability is Availability.QUEUED:
            entry["consequence"] = str(contract.consequence)
            if contract.consequence is Consequence.HIGH:
                entry["approval"] = "REQUIRED"
        tools.append(entry)

    manifest: dict[str, Any] = {
        "mode": str(mode),
        "identity": {"state": str(identity), "grant_ttl_remaining_s": identity_ttl_s},
        "time_trust": str(time_trust),
        "tools": tools,
    }
    if tier is not None:
        manifest["tier"] = {"name": tier.name, "rank": tier.rank, "model": tier.model}
    if budget is not None:
        manifest["budget"] = {
            "tokens_remaining": budget.tokens_remaining,
            "seconds_remaining": budget.seconds_remaining,
            "power": budget.power,
        }
    return manifest


def manifest_hash(manifest: dict[str, Any]) -> str:
    """The hash stamped on every decision made under this manifest.

    It is over the canonical form, so a decision can be tied to exactly what the
    model was told rather than to a description of it.
    """
    return content_hash(manifest)


def render_table(manifest: dict[str, Any]) -> str:
    """The manifest as the model sees it: a table, not JSON.

    Rendered rather than dumped because this goes into a prompt, and the column
    that matters most is the reason a tool is unavailable.
    """
    lines = [
        f"Mode: {manifest['mode']}   "
        f"Identity: {manifest['identity']['state']}   "
        f"Time trust: {manifest['time_trust']}",
        "",
        "| tool | availability | age | note |",
        "| --- | --- | --- | --- |",
    ]
    for tool in manifest["tools"]:
        age = tool.get("data_age_s")
        note = tool.get("reason") or ""
        if tool.get("approval") == "REQUIRED":
            note = (note + "; approval required").lstrip("; ")
        lines.append(
            f"| {tool['name']} | {tool['availability']} | {'' if age is None else age} | {note} |"
        )
    return "\n".join(lines)


def canonical_manifest(manifest: dict[str, Any]) -> bytes:
    return canonical_json(manifest)
