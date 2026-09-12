# SPDX-License-Identifier: Apache-2.0
"""The outage scenario's tools, their contracts, and their precondition checkers.

The contracts are the interesting part. Each one is a claim about what this tool
does when its backend is gone, and together they are what makes the scenario more
than a story: the same agent, the same tools, in a world where three of the
backends have stopped answering, behaves differently because the contracts say it
must.

The backends here are in-process dictionaries rather than HTTP services. The
headless scenario runs deterministically in one process; the live demo swaps these
for real fake services over loopback without the contracts changing at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from deadreckoning.tools.contract import (
    Approval,
    Consequence,
    HydrateSpec,
    OfflinePolicy,
    PreconditionSpec,
    SideEffect,
    ToolContract,
)
from deadreckoning.tools.registry import ToolRegistry

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def load_fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / f"{name}.json").read_text())


@dataclass(slots=True)
class Backends:
    """In-process stand-ins for the utility's systems.

    They record what was done to them, because the scenario's assertions are about
    side effects having happened exactly once, which is not observable from the
    agent's own log alone.
    """

    assets: dict[str, Any] = field(default_factory=lambda: load_fixture("assets"))
    tickets: dict[str, Any] = field(default_factory=lambda: load_fixture("tickets"))
    crews: dict[str, Any] = field(default_factory=lambda: load_fixture("crews"))
    load_by_feeder: dict[str, int] = field(
        default_factory=lambda: {"F-31": 74, "F-32": 91, "F-33": 55}
    )
    weather: dict[str, Any] = field(default_factory=lambda: {"area": "SB-3", "wind_kph": 40})
    field_reports: dict[str, list[dict[str, Any]]] = field(
        default_factory=dict[str, list[dict[str, Any]]]
    )
    executed: list[tuple[str, dict[str, Any]]] = field(
        default_factory=list[tuple[str, dict[str, Any]]]
    )

    def ticket(self, ticket_id: str) -> dict[str, Any] | None:
        return next((t for t in self.tickets["tickets"] if t["id"] == ticket_id), None)

    def crew(self, crew_id: str) -> dict[str, Any] | None:
        return next((c for c in self.crews["crews"] if c["id"] == crew_id), None)

    def asset(self, asset_id: str) -> dict[str, Any] | None:
        for feeder in self.assets["feeders"]:
            if feeder["id"] == asset_id:
                return feeder
        for customer in self.assets["priority_customers"]:
            if customer["id"] == asset_id:
                return customer
        return None


CONTRACTS: dict[str, ToolContract] = {
    "lookup_asset": ToolContract(
        name="lookup_asset",
        description="Look up a feeder or a priority customer.",
        backend="gis-api",
        offline_policy=OfflinePolicy.LOCAL,
        side_effect=SideEffect.NONE,
        consequence=Consequence.LOW,
        staleness_budget_s=86400,
        hydrate=HydrateSpec(subject_types=["feeder", "customer"], arg_from_subject="asset_id"),
    ),
    "list_open_tickets": ToolContract(
        name="list_open_tickets",
        description="Every open outage ticket.",
        backend="ticket-api",
        offline_policy=OfflinePolicy.LOCAL,
        side_effect=SideEffect.NONE,
        consequence=Consequence.LOW,
        staleness_budget_s=3600,
        hydrate=HydrateSpec(static_args=[{}]),
    ),
    "lookup_crew": ToolContract(
        name="lookup_crew",
        description="A crew's base and whether it is free.",
        backend="dispatch-api",
        offline_policy=OfflinePolicy.LOCAL,
        side_effect=SideEffect.NONE,
        consequence=Consequence.LOW,
        staleness_budget_s=86400,
        hydrate=HydrateSpec(subject_types=["crew"], arg_from_subject="crew_id"),
    ),
    "get_field_reports": ToolContract(
        name="get_field_reports",
        description="Observations entered on this truck. Never fetched from anywhere.",
        backend=None,
        offline_policy=OfflinePolicy.LOCAL,
        side_effect=SideEffect.NONE,
        consequence=Consequence.LOW,
        staleness_budget_s=None,
        local_source="field_input",
    ),
    "get_live_load": ToolContract(
        name="get_live_load",
        description="Present load on a feeder. Online only; there is no useful cached value.",
        backend="scada-api",
        offline_policy=OfflinePolicy.FAIL,
        side_effect=SideEffect.NONE,
        consequence=Consequence.LOW,
    ),
    "weather_forecast": ToolContract(
        name="weather_forecast",
        description="Forecast for an area.",
        backend="weather-api",
        offline_policy=OfflinePolicy.LOCAL,
        side_effect=SideEffect.NONE,
        consequence=Consequence.LOW,
        staleness_budget_s=21600,
        fail_when_stale=True,
        hydrate=HydrateSpec(static_args=[{"area": "SB-3"}]),
    ),
    "set_ticket_priority": ToolContract(
        name="set_ticket_priority",
        description="Set a ticket's priority.",
        backend="ticket-api",
        offline_policy=OfflinePolicy.QUEUE,
        side_effect=SideEffect.IDEMPOTENT,
        consequence=Consequence.MEDIUM,
        preconditions=[PreconditionSpec(check="ticket_open", args_from={"ticket_id": "ticket_id"})],
        idempotency_key="prio:{ticket_id}:{priority}",
        expiry_s=14400,
    ),
    "dispatch_crew": ToolContract(
        name="dispatch_crew",
        description="Send a crew to a ticket.",
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
    ),
    "notify_customer": ToolContract(
        name="notify_customer",
        description="Send a customer a notification.",
        backend="notify-api",
        offline_policy=OfflinePolicy.QUEUE,
        side_effect=SideEffect.IDEMPOTENT,
        consequence=Consequence.LOW,
        preconditions=[PreconditionSpec(check="ticket_open", args_from={"ticket_id": "ticket_id"})],
        idempotency_key="notify:{customer_id}:{ticket_id}:{template}",
        expiry_s=7200,
    ),
}


def build_registry(backends: Backends) -> ToolRegistry:
    registry = ToolRegistry()

    def lookup_asset(args: dict[str, Any]) -> Any:
        return backends.asset(str(args["asset_id"]))

    def list_open_tickets(_args: dict[str, Any]) -> Any:
        return [t for t in backends.tickets["tickets"] if t["open"]]

    def lookup_crew(args: dict[str, Any]) -> Any:
        return backends.crew(str(args["crew_id"]))

    def get_field_reports(args: dict[str, Any]) -> Any:
        return backends.field_reports.get(str(args["ticket_id"]), [])

    def get_live_load(args: dict[str, Any]) -> Any:
        return {
            "feeder_id": args["feeder_id"],
            "load_pct": backends.load_by_feeder[str(args["feeder_id"])],
        }

    def weather_forecast(_args: dict[str, Any]) -> Any:
        return dict(backends.weather)

    def set_ticket_priority(args: dict[str, Any]) -> Any:
        ticket = backends.ticket(str(args["ticket_id"]))
        if ticket is not None:
            ticket["priority"] = args["priority"]
        backends.executed.append(("set_ticket_priority", dict(args)))
        return {"ok": True, "ticket_id": args["ticket_id"], "priority": args["priority"]}

    def dispatch_crew(args: dict[str, Any]) -> Any:
        ticket = backends.ticket(str(args["ticket_id"]))
        crew = backends.crew(str(args["crew_id"]))
        if ticket is not None:
            ticket["assigned"] = True
        if crew is not None:
            crew["available"] = False
        backends.executed.append(("dispatch_crew", dict(args)))
        return {"ok": True, "crew_id": args["crew_id"], "ticket_id": args["ticket_id"]}

    def notify_customer(args: dict[str, Any]) -> Any:
        backends.executed.append(("notify_customer", dict(args)))
        return {"ok": True, "customer_id": args["customer_id"]}

    implementations = {
        "lookup_asset": lookup_asset,
        "list_open_tickets": list_open_tickets,
        "lookup_crew": lookup_crew,
        "get_field_reports": get_field_reports,
        "get_live_load": get_live_load,
        "weather_forecast": weather_forecast,
        "set_ticket_priority": set_ticket_priority,
        "dispatch_crew": dispatch_crew,
        "notify_customer": notify_customer,
    }
    for name, contract in CONTRACTS.items():
        registry.register(contract, live=implementations[name])
    return registry


# --- precondition checkers ---------------------------------------------------
# Each reads the live backend when it can and the local snapshot otherwise, and
# returns both whether it holds and what it saw, because the value observed at
# deferral is compared against the value at drain.


def ticket_open(backends: Backends, ticket_id: str) -> tuple[bool, Any]:
    ticket = backends.ticket(ticket_id)
    return (bool(ticket and ticket["open"]), ticket["open"] if ticket else None)


def ticket_unassigned(backends: Backends, ticket_id: str) -> tuple[bool, Any]:
    ticket = backends.ticket(ticket_id)
    return (bool(ticket and not ticket["assigned"]), not ticket["assigned"] if ticket else None)


def crew_available(backends: Backends, crew_id: str) -> tuple[bool, Any]:
    crew = backends.crew(crew_id)
    return (bool(crew and crew["available"]), crew["available"] if crew else None)


CHECKERS = {
    "ticket_open": ticket_open,
    "ticket_unassigned": ticket_unassigned,
    "crew_available": crew_available,
}
