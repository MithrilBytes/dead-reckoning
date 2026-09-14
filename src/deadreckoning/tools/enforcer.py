# SPDX-License-Identifier: Apache-2.0
"""Deciding how a tool call is dispatched, and refusing a decision that ignored it.

This is the part of the runtime that does not trust the model. Asking a model
nicely not to answer from data it does not have works until the moment it
matters: under pressure, with a plausible-looking gap, a model will fill it. So
the dispatch path is chosen here from the tool's declared contract and the
current health of its backend, and the model is told what happened instead of
being consulted about it.

The second half is the enforcement of abstention. A model that calls an
unavailable tool, gets told it is unavailable, and then produces a confident final
answer that depends on it has done the one thing this system exists to prevent.
The runtime converts that into an abstention, keeps what the model actually said,
and records that it had to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from deadreckoning.health import HealthState
from deadreckoning.local_store import LocalStore
from deadreckoning.tools.contract import Availability, OfflinePolicy, ToolContract
from deadreckoning.tools.registry import ToolRegistry

USABLE = frozenset({HealthState.HEALTHY, HealthState.SLOW})


class OutboxPort(Protocol):
    """What the enforcer needs from the outbox.

    Narrow on purpose: the enforcer's job ends at recording the intent, and
    everything about draining it belongs to the outbox itself.
    """

    def defer(self, contract: ToolContract, args: dict[str, Any]) -> str: ...


@dataclass(frozen=True, slots=True)
class Dispatch:
    """The result of one tool call, in the vocabulary the model and log share."""

    tool: str
    availability: Availability
    result: Any = None
    reason: str | None = None
    data_age_s: int | None = None
    content_hash: str | None = None
    outbox_id: str | None = None
    latency_ms: int | None = None

    @property
    def usable_answer(self) -> bool:
        """Whether this produced something a decision may rest on."""
        return self.availability in (
            Availability.LIVE,
            Availability.LIVE_SLOW,
            Availability.LOCAL,
            Availability.LOCAL_STALE,
        )


class ToolEnforcer:
    """Chooses the path for every call. The model never does."""

    def __init__(
        self,
        registry: ToolRegistry,
        store: LocalStore,
        health: dict[str, HealthState],
        now_ms: int,
        outbox: OutboxPort | None = None,
    ) -> None:
        self.registry = registry
        self.store = store
        self.health = health
        self.now_ms = now_ms
        self.outbox = outbox

    def backend_state(self, contract: ToolContract) -> HealthState:
        """A tool with no backend is always reachable; it depends on nothing.

        An unprobed backend is treated as unreachable, not as working. UNKNOWN
        means nobody has checked, and acting on an unchecked dependency is the
        assumption this runtime exists to avoid.
        """
        if contract.always_local:
            return HealthState.HEALTHY
        return self.health.get(contract.backend or "", HealthState.UNKNOWN)

    def availability(self, contract: ToolContract) -> tuple[Availability, str | None]:
        """What the manifest should say about this tool right now.

        Pure: it reports without running anything, so the manifest can be built,
        hashed and forecast without side effects.
        """
        state = self.backend_state(contract)
        if state in USABLE:
            slow = state is HealthState.SLOW
            if (
                slow
                and contract.offline_policy is OfflinePolicy.LOCAL
                and contract.prefer_local_when_slow
            ):
                return self._local_availability(contract)
            return (Availability.LIVE_SLOW if slow else Availability.LIVE), None

        reason = f"backend {contract.backend} is {state}"
        if contract.offline_policy is OfflinePolicy.FAIL:
            return Availability.UNAVAILABLE, reason
        if contract.offline_policy is OfflinePolicy.QUEUE:
            return Availability.QUEUED, reason
        return self._local_availability(contract)

    def _local_availability(self, contract: ToolContract) -> tuple[Availability, str | None]:
        """LOCAL tools report on their freshest row, since the manifest is per tool.

        A tool with no rows at all is unavailable, not merely stale: there is
        nothing to be stale about, and saying LOCAL would promise an answer that
        does not exist.
        """
        if contract.local_source is not None and self.store.count(contract.name) == 0:
            return Availability.LOCAL, None
        if self.store.count(contract.name) == 0:
            return Availability.UNAVAILABLE, f"no local data for {contract.name}"
        return Availability.LOCAL, None

    def dispatch(self, name: str, args: dict[str, Any]) -> Dispatch:
        """Run one call down whichever path its contract and the world allow.

        Only backend health and offline policy choose the path. Identity and
        approval are not consulted here yet, so a call against a healthy backend
        runs live whatever the identity state or the tool's approval setting.
        They are applied only to queued calls, when the outbox entry is created
        and again when the outbox drains.
        """
        tool = self.registry.get(name)
        contract = tool.contract
        state = self.backend_state(contract)

        if state in USABLE:
            prefer_local = (
                state is HealthState.SLOW
                and contract.offline_policy is OfflinePolicy.LOCAL
                and contract.prefer_local_when_slow
            )
            if not prefer_local:
                return self._live(tool.live, contract, args, slow=state is HealthState.SLOW)

        reason = f"backend {contract.backend} is {state}"
        if contract.offline_policy is OfflinePolicy.FAIL:
            return Dispatch(contract.name, Availability.UNAVAILABLE, reason=reason)
        if contract.offline_policy is OfflinePolicy.QUEUE:
            return self._queue(contract, args, reason)
        return self._local(tool.local, contract, args)

    def _live(
        self, live: Any, contract: ToolContract, args: dict[str, Any], slow: bool
    ) -> Dispatch:
        if live is None:
            return Dispatch(
                contract.name,
                Availability.UNAVAILABLE,
                reason=f"{contract.name} has no live implementation registered",
            )
        result = live(args)
        return Dispatch(
            contract.name,
            Availability.LIVE_SLOW if slow else Availability.LIVE,
            result=result,
        )

    def _queue(self, contract: ToolContract, args: dict[str, Any], reason: str) -> Dispatch:
        if self.outbox is None:
            return Dispatch(
                contract.name,
                Availability.UNAVAILABLE,
                reason=f"{reason}, and no outbox is attached to defer it",
            )
        outbox_id = self.outbox.defer(contract, args)
        return Dispatch(contract.name, Availability.QUEUED, outbox_id=outbox_id, reason=reason)

    def _local(self, local: Any, contract: ToolContract, args: dict[str, Any]) -> Dispatch:
        row = self.store.get(contract.name, args)
        if row is None:
            if local is not None:
                result = local(args)
                return Dispatch(contract.name, Availability.LOCAL, result=result, data_age_s=None)
            return Dispatch(
                contract.name,
                Availability.UNAVAILABLE,
                reason=f"no local data for {contract.name} with these arguments",
            )
        age = row.age_s(self.now_ms)
        stale = contract.staleness_budget_s is not None and age > contract.staleness_budget_s
        if stale and contract.fail_when_stale:
            return Dispatch(
                contract.name,
                Availability.UNAVAILABLE,
                reason=(
                    f"local data for {contract.name} is {age}s old, past its"
                    f" {contract.staleness_budget_s}s budget, and this tool refuses stale data"
                ),
                data_age_s=age,
            )
        return Dispatch(
            contract.name,
            Availability.LOCAL_STALE if stale else Availability.LOCAL,
            result=row.content,
            data_age_s=age,
            content_hash=row.content_hash,
        )


@dataclass(frozen=True, slots=True)
class EnforcementResult:
    """Whether a decision may stand as the model wrote it."""

    allowed: bool
    unavailable_dependencies: list[str]

    @property
    def reason(self) -> str:
        joined = ", ".join(self.unavailable_dependencies)
        return f"the decision depends on {joined}, which returned UNAVAILABLE during this task"


def validate_decision(
    depends_on: list[str], dispatches: list[Dispatch], abstained: bool
) -> EnforcementResult:
    """Refuse a final answer that rests on a tool which never answered.

    Checked against what dispatch actually returned rather than against the tool's
    declared policy. A FAIL tool with its backend down is only the commonest way
    to reach UNAVAILABLE: a LOCAL tool past its staleness budget with
    `fail_when_stale` set is equally unavailable, and scoping this to policy would
    let that path through, which is the one thing the rule exists to stop.
    """
    if abstained:
        return EnforcementResult(allowed=True, unavailable_dependencies=[])
    unavailable = sorted(
        {d.tool for d in dispatches if d.availability is Availability.UNAVAILABLE} & set(depends_on)
    )
    return EnforcementResult(allowed=not unavailable, unavailable_dependencies=unavailable)
