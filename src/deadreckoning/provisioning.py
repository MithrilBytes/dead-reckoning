# SPDX-License-Identifier: Apache-2.0
"""Filling the local store while the link is up, so it has answers when it is not.

A LOCAL offline policy is a promise that a tool can answer when its backend is
gone. The promise is worth exactly as much as the rows behind it, and until this
module existed nothing put rows there: the store was read by every local
substitute and written by nothing.

Hydration is declared on the contract, the same place degradation is, and runs
over a flat, explicit working set. The runtime expands nothing. Computing a
closure would need a declared model of subject types and the relations between
them, and modelling domain state is a stated non-goal; the list comes from
tooling that already knows the domain.

The subtle rule is in `_capture`. A refetch that returns identical content still
advances the row's timestamp, because age measures time since the source last
confirmed the value, not time since the value last changed. Getting that backwards
makes a node report data as stale that it has just confirmed is current, and
abstain when it did not need to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from deadreckoning.health import HealthState
from deadreckoning.local_store import LocalStore
from deadreckoning.tools.contract import OfflinePolicy, ToolContract
from deadreckoning.tools.registry import ToolRegistry

USABLE = frozenset({HealthState.HEALTHY, HealthState.SLOW})


class SkipReason(StrEnum):
    BATTERY_LOW = "BATTERY_LOW"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    BACKEND_UNAVAILABLE = "BACKEND_UNAVAILABLE"


class Trigger(StrEnum):
    MANUAL = "MANUAL"
    MAINTENANCE = "MAINTENANCE"
    POST_RECONCILIATION = "POST_RECONCILIATION"


@dataclass(slots=True)
class ToolCounts:
    written: int = 0
    revalidated_unchanged: int = 0
    evicted: int = 0
    failures: dict[str, int] = field(default_factory=dict[str, int])

    def as_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "written": self.written,
            "revalidated_unchanged": self.revalidated_unchanged,
            "evicted": self.evicted,
        }
        if self.failures:
            body["failures"] = dict(self.failures)
        return body


@dataclass(slots=True)
class ProvisionSummary:
    """One pass, in the shape the record body takes.

    Per tool rather than per row: the per-row provenance already lives in the
    store, and a log that listed every row would bury the decisions beside it.
    """

    trigger: Trigger
    working_set_size: int
    tools: dict[str, ToolCounts] = field(default_factory=dict[str, ToolCounts])
    skipped_reason: SkipReason | None = None

    def counts_for(self, tool: str) -> ToolCounts:
        return self.tools.setdefault(tool, ToolCounts())

    @property
    def rows_written(self) -> int:
        return sum(c.written for c in self.tools.values())

    def as_body(self) -> dict[str, Any]:
        return {
            "trigger": str(self.trigger),
            "working_set_size": self.working_set_size,
            "tools": {name: counts.as_dict() for name, counts in sorted(self.tools.items())},
            "skipped_reason": str(self.skipped_reason) if self.skipped_reason else None,
        }


def subject_type(subject: str) -> str:
    """The type prefix of a `<type>:<id>` identifier.

    This is the only structure the runtime relies on, and it relies on it only
    here. Everywhere else a subject is an opaque string.
    """
    return subject.split(":", 1)[0] if ":" in subject else subject


def subject_id(subject: str) -> str:
    return subject.split(":", 1)[1] if ":" in subject else subject


def planned_calls(
    contract: ToolContract, working_set: list[str]
) -> list[tuple[dict[str, Any], str | None]]:
    """Every call this pass would make for one tool, with the subject each is for.

    Pure, so a dry run can show exactly what would be fetched without fetching it.
    """
    if contract.hydrate is None:
        return []
    calls: list[tuple[dict[str, Any], str | None]] = []
    spec = contract.hydrate
    if spec.subject_types and spec.arg_from_subject:
        wanted = set(spec.subject_types)
        calls.extend(
            ({spec.arg_from_subject: subject_id(subject)}, subject)
            for subject in working_set
            if subject_type(subject) in wanted
        )
    calls.extend((dict(args), None) for args in spec.static_args)
    return calls


class Provisioner:
    """Fills the local store for a declared working set."""

    def __init__(
        self,
        registry: ToolRegistry,
        store: LocalStore,
        health: dict[str, HealthState],
        max_rows_per_tool: int = 5000,
        max_calls_per_pass: int = 500,
    ) -> None:
        self.registry = registry
        self.store = store
        self.health = health
        self.max_rows_per_tool = max_rows_per_tool
        self.max_calls_per_pass = max_calls_per_pass

    def hydratable(self) -> list[ToolContract]:
        return [
            c
            for c in self.registry.contracts()
            if c.offline_policy is OfflinePolicy.LOCAL and c.hydrate is not None
        ]

    def backend_usable(self, contract: ToolContract) -> bool:
        if contract.always_local:
            return True
        return self.health.get(contract.backend or "", HealthState.UNKNOWN) in USABLE

    def run(
        self,
        working_set: list[str],
        now_ms: int,
        trigger: Trigger = Trigger.MANUAL,
        power: str = "MAINS",
        dry_run: bool = False,
    ) -> ProvisionSummary:
        summary = ProvisionSummary(trigger=trigger, working_set_size=len(working_set))

        if power == "BATTERY_LOW":
            # Unconditional and not configurable. Speculative work must never
            # compete with the task in front of it.
            summary.skipped_reason = SkipReason.BATTERY_LOW
            return summary

        candidates = self.hydratable()
        if candidates and not any(self.backend_usable(c) for c in candidates):
            summary.skipped_reason = SkipReason.BACKEND_UNAVAILABLE
            return summary

        calls_made = 0
        for contract in candidates:
            if not self.backend_usable(contract):
                continue
            counts = summary.counts_for(contract.name)
            for args, subject in planned_calls(contract, working_set):
                if calls_made >= self.max_calls_per_pass:
                    summary.skipped_reason = SkipReason.BUDGET_EXHAUSTED
                    return summary
                calls_made += 1
                if dry_run:
                    counts.written += 1
                    continue
                self._capture(contract, args, subject, now_ms, counts)
            counts.evicted += self._evict(contract, working_set)
        return summary

    def _capture(
        self,
        contract: ToolContract,
        args: dict[str, Any],
        subject: str | None,
        now_ms: int,
        counts: ToolCounts,
    ) -> None:
        tool = self.registry.get(contract.name)
        if tool.live is None:
            counts.failures["NO_IMPLEMENTATION"] = counts.failures.get("NO_IMPLEMENTATION", 0) + 1
            return
        try:
            content = tool.live(args)
        except Exception as exc:  # noqa: BLE001
            # Classified rather than swallowed: a hydration attempt against a dead
            # backend is an observation like any other, and the caller feeds it to
            # the health monitor.
            name = type(exc).__name__
            counts.failures[name] = counts.failures.get(name, 0) + 1
            return

        before = self.store.get(contract.name, args)
        row = self.store.put(
            contract.name,
            args,
            content,
            captured_ms=now_ms,
            subject=subject,
            source=contract.backend,
        )
        if before is not None and before.content_hash == row.content_hash:
            # Same bytes, fresh confirmation. The timestamp moves and the hash does
            # not, which is what lets a reader see that nothing changed while still
            # treating the row as current (RFC 9111 section 4.3.4 does the same for
            # a revalidated response).
            counts.revalidated_unchanged += 1
        else:
            counts.written += 1

    def _evict(self, contract: ToolContract, working_set: list[str]) -> int:
        """Trim a tool back to its cap, oldest first, never touching what cannot be refetched."""
        cap = min(
            contract.hydrate.max_rows if contract.hydrate else self.max_rows_per_tool,
            self.max_rows_per_tool,
        )
        held = self.store.count(contract.name)
        if held <= cap:
            return 0
        return self.store.evict_oldest(contract.name, held - cap, keep_subjects=set(working_set))
