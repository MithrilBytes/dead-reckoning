# SPDX-License-Identifier: Apache-2.0
"""Executing deferred actions, but only if their justification still holds.

Draining is where the project's central claim is either true or it is not. A queue
that replays what it was told to do is a queue. This one re-checks why it was told
to, and an intent whose reason has evaporated does not fire: it becomes a question
for the agent, carrying both the value observed when the decision was made and the
value observed now, so that whoever reads it can see exactly what changed.

Order is by stamp and serialised per subject, so two actions about the same ticket
cannot interleave. Every entry is attempted at most once per pass, and an entry
found in EXECUTING after a crash is never blindly retried: it is verified if the
tool can verify, and surfaced for a human if it cannot.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from deadreckoning.identity import AuthorityOutcome, authority_for
from deadreckoning.outbox import Entry, Outbox, OutboxState, Transition
from deadreckoning.records import IdentityState
from deadreckoning.tools.preconditions import CheckerRegistry
from deadreckoning.tools.registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class PreconditionDelta:
    """What a check said then, and what it says now."""

    check: str
    args: dict[str, Any]
    observed_value: Any
    current_value: Any

    @property
    def changed(self) -> bool:
        return self.observed_value != self.current_value

    def as_dict(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "args": self.args,
            "observed_value": self.observed_value,
            "current_value": self.current_value,
            "holds": False,
        }


@dataclass(slots=True)
class DrainResult:
    executed: list[str] = field(default_factory=list[str])
    precondition_failed: list[str] = field(default_factory=list[str])
    expired: list[str] = field(default_factory=list[str])
    awaiting_approval: list[str] = field(default_factory=list[str])
    identity_blocked: list[str] = field(default_factory=list[str])
    failed: list[str] = field(default_factory=list[str])
    transitions: list[Transition] = field(default_factory=list[Transition])
    deltas: dict[str, list[PreconditionDelta]] = field(
        default_factory=dict[str, list[PreconditionDelta]]
    )

    @property
    def attempted(self) -> int:
        return (
            len(self.executed)
            + len(self.precondition_failed)
            + len(self.expired)
            + len(self.awaiting_approval)
            + len(self.identity_blocked)
            + len(self.failed)
        )


def _always_usable(_name: str) -> bool:
    return True


class Drainer:
    """Runs one drain pass over the outbox."""

    def __init__(
        self,
        outbox: Outbox,
        registry: ToolRegistry,
        checkers: CheckerRegistry,
        *,
        identity: IdentityState,
        connected: bool,
        now_ms: int,
        backend_usable: Callable[[str], bool] | None = None,
    ) -> None:
        self.outbox = outbox
        self.registry = registry
        self.checkers = checkers
        self.identity = identity
        self.connected = connected
        self.now_ms = now_ms
        self.backend_usable: Callable[[str], bool] = backend_usable or _always_usable

    def run(self) -> DrainResult:
        result = DrainResult()
        seen_subjects: set[str] = set()
        for entry in self.outbox.drainable():
            if entry.subject is not None and entry.subject in seen_subjects:
                # Serialised per subject: two actions about one ticket must not
                # interleave, and the second may depend on what the first did.
                continue
            if entry.subject is not None:
                seen_subjects.add(entry.subject)
            self._drain_one(entry, result)
        return result

    def _drain_one(self, entry: Entry, result: DrainResult) -> None:
        contract = self.registry.get(entry.tool).contract

        if entry.expired(self.now_ms):
            # Checked before anything else. An expired intent is never executed,
            # however healthy the world looks now.
            result.expired.append(entry.id)
            result.transitions.append(
                self.outbox.transition(entry, OutboxState.EXPIRED, reason="EXPIRED")
            )
            return

        authority = authority_for(contract, self.identity, self.connected)
        if authority.outcome is AuthorityOutcome.DENY:
            result.identity_blocked.append(entry.id)
            result.transitions.append(
                self.outbox.transition(
                    entry,
                    OutboxState.CANCELLED,
                    reason="IDENTITY_INSUFFICIENT",
                    detail={"authority": authority.reason},
                )
            )
            return
        if authority.outcome is AuthorityOutcome.QUEUE_ONLY:
            # Authority is re-evaluated at drain, not only at creation. An entry
            # approved under a fresh identity must not execute unchanged hours
            # later under an expired one.
            result.identity_blocked.append(entry.id)
            result.transitions.append(
                Transition(
                    entry.id,
                    OutboxState.READY,
                    OutboxState.READY,
                    reason="IDENTITY_INSUFFICIENT",
                    detail={"authority": authority.reason},
                )
            )
            return
        if authority.needs_approval and not (entry.approval and entry.approval.granted):
            result.awaiting_approval.append(entry.id)
            result.transitions.append(
                self.outbox.transition(
                    entry,
                    OutboxState.AWAITING_APPROVAL,
                    reason="APPROVAL_REQUIRED",
                    detail={"authority": authority.reason},
                )
            )
            return

        if entry.args and not self.backend_usable(contract.backend or ""):
            return  # stays READY; the maintenance pass will come back to it

        deltas = self._recheck(entry)
        if deltas:
            result.precondition_failed.append(entry.id)
            result.deltas[entry.id] = deltas
            entry.preconditions = [d.as_dict() for d in deltas]
            result.transitions.append(
                self.outbox.transition(
                    entry,
                    OutboxState.PRECONDITION_FAILED,
                    reason="PRECONDITION_FAILED",
                    detail={"deltas": [d.as_dict() for d in deltas]},
                )
            )
            return

        self._execute(entry, result)

    def _recheck(self, entry: Entry) -> list[PreconditionDelta]:
        """Re-run every check against the world as it is now.

        Against the live backend, which is reachable by definition at this point.
        A check that passed on a cached snapshot hours ago proves nothing about
        the world the action is about to touch.
        """
        failed: list[PreconditionDelta] = []
        for stored in entry.preconditions:
            observation = self.checkers.run(str(stored["check"]), dict(stored["args"]))
            if not observation.holds:
                failed.append(
                    PreconditionDelta(
                        check=observation.check,
                        args=observation.args,
                        observed_value=stored.get("observed_value"),
                        current_value=observation.value,
                    )
                )
        return failed

    def _execute(self, entry: Entry, result: DrainResult) -> None:
        tool = self.registry.get(entry.tool)
        entry.attempts += 1
        self.outbox.transition(entry, OutboxState.EXECUTING, reason="EXECUTING")
        if tool.live is None:
            entry.last_error = "no live implementation registered"
            result.failed.append(entry.id)
            result.transitions.append(
                self.outbox.transition(entry, OutboxState.FAILED, reason="NO_IMPLEMENTATION")
            )
            return
        try:
            tool.live(entry.args)
        except Exception as exc:  # noqa: BLE001
            entry.last_error = f"{type(exc).__name__}: {exc}"
            result.failed.append(entry.id)
            result.transitions.append(
                self.outbox.transition(
                    entry,
                    OutboxState.FAILED,
                    reason="EXECUTION_FAILED",
                    detail={"error": entry.last_error},
                )
            )
            return
        result.executed.append(entry.id)
        result.transitions.append(self.outbox.transition(entry, OutboxState.DONE, reason="DONE"))


def reconcile_interrupted(outbox: Outbox, registry: ToolRegistry) -> list[Transition]:
    """Resolve entries found mid-execution after a crash.

    Never by retrying. Somewhere between the call going out and the record coming
    back, a crew may already have been dispatched, and the one thing that must not
    happen is dispatching a second. If the tool can say whether its effect landed,
    it is asked; if it cannot, the entry is failed loudly and a human decides.
    """
    transitions: list[Transition] = []
    for entry in outbox.all(OutboxState.EXECUTING):
        tool = registry.get(entry.tool)
        if tool.verify is None:
            entry.last_error = "UNKNOWN_EXECUTION_STATE"
            transitions.append(
                outbox.transition(
                    entry,
                    OutboxState.FAILED,
                    reason="UNKNOWN_EXECUTION_STATE",
                    detail={
                        "note": "found mid-execution after a restart and this tool cannot"
                        " verify whether its effect landed, so it must not be retried"
                    },
                )
            )
            continue
        verdict = tool.verify(entry.idempotency_key)
        if verdict == "executed":
            transitions.append(
                outbox.transition(entry, OutboxState.DONE, reason="VERIFIED_EXECUTED")
            )
        elif verdict == "not_executed":
            transitions.append(
                outbox.transition(entry, OutboxState.READY, reason="VERIFIED_NOT_EXECUTED")
            )
        else:
            entry.last_error = "UNKNOWN_EXECUTION_STATE"
            transitions.append(
                outbox.transition(entry, OutboxState.FAILED, reason="UNKNOWN_EXECUTION_STATE")
            )
    return transitions


@dataclass(frozen=True, slots=True)
class ReDecideTask:
    """What the agent is asked when a deferred action's justification has gone.

    Not a retry and not a cancellation. The agent is shown what it believed, what
    is true now, and the action it had intended, and decides again. Cancel, modify
    or proceed with approval are all legitimate answers, and which one is right
    depends on facts only the agent has.
    """

    task_id: str
    original_decision_id: str
    outbox_id: str
    tool: str
    args: dict[str, Any]
    subject: str | None
    deltas: list[PreconditionDelta]

    def instructions(self) -> str:
        rendered_args = ", ".join(f"{k}={v!r}" for k, v in self.args.items())
        lines = [
            "An action you decided on earlier has not been taken, because a condition"
            " that justified it is no longer true.",
            "",
            f"Intended action: {self.tool}({rendered_args})",
            "",
            "What changed:",
        ]
        lines.extend(
            f"  {d.check}({', '.join(f'{k}={v!r}' for k, v in d.args.items())}):"
            f" was {d.observed_value!r} when you decided, is {d.current_value!r} now"
            for d in self.deltas
        )
        lines += [
            "",
            "Decide again. You may cancel the action, modify it, or ask to proceed anyway,"
            " in which case it will wait for a human. Your decision supersedes the original.",
        ]
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "original_decision_id": self.original_decision_id,
            "outbox_id": self.outbox_id,
            "tool": self.tool,
            "args": self.args,
            "subject": self.subject,
            "deltas": [d.as_dict() for d in self.deltas],
        }


def redecide_tasks(
    outbox: Outbox, result: DrainResult, prefix: str = "redecide"
) -> list[ReDecideTask]:
    """One task per entry whose preconditions failed, carrying the delta."""
    tasks: list[ReDecideTask] = []
    for entry_id, deltas in result.deltas.items():
        entry = outbox.get(entry_id)
        if entry is None:
            continue
        tasks.append(
            ReDecideTask(
                task_id=f"{prefix}-{entry_id}",
                original_decision_id=entry.decision_id,
                outbox_id=entry.id,
                tool=entry.tool,
                args=entry.args,
                subject=entry.subject,
                deltas=deltas,
            )
        )
        entry.re_decide_task_id = tasks[-1].task_id
        outbox.transition(
            entry,
            OutboxState.PRECONDITION_FAILED,
            reason="RE_DECIDE_CREATED",
            detail={"re_decide_task_id": tasks[-1].task_id},
        )
    return tasks
