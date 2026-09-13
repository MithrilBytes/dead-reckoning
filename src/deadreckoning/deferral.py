# SPDX-License-Identifier: Apache-2.0
"""Turning a tool call the world cannot take right now into a recorded intent.

The enforcer decides that a call must be deferred. This is what actually defers
it, and the interesting work is not the queuing: it is capturing, at this moment,
the checks that made the call the right thing to do. Hours later those are what
the drain re-runs, and an intent that arrives with no record of its own
justification is just a replayed command.

Preconditions are evaluated against the best source available, which while
disconnected is the local snapshot rather than the backend. The source is recorded
alongside the value, because "the crew was free according to a four hour old
cache" and "the crew was free according to dispatch" are different claims and the
adjudicator needs to tell them apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from deadreckoning.identity import Authority, AuthorityOutcome, authority_for
from deadreckoning.outbox import DuplicateIntentError, Entry, Outbox
from deadreckoning.records import IdentityState
from deadreckoning.tools.contract import ToolContract
from deadreckoning.tools.preconditions import CheckerRegistry, Observation


@dataclass(slots=True)
class DeferralContext:
    """Who is deferring, about what, and under what authority."""

    decision_id: str
    subject: str | None
    identity: IdentityState
    connected: bool
    observed_source: str = "LOCAL"


class OutboxDeferrer:
    """The enforcer's queue path, backed by the real outbox.

    Implements the narrow port the enforcer depends on, so the enforcer stays
    ignorant of approval, preconditions and identity, which are not its business.
    """

    def __init__(
        self,
        outbox: Outbox,
        checkers: CheckerRegistry,
        next_id: Any,
        now_ms: Any,
        hlc: Any,
    ) -> None:
        self.outbox = outbox
        self.checkers = checkers
        self._next_id = next_id
        self._now_ms = now_ms
        self._hlc = hlc
        self.context: DeferralContext | None = None
        self.duplicates: list[str] = []
        self.created: list[str] = []

    def authority(self, contract: ToolContract) -> Authority:
        context = self.context
        if context is None:
            return Authority(AuthorityOutcome.ALLOW, "no context")
        return authority_for(contract, context.identity, context.connected)

    def capture(self, contract: ToolContract, args: dict[str, Any]) -> list[Observation]:
        """Run each declared check now, and keep what it saw.

        A checker that is not registered is a startup error rather than something
        to discover here, so anything reaching this point is expected to run.
        """
        observations: list[Observation] = []
        source = self.context.observed_source if self.context else "LOCAL"
        for spec in contract.preconditions:
            checker_args = {
                checker_arg: args[call_arg]
                for checker_arg, call_arg in spec.args_from.items()
                if call_arg in args
            }
            observations.append(self.checkers.run(spec.check, checker_args, source=source))
        return observations

    def defer(self, contract: ToolContract, args: dict[str, Any]) -> str:
        """Record the intent. Returns the entry id, or the existing one if duplicate.

        A duplicate is not an error to the caller. Two callers reaching the same
        conclusion about the same ticket is a normal thing for an agent to do, and
        the right response is one action, not a failed step.
        """
        context = self.context
        assert context is not None, "deferral needs a context; the loop sets one per step"
        authority = self.authority(contract)
        try:
            entry = self.outbox.defer(
                contract=contract,
                args=args,
                subject=context.subject,
                decision_id=context.decision_id,
                deferral_record_id=context.decision_id,
                observations=self.capture(contract, args),
                created_hlc=self._hlc(),
                now_ms=self._now_ms(),
                approval_required=authority.needs_approval,
                approval_reason=authority.reason if authority.needs_approval else None,
                entry_id=self._next_id(),
            )
        except DuplicateIntentError as exc:
            self.duplicates.append(exc.key)
            return exc.existing_id
        self.created.append(entry.id)
        return entry.id

    def entry(self, entry_id: str) -> Entry | None:
        return self.outbox.get(entry_id)

    def link_to_decision(self, decision_id: str) -> list[str]:
        """Point everything deferred during this task at the decision it serves."""
        linked = list(self.created)
        if linked:
            self.outbox.link_decision(linked, decision_id)
        self.created.clear()
        return linked
