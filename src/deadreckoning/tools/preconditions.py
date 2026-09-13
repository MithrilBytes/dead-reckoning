# SPDX-License-Identifier: Apache-2.0
"""The checks that justified a deferred action, and re-run before it fires.

This is the mechanism that makes a queued side effect safe rather than merely
delayed. When the agent decides to dispatch a crew and cannot, the reason it
decided so is captured alongside the intent: the crew was free, the ticket was
unassigned. Hours later, when the link returns, those are checked again against
the world as it now is. If headquarters sent someone in the meantime, the action
does not fire.

A checker returns both whether it holds and what it saw, because the value at
deferral and the value at drain are compared and shown to whoever has to
adjudicate the difference.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

CheckerFn = Callable[..., tuple[bool, Any]]


@dataclass(frozen=True, slots=True)
class Observation:
    """One check, and the value behind it."""

    check: str
    args: dict[str, Any]
    holds: bool
    value: Any
    source: str = "LIVE"

    def as_dict(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "args": self.args,
            "observed_value": self.value,
            "observed_source": self.source,
        }


class UnknownCheckerError(KeyError):
    """A contract named a check nothing registered."""


class CheckerRegistry:
    """Named checks a contract may refer to.

    Deliberately a registry rather than free functions on the contract: a
    precondition has to be runnable twice, once when the decision is made and once
    when the action fires, and those happen in different processes hours apart.
    """

    def __init__(self) -> None:
        self._checkers: dict[str, CheckerFn] = {}

    def register(self, name: str, fn: CheckerFn) -> None:
        if name in self._checkers:
            raise ValueError(f"{name} is already registered as a precondition check")
        self._checkers[name] = fn

    def names(self) -> list[str]:
        return sorted(self._checkers)

    def run(self, name: str, args: dict[str, Any], source: str = "LIVE") -> Observation:
        if name not in self._checkers:
            raise UnknownCheckerError(
                f"{name!r} is not a registered precondition check. Known: {self.names()}"
            )
        holds, value = self._checkers[name](**args)
        return Observation(check=name, args=args, holds=bool(holds), value=value, source=source)

    def validate(self, required: list[str]) -> None:
        """Fail at startup rather than at drain, which is hours later and in a truck."""
        missing = sorted(set(required) - set(self._checkers))
        if missing:
            raise UnknownCheckerError(
                f"contracts reference preconditions nothing registers: {', '.join(missing)}"
            )
