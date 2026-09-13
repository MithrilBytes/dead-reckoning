# SPDX-License-Identifier: Apache-2.0
"""Power, and the ceilings a task runs under.

A task that quietly stops thinking when it runs out of tokens has produced an
answer shaped like a conclusion and arrived at by truncation. Every ceiling here
ends a task with an escalation instead, so the log says the task ran out rather
than implying it finished.

Power is separate from health and deliberately so. A truck on its own battery has
not lost anything; it is choosing to spend less, and that choice shows up in which
tier answers rather than in what is reachable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class PowerState(StrEnum):
    MAINS = "MAINS"
    BATTERY_HIGH = "BATTERY_HIGH"
    BATTERY_LOW = "BATTERY_LOW"
    UNKNOWN = "UNKNOWN"


class ResourceSensor(Protocol):
    """Where the node's power state comes from.

    A protocol because real hardware sensing is a stated non-goal: what the
    runtime needs is the reading, not the thermometer.
    """

    def power_state(self) -> PowerState: ...


class FakeResourceSensor:
    """A sensor an operator can set, so the power rule is demonstrable."""

    def __init__(self, state: PowerState = PowerState.MAINS) -> None:
        self._state = state

    def power_state(self) -> PowerState:
        return self._state

    def set(self, state: PowerState) -> PowerState:
        previous, self._state = self._state, state
        return previous


class BudgetExhaustedError(RuntimeError):
    """A ceiling was reached. Carries which one, so the record can say."""

    def __init__(self, which: str, limit: int, consumed: int) -> None:
        super().__init__(f"{which} budget exhausted: {consumed} of {limit}")
        self.which = which
        self.limit = limit
        self.consumed = consumed


@dataclass(slots=True)
class TaskBudget:
    """What one task may spend, and what it has spent.

    Tokens are counted from the endpoint's reported usage where it reports any,
    and from a declared per-turn cost otherwise, because `tokens_remaining` goes
    into the manifest and the manifest hash is stamped on every decision. An
    unmeasured budget would make that hash depend on something nothing records.
    """

    max_tokens_total: int = 20000
    max_wall_s: int = 600
    max_steps: int = 12
    tokens_consumed: int = 0
    steps_taken: int = 0
    started_ms: int = 0

    def tokens_remaining(self) -> int:
        return max(0, self.max_tokens_total - self.tokens_consumed)

    def seconds_remaining(self, now_ms: int) -> int:
        if not self.started_ms:
            return self.max_wall_s
        elapsed = (now_ms - self.started_ms) / 1000
        return max(0, int(self.max_wall_s - elapsed))

    def request_tokens(self, tier_max_tokens: int) -> int:
        """Ask for the smaller of what the tier allows and what is left.

        A task near its ceiling asks for less rather than overrunning and being
        truncated, which is the difference between a short answer and a severed one.
        """
        return max(1, min(tier_max_tokens, self.tokens_remaining()))

    def spend(self, tokens: int, now_ms: int) -> None:
        self.tokens_consumed += max(0, tokens)
        self.steps_taken += 1
        self.check(now_ms)

    def check(self, now_ms: int) -> None:
        if self.steps_taken > self.max_steps:
            raise BudgetExhaustedError("steps", self.max_steps, self.steps_taken)
        if self.tokens_consumed > self.max_tokens_total:
            raise BudgetExhaustedError("tokens", self.max_tokens_total, self.tokens_consumed)
        if self.started_ms and self.seconds_remaining(now_ms) <= 0:
            raise BudgetExhaustedError("wall", self.max_wall_s, self.max_wall_s)
