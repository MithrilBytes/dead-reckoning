# SPDX-License-Identifier: Apache-2.0
"""Deterministic fault injection at the transport boundary.

A demo of a system that survives outages is worthless if the audience cannot tell
whether the outage was real. So every injected fault is recorded as injected, and
the log distinguishes a link that failed from a link somebody switched off for the
camera. That honesty is the point of this module, more than the injection itself.

The injector sits inside the transport, not in front of the health monitor, so an
injected fault travels the same path as a real one: the client raises, the
classifier names it, the breaker counts it. Nothing downstream has a special case
for chaos, which is what makes the demo evidence rather than theatre.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from deadreckoning.health import FailureClass


class ChaosDisabledError(RuntimeError):
    """Fault injection was requested while it is switched off."""


class InjectedFaultError(Exception):
    """Raised by the transport when a fault is armed for this dependency."""

    def __init__(self, dependency: str, failure_class: FailureClass) -> None:
        super().__init__(f"injected {failure_class} on {dependency}")
        self.dependency = dependency
        self.failure_class = failure_class


class Clock(Protocol):
    def __call__(self) -> float: ...


@dataclass(frozen=True, slots=True)
class Fault:
    """One armed fault. `until` is None for an indefinite one."""

    failure_class: FailureClass | None = None
    latency_ms: int | None = None
    drop_pct: float | None = None
    until: float | None = None

    def expired(self, now: float) -> bool:
        return self.until is not None and now >= self.until

    def as_dict(self) -> dict[str, object]:
        return {
            "failure_class": str(self.failure_class) if self.failure_class else None,
            "latency_ms": self.latency_ms,
            "drop_pct": self.drop_pct,
            "until": self.until,
        }


class FaultInjector:
    """Per dependency faults, armed and lifted explicitly.

    Inert unless enabled, and it refuses to enable under a production profile. A
    runtime that can be told to break itself in production is a liability whatever
    the operator intended.
    """

    def __init__(self, enabled: bool, profile: str, now: Clock) -> None:
        if enabled and profile == "production":
            raise ChaosDisabledError(
                "fault injection cannot be enabled while the profile is 'production'"
            )
        self._enabled = enabled
        self._now = now
        self._faults: dict[str, Fault] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _require_enabled(self) -> None:
        if not self._enabled:
            raise ChaosDisabledError("fault injection is disabled; set [chaos] enabled = true")

    def arm(
        self,
        dependency: str,
        failure_class: FailureClass | None = None,
        latency_ms: int | None = None,
        drop_pct: float | None = None,
        for_seconds: float | None = None,
    ) -> Fault:
        self._require_enabled()
        fault = Fault(
            failure_class=failure_class,
            latency_ms=latency_ms,
            drop_pct=drop_pct,
            until=None if for_seconds is None else self._now() + for_seconds,
        )
        self._faults[dependency] = fault
        return fault

    def restore(self, dependency: str) -> Fault | None:
        """Lift the fault on one dependency, returning what was lifted."""
        self._require_enabled()
        return self._faults.pop(dependency, None)

    def restore_all(self) -> dict[str, Fault]:
        """Lift every armed fault at once.

        One call, one derivation. Restoring dependencies one at a time would walk
        the mode controller through a series of intermediate states that never
        really existed, and fill the log with transitions nobody experienced.

        It deliberately does not touch the power state, which is an independent
        condition. Conflating the two would misattribute a routing change.
        """
        self._require_enabled()
        lifted = dict(self._faults)
        self._faults.clear()
        return lifted

    def load(self, faults: dict[str, Fault]) -> None:
        """Reinstate persisted faults when a process resumes."""
        if not self._enabled:
            return
        self._faults.update(faults)

    def armed(self, dependency: str) -> Fault | None:
        fault = self._faults.get(dependency)
        if fault is None:
            return None
        if fault.expired(self._now()):
            del self._faults[dependency]
            return None
        return fault

    def all_armed(self) -> dict[str, Fault]:
        return {name: f for name in list(self._faults) if (f := self.armed(name)) is not None}

    def check(self, dependency: str) -> int:
        """Called by the transport before a real call.

        Raises if a failure is armed, and otherwise returns the latency in
        milliseconds to add. Latency alone is not a failure: it may or may not
        push the dependency past its threshold, and the health monitor decides
        that, not the injector.
        """
        fault = self.armed(dependency)
        if fault is None:
            return 0
        if fault.failure_class is not None:
            raise InjectedFaultError(dependency, fault.failure_class)
        return fault.latency_ms or 0
