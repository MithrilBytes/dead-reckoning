# SPDX-License-Identifier: Apache-2.0
"""Failure classification, circuit breakers, and the health of each dependency.

Most systems record that a call failed. This one records *how*, because the
difference decides what the agent may do next. A name that will not resolve, a
port that refuses, a handshake that fails because the clock is wrong, and a token
that has expired all look alike to a retry loop and mean entirely different
things to a truck in a storm: one is the network, one is the server, one is this
machine's own clock, and one is an identity problem that no amount of waiting will
fix.

A certificate error is the sharpest case. On a laptop whose clock has drifted
after hours off the grid, "certificate not yet valid" is almost never an attack;
it is the clock. Reporting it as a security failure sends a crew chasing a
compromise that is not there, so the classification depends on whether this node
currently trusts its own clock.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum


class FailureClass(StrEnum):
    OK = "OK"
    DNS_FAILURE = "DNS_FAILURE"
    CONNECT_REFUSED = "CONNECT_REFUSED"
    CONNECT_TIMEOUT = "CONNECT_TIMEOUT"
    TLS_CLOCK_SKEW = "TLS_CLOCK_SKEW"
    TLS_OTHER = "TLS_OTHER"
    AUTH_FAILURE = "AUTH_FAILURE"
    RATE_LIMITED = "RATE_LIMITED"
    SERVER_ERROR = "SERVER_ERROR"
    READ_TIMEOUT = "READ_TIMEOUT"
    SLOW_RESPONSE = "SLOW_RESPONSE"
    PROTOCOL_ERROR = "PROTOCOL_ERROR"


class HealthState(StrEnum):
    HEALTHY = "HEALTHY"
    SLOW = "SLOW"
    UNREACHABLE = "UNREACHABLE"
    AUTH_BROKEN = "AUTH_BROKEN"
    UNKNOWN = "UNKNOWN"


class BreakerState(StrEnum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class ObservationSource(StrEnum):
    PASSIVE = "PASSIVE"
    CANARY = "CANARY"
    FAULT_INJECTION = "FAULT_INJECTION"
    FAULT_RESTORE = "FAULT_RESTORE"


USABLE_STATES = frozenset({HealthState.HEALTHY, HealthState.SLOW})
"""States a caller may act on. Everything else, UNKNOWN included, is not usable:
a dependency nobody has checked is not a dependency known to work."""

TRIP_IMMEDIATELY = frozenset({FailureClass.DNS_FAILURE, FailureClass.CONNECT_REFUSED})
"""Classes that need only one confirmation rather than a run of failures. A name
that does not resolve and a port that refuses are answers, not timeouts, and
waiting for a third identical answer helps nobody."""

REMEDIATION: dict[FailureClass, str] = {
    FailureClass.DNS_FAILURE: (
        "name did not resolve: check the resolver or the link, not the server"
    ),
    FailureClass.CONNECT_REFUSED: ("nothing listening: the service is down or the port is wrong"),
    FailureClass.CONNECT_TIMEOUT: "no answer at all: likely the link rather than the service",
    FailureClass.TLS_CLOCK_SKEW: (
        "certificate rejected while this node does not trust its own clock: "
        "almost certainly skew, not a compromised endpoint. "
        "Re-sync time before suspecting the peer."
    ),
    FailureClass.TLS_OTHER: "handshake failed for a reason other than validity dates",
    FailureClass.AUTH_FAILURE: (
        "credentials rejected: waiting will not fix this, identity must refresh"
    ),
    FailureClass.RATE_LIMITED: "throttled: back off, the dependency is healthy",
    FailureClass.SERVER_ERROR: "the dependency answered but failed internally",
    FailureClass.READ_TIMEOUT: (
        "connected but no response in time: the service is overloaded or stuck"
    ),
    FailureClass.SLOW_RESPONSE: "answering, but past its threshold",
    FailureClass.PROTOCOL_ERROR: "answered with something this client cannot parse",
    FailureClass.OK: "",
}


@dataclass(frozen=True, slots=True)
class BreakerPolicy:
    """Breaker defaults. Held as data so tests can drive a breaker without waiting.

    The closed, open and half-open states follow the circuit breaker in Michael
    Nygard, Release It! (Pragmatic Bookshelf, 2007).
    """

    open_after_consecutive_failures: int = 3
    close_after_consecutive_successes: int = 2
    half_open_backoff_initial_s: float = 5.0
    half_open_backoff_max_s: float = 120.0


@dataclass(frozen=True, slots=True)
class DependencyHealth:
    """One dependency's assessment. Immutable; every change produces a new value.

    `needs_reconciliation` is sticky. It is set when the dependency becomes
    unusable and cleared only once a reconciliation that covered it has finished,
    so a dependency that recovers through SLOW, or that flaps twice on the way
    back, still reconciles exactly once.
    """

    name: str
    state: HealthState = HealthState.UNKNOWN
    breaker: BreakerState = BreakerState.CLOSED
    last_failure_class: FailureClass | None = None
    last_latency_ms: int | None = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    backoff_s: float = 0.0
    needs_reconciliation: bool = False

    @property
    def usable(self) -> bool:
        return self.state in USABLE_STATES

    def snapshot(self, dependency_type: str) -> dict[str, object]:
        return {
            "type": dependency_type,
            "state": str(self.state),
            "breaker": str(self.breaker),
            "last_failure_class": str(self.last_failure_class) if self.last_failure_class else None,
            "last_latency_ms": self.last_latency_ms,
        }


@dataclass(frozen=True, slots=True)
class Transition:
    """What one observation did. Emitted as a record only when something changed."""

    dependency: str
    before: DependencyHealth
    after: DependencyHealth
    failure_class: FailureClass
    source: ObservationSource
    latency_ms: int | None

    @property
    def state_changed(self) -> bool:
        return self.before.state is not self.after.state

    @property
    def remediation(self) -> str:
        return REMEDIATION.get(self.failure_class, "")


def classify_status(status_code: int) -> FailureClass:
    """Map an HTTP status to a class.

    401 and 403 are deliberately not `UNREACHABLE`. The dependency answered; it
    refused. That distinction is what lets identity continuity react instead of
    the breaker waiting for a link that is already up.
    """
    if status_code in (401, 403):
        return FailureClass.AUTH_FAILURE
    if status_code == 429:
        return FailureClass.RATE_LIMITED
    if status_code >= 500:
        return FailureClass.SERVER_ERROR
    if status_code >= 400:
        return FailureClass.PROTOCOL_ERROR
    return FailureClass.OK


def classify_latency(
    latency_ms: int, slow_threshold_ms: int | None, current: FailureClass = FailureClass.OK
) -> FailureClass:
    """A successful but late answer is its own class, not a success and not a failure."""
    if current is not FailureClass.OK:
        return current
    if slow_threshold_ms is None:
        return FailureClass.OK
    return FailureClass.SLOW_RESPONSE if latency_ms > slow_threshold_ms else FailureClass.OK


class Breaker:
    """Per dependency circuit breaker.

    Passive observations from real calls feed this exactly like active probes. A
    call that has already failed must not wait for the next canary to open the
    breaker; the agent's very next step needs to see the truth.
    """

    def __init__(self, policy: BreakerPolicy | None = None) -> None:
        self.policy = policy or BreakerPolicy()

    def observe(
        self, health: DependencyHealth, failure_class: FailureClass, latency_ms: int | None
    ) -> DependencyHealth:
        succeeded = failure_class in (FailureClass.OK, FailureClass.SLOW_RESPONSE)
        if failure_class is FailureClass.RATE_LIMITED:
            # Throttling says the dependency is alive and busy. Counting it as a
            # failure would open the breaker on a service that is working.
            return replace(health, last_failure_class=failure_class, last_latency_ms=latency_ms)
        return (
            self._on_success(health, failure_class, latency_ms)
            if succeeded
            else self._on_failure(health, failure_class, latency_ms)
        )

    def _on_success(
        self, health: DependencyHealth, failure_class: FailureClass, latency_ms: int | None
    ) -> DependencyHealth:
        successes = health.consecutive_successes + 1
        closing = (
            health.breaker is BreakerState.CLOSED
            or successes >= self.policy.close_after_consecutive_successes
        )
        breaker = BreakerState.CLOSED if closing else health.breaker
        state = (
            HealthState.SLOW if failure_class is FailureClass.SLOW_RESPONSE else HealthState.HEALTHY
        )
        return replace(
            health,
            breaker=breaker,
            state=state if breaker is BreakerState.CLOSED else health.state,
            last_failure_class=failure_class,
            last_latency_ms=latency_ms,
            consecutive_failures=0,
            consecutive_successes=0 if closing else successes,
            backoff_s=0.0 if closing else health.backoff_s,
        )

    def _on_failure(
        self, health: DependencyHealth, failure_class: FailureClass, latency_ms: int | None
    ) -> DependencyHealth:
        failures = health.consecutive_failures + 1
        trip = (
            failures >= self.policy.open_after_consecutive_failures
            or (failure_class in TRIP_IMMEDIATELY and failures >= 2)
            or health.breaker is BreakerState.HALF_OPEN
        )
        breaker = BreakerState.OPEN if trip else health.breaker
        if breaker is BreakerState.OPEN:
            state = (
                HealthState.AUTH_BROKEN
                if failure_class is FailureClass.AUTH_FAILURE
                else HealthState.UNREACHABLE
            )
            backoff = (
                self.policy.half_open_backoff_initial_s
                if health.backoff_s == 0.0
                else min(health.backoff_s * 2, self.policy.half_open_backoff_max_s)
            )
        else:
            state = health.state
            backoff = health.backoff_s
        return replace(
            health,
            breaker=breaker,
            state=state,
            last_failure_class=failure_class,
            last_latency_ms=latency_ms,
            consecutive_failures=failures,
            consecutive_successes=0,
            backoff_s=backoff,
            needs_reconciliation=health.needs_reconciliation
            or state in (HealthState.UNREACHABLE, HealthState.AUTH_BROKEN),
        )


class HealthMonitor:
    """Holds the health vector and turns observations into transitions."""

    def __init__(
        self,
        dependencies: dict[str, str],
        policy: BreakerPolicy | None = None,
        slow_thresholds: dict[str, int] | None = None,
    ) -> None:
        self._types = dict(dependencies)
        self._slow = dict(slow_thresholds or {})
        self._breaker = Breaker(policy)
        self._health: dict[str, DependencyHealth] = {
            name: DependencyHealth(name=name) for name in dependencies
        }

    @property
    def vector(self) -> dict[str, DependencyHealth]:
        return dict(self._health)

    def get(self, dependency: str) -> DependencyHealth:
        return self._health[dependency]

    def dependency_type(self, dependency: str) -> str:
        return self._types[dependency]

    def slow_threshold(self, dependency: str) -> int | None:
        return self._slow.get(dependency)

    def observe(
        self,
        dependency: str,
        failure_class: FailureClass,
        latency_ms: int | None = None,
        source: ObservationSource = ObservationSource.PASSIVE,
    ) -> Transition:
        if dependency not in self._health:
            raise KeyError(
                f"{dependency!r} is not declared. The runtime must not contact anything undeclared,"
                " and must not track health for it either."
            )
        before = self._health[dependency]
        after = self._breaker.observe(before, failure_class, latency_ms)
        self._health[dependency] = after
        return Transition(
            dependency=dependency,
            before=before,
            after=after,
            failure_class=failure_class,
            source=source,
            latency_ms=latency_ms,
        )

    def restore(self, health: DependencyHealth) -> None:
        """Reinstate a persisted assessment, for a process that is resuming.

        Health outlives a single command: a fault armed by one invocation has to
        still be visible to the next, or the CLI would forget what it had been
        told between one line of a demo and the next.
        """
        if health.name in self._health:
            self._health[health.name] = health

    def clear_reconciliation(self, dependencies: list[str]) -> None:
        """Called when a reconciliation covering these dependencies has finished."""
        for name in dependencies:
            if name in self._health:
                self._health[name] = replace(self._health[name], needs_reconciliation=False)

    def pending_reconciliation(self) -> list[str]:
        return sorted(name for name, health in self._health.items() if health.needs_reconciliation)

    def snapshot(self) -> dict[str, dict[str, object]]:
        return {
            name: health.snapshot(self._types[name])
            for name, health in sorted(self._health.items())
        }
