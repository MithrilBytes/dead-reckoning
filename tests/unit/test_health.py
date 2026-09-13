# SPDX-License-Identifier: Apache-2.0
"""Failure classification and breaker behaviour.

The matrix test is the point of this file. Every class in the taxonomy has to be
reachable and has to land the dependency in the right state, because the whole
design downstream, which tool may run, which tier answers, whether identity
refreshes, is a function of that state and nothing else.
"""

from __future__ import annotations

import httpx
import pytest

from deadreckoning.chaos import InjectedFaultError
from deadreckoning.health import (
    Breaker,
    BreakerPolicy,
    BreakerState,
    DependencyHealth,
    FailureClass,
    HealthMonitor,
    HealthState,
    ObservationSource,
    classify_latency,
    classify_status,
)
from deadreckoning.transport import classify_exception, date_header_offset_ms


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (200, FailureClass.OK),
        (204, FailureClass.OK),
        (400, FailureClass.PROTOCOL_ERROR),
        (401, FailureClass.AUTH_FAILURE),
        (403, FailureClass.AUTH_FAILURE),
        (429, FailureClass.RATE_LIMITED),
        (500, FailureClass.SERVER_ERROR),
        (503, FailureClass.SERVER_ERROR),
    ],
)
def test_status_codes_classify(status: int, expected: FailureClass) -> None:
    assert classify_status(status) is expected


def test_auth_failure_is_not_unreachable() -> None:
    """The dependency answered and refused. Those are different problems.

    Treating a rejected token as a dead link would have the node wait for a
    network that is already up, while the thing that actually needs to happen,
    refreshing identity, never gets triggered.
    """
    monitor = HealthMonitor({"idp": "IDP"})
    for _ in range(3):
        monitor.observe("idp", FailureClass.AUTH_FAILURE)
    assert monitor.get("idp").state is HealthState.AUTH_BROKEN


@pytest.mark.parametrize(
    ("exc", "trusted", "expected"),
    [
        (httpx.ConnectTimeout("timed out"), True, FailureClass.CONNECT_TIMEOUT),
        (httpx.ReadTimeout("slow"), True, FailureClass.READ_TIMEOUT),
        (
            httpx.ConnectError("[Errno 8] nodename nor servname provided"),
            True,
            FailureClass.DNS_FAILURE,
        ),
        (httpx.ConnectError("Connection refused"), True, FailureClass.CONNECT_REFUSED),
        (
            httpx.ConnectError("certificate verify failed: certificate has expired"),
            True,
            FailureClass.TLS_OTHER,
        ),
        (
            httpx.ConnectError("certificate verify failed: certificate is not yet valid"),
            False,
            FailureClass.TLS_CLOCK_SKEW,
        ),
        (
            httpx.ConnectError("certificate verify failed: self signed"),
            False,
            FailureClass.TLS_OTHER,
        ),
        (httpx.LocalProtocolError("bad frame"), True, FailureClass.PROTOCOL_ERROR),
        (InjectedFaultError("gis-api", FailureClass.DNS_FAILURE), True, FailureClass.DNS_FAILURE),
    ],
)
def test_transport_exceptions_classify(
    exc: BaseException, trusted: bool, expected: FailureClass
) -> None:
    assert classify_exception(exc, trusted) is expected


def test_a_certificate_error_reads_as_skew_only_when_the_clock_is_doubted() -> None:
    """The same exception, two meanings, decided by what the node knows about itself."""
    expired = httpx.ConnectError("certificate verify failed: certificate has expired")
    assert classify_exception(expired, time_trust_is_trusted=False) is FailureClass.TLS_CLOCK_SKEW
    assert classify_exception(expired, time_trust_is_trusted=True) is FailureClass.TLS_OTHER


def test_an_unrecognised_exception_is_reraised_not_swallowed() -> None:
    with pytest.raises(ValueError, match="something else"):
        classify_exception(ValueError("something else"), True)


def test_slow_is_a_success_not_a_failure() -> None:
    assert classify_latency(9000, 8000) is FailureClass.SLOW_RESPONSE
    assert classify_latency(100, 8000) is FailureClass.OK
    monitor = HealthMonitor({"gis": "TOOL_BACKEND"}, slow_thresholds={"gis": 2000})
    monitor.observe("gis", FailureClass.SLOW_RESPONSE, latency_ms=3000)
    assert monitor.get("gis").state is HealthState.SLOW
    assert monitor.get("gis").usable


def test_a_dependency_with_no_threshold_never_reaches_slow() -> None:
    assert classify_latency(999_999, None) is FailureClass.OK


def test_rate_limiting_does_not_open_the_breaker() -> None:
    """Throttling means the dependency is alive and busy.

    Backing off is right. Declaring it down is not.
    """
    monitor = HealthMonitor({"api": "TOOL_BACKEND"})
    monitor.observe("api", FailureClass.OK, latency_ms=10)
    for _ in range(10):
        monitor.observe("api", FailureClass.RATE_LIMITED)
    assert monitor.get("api").breaker is BreakerState.CLOSED
    assert monitor.get("api").state is HealthState.HEALTHY


def test_the_breaker_opens_after_three_consecutive_failures() -> None:
    breaker = Breaker(BreakerPolicy())
    health = DependencyHealth(name="api")
    for _ in range(2):
        health = breaker.observe(health, FailureClass.SERVER_ERROR, None)
        assert health.breaker is BreakerState.CLOSED
    health = breaker.observe(health, FailureClass.SERVER_ERROR, None)
    assert health.breaker is BreakerState.OPEN
    assert health.state is HealthState.UNREACHABLE


def test_a_refused_port_trips_on_the_second_answer_not_the_third() -> None:
    """A refusal is an answer, not a timeout. Waiting for a third identical one helps nobody."""
    breaker = Breaker(BreakerPolicy())
    health = DependencyHealth(name="api")
    health = breaker.observe(health, FailureClass.CONNECT_REFUSED, None)
    assert health.breaker is BreakerState.CLOSED
    health = breaker.observe(health, FailureClass.CONNECT_REFUSED, None)
    assert health.breaker is BreakerState.OPEN


def test_backoff_grows_and_is_capped() -> None:
    breaker = Breaker(BreakerPolicy(half_open_backoff_initial_s=5, half_open_backoff_max_s=20))
    health = DependencyHealth(name="api")
    seen: list[float] = []
    for _ in range(8):
        health = breaker.observe(health, FailureClass.SERVER_ERROR, None)
        if health.backoff_s:
            seen.append(health.backoff_s)
    assert seen[0] == 5
    assert max(seen) <= 20


def test_a_passive_observation_opens_the_breaker_without_waiting_for_a_canary() -> None:
    """A real call that just failed must not wait for the next probe to be believed."""
    monitor = HealthMonitor({"api": "TOOL_BACKEND"})
    transitions = [
        monitor.observe("api", FailureClass.CONNECT_TIMEOUT, source=ObservationSource.PASSIVE)
        for _ in range(3)
    ]
    assert transitions[-1].after.state is HealthState.UNREACHABLE
    assert transitions[-1].state_changed


def test_reconciliation_is_sticky_until_cleared() -> None:
    """A dependency that recovers through SLOW, or flaps, must reconcile exactly once."""
    monitor = HealthMonitor({"hub": "SYNC_HUB"}, slow_thresholds={"hub": 1000})
    for _ in range(3):
        monitor.observe("hub", FailureClass.CONNECT_TIMEOUT)
    assert monitor.pending_reconciliation() == ["hub"]

    monitor.observe("hub", FailureClass.SLOW_RESPONSE, latency_ms=2000)
    monitor.observe("hub", FailureClass.SLOW_RESPONSE, latency_ms=2000)
    assert monitor.get("hub").usable
    assert monitor.pending_reconciliation() == ["hub"], "recovery alone must not clear the flag"

    monitor.clear_reconciliation(["hub"])
    assert monitor.pending_reconciliation() == []


def test_an_undeclared_dependency_is_refused() -> None:
    monitor = HealthMonitor({"api": "TOOL_BACKEND"})
    with pytest.raises(KeyError, match="not declared"):
        monitor.observe("somewhere-else", FailureClass.OK)


def test_every_failure_class_is_reachable_and_lands_somewhere_defined() -> None:
    """No class in the taxonomy may be decorative."""
    monitor = HealthMonitor({"d": "TOOL_BACKEND"}, slow_thresholds={"d": 100})
    landed: dict[FailureClass, HealthState] = {}
    for failure_class in FailureClass:
        fresh = HealthMonitor({"d": "TOOL_BACKEND"}, slow_thresholds={"d": 100})
        for _ in range(3):
            fresh.observe("d", failure_class, latency_ms=500)
        landed[failure_class] = fresh.get("d").state
    assert landed[FailureClass.OK] is HealthState.HEALTHY
    assert landed[FailureClass.SLOW_RESPONSE] is HealthState.SLOW
    assert landed[FailureClass.AUTH_FAILURE] is HealthState.AUTH_BROKEN
    assert landed[FailureClass.RATE_LIMITED] is HealthState.UNKNOWN
    for failure_class in (
        FailureClass.DNS_FAILURE,
        FailureClass.CONNECT_REFUSED,
        FailureClass.CONNECT_TIMEOUT,
        FailureClass.TLS_CLOCK_SKEW,
        FailureClass.TLS_OTHER,
        FailureClass.SERVER_ERROR,
        FailureClass.READ_TIMEOUT,
        FailureClass.PROTOCOL_ERROR,
    ):
        assert landed[failure_class] is HealthState.UNREACHABLE, failure_class
    assert set(landed) == set(FailureClass)
    assert monitor.get("d").state is HealthState.UNKNOWN


def test_the_date_header_offset_is_signed_and_tolerant() -> None:
    local = 1_700_000_000_000
    assert date_header_offset_ms(None, local) is None
    assert date_header_offset_ms("not a date", local) is None
    offset = date_header_offset_ms("Tue, 14 Nov 2023 22:13:00 GMT", local)
    assert offset is not None
