# SPDX-License-Identifier: Apache-2.0
"""Shared HTTP transport behaviour: timing, fault injection, and classification.

Every outbound call in this runtime passes through here, whether it is going to a
model endpoint or a tool backend, so that all of them are timed the same way, see
the same injected faults, and have their failures named by the same rules. A tool
backend whose timeouts were classified differently from a model tier would give
the mode controller two incompatible pictures of the same network.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from email.utils import parsedate_to_datetime

import httpx

from deadreckoning.chaos import InjectedFaultError
from deadreckoning.health import FailureClass


def classify_exception(exc: BaseException, time_trust_is_trusted: bool) -> FailureClass:
    """Map a transport exception to a class, given what this node thinks of its clock.

    The SDK for these endpoints collapses most of this into one connection error,
    which is precisely the detail the mode controller needs, so the client is
    hand-rolled and this function is why.

    The certificate branch is the one that earns its keep. A validity error while
    the clock is untrusted is almost certainly skew, not an attack, and saying so
    is the difference between re-syncing time and dispatching someone to
    investigate a breach that never happened.
    """
    if isinstance(exc, InjectedFaultError):
        return exc.failure_class
    if isinstance(exc, httpx.ConnectTimeout):
        return FailureClass.CONNECT_TIMEOUT
    if isinstance(exc, httpx.ReadTimeout | httpx.WriteTimeout | httpx.PoolTimeout):
        return FailureClass.READ_TIMEOUT
    if isinstance(exc, httpx.ConnectError):
        text = str(exc).lower()
        if _looks_like_certificate_validity(text):
            return FailureClass.TLS_OTHER if time_trust_is_trusted else FailureClass.TLS_CLOCK_SKEW
        if "certificate" in text or "ssl" in text or "tls" in text:
            return FailureClass.TLS_OTHER
        if _looks_like_name_resolution(text):
            return FailureClass.DNS_FAILURE
        if "refused" in text:
            return FailureClass.CONNECT_REFUSED
        return FailureClass.CONNECT_TIMEOUT
    if isinstance(exc, httpx.ProtocolError | httpx.DecodingError):
        return FailureClass.PROTOCOL_ERROR
    if isinstance(exc, httpx.TransportError):
        return FailureClass.CONNECT_TIMEOUT
    raise exc


def _looks_like_certificate_validity(text: str) -> bool:
    return ("certificate" in text or "certificate_verify_failed" in text) and (
        "not yet valid" in text or "has expired" in text or "expired" in text
    )


def _looks_like_name_resolution(text: str) -> bool:
    return (
        "name or service not known" in text
        or "nodename nor servname" in text
        or "temporary failure in name resolution" in text
        or "getaddrinfo" in text
        or "name resolution" in text
    )


@dataclass(frozen=True, slots=True)
class CallOutcome:
    """What one HTTP call produced, in the terms the health monitor speaks."""

    failure_class: FailureClass
    latency_ms: int
    status_code: int | None = None
    body: object | None = None
    date_header_offset_ms: int | None = None

    @property
    def ok(self) -> bool:
        return self.failure_class in (FailureClass.OK, FailureClass.SLOW_RESPONSE)


def date_header_offset_ms(header_value: str | None, local_ms: int) -> int | None:
    """Signed offset between this machine's clock and a server's `Date`.

    Taken from traffic the node was making anyway, which is why it is worth
    reading: free evidence about the one thing a disconnected node cannot check
    for itself. Positive means the local clock is ahead.
    """
    if not header_value:
        return None
    try:
        parsed = parsedate_to_datetime(header_value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.UTC)
    return local_ms - int(parsed.timestamp() * 1000)
