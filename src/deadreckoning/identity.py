# SPDX-License-Identifier: Apache-2.0
"""What this node is still allowed to do as its credentials age.

An agent that loses contact with its identity provider has two bad options and one
good one. It can keep acting as though nothing happened, which is how a revoked
operator keeps dispatching crews for another eight hours. It can stop entirely,
which strands a truck that can still usefully triage. Or it can keep its
authority and shed it gradually, which is what this does.

The table is deliberately asymmetric. Reading stays available at every level,
because reading harms nobody and a crew that cannot look anything up is a crew
that guesses. Writing tightens as the grant ages, and a node with no identity at
all writes nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from deadreckoning.records import IdentityState
from deadreckoning.tools.contract import Approval, Consequence, SideEffect, ToolContract


class AuthorityOutcome(StrEnum):
    """What the authority table does to a call stage 1 already placed."""

    ALLOW = "ALLOW"
    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"
    QUEUE_ONLY = "QUEUE_ONLY"
    DENY = "DENY"


@dataclass(frozen=True, slots=True)
class Authority:
    outcome: AuthorityOutcome
    reason: str

    @property
    def denied(self) -> bool:
        return self.outcome is AuthorityOutcome.DENY

    @property
    def needs_approval(self) -> bool:
        return self.outcome in (AuthorityOutcome.REQUIRE_APPROVAL, AuthorityOutcome.QUEUE_ONLY)


def authority_for(contract: ToolContract, identity: IdentityState, connected: bool) -> Authority:
    """Stage two of dispatch. It can tighten what stage one decided, never loosen it.

    The ordering matters: a read is checked first and almost always allowed, so
    that no amount of identity decay stops a node from looking things up.
    """
    if contract.side_effect is SideEffect.NONE:
        if identity is IdentityState.NONE and not contract.always_local:
            return Authority(
                AuthorityOutcome.DENY,
                "identity NONE: this node has never authenticated, so it may only read"
                " what is already on it",
            )
        return Authority(AuthorityOutcome.ALLOW, "read-only")

    if identity is IdentityState.NONE:
        return Authority(AuthorityOutcome.DENY, "identity NONE: no side effects without identity")

    if identity is IdentityState.STALE:
        return Authority(
            AuthorityOutcome.QUEUE_ONLY,
            "identity STALE: the cached grant has outlived its TTL, so this waits for a human",
        )

    if identity is IdentityState.CACHED:
        if contract.consequence is Consequence.HIGH:
            return Authority(
                AuthorityOutcome.REQUIRE_APPROVAL,
                "identity CACHED and consequence HIGH",
            )
        if contract.side_effect is SideEffect.NON_IDEMPOTENT:
            return Authority(
                AuthorityOutcome.REQUIRE_APPROVAL,
                "identity CACHED and the action cannot be safely repeated",
            )
        return _by_policy(contract, connected)

    return _by_policy(contract, connected)


def _by_policy(contract: ToolContract, connected: bool) -> Authority:
    if contract.approval is Approval.ALWAYS:
        return Authority(AuthorityOutcome.REQUIRE_APPROVAL, "tool policy: approval ALWAYS")
    if contract.approval is Approval.WHEN_NOT_CONNECTED and not connected:
        return Authority(
            AuthorityOutcome.REQUIRE_APPROVAL, "tool policy: approval when not CONNECTED"
        )
    return Authority(AuthorityOutcome.ALLOW, "within authority")


@dataclass(frozen=True, slots=True)
class IdentityAssessment:
    state: IdentityState
    ttl_remaining_s: int | None
    reason: str


class IdentityTracker:
    """Tracks how fresh this node's grant is, and never invents freshness.

    Like time trust, it holds no clock: the caller passes the instant, so the
    decay is testable without waiting eight hours for a grant to go stale.
    """

    def __init__(self, fresh_window_s: int = 900, cached_grant_ttl_s: int = 28800) -> None:
        self.fresh_window_s = fresh_window_s
        self.cached_grant_ttl_s = cached_grant_ttl_s
        self._granted_ms: int | None = None
        self._last_contact_ms: int | None = None
        self._idp_reachable = True

    def granted(self, at_ms: int) -> None:
        self._granted_ms = at_ms
        self._last_contact_ms = at_ms
        self._idp_reachable = True

    def contacted(self, at_ms: int) -> None:
        """The provider answered, so the grant is confirmed current."""
        self._last_contact_ms = at_ms
        self._idp_reachable = True

    def unreachable(self) -> None:
        self._idp_reachable = False

    def assess(self, now_ms: int) -> IdentityAssessment:
        if self._granted_ms is None:
            return IdentityAssessment(IdentityState.NONE, None, "never authenticated")
        age_s = (now_ms - self._granted_ms) / 1000
        remaining = max(0, int(self.cached_grant_ttl_s - age_s))
        if self._idp_reachable and self._last_contact_ms is not None:
            since_contact_s = (now_ms - self._last_contact_ms) / 1000
            if since_contact_s <= self.fresh_window_s:
                return IdentityAssessment(IdentityState.FRESH, remaining, "confirmed recently")
        if age_s <= self.cached_grant_ttl_s:
            return IdentityAssessment(
                IdentityState.CACHED, remaining, "grant still inside its TTL, provider unreachable"
            )
        return IdentityAssessment(IdentityState.STALE, 0, "cached grant has outlived its TTL")
