# SPDX-License-Identifier: Apache-2.0
"""The authority table, every cell, and how a grant decays."""

from __future__ import annotations

import pytest

from deadreckoning.identity import AuthorityOutcome, IdentityTracker, authority_for
from deadreckoning.records import IdentityState
from deadreckoning.tools.contract import (
    Approval,
    Consequence,
    OfflinePolicy,
    SideEffect,
    ToolContract,
)

HOUR = 3_600_000


def tool(side_effect: SideEffect, consequence: Consequence, **kwargs: object) -> ToolContract:
    if side_effect is SideEffect.NONE:
        return ToolContract(
            name="read",
            backend="api",
            offline_policy=OfflinePolicy.FAIL,
            side_effect=side_effect,
            consequence=consequence,
            **kwargs,  # pyright: ignore[reportArgumentType]
        )
    return ToolContract(
        name="write",
        backend="api",
        offline_policy=OfflinePolicy.QUEUE,
        side_effect=side_effect,
        consequence=consequence,
        idempotency_key="k:{x}" if side_effect is SideEffect.NON_IDEMPOTENT else None,
        expiry_s=3600,
        **kwargs,  # pyright: ignore[reportArgumentType]
    )


READ = tool(SideEffect.NONE, Consequence.LOW)
LOW_WRITE = tool(SideEffect.IDEMPOTENT, Consequence.LOW)
RISKY_WRITE = tool(SideEffect.NON_IDEMPOTENT, Consequence.MEDIUM)
HIGH_WRITE = tool(SideEffect.NON_IDEMPOTENT, Consequence.HIGH)


@pytest.mark.parametrize("identity", list(IdentityState))
def test_reading_survives_every_identity_state_for_a_local_tool(identity: IdentityState) -> None:
    """A crew that cannot look anything up is a crew that guesses."""
    local_read = ToolContract(
        name="local",
        backend=None,
        offline_policy=OfflinePolicy.LOCAL,
        side_effect=SideEffect.NONE,
        consequence=Consequence.LOW,
        staleness_budget_s=None,
        local_source="field_input",
    )
    assert authority_for(local_read, identity, connected=False).outcome is AuthorityOutcome.ALLOW


@pytest.mark.parametrize(
    ("identity", "contract", "expected"),
    [
        (IdentityState.FRESH, READ, AuthorityOutcome.ALLOW),
        (IdentityState.FRESH, LOW_WRITE, AuthorityOutcome.ALLOW),
        (IdentityState.FRESH, RISKY_WRITE, AuthorityOutcome.ALLOW),
        (IdentityState.CACHED, READ, AuthorityOutcome.ALLOW),
        (IdentityState.CACHED, LOW_WRITE, AuthorityOutcome.ALLOW),
        (IdentityState.CACHED, RISKY_WRITE, AuthorityOutcome.REQUIRE_APPROVAL),
        (IdentityState.CACHED, HIGH_WRITE, AuthorityOutcome.REQUIRE_APPROVAL),
        (IdentityState.STALE, READ, AuthorityOutcome.ALLOW),
        (IdentityState.STALE, LOW_WRITE, AuthorityOutcome.QUEUE_ONLY),
        (IdentityState.STALE, HIGH_WRITE, AuthorityOutcome.QUEUE_ONLY),
        (IdentityState.NONE, READ, AuthorityOutcome.DENY),
        (IdentityState.NONE, LOW_WRITE, AuthorityOutcome.DENY),
        (IdentityState.NONE, HIGH_WRITE, AuthorityOutcome.DENY),
    ],
)
def test_the_authority_table_cell_by_cell(
    identity: IdentityState, contract: ToolContract, expected: AuthorityOutcome
) -> None:
    assert authority_for(contract, identity, connected=True).outcome is expected


def test_authority_only_ever_tightens() -> None:
    """Stage two of dispatch may narrow what stage one allowed, never widen it."""
    order = {
        AuthorityOutcome.ALLOW: 0,
        AuthorityOutcome.REQUIRE_APPROVAL: 1,
        AuthorityOutcome.QUEUE_ONLY: 2,
        AuthorityOutcome.DENY: 3,
    }
    for contract in (LOW_WRITE, RISKY_WRITE, HIGH_WRITE):
        severities = [
            order[authority_for(contract, identity, connected=True).outcome]
            for identity in (
                IdentityState.FRESH,
                IdentityState.CACHED,
                IdentityState.STALE,
                IdentityState.NONE,
            )
        ]
        assert severities == sorted(severities), f"{contract.name} loosened as identity decayed"


def test_a_when_not_connected_policy_fires_only_when_disconnected() -> None:
    contract = tool(
        SideEffect.NON_IDEMPOTENT, Consequence.HIGH, approval=Approval.WHEN_NOT_CONNECTED
    )
    assert authority_for(contract, IdentityState.FRESH, connected=True).outcome is (
        AuthorityOutcome.ALLOW
    )
    assert authority_for(contract, IdentityState.FRESH, connected=False).outcome is (
        AuthorityOutcome.REQUIRE_APPROVAL
    )


def test_an_always_policy_fires_even_fully_connected_and_fresh() -> None:
    contract = tool(SideEffect.IDEMPOTENT, Consequence.LOW, approval=Approval.ALWAYS)
    assert authority_for(contract, IdentityState.FRESH, connected=True).outcome is (
        AuthorityOutcome.REQUIRE_APPROVAL
    )


def test_every_refusal_says_why() -> None:
    for identity in (IdentityState.CACHED, IdentityState.STALE, IdentityState.NONE):
        assert authority_for(HIGH_WRITE, identity, connected=False).reason


def test_a_grant_decays_from_fresh_through_cached_to_stale() -> None:
    tracker = IdentityTracker(fresh_window_s=900, cached_grant_ttl_s=8 * 3600)
    tracker.granted(at_ms=0)
    assert tracker.assess(60_000).state is IdentityState.FRESH

    tracker.unreachable()
    assert tracker.assess(HOUR).state is IdentityState.CACHED
    assert tracker.assess(HOUR * 4).state is IdentityState.CACHED
    assert tracker.assess(HOUR * 9).state is IdentityState.STALE


def test_a_node_that_never_authenticated_has_no_identity() -> None:
    assert IdentityTracker().assess(0).state is IdentityState.NONE


def test_reaching_the_provider_again_restores_freshness() -> None:
    tracker = IdentityTracker(fresh_window_s=900, cached_grant_ttl_s=8 * 3600)
    tracker.granted(at_ms=0)
    tracker.unreachable()
    assert tracker.assess(HOUR).state is IdentityState.CACHED
    tracker.contacted(at_ms=HOUR)
    assert tracker.assess(HOUR + 1000).state is IdentityState.FRESH


def test_the_remaining_ttl_is_reported_so_the_model_can_see_it_shrinking() -> None:
    tracker = IdentityTracker(cached_grant_ttl_s=8 * 3600)
    tracker.granted(at_ms=0)
    tracker.unreachable()
    assert tracker.assess(HOUR * 2).ttl_remaining_s == 6 * 3600
