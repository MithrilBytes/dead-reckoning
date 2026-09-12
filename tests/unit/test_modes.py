# SPDX-License-Identifier: Apache-2.0
"""Mode derivation, tabulated.

The derivation is a pure function precisely so that it can be enumerated rather
than sampled, and the exhaustive test below is the reason that purity is a
requirement and not a preference.
"""

from __future__ import annotations

import itertools

import pytest

from deadreckoning.health import DependencyHealth, HealthState
from deadreckoning.modes import ModeController, ModeInputs, derive_mode
from deadreckoning.records import Mode

TYPES = {"frontier": "MODEL_TIER", "local-q4": "MODEL_TIER", "gis": "TOOL_BACKEND"}
KINDS = {"frontier": "remote", "local-q4": "local"}


def inputs(
    frontier: HealthState,
    local: HealthState,
    gis: HealthState,
    previous: Mode | None = None,
    **kwargs: object,
) -> ModeInputs:
    return ModeInputs(
        health={
            "frontier": DependencyHealth("frontier", state=frontier),
            "local-q4": DependencyHealth("local-q4", state=local),
            "gis": DependencyHealth("gis", state=gis),
        },
        types=TYPES,
        tier_kinds=KINDS,
        previous=previous,
        **kwargs,  # pyright: ignore[reportArgumentType]
    )


H, S, U, A, K = (
    HealthState.HEALTHY,
    HealthState.SLOW,
    HealthState.UNREACHABLE,
    HealthState.AUTH_BROKEN,
    HealthState.UNKNOWN,
)


def test_the_derivation_is_total_over_every_health_permutation() -> None:
    """Three dependencies, five states each: every one of the 125 has an answer."""
    seen: set[Mode] = set()
    for frontier, local, gis in itertools.product([H, S, U, A, K], repeat=3):
        mode = derive_mode(inputs(frontier, local, gis, previous=Mode.CONNECTED))
        assert isinstance(mode, Mode)
        seen.add(mode)
    assert {Mode.CONNECTED, Mode.DEGRADED, Mode.ISLANDED} <= seen


def test_the_derivation_is_deterministic() -> None:
    for frontier, local, gis in itertools.product([H, S, U, A, K], repeat=3):
        first = derive_mode(inputs(frontier, local, gis, previous=Mode.DEGRADED))
        again = derive_mode(inputs(frontier, local, gis, previous=Mode.DEGRADED))
        assert first is again


@pytest.mark.parametrize(
    ("frontier", "local", "gis", "expected"),
    [
        (H, H, H, Mode.CONNECTED),
        (S, H, H, Mode.CONNECTED),
        (H, H, S, Mode.CONNECTED),
        (H, H, U, Mode.DEGRADED),
        (H, U, H, Mode.DEGRADED),
        (H, H, A, Mode.DEGRADED),
        (U, H, H, Mode.ISLANDED),
        (A, H, H, Mode.ISLANDED),
        (U, U, U, Mode.ISLANDED),
        (K, H, H, Mode.ISLANDED),
        (H, K, H, Mode.DEGRADED),
    ],
)
def test_the_named_cases(
    frontier: HealthState, local: HealthState, gis: HealthState, expected: Mode
) -> None:
    assert derive_mode(inputs(frontier, local, gis, previous=Mode.CONNECTED)) is expected


def test_an_unprobed_dependency_holds_the_node_out_of_connected() -> None:
    """UNKNOWN is not HEALTHY. A node has not verified what it has not checked."""
    assert derive_mode(inputs(H, K, H, previous=Mode.CONNECTED)) is Mode.DEGRADED


def test_a_slow_link_is_a_working_link() -> None:
    assert derive_mode(inputs(S, S, S, previous=Mode.CONNECTED)) is Mode.CONNECTED


def test_a_peer_that_is_switched_off_does_not_degrade_the_node() -> None:
    base = inputs(H, H, H, previous=Mode.CONNECTED)
    with_peer = ModeInputs(
        health={**base.health, "truck-12": DependencyHealth("truck-12", state=U)},
        types={**TYPES, "truck-12": "PEER"},
        tier_kinds=KINDS,
        previous=Mode.CONNECTED,
    )
    assert derive_mode(with_peer) is Mode.CONNECTED


def test_a_surviving_local_tier_does_not_make_the_node_connected() -> None:
    """The whole point of ISLANDED: a model on the truck is not a link to anywhere."""
    assert derive_mode(inputs(U, H, H, previous=Mode.CONNECTED)) is Mode.ISLANDED


def test_recovery_triggers_reconnecting_from_slow_as_well_as_healthy() -> None:
    for recovered in (H, S):
        state = ModeInputs(
            health={
                "frontier": DependencyHealth(
                    "frontier", state=recovered, needs_reconciliation=True
                ),
                "local-q4": DependencyHealth("local-q4", state=H),
                "gis": DependencyHealth("gis", state=H),
            },
            types=TYPES,
            tier_kinds=KINDS,
            previous=Mode.ISLANDED,
            reconciliation_work=frozenset({"frontier"}),
        )
        assert derive_mode(state) is Mode.RECONNECTING, recovered


def test_recovery_with_no_pending_work_does_not_reconcile() -> None:
    state = ModeInputs(
        health={
            "frontier": DependencyHealth("frontier", state=H, needs_reconciliation=True),
            "local-q4": DependencyHealth("local-q4", state=H),
            "gis": DependencyHealth("gis", state=H),
        },
        types=TYPES,
        tier_kinds=KINDS,
        previous=Mode.ISLANDED,
        reconciliation_work=frozenset(),
    )
    assert derive_mode(state) is Mode.CONNECTED


def test_reconnecting_is_absorbing_while_the_sequence_runs() -> None:
    """A dependency returning mid-sequence must not knock the node out of the work."""
    state = inputs(H, H, H, previous=Mode.RECONNECTING, reconciliation_in_progress=True)
    assert derive_mode(state) is Mode.RECONNECTING


def test_losing_the_remote_tiers_mid_sequence_aborts_to_islanded() -> None:
    state = inputs(U, H, H, previous=Mode.RECONNECTING, reconciliation_in_progress=True)
    assert derive_mode(state) is Mode.ISLANDED


def test_once_the_sequence_ends_the_mode_re_derives() -> None:
    state = inputs(H, H, H, previous=Mode.RECONNECTING, reconciliation_in_progress=False)
    assert derive_mode(state) is Mode.CONNECTED


def test_going_down_is_immediate_and_coming_up_waits() -> None:
    controller = ModeController(up_dwell_s=10, initial=Mode.CONNECTED)
    assert controller.evaluate(Mode.ISLANDED, now=0.0) is Mode.ISLANDED
    assert controller.evaluate(Mode.CONNECTED, now=1.0) is None
    assert controller.evaluate(Mode.CONNECTED, now=5.0) is None
    assert controller.evaluate(Mode.CONNECTED, now=11.0) is Mode.CONNECTED


def test_a_flapping_link_produces_one_upward_change_not_many() -> None:
    controller = ModeController(up_dwell_s=10, initial=Mode.CONNECTED)
    controller.evaluate(Mode.ISLANDED, now=0.0)
    upward = 0
    for tick in range(1, 40):
        now = float(tick)
        proposed = Mode.CONNECTED if tick % 2 else Mode.ISLANDED
        if controller.evaluate(proposed, now) is Mode.CONNECTED:
            upward += 1
    assert upward == 0, "a link that keeps dropping never satisfies the dwell"


def test_a_link_that_stays_up_does_satisfy_the_dwell() -> None:
    controller = ModeController(up_dwell_s=10, initial=Mode.ISLANDED)
    changes = [controller.evaluate(Mode.CONNECTED, float(t)) for t in range(0, 20)]
    assert changes.count(Mode.CONNECTED) == 1


def test_reconnecting_is_never_delayed_by_the_dwell() -> None:
    """Postponing reconciliation postpones the work rather than stabilising anything."""
    controller = ModeController(up_dwell_s=10, initial=Mode.ISLANDED)
    assert controller.evaluate(Mode.RECONNECTING, now=0.0) is Mode.RECONNECTING
