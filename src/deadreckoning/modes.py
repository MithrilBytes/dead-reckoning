# SPDX-License-Identifier: Apache-2.0
"""Deriving the operating mode from the health of everything the node depends on.

`derive_mode` is a pure function, and deliberately so. It reads no clock, no
database and no ambient state, which is what allows every combination of inputs to
be tabulated and tested rather than sampled. The one time-dependent rule, the
dwell that stops a flapping link from rewriting the mode every few seconds, lives
in the controller that calls it.

The asymmetry between falling and rising is the interesting part. Losing a
dependency takes effect at once, because an agent that believes it still has a
capability it has lost will fabricate. Regaining one waits, because a link that
comes back for three seconds and disappears again has not really come back, and
acting on it would start reconciliation work that cannot finish.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from deadreckoning.health import USABLE_STATES, DependencyHealth
from deadreckoning.records import Mode

REMOTE_TIER_KINDS = frozenset({"remote", "onprem"})
"""Kinds that mean "not on this machine". A local tier surviving an outage does
not make the node connected, and is exactly what ISLANDED describes having."""


@dataclass(frozen=True, slots=True)
class ModeInputs:
    """Everything the derivation is allowed to see.

    Keeping this a single value makes the purity claim checkable: if it is not in
    here, the derivation cannot have used it.
    """

    health: dict[str, DependencyHealth]
    types: dict[str, str]
    tier_kinds: dict[str, str]
    previous: Mode | None
    reconciliation_in_progress: bool = False
    reconciliation_work: frozenset[str] = frozenset()
    remote_tiers_failed_mid_sequence: bool = False


def _remote_tier_usable(inputs: ModeInputs) -> bool:
    return any(
        inputs.health[name].state in USABLE_STATES
        for name, kind in inputs.tier_kinds.items()
        if kind in REMOTE_TIER_KINDS and name in inputs.health
    )


def derive_mode(inputs: ModeInputs) -> Mode:
    """Mode from the health vector and the previous mode. No other inputs."""
    remote_usable = _remote_tier_usable(inputs)

    # 0. Reconciliation is a sequence, not a state of the health vector. A
    #    dependency coming back mid-sequence must not knock the node out of work
    #    it has already started.
    if inputs.previous is Mode.RECONNECTING and inputs.reconciliation_in_progress:
        return Mode.ISLANDED if not remote_usable else Mode.RECONNECTING

    # 1. Something we were waiting on is usable again, and there is work that
    #    needed it. SLOW counts: every consumer of a recovered dependency accepts
    #    it, so gating reconciliation on HEALTHY alone would strand queued work on
    #    a node that recovered over a congested link.
    if inputs.previous in (Mode.DEGRADED, Mode.ISLANDED):
        recovered = {
            name
            for name, health in inputs.health.items()
            if health.needs_reconciliation and health.state in USABLE_STATES
        }
        if recovered & inputs.reconciliation_work:
            return Mode.RECONNECTING

    # 2. No model off this machine. This is the condition the whole runtime is
    #    named for, so it is tested before the happier cases.
    if not remote_usable:
        return Mode.ISLANDED

    # 3. Everything that gates capability is usable. Peers are excluded: a node is
    #    not degraded merely because a peer is switched off.
    gating = [health for name, health in inputs.health.items() if inputs.types.get(name) != "PEER"]
    if gating and all(health.state in USABLE_STATES for health in gating):
        return Mode.CONNECTED

    # 4. Something is missing, but a model is still reachable.
    return Mode.DEGRADED


class ModeController:
    """Applies the dwell and reports transitions worth recording.

    Downward transitions land immediately. Upward ones must hold for `up_dwell`
    first, so a link that flaps produces one mode change rather than a dozen.
    """

    SEVERITY: ClassVar[dict[Mode, int]] = {
        Mode.CONNECTED: 0,
        Mode.DEGRADED: 1,
        Mode.RECONNECTING: 1,
        Mode.ISLANDED: 2,
    }

    def __init__(self, up_dwell_s: float, initial: Mode = Mode.ISLANDED) -> None:
        self.up_dwell_s = up_dwell_s
        self.mode = initial
        self._candidate: Mode | None = None
        self._candidate_since: float | None = None

    def _is_upward(self, proposed: Mode) -> bool:
        return self.SEVERITY[proposed] < self.SEVERITY[self.mode]

    def evaluate(self, proposed: Mode, now: float) -> Mode | None:
        """Return the new mode if it changed, else None.

        RECONNECTING is never delayed even though it can look upward: it is the
        act of doing reconciliation work, and postponing it would postpone the
        work rather than stabilise anything.
        """
        if proposed is self.mode:
            self._candidate = None
            self._candidate_since = None
            return None

        if not self._is_upward(proposed) or proposed is Mode.RECONNECTING:
            self.mode = proposed
            self._candidate = None
            self._candidate_since = None
            return proposed

        if self._candidate is not proposed:
            self._candidate = proposed
            self._candidate_since = now

        assert self._candidate_since is not None
        if now - self._candidate_since >= self.up_dwell_s:
            self.mode = proposed
            self._candidate = None
            self._candidate_since = None
            return proposed
        return None

    @property
    def pending(self) -> Mode | None:
        """An upward transition waiting out its dwell, if any."""
        return self._candidate
