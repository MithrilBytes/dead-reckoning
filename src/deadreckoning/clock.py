# SPDX-License-Identifier: Apache-2.0
"""Hybrid logical clocks.

A node that has been offline for hours, whose operator has nudged the system
clock, still has to produce timestamps that order correctly against a peer it has
never met. Wall time cannot do that and a Lamport counter throws away the wall
time a human needs to read the log. A hybrid logical clock keeps both: a physical
component that tracks real time to within the clock's drift, and a logical counter
that breaks ties and preserves causality when the physical component stalls or
goes backwards.

Kulkarni, Demirbas, Madappa, Avva and Leone, "Logical Physical Clocks and
Consistent Snapshotting in Globally Distributed Systems", OPODIS 2014.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

MAX_LOGICAL = 65535
"""Ceiling on the logical counter, fixed by the wire format.

Reaching it means this node emitted 65536 events inside a single millisecond
without the physical clock advancing. Rather than overflow the field or stall, the
clock borrows a millisecond from the future and resets the counter, which keeps
the sequence strictly increasing and stays inside the format.
"""


class MonotonicSource(Protocol):
    """Supplies wall clock milliseconds. Injectable so tests can drive time."""

    def __call__(self) -> int: ...


@dataclass(frozen=True, order=False, slots=True)
class HLC:
    """A point in causal time. Ordered by physical, then logical, then node id.

    The node id participates in ordering only to make it total: two events on
    different nodes with identical physical and logical parts are concurrent in
    the causal sense, and the tiebreak is arbitrary but stable.
    """

    physical_ms: int
    logical: int
    node_id: str

    def __post_init__(self) -> None:
        if self.physical_ms < 0:
            raise ValueError(f"physical_ms must not be negative, got {self.physical_ms}")
        if not 0 <= self.logical <= MAX_LOGICAL:
            raise ValueError(f"logical must be within 0..{MAX_LOGICAL}, got {self.logical}")
        if not self.node_id:
            raise ValueError("node_id must not be empty")

    @property
    def _key(self) -> tuple[int, int, str]:
        return (self.physical_ms, self.logical, self.node_id)

    def __lt__(self, other: HLC) -> bool:
        return self._key < other._key

    def __le__(self, other: HLC) -> bool:
        return self._key <= other._key

    def __gt__(self, other: HLC) -> bool:
        return self._key > other._key

    def __ge__(self, other: HLC) -> bool:
        return self._key >= other._key

    def as_dict(self) -> dict[str, int | str]:
        return {
            "physical_ms": self.physical_ms,
            "logical": self.logical,
            "node_id": self.node_id,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> HLC:
        physical = raw["physical_ms"]
        logical = raw["logical"]
        node_id = raw["node_id"]
        if not isinstance(physical, int) or not isinstance(logical, int):
            raise TypeError("hlc physical_ms and logical must be integers")
        if not isinstance(node_id, str):
            raise TypeError("hlc node_id must be a string")
        return cls(physical_ms=physical, logical=logical, node_id=node_id)

    def __str__(self) -> str:
        return f"{self.physical_ms}-{self.logical}-{self.node_id}"


def _bounded(physical_ms: int, logical: int, node_id: str) -> HLC:
    """Carry a logical overflow into the physical component.

    Only reachable under a burst of more than MAX_LOGICAL events in one
    millisecond. Advancing physical keeps the result strictly greater than the
    previous stamp, which is the property everything else relies on.
    """
    while logical > MAX_LOGICAL:
        physical_ms += 1
        logical -= MAX_LOGICAL + 1
    return HLC(physical_ms=physical_ms, logical=logical, node_id=node_id)


class HybridLogicalClock:
    """Per node clock. Never goes backwards, including across a restart.

    The caller is responsible for persisting `last` and handing it back at
    construction. This class holds no storage and does no I/O.
    """

    def __init__(self, node_id: str, now_ms: MonotonicSource, last: HLC | None = None) -> None:
        if not node_id:
            raise ValueError("node_id must not be empty")
        if last is not None and last.node_id != node_id:
            raise ValueError(f"restored clock belongs to node {last.node_id!r}, not {node_id!r}")
        self._node_id = node_id
        self._now_ms = now_ms
        self._last = last if last is not None else HLC(0, 0, node_id)

    @property
    def last(self) -> HLC:
        """The most recent stamp issued. Persist this; never let it regress."""
        return self._last

    def now(self) -> HLC:
        """Stamp a local event.

        If the wall clock has not advanced past the last stamp, including the case
        where it moved backwards, the logical counter carries the ordering instead.
        """
        wall = self._now_ms()
        physical = max(wall, self._last.physical_ms)
        logical = self._last.logical + 1 if physical == self._last.physical_ms else 0
        self._last = _bounded(physical, logical, self._node_id)
        return self._last

    def observe(self, remote: HLC) -> HLC:
        """Merge a stamp received from another node and return the local stamp.

        After this, the local clock is strictly ahead of both what it had and what
        it was told, so anything stamped next is causally after the message.
        """
        wall = self._now_ms()
        physical = max(wall, self._last.physical_ms, remote.physical_ms)
        if physical == self._last.physical_ms and physical == remote.physical_ms:
            logical = max(self._last.logical, remote.logical) + 1
        elif physical == self._last.physical_ms:
            logical = self._last.logical + 1
        elif physical == remote.physical_ms:
            logical = remote.logical + 1
        else:
            logical = 0
        self._last = _bounded(physical, logical, self._node_id)
        return self._last


class TimeTrust(StrEnum):
    TRUSTED = "TRUSTED"
    DRIFTING = "DRIFTING"
    UNTRUSTED = "UNTRUSTED"


class TrustSource(StrEnum):
    TIME_SOURCE = "TIME_SOURCE"
    DATE_HEADER = "DATE_HEADER"
    STALENESS = "STALENESS"
    BACKWARD_JUMP = "BACKWARD_JUMP"


@dataclass(frozen=True, slots=True)
class TrustAssessment:
    trust: TimeTrust
    source: TrustSource
    offset_ms: int | None


class TimeTrustTracker:
    """How much this node believes its own clock.

    This exists because of one specific failure that looks like an attack and is
    not. A laptop that has been off the grid for hours comes back with a clock
    that has drifted; the first TLS handshake fails with "certificate not yet
    valid"; and a system that reports that as a security failure sends a crew
    chasing a compromise that never happened. Knowing whether the clock is
    trustworthy is what lets the runtime say "this is probably skew" instead.

    It holds no clock of its own. Callers pass the current time, so the whole
    thing is testable without waiting.
    """

    def __init__(
        self,
        fresh_s: int = 3600,
        stale_s: int = 86400,
        skew_threshold_s: int = 300,
    ) -> None:
        self.fresh_s = fresh_s
        self.stale_s = stale_s
        self.skew_threshold_ms = skew_threshold_s * 1000
        self._last_check_ms: int | None = None
        self._last_offset_ms: int | None = None
        self._jumped_backwards = False
        self._opportunistic_skew = False

    def observe_time_source(self, offset_ms: int, at_ms: int) -> None:
        """An authoritative check against a declared time source."""
        self._last_check_ms = at_ms
        self._last_offset_ms = offset_ms
        self._opportunistic_skew = abs(offset_ms) > self.skew_threshold_ms
        if not self._opportunistic_skew:
            # A good check is the only thing that clears a suspected jump: the
            # clock has been vouched for by something outside this machine.
            self._jumped_backwards = False

    def observe_date_header(self, offset_ms: int) -> None:
        """An opportunistic comparison against any successful response.

        Free evidence, taken from traffic the node was making anyway. It can
        reduce trust but never restore it, because a `Date` header is not an
        authenticated time source and a compromised or merely wrong peer should
        not be able to talk this node into believing its clock.
        """
        self._last_offset_ms = offset_ms
        if abs(offset_ms) > self.skew_threshold_ms:
            self._opportunistic_skew = True

    def note_backward_jump(self, delta_ms: int) -> None:
        if abs(delta_ms) > self.skew_threshold_ms:
            self._jumped_backwards = True

    def assess(self, now_ms: int) -> TrustAssessment:
        if self._jumped_backwards:
            return TrustAssessment(
                TimeTrust.UNTRUSTED, TrustSource.BACKWARD_JUMP, self._last_offset_ms
            )
        if self._last_check_ms is None:
            return TrustAssessment(TimeTrust.UNTRUSTED, TrustSource.STALENESS, None)
        age_s = (now_ms - self._last_check_ms) / 1000
        if self._opportunistic_skew:
            source = TrustSource.DATE_HEADER
            trust = TimeTrust.DRIFTING if age_s <= self.stale_s else TimeTrust.UNTRUSTED
            return TrustAssessment(trust, source, self._last_offset_ms)
        if age_s <= self.fresh_s:
            return TrustAssessment(TimeTrust.TRUSTED, TrustSource.TIME_SOURCE, self._last_offset_ms)
        if age_s <= self.stale_s:
            return TrustAssessment(TimeTrust.DRIFTING, TrustSource.STALENESS, self._last_offset_ms)
        return TrustAssessment(TimeTrust.UNTRUSTED, TrustSource.STALENESS, self._last_offset_ms)
