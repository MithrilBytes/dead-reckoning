# SPDX-License-Identifier: Apache-2.0
"""Choosing which model answers, and stamping the choice onto what it decides.

Two rules, and they disagree on purpose. R1 takes the best model available.
Under a low battery the power rule takes the cheapest one that still meets the
task's floor, which on a truck working an outage is usually the right trade. The
router computes both, uses one, and records the difference, because a decision
made at reduced fidelity to save power is a decision somebody should be able to
find later and look at again.

The floor itself is the part that stops this being a fallback chain. A task class
can say it will not be answered below a certain fidelity, and if nothing meets
that, the runtime escalates rather than quietly answering worse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from deadreckoning.budget import PowerState
from deadreckoning.config import TierConfig, TierKind
from deadreckoning.health import HealthState

USABLE = frozenset({HealthState.HEALTHY, HealthState.SLOW})


class MinRankUnmetError(RuntimeError):
    """No available tier is good enough for this task class.

    Raised rather than silently downgrading. The task escalates, and the record
    carries no tier or params because no model was ever called.
    """

    def __init__(self, task_class: str, min_rank: int, available: list[str]) -> None:
        super().__init__(
            f"task class {task_class!r} requires rank <= {min_rank};"
            f" usable tiers are {available or 'none'}"
        )
        self.task_class = task_class
        self.min_rank = min_rank
        self.available = available


@dataclass(frozen=True, slots=True)
class Selection:
    """Which tier answers, what else could have, and why review may be needed."""

    tier: TierConfig
    unconstrained: TierConfig
    tiers_available: list[str]
    review_required: bool
    review_reason: str | None
    power: PowerState

    @property
    def power_constrained(self) -> bool:
        return self.tier.name != self.unconstrained.name

    def stamp(self) -> dict[str, Any]:
        """The tier fields every model-derived record carries."""
        return {
            "name": self.tier.name,
            "rank": self.tier.rank,
            "kind": str(self.tier.kind),
            "model": self.tier.model,
        }


class Router:
    """Selects a tier per task class, health vector and power state."""

    def __init__(self, tiers: list[TierConfig]) -> None:
        self.tiers = sorted(tiers, key=lambda t: t.rank)

    def usable(self, health: dict[str, HealthState]) -> list[TierConfig]:
        return [
            tier
            for tier in self.tiers
            if tier.scripted or health.get(tier.name, HealthState.UNKNOWN) in USABLE
        ]

    def select(
        self,
        *,
        task_class: str,
        min_rank: int,
        review_above_rank: int,
        health: dict[str, HealthState],
        power: PowerState = PowerState.MAINS,
    ) -> Selection:
        usable = self.usable(health)
        eligible = [tier for tier in usable if tier.rank <= min_rank]
        if not eligible:
            raise MinRankUnmetError(task_class, min_rank, [t.name for t in usable])

        # R1 alone: the best thing available that meets the floor.
        unconstrained = eligible[0]

        chosen = unconstrained
        if power is PowerState.BATTERY_LOW and unconstrained.kind is TierKind.LOCAL:
            # The power rule ranges over local tiers only, and applies only when
            # the node was going to run a model locally anyway. Rank tracks the
            # cost of local inference, where a larger or less quantised model
            # burns more battery. It is the wrong proxy for a remote call, which
            # costs radio rather than compute and may well be cheaper than running
            # anything on the truck, so a reachable remote tier is left alone.
            local = [t for t in eligible if t.kind is TierKind.LOCAL]
            if local:
                chosen = local[-1]

        review_required, review_reason = self._review(chosen, unconstrained, review_above_rank)
        return Selection(
            tier=chosen,
            unconstrained=unconstrained,
            tiers_available=[t.name for t in usable],
            review_required=review_required,
            review_reason=review_reason,
            power=power,
        )

    def _review(
        self, chosen: TierConfig, unconstrained: TierConfig, review_above_rank: int
    ) -> tuple[bool, str | None]:
        """Whether this decision gets looked at again, and which rule says so.

        The distinction the reason draws is the one a reviewer needs: was this
        reviewed because the tier was below the threshold anyway, or only because
        the node was conserving power and would not have been on mains.
        """
        if chosen.rank <= review_above_rank:
            return False, None
        if chosen.name != unconstrained.name and unconstrained.rank <= review_above_rank:
            return True, "POWER_CONSTRAINED"
        return True, "TIER_BELOW_REVIEW_THRESHOLD"

    def next_after(
        self, failed: TierConfig, min_rank: int, health: dict[str, HealthState]
    ) -> TierConfig | None:
        """The tier a mid-call failure falls to: the next worse one still usable.

        One step, not a cascade. A chain of retries through every tier turns a
        transient failure into a long silence, and the mode controller would
        rather hear about the failure now.
        """
        return next(
            (
                tier
                for tier in self.usable(health)
                if tier.rank > failed.rank and tier.rank <= min_rank
            ),
            None,
        )
