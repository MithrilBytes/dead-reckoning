# SPDX-License-Identifier: Apache-2.0
"""Tier selection, the power rule, and which of them explains a review."""

from __future__ import annotations

import pytest

from deadreckoning.budget import (
    BudgetExhaustedError,
    FakeResourceSensor,
    PowerState,
    TaskBudget,
)
from deadreckoning.config import TierConfig, TierKind
from deadreckoning.health import HealthState
from deadreckoning.router import MinRankUnmetError, Router

FRONTIER = TierConfig(name="frontier", rank=0, kind=TierKind.REMOTE, model="big", base_url="u")
LOCAL_Q8 = TierConfig(name="local-q8", rank=1, kind=TierKind.LOCAL, model="q8", base_url="u")
LOCAL_Q4 = TierConfig(name="local-q4", rank=2, kind=TierKind.LOCAL, model="q4", base_url="u")
ALL = [FRONTIER, LOCAL_Q8, LOCAL_Q4]


def health(**states: HealthState) -> dict[str, HealthState]:
    base = dict.fromkeys(("frontier", "local-q8", "local-q4"), HealthState.HEALTHY)
    return {**base, **states}


def select(router: Router, **kwargs: object):
    return router.select(
        task_class=str(kwargs.pop("task_class", "triage")),
        min_rank=int(kwargs.pop("min_rank", 2)),  # pyright: ignore[reportArgumentType]
        review_above_rank=int(kwargs.pop("review_above_rank", 0)),  # pyright: ignore[reportArgumentType]
        health=kwargs.pop("health", health()),  # pyright: ignore[reportArgumentType]
        power=kwargs.pop("power", PowerState.MAINS),  # pyright: ignore[reportArgumentType]
    )


def test_the_best_available_tier_wins_on_mains() -> None:
    assert select(Router(ALL)).tier.name == "frontier"


def test_an_unreachable_tier_is_not_selected() -> None:
    chosen = select(Router(ALL), health=health(frontier=HealthState.UNREACHABLE))
    assert chosen.tier.name == "local-q8"


def test_a_slow_tier_is_still_a_usable_tier() -> None:
    chosen = select(Router(ALL), health=health(frontier=HealthState.SLOW))
    assert chosen.tier.name == "frontier"


def test_an_unprobed_tier_is_not_usable() -> None:
    """UNKNOWN means nobody has checked, which is not the same as working."""
    chosen = select(Router(ALL), health=health(frontier=HealthState.UNKNOWN))
    assert chosen.tier.name == "local-q8"


def test_the_floor_excludes_tiers_that_are_not_good_enough() -> None:
    chosen = select(Router(ALL), min_rank=1, health=health(frontier=HealthState.UNREACHABLE))
    assert chosen.tier.name == "local-q8"
    assert "local-q4" in chosen.tiers_available, "still reachable, just not good enough"


def test_a_floor_nothing_meets_escalates_rather_than_answering_worse() -> None:
    """This is what stops the ladder being a fallback chain."""
    with pytest.raises(MinRankUnmetError) as caught:
        select(
            Router(ALL),
            min_rank=1,
            health=health(
                frontier=HealthState.UNREACHABLE, **{"local-q8": HealthState.UNREACHABLE}
            ),
        )
    assert caught.value.min_rank == 1
    assert "local-q4" in caught.value.available


def test_a_low_battery_takes_the_cheapest_acceptable_local_tier() -> None:
    chosen = select(
        Router(ALL),
        min_rank=2,
        health=health(frontier=HealthState.UNREACHABLE),
        power=PowerState.BATTERY_LOW,
    )
    assert chosen.tier.name == "local-q4"
    assert chosen.unconstrained.name == "local-q8", "what it would have taken on mains"
    assert chosen.power_constrained


def test_the_power_rule_respects_the_floor() -> None:
    chosen = select(
        Router(ALL),
        min_rank=1,
        health=health(frontier=HealthState.UNREACHABLE),
        power=PowerState.BATTERY_LOW,
    )
    assert chosen.tier.name == "local-q8", "rank 2 is cheaper but the floor forbids it"


def test_the_power_rule_leaves_a_reachable_remote_tier_alone() -> None:
    """Rank tracks the cost of local inference. It is the wrong proxy for a remote
    call, which costs radio rather than compute and may be cheaper than running a
    model on the truck."""
    chosen = select(Router(ALL), power=PowerState.BATTERY_LOW)
    assert chosen.tier.name == "frontier"
    assert not chosen.power_constrained


def test_a_decision_below_the_threshold_is_flagged_for_review() -> None:
    chosen = select(
        Router(ALL), review_above_rank=0, health=health(frontier=HealthState.UNREACHABLE)
    )
    assert chosen.review_required
    assert chosen.review_reason == "TIER_BELOW_REVIEW_THRESHOLD"


def test_a_decision_at_or_above_the_threshold_is_not() -> None:
    chosen = select(Router(ALL), review_above_rank=0)
    assert not chosen.review_required
    assert chosen.review_reason is None


def test_power_constrained_is_recorded_when_that_is_the_only_reason() -> None:
    """On mains this decision would not have been reviewed at all. The reason says so."""
    chosen = select(
        Router(ALL),
        min_rank=2,
        review_above_rank=1,
        health=health(frontier=HealthState.UNREACHABLE),
        power=PowerState.BATTERY_LOW,
    )
    assert chosen.tier.name == "local-q4"
    assert chosen.review_required
    assert chosen.review_reason == "POWER_CONSTRAINED"


def test_the_threshold_reason_wins_when_it_would_have_applied_anyway() -> None:
    chosen = select(
        Router(ALL),
        min_rank=2,
        review_above_rank=0,
        health=health(frontier=HealthState.UNREACHABLE),
        power=PowerState.BATTERY_LOW,
    )
    assert chosen.review_reason == "TIER_BELOW_REVIEW_THRESHOLD", (
        "rank 1 would have been reviewed too, so power is not the cause"
    )


def test_the_stamp_carries_what_a_record_needs() -> None:
    stamp = select(Router(ALL)).stamp()
    assert stamp == {"name": "frontier", "rank": 0, "kind": "remote", "model": "big"}


def test_tiers_available_is_what_was_usable_at_selection_time() -> None:
    chosen = select(Router(ALL), health=health(frontier=HealthState.UNREACHABLE))
    assert chosen.tiers_available == ["local-q8", "local-q4"]


def test_a_mid_call_failure_falls_one_step_not_a_cascade() -> None:
    router = Router(ALL)
    assert router.next_after(FRONTIER, min_rank=2, health=health()).name == "local-q8"  # pyright: ignore[reportOptionalMemberAccess]
    assert router.next_after(LOCAL_Q4, min_rank=2, health=health()) is None


def test_a_fallback_still_respects_the_floor() -> None:
    assert Router(ALL).next_after(LOCAL_Q8, min_rank=1, health=health()) is None


def test_a_task_asks_for_less_rather_than_being_truncated() -> None:
    budget = TaskBudget(max_tokens_total=1000)
    budget.tokens_consumed = 900
    assert budget.request_tokens(tier_max_tokens=2048) == 100


def test_exceeding_a_ceiling_raises_rather_than_truncating() -> None:
    budget = TaskBudget(max_tokens_total=100, max_steps=2, started_ms=0)
    budget.spend(60, now_ms=1000)
    with pytest.raises(BudgetExhaustedError) as caught:
        budget.spend(60, now_ms=2000)
    assert caught.value.which == "tokens"


def test_the_step_ceiling_is_enforced_too() -> None:
    budget = TaskBudget(max_steps=1, started_ms=0)
    budget.spend(1, now_ms=1)
    with pytest.raises(BudgetExhaustedError, match="steps"):
        budget.spend(1, now_ms=2)


def test_the_sensor_is_settable_so_the_power_rule_is_demonstrable() -> None:
    sensor = FakeResourceSensor()
    assert sensor.power_state() is PowerState.MAINS
    assert sensor.set(PowerState.BATTERY_LOW) is PowerState.MAINS
    assert sensor.power_state() is PowerState.BATTERY_LOW
