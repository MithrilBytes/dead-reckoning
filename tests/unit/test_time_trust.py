# SPDX-License-Identifier: Apache-2.0
"""Whether this node believes its own clock, which decides how it reads a
certificate error."""

from __future__ import annotations

from deadreckoning.clock import TimeTrust, TimeTrustTracker, TrustSource

HOUR = 3_600_000


def tracker() -> TimeTrustTracker:
    return TimeTrustTracker(fresh_s=3600, stale_s=86400, skew_threshold_s=300)


def test_a_node_that_has_checked_nothing_trusts_nothing() -> None:
    assert tracker().assess(now_ms=0).trust is TimeTrust.UNTRUSTED


def test_a_recent_close_check_is_trusted() -> None:
    t = tracker()
    t.observe_time_source(offset_ms=120, at_ms=0)
    assert t.assess(now_ms=60_000).trust is TimeTrust.TRUSTED


def test_trust_decays_to_drifting_then_to_untrusted() -> None:
    t = tracker()
    t.observe_time_source(offset_ms=0, at_ms=0)
    assert t.assess(now_ms=HOUR - 1).trust is TimeTrust.TRUSTED
    assert t.assess(now_ms=HOUR * 5).trust is TimeTrust.DRIFTING
    assert t.assess(now_ms=HOUR * 30).trust is TimeTrust.UNTRUSTED


def test_a_large_offset_from_the_time_source_is_not_trusted() -> None:
    t = tracker()
    t.observe_time_source(offset_ms=600_000, at_ms=0)
    assert t.assess(now_ms=1000).trust is TimeTrust.DRIFTING


def test_a_date_header_can_lower_trust() -> None:
    """Free evidence, taken from traffic the node was making anyway."""
    t = tracker()
    t.observe_time_source(offset_ms=0, at_ms=0)
    assert t.assess(now_ms=1000).trust is TimeTrust.TRUSTED
    t.observe_date_header(offset_ms=900_000)
    assessment = t.assess(now_ms=2000)
    assert assessment.trust is TimeTrust.DRIFTING
    assert assessment.source is TrustSource.DATE_HEADER


def test_a_date_header_cannot_restore_trust() -> None:
    """A Date header is not an authenticated source. A wrong or hostile peer must
    not be able to talk this node into believing its clock."""
    t = tracker()
    t.observe_time_source(offset_ms=0, at_ms=0)
    t.observe_date_header(offset_ms=900_000)
    t.observe_date_header(offset_ms=0)
    assert t.assess(now_ms=2000).trust is TimeTrust.DRIFTING


def test_a_backward_jump_destroys_trust_outright() -> None:
    t = tracker()
    t.observe_time_source(offset_ms=0, at_ms=0)
    t.note_backward_jump(delta_ms=-600_000)
    assessment = t.assess(now_ms=1000)
    assert assessment.trust is TimeTrust.UNTRUSTED
    assert assessment.source is TrustSource.BACKWARD_JUMP


def test_a_small_backward_step_is_tolerated() -> None:
    t = tracker()
    t.observe_time_source(offset_ms=0, at_ms=0)
    t.note_backward_jump(delta_ms=-50)
    assert t.assess(now_ms=1000).trust is TimeTrust.TRUSTED


def test_only_a_real_check_clears_a_jump() -> None:
    t = tracker()
    t.note_backward_jump(delta_ms=-600_000)
    t.observe_date_header(offset_ms=0)
    assert t.assess(now_ms=1000).trust is TimeTrust.UNTRUSTED
    t.observe_time_source(offset_ms=10, at_ms=1000)
    assert t.assess(now_ms=2000).trust is TimeTrust.TRUSTED
