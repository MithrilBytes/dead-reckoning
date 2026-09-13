# SPDX-License-Identifier: Apache-2.0
"""Fault injection, including the parts that keep a recording honest."""

from __future__ import annotations

import pytest

from deadreckoning.chaos import ChaosDisabledError, FaultInjector, InjectedFaultError
from deadreckoning.health import FailureClass


def injector(enabled: bool = True, profile: str = "demo", clock: list[float] | None = None):
    ticks = clock if clock is not None else [0.0]
    return FaultInjector(enabled=enabled, profile=profile, now=lambda: ticks[0])


def test_it_refuses_to_arm_in_production() -> None:
    """A runtime that can be told to break itself in production is a liability."""
    with pytest.raises(ChaosDisabledError, match="production"):
        FaultInjector(enabled=True, profile="production", now=lambda: 0.0)


def test_production_with_chaos_off_is_fine() -> None:
    assert FaultInjector(enabled=False, profile="production", now=lambda: 0.0).enabled is False


def test_it_is_inert_unless_enabled() -> None:
    disabled = injector(enabled=False)
    with pytest.raises(ChaosDisabledError, match="disabled"):
        disabled.arm("gis", FailureClass.DNS_FAILURE)
    assert disabled.check("gis") == 0


def test_an_armed_fault_raises_at_the_transport() -> None:
    fi = injector()
    fi.arm("scada", FailureClass.CONNECT_TIMEOUT)
    with pytest.raises(InjectedFaultError) as caught:
        fi.check("scada")
    assert caught.value.failure_class is FailureClass.CONNECT_TIMEOUT
    assert fi.check("gis") == 0, "other dependencies are untouched"


def test_latency_is_not_a_failure() -> None:
    """Added latency may or may not cross a threshold. The monitor decides, not the injector."""
    fi = injector()
    fi.arm("gis", latency_ms=250)
    assert fi.check("gis") == 250


def test_a_timed_fault_expires_on_its_own() -> None:
    ticks = [0.0]
    fi = injector(clock=ticks)
    fi.arm("gis", FailureClass.SERVER_ERROR, for_seconds=30)
    with pytest.raises(InjectedFaultError):
        fi.check("gis")
    ticks[0] = 31.0
    assert fi.check("gis") == 0


def test_restore_lifts_one_and_reports_what_it_lifted() -> None:
    fi = injector()
    fi.arm("gis", FailureClass.DNS_FAILURE)
    fi.arm("scada", FailureClass.CONNECT_TIMEOUT)
    lifted = fi.restore("gis")
    assert lifted is not None
    assert lifted.failure_class is FailureClass.DNS_FAILURE
    assert fi.check("gis") == 0
    with pytest.raises(InjectedFaultError):
        fi.check("scada")


def test_restore_all_lifts_everything_in_one_go() -> None:
    """One call, one derivation. Restoring one at a time would walk the mode
    controller through states the node never really occupied."""
    fi = injector()
    for name in ("gis", "scada", "ticket-api"):
        fi.arm(name, FailureClass.CONNECT_TIMEOUT)
    lifted = fi.restore_all()
    assert sorted(lifted) == ["gis", "scada", "ticket-api"]
    assert all(fi.check(name) == 0 for name in lifted)
    assert fi.all_armed() == {}


def test_restoring_something_that_was_never_armed_is_not_an_error() -> None:
    assert injector().restore("gis") is None


def test_armed_faults_are_enumerable_for_the_log() -> None:
    """An operator has to be able to see what is currently faked."""
    fi = injector()
    fi.arm("gis", FailureClass.DNS_FAILURE)
    fi.arm("scada", latency_ms=100)
    armed = fi.all_armed()
    assert set(armed) == {"gis", "scada"}
    assert armed["gis"].as_dict()["failure_class"] == "DNS_FAILURE"
    assert armed["scada"].as_dict()["latency_ms"] == 100


@pytest.mark.parametrize("failure_class", list(FailureClass))
def test_every_class_in_the_taxonomy_can_be_injected(failure_class: FailureClass) -> None:
    """If a class cannot be injected it cannot be tested, and it may as well not exist."""
    fi = injector()
    fi.arm("d", failure_class)
    with pytest.raises(InjectedFaultError) as caught:
        fi.check("d")
    assert caught.value.failure_class is failure_class
