# SPDX-License-Identifier: Apache-2.0
"""`dr chaos`: staged faults, recorded as staged.

Its own module because fault injection is a distinct surface, and because an
operator checking a recording needs to read the whole command in one piece to
satisfy themselves that what they saw was what was injected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from deadreckoning.budget import PowerState
from deadreckoning.chaos import ChaosDisabledError
from deadreckoning.cli_support import (
    ConfigOption,
    JsonOption,
    emit,
    fail,
    open_node,
    stdout,
)
from deadreckoning.health import FailureClass


def chaos(
    dependency: Annotated[
        str | None, typer.Argument(help="Dependency to affect. Omit with --restore-all.")
    ] = None,
    config: ConfigOption = Path("dr.toml"),
    failure: Annotated[str | None, typer.Option("--fail", help="Failure class to inject.")] = None,
    latency: Annotated[int | None, typer.Option("--latency", help="Milliseconds to add.")] = None,
    for_seconds: Annotated[
        float | None, typer.Option("--for", help="Lift the fault automatically after this long.")
    ] = None,
    restore: Annotated[
        bool, typer.Option("--restore", help="Lift this dependency's fault.")
    ] = False,
    restore_all: Annotated[
        bool, typer.Option("--restore-all", help="Lift every fault, in one derivation.")
    ] = False,
    power: Annotated[
        str | None,
        typer.Option("--power", help="Set the power state: MAINS, BATTERY_HIGH, BATTERY_LOW."),
    ] = None,
    as_json: JsonOption = False,
) -> None:
    """Inject a fault, and record that it was injected.

    Everything here is written to the log as an injection, so a recording can
    never be mistaken for a natural outage.
    """
    if failure is not None and failure not in set(FailureClass):
        fail(f"unknown failure class {failure!r}. Known: {', '.join(sorted(FailureClass))}")
    if power is not None:
        if power not in set(PowerState):
            fail(f"unknown power state {power!r}. Known: {', '.join(sorted(PowerState))}")
        _set_power(config, PowerState(power), as_json)
        return
    if not restore_all and dependency is None:
        fail("name a dependency, pass --restore-all, or set --power")

    with open_node(config) as node:
        try:
            if restore_all:
                lifted = node.lift_faults(None)
                payload: dict[str, Any] = {"lifted": lifted, "mode": str(node.mode)}
            elif restore:
                assert dependency is not None
                lifted = node.lift_faults(dependency)
                payload = {"lifted": lifted, "mode": str(node.mode)}
            else:
                assert dependency is not None
                node.arm_fault(
                    dependency,
                    failure_class=FailureClass(failure) if failure else None,
                    latency_ms=latency,
                    for_seconds=for_seconds,
                )
                payload = {
                    "armed": dependency,
                    "failure_class": failure,
                    "latency_ms": latency,
                    "mode": str(node.mode),
                    "state": str(node.monitor.get(dependency).state),
                }
        except ChaosDisabledError as exc:
            fail(str(exc))
            raise AssertionError("unreachable") from exc
        except KeyError as exc:
            fail(f"{exc.args[0]}")
            raise AssertionError("unreachable") from exc

    def render(p: dict[str, Any]) -> None:
        if "armed" in p:
            what = p["failure_class"] or f"{p['latency_ms']} ms latency"
            stdout.print(f"injected [yellow]{what}[/yellow] on {p['armed']}")
            stdout.print(f"{p['armed']} is now {p['state']}; mode {p['mode']}")
        else:
            lifted = ", ".join(p["lifted"]) or "nothing"
            stdout.print(f"restored {lifted}; mode {p['mode']}")

    emit(payload, as_json, render)


def _set_power(config: Path, state: PowerState, as_json: bool) -> None:
    """Change the power state, and record the change.

    Power is not a dependency: it has no health state and nothing in the
    dependency type enum describes it, so this emits a RESOURCE_CHANGE rather
    than pretending it is a health event. The routing that follows turns on it,
    so it has to be explicable from the log alone.
    """
    from deadreckoning.records import RecordKind

    with open_node(config) as node:
        previous = node.power
        node.set_power(state)
        node.emit(
            RecordKind.RESOURCE_CHANGE,
            body={
                "from_power": str(previous),
                "to_power": str(state),
                "source": "FAULT_INJECTION",
            },
        )
        payload = {"from_power": str(previous), "to_power": str(state)}

    def render(p: dict[str, Any]) -> None:
        stdout.print(f"power {p['from_power']} -> [bold]{p['to_power']}[/bold]")

    emit(payload, as_json, render)
