# SPDX-License-Identifier: Apache-2.0
"""The reference scenario, driven end to end.

A storm takes out substation SB-3. Two repair trucks run a node each. The cell
towers they reach headquarters through are on grid power, so as the outage
spreads the trucks lose connectivity because of the very outage they were sent to
fix.

Run it with `python -m demo.scenario_outage`. Everything is in process and on
loopback: scripted models, an in-process view of the utility's systems, and a
real hub over a real socket. No network, no API key, no model download.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import threading
import time
from pathlib import Path

from hub.server import serve

from deadreckoning.agent import Task
from deadreckoning.drain import Drainer, redecide_tasks
from deadreckoning.health import FailureClass, HealthState
from deadreckoning.llm.scripted import abstain, escalate, final, tool_call
from deadreckoning.manifest import TierView
from deadreckoning.outbox import OutboxState
from deadreckoning.records import IdentityState, Mode, Record, RecordKind
from deadreckoning.sync.conflicts import plan_from, undetected
from demo.harness import BACKENDS, Truck


def say(text: str = "") -> None:
    print(text, flush=True)


def beat(title: str) -> None:
    say()
    say(f"\033[1m{title}\033[0m")
    say("-" * len(title))


def summarise(records: list[Record], *kinds: RecordKind) -> str:
    wanted = set(kinds) if kinds else None
    return ", ".join(str(r.kind) for r in records if wanted is None or r.kind in wanted)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("./data/demo"))
    parser.add_argument("--keep", action="store_true", help="Keep the data directory.")
    args = parser.parse_args()

    root: Path = args.data_dir
    if root.exists() and not args.keep:
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    server, hub = serve(root / "hub", port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    hub_url = f"http://127.0.0.1:{server.server_address[1]}"

    seven = Truck("truck-7", root / "truck-7", hub_url)
    twelve = Truck("truck-12", root / "truck-12", hub_url)
    frontier, local = (
        TierView("frontier", 0, "scripted-frontier"),
        TierView("local-q4", 2, "scripted-local"),
    )

    try:
        beat("Phase 1  Connected. Two trucks, full fidelity.")
        records = seven.run(
            Task("T-101", "triage", "Triage T-101.", subject="ticket:T-101"),
            {
                ("T-101", 1): tool_call("lookup_asset", asset_id="F-31"),
                ("T-101", 2): tool_call("get_live_load", feeder_id="F-31"),
                ("T-101", 3): final(
                    "ticket:T-101",
                    "priority",
                    "P1",
                    "hospital on 8h backup and the feeder is loaded",
                    depends_on=["lookup_asset", "get_live_load"],
                ),
            },
            seven.health(),
            Mode.CONNECTED,
            frontier,
        )
        live = [r for r in records if r.kind is RecordKind.TOOL_RESULT]
        say(f"  truck-7 decided T-101 = P1 at rank 0, {len(live)} tools live")
        say("  nothing queued: everything is reachable")

        beat("Phase 2  SCADA is gone. The sensor ticket cannot be answered.")
        seven.node.arm_fault("scada-api", failure_class=FailureClass.CONNECT_TIMEOUT)
        degraded = seven.health(**{"scada-api": HealthState.UNREACHABLE})
        records = seven.run(
            Task("T-106", "triage", "Triage T-106.", subject="ticket:T-106"),
            {
                ("T-106", 1): tool_call("get_live_load", feeder_id="F-32"),
                ("T-106", 2): abstain(
                    "ticket:T-106",
                    "priority",
                    "priority depends on live feeder load, which is unavailable",
                    ["get_live_load"],
                    partial={"value_if_load_high": "P1", "value_if_load_low": "P3"},
                ),
            },
            degraded,
            Mode.DEGRADED,
            frontier,
        )
        say("  truck-7 abstained on T-106 and gave the conditional answer")
        say("  no FINAL was written: " + str(not any(r.kind is RecordKind.FINAL for r in records)))

        records = seven.run(
            Task("T-103", "triage", "Triage T-103.", subject="ticket:T-103"),
            {
                ("T-103", 1): tool_call("lookup_asset", asset_id="F-33"),
                ("T-103", 2): final(
                    "ticket:T-103",
                    "priority",
                    "P1",
                    "water pumping on 4h backup is P1 regardless of load",
                    depends_on=["lookup_asset"],
                ),
            },
            degraded,
            Mode.DEGRADED,
            frontier,
        )
        say("  truck-7 still decided T-103 = P1: it never needed the dead sensor")

        beat("Phase 3  Everything drops. Both trucks are on their own.")
        islanded: dict[str, HealthState] = dict.fromkeys(BACKENDS, HealthState.UNREACHABLE)
        seven.run(
            Task("T-104", "triage", "Triage T-104.", subject="ticket:T-104"),
            {
                ("T-104", 1): tool_call("get_field_reports", ticket_id="T-104"),
                ("T-104", 2): tool_call(
                    "set_ticket_priority",
                    ticket_id="T-104",
                    priority="P2",
                    rationale="ticket text only",
                ),
                ("T-104", 3): final(
                    "ticket:T-104",
                    "priority",
                    "P2",
                    "the ticket says lines down; no field report on this truck",
                    depends_on=["get_field_reports"],
                ),
            },
            islanded,
            Mode.ISLANDED,
            local,
        )
        say("  truck-7 has only the ticket text and calls T-104 a P2, at rank 2")

        twelve.backends.field_reports["T-104"] = [
            {"observation": "energized conductor on ground, 40m from school entrance"}
        ]
        twelve.run(
            Task("T-104", "triage", "Triage T-104.", subject="ticket:T-104"),
            {
                ("T-104", 1): tool_call("get_field_reports", ticket_id="T-104"),
                ("T-104", 2): tool_call(
                    "set_ticket_priority",
                    ticket_id="T-104",
                    priority="P1",
                    rationale="live conductor beside a school",
                ),
                ("T-104", 3): final(
                    "ticket:T-104",
                    "priority",
                    "P1",
                    "a crew saw an energized conductor on the ground by the school",
                    depends_on=["get_field_reports"],
                ),
            },
            islanded,
            Mode.ISLANDED,
            local,
        )
        say("  truck-12 has a field report of a live conductor and calls it a P1")
        say("  both queued a priority change; neither has heard of the other")

        seven.run(
            Task("D-101", "triage", "Dispatch to T-101.", subject="ticket:T-101"),
            {
                ("D-101", 1): tool_call("dispatch_crew", crew_id="C-1", ticket_id="T-101"),
                ("D-101", 2): final("ticket:T-101", "dispatch", "C-1", "hospital first"),
            },
            islanded,
            Mode.ISLANDED,
            local,
        )
        dispatch_entry = next(e for e in seven.outbox.all() if e.tool == "dispatch_crew")
        say(f"  dispatch to T-101 is {dispatch_entry.state}: a person has to say yes")

        seven.run(
            Task("T-108", "triage", "Triage T-108.", subject="ticket:T-108"),
            {("T-108", 1): escalate("ticket:T-108", "smell of burning near a transformer")},
            islanded,
            Mode.ISLANDED,
            local,
        )
        say("  T-108 escalated to a human: outside the agent's remit")

        beat("Phase 4  The link returns.")
        seven.outbox.approve(dispatch_entry, by="dispatcher-jlee", note="confirmed by radio")
        say("  the dispatcher approves the crew by radio")

        seven.backends.ticket("T-101")["assigned"] = True  # type: ignore[index]
        say("  meanwhile headquarters has already sent its own crew to T-101")

        for truck in (twelve, seven, twelve):
            pushed, pulled = truck.sync()
            if pushed or pulled:
                say(f"  {truck.node_id} synced: pushed {pushed}, pulled {pulled}")
        say("  both logs now hold both trucks' decisions")

        # Conflicts are detected before anything drains. That ordering is why
        # the sequence is push, pull, review, drain: an action queued on the
        # strength of a decision under dispute must not fire while a person is
        # still deciding whether the decision was right.
        found = undetected(list(seven.node.store.iter_records()), seven.node_id)
        for conflict in found:
            traced = seven.outbox.tracing_to(set(conflict.record_ids))
            held: list[str] = []
            for entry in traced:
                if not entry.terminal:
                    entry.hold = {
                        "conflict_id": conflict.dedupe_key,
                        "previous_state": str(entry.state),
                    }
                    seven.outbox.transition(
                        entry, OutboxState.ON_HOLD, reason="CONCURRENT_DECISION"
                    )
                    held.append(entry.id)
            seven.node.emit(
                RecordKind.CONFLICT,
                body=conflict.as_body(seven.node_id, held, []),
                subject=conflict.subject,
            )
            say(
                f"  conflict on {conflict.subject}: "
                + " against ".join(f"{v!r}" for v in conflict.values.values())
            )
            say(f"  {len(held)} queued action(s) held while a person decides")

        drained = Drainer(
            seven.outbox,
            seven.registry,
            seven.checkers,
            identity=IdentityState.FRESH,
            connected=True,
            now_ms=seven.node.clock.last.physical_ms,
        ).run()
        say(
            f"  truck-7 drained: {len(drained.executed)} executed,"
            f" {len(drained.precondition_failed)} refused on preconditions"
        )
        for task in redecide_tasks(seven.outbox, drained):
            delta = task.deltas[0]
            say(
                f"  {task.tool} did not fire: {delta.check} was"
                f" {delta.observed_value!r} and is {delta.current_value!r}"
            )

        beat("Phase 5  The disagreement surfaces, and a person decides.")
        if found:
            conflict = found[0]
            chosen = next(rid for rid, value in conflict.values.items() if value == "P1")
            resolution = seven.node.emit(
                RecordKind.RESOLUTION,
                body={
                    "conflict_ids": [],
                    "dedupe_keys": [conflict.dedupe_key],
                    "chosen": chosen,
                    "effective_outcome": {"key": "priority", "value": "P1"},
                    "by": "supervisor-mchen",
                    "note": "field report confirms an energized conductor",
                    "released_outbox_ids": [],
                    "cancelled_outbox_ids": [],
                },
                subject=conflict.subject,
                supersedes=sorted(conflict.record_ids),
            )
            plan = plan_from(resolution)
            for entry in seven.outbox.all(OutboxState.ON_HOLD):
                verdict = plan.verdict(entry.decision_id)
                if verdict == "CANCEL":
                    seven.outbox.transition(
                        entry, OutboxState.CANCELLED, reason="CANCELLED_BY_RESOLUTION"
                    )
            say("  supervisor-mchen chose P1; truck-7's P2 change is cancelled")

        beat("Verify")
        for truck in (seven, twelve):
            truck.sync()
            broken = truck.node.store.verify_all()
            state = "ok" if all(b is None for b in broken.values()) else "BROKEN"
            say(
                f"  {truck.node_id}: {state}, {truck.node.store.count()} records"
                f" across {len(broken)} node(s)"
            )
        hub_broken = hub.store.verify_all()
        say(
            f"  hq-hub: {'ok' if all(b is None for b in hub_broken.values()) else 'BROKEN'},"
            f" {hub.store.count()} records across {len(hub_broken)} node(s)"
        )

        beat("The whole story for ticket:T-104, from truck-7")
        for record in seven.node.store.iter_records(subject="ticket:T-104"):
            detail = ""
            if record.kind is RecordKind.FINAL:
                detail = f" {record.body['outcome']['value']} ({record.node_id})"
            elif record.kind is RecordKind.CONFLICT:
                detail = f" {record.body['subtype']}"
            elif record.kind is RecordKind.RESOLUTION:
                detail = f" chose {record.body['effective_outcome']['value']}"
            say(f"  {record.kind}{detail}")
        say()
        return 0
    finally:
        seven.close()
        twelve.close()
        server.shutdown()
        hub.close()
        time.sleep(0.1)


if __name__ == "__main__":
    sys.exit(main())
