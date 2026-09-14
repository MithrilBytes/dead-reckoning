# SPDX-License-Identifier: Apache-2.0
"""`dr sync`, `dr conflicts`, `dr resolve`: exchanging logs and adjudicating them.

`dr conflicts` is the command this project exists to be able to offer. It shows
two decisions that cannot both stand, side by side, with the tier and mode each
was made under and the reasoning each gave, so a person can see that both were
reasonable and decide which one the world should act on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer
from rich.table import Table

from deadreckoning.cli_support import ConfigOption, JsonOption, emit, fail, open_node, stdout
from deadreckoning.node import Node
from deadreckoning.outbox import Outbox, OutboxState
from deadreckoning.records import RecordKind
from deadreckoning.sync.client import SyncClient
from deadreckoning.sync.conflicts import dedupe_key as compute_dedupe_key
from deadreckoning.sync.conflicts import outcome_of, plan_from, undetected


def sync(
    config: ConfigOption = Path("dr.toml"),
    with_: Annotated[str | None, typer.Option("--with", help="Hub or peer base URL.")] = None,
    as_json: JsonOption = False,
) -> None:
    """Exchange records with a hub or peer, push first, then pull and merge.

    Pushing before pulling is not arbitrary. It gives the other side the earliest
    chance to detect a conflict before it drains actions of its own.
    """
    with open_node(config) as node:
        target = with_ or _configured_hub(node)
        if target is None:
            fail("no sync target: pass --with, or declare a SYNC_HUB dependency")
            raise AssertionError("unreachable")
        client = SyncClient(target, peer_name=target)
        held = list(node.store.iter_records())
        try:
            peer_vector = client.head()
            pushed = client.push(held, peer_vector)
            node.emit(RecordKind.SYNC, body=pushed.as_body())
            known = sorted(set(peer_vector) | {node.node_id})
            pulled = client.pull(held, known)
        except Exception as exc:
            fail(f"sync with {target} failed: {type(exc).__name__}: {exc}")
            raise AssertionError("unreachable") from exc

        for record in pulled.accepted:
            node.store.append(record)
        found = undetected(list(node.store.iter_records()), node.node_id)
        pulled.conflicts_detected = len(found)
        node.emit(RecordKind.SYNC, body=pulled.as_body())
        for conflict in found:
            _record_conflict(node, conflict)
        payload = {
            "peer": target,
            "pushed": pushed.pushed,
            "pulled": pulled.pulled,
            "conflicts_detected": len(found),
        }

    def render(p: dict[str, Any]) -> None:
        stdout.print(
            f"pushed {p['pushed']}, pulled {p['pulled']} from {p['peer']};"
            f" {p['conflicts_detected']} conflict(s) detected"
        )

    emit(payload, as_json, render)


def _configured_hub(node: Node) -> str | None:
    hub = next((d for d in node.config.dependencies if str(d.type) in ("SYNC_HUB", "PEER")), None)
    return hub.base_url if hub else None


def _record_conflict(node: Any, conflict: Any) -> None:
    """Write the conflict, and hold anything that traced to either decision.

    Entries already done are listed rather than hidden: an adjudicator has to know
    that one of these actions has already happened.
    """
    box = Outbox(node.database, node.node_id)
    traced = box.tracing_to(set(conflict.record_ids))
    held: list[str] = []
    already: list[str] = []
    for entry in traced:
        if entry.state is OutboxState.DONE:
            already.append(entry.id)
        elif not entry.terminal:
            entry.hold = {"conflict_id": conflict.dedupe_key, "previous_state": str(entry.state)}
            box.transition(entry, OutboxState.ON_HOLD, reason="CONCURRENT_DECISION")
            held.append(entry.id)
    node.emit(
        RecordKind.CONFLICT,
        body=conflict.as_body(node.node_id, held, already),
        subject=conflict.subject,
    )


def conflicts(config: ConfigOption = Path("dr.toml"), as_json: JsonOption = False) -> None:
    """Open disagreements, both sides shown, grouped by their shared identity."""
    with open_node(config) as node:
        records = list(node.store.iter_records())
        by_id = {r.id: r for r in records}
        resolved = {
            key
            for r in records
            if r.kind is RecordKind.RESOLUTION
            for key in r.body.get("dedupe_keys", [])
        }
        groups: dict[str, dict[str, Any]] = {}
        for record in records:
            if record.kind is not RecordKind.CONFLICT:
                continue
            key = str(record.body["dedupe_key"])
            if key in resolved:
                continue
            group = groups.setdefault(
                key,
                {
                    "dedupe_key": key,
                    "subtype": record.body["subtype"],
                    "subject": record.body["subject"],
                    "outcome_key": record.body["outcome_key"],
                    "detected_by": [],
                    "sides": [],
                    "held_outbox_ids": [],
                    "already_executed_outbox_ids": [],
                },
            )
            group["detected_by"].append(record.node_id)
            group["held_outbox_ids"] += record.body.get("held_outbox_ids", [])
            group["already_executed_outbox_ids"] += record.body.get(
                "already_executed_outbox_ids", []
            )
            if not group["sides"]:
                for record_id in record.body["record_ids"]:
                    side = by_id.get(record_id)
                    if side is None:
                        continue
                    group["sides"].append(
                        {
                            "record_id": side.id,
                            "node": side.node_id,
                            "value": outcome_of(side).get("value"),
                            "tier": (side.tier or {}).get("name"),
                            "rank": (side.tier or {}).get("rank"),
                            "mode": str(side.mode),
                            "rationale": side.body.get("rationale") or side.body.get("reason"),
                        }
                    )
        payload = {"conflicts": list(groups.values())}

    def render(p: dict[str, Any]) -> None:
        if not p["conflicts"]:
            stdout.print("[dim]no open conflicts[/dim]")
            return
        for group in p["conflicts"]:
            stdout.print(
                f"[bold]{group['subject']}[/bold]  {group['outcome_key']}"
                f"  [dim]{group['subtype']}  {group['dedupe_key'][:16]}...[/dim]"
            )
            table = Table(box=None, pad_edge=False)
            for column in ("record", "node", "value", "tier", "mode", "reasoning"):
                table.add_column(column, overflow="fold")
            for side in group["sides"]:
                table.add_row(
                    side["record_id"][:20],
                    side["node"],
                    str(side["value"]),
                    f"{side['tier']} (rank {side['rank']})",
                    side["mode"],
                    side["rationale"] or "",
                )
            stdout.print(table)
            if group["held_outbox_ids"]:
                stdout.print(f"  held: {', '.join(sorted(set(group['held_outbox_ids'])))}")
            if group["already_executed_outbox_ids"]:
                stdout.print(
                    "  [yellow]already executed:"
                    f" {', '.join(sorted(set(group['already_executed_outbox_ids'])))}[/yellow]"
                )
            stdout.print("")

    emit(payload, as_json, render)


def resolve(
    key: Annotated[str, typer.Argument(help="Conflict id or dedupe key.")],
    by: Annotated[str, typer.Option("--by", help="Who is deciding.")],
    config: ConfigOption = Path("dr.toml"),
    choose: Annotated[
        str | None, typer.Option("--choose", help="Record id whose outcome stands.")
    ] = None,
    override_key: Annotated[str | None, typer.Option("--key", help="Override outcome key.")] = None,
    override_value: Annotated[
        str | None, typer.Option("--value", help="Override outcome value.")
    ] = None,
    note: Annotated[str | None, typer.Option("--note")] = None,
    as_json: JsonOption = False,
) -> None:
    """Settle a conflict. A person decides; the runtime records and enacts it."""
    if not choose and not (override_key and override_value):
        fail("pass --choose <record id>, or --key and --value to override both sides")

    with open_node(config) as node:
        records = list(node.store.iter_records())
        conflict_records = [
            r
            for r in records
            if r.kind is RecordKind.CONFLICT and key in (r.id, r.body.get("dedupe_key"))
        ]
        if not conflict_records:
            fail(f"no open conflict matching {key!r}")
            raise AssertionError("unreachable")

        record_ids: set[str] = set()
        dedupe_keys: set[str] = set()
        for record in conflict_records:
            record_ids.update(record.body["record_ids"])
            dedupe_keys.add(str(record.body["dedupe_key"]))
        subject = str(conflict_records[0].body["subject"])

        by_id = {r.id: r for r in records}
        if choose and choose not in record_ids:
            fail(f"{choose!r} is not one of the records in this conflict")
        effective = (
            outcome_of(by_id[choose]) if choose else {"key": override_key, "value": override_value}
        )

        resolution = node.emit(
            RecordKind.RESOLUTION,
            body={
                "conflict_ids": [r.id for r in conflict_records],
                "dedupe_keys": sorted(dedupe_keys),
                "chosen": choose,
                "effective_outcome": effective,
                "by": by,
                "note": note,
                "released_outbox_ids": [],
                "cancelled_outbox_ids": [],
            },
            subject=subject,
            supersedes=sorted(record_ids),
        )
        released, cancelled = _apply(node, resolution)
        payload = {
            "resolution_id": resolution.id,
            "chosen": choose,
            "effective_outcome": effective,
            "released": released,
            "cancelled": cancelled,
            "dedupe_keys": sorted(dedupe_keys),
        }

    def render(p: dict[str, Any]) -> None:
        stdout.print(
            f"resolved {p['dedupe_keys'][0][:16]}... as"
            f" {p['effective_outcome']['key']} = {p['effective_outcome']['value']!r}"
        )
        if p["released"]:
            stdout.print(f"  released: {', '.join(p['released'])}")
        if p["cancelled"]:
            stdout.print(f"  cancelled: {', '.join(p['cancelled'])}")

    emit(payload, as_json, render)


def _apply(node: Any, resolution: Any) -> tuple[list[str], list[str]]:
    """Enact a resolution on this node's held entries.

    The same predicate every node applies when it merges this record, so the
    resolving node and its peers reach the same state without coordinating.
    """
    plan = plan_from(resolution)
    box = Outbox(node.database, node.node_id)
    released: list[str] = []
    cancelled: list[str] = []
    for entry in box.all(OutboxState.ON_HOLD):
        verdict = plan.verdict(entry.decision_id)
        if verdict == "RELEASE":
            previous = (entry.hold or {}).get("previous_state", str(OutboxState.READY))
            transition = box.transition(
                entry, OutboxState(previous), reason="RELEASED_BY_RESOLUTION"
            )
            released.append(entry.id)
            node.emit(RecordKind.OUTBOX_STATE, body=transition.as_body(), subject=entry.subject)
        elif verdict == "CANCEL":
            transition = box.transition(
                entry, OutboxState.CANCELLED, reason="CANCELLED_BY_RESOLUTION"
            )
            cancelled.append(entry.id)
            node.emit(RecordKind.OUTBOX_STATE, body=transition.as_body(), subject=entry.subject)
    return released, cancelled


__all__ = ["compute_dedupe_key", "conflicts", "resolve", "sync"]
