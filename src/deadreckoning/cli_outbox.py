# SPDX-License-Identifier: Apache-2.0
"""`dr outbox`, `dr approve`, `dr reject`: the human end of a deferred action.

These exist because some actions should not happen without a person, and a person
needs to be able to see what is waiting, what justified it, and how old that
justification is, without reading a database by hand.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer
from rich.table import Table

from deadreckoning.cli_support import ConfigOption, JsonOption, emit, fail, open_node, stdout
from deadreckoning.outbox import Outbox, OutboxState
from deadreckoning.records import RecordKind


def _entry_summary(entry: Any, now_ms: int) -> dict[str, Any]:
    return {
        "id": entry.id,
        "state": str(entry.state),
        "tool": entry.tool,
        "args": entry.args,
        "subject": entry.subject,
        "consequence": str(entry.consequence),
        "idempotency_key": entry.idempotency_key,
        "expires_in_s": max(0, (entry.expires_at_ms - now_ms) // 1000),
        "expired": entry.expired(now_ms),
        "approval": entry.approval.as_dict() if entry.approval else None,
        "preconditions": entry.preconditions,
        "attempts": entry.attempts,
        "last_error": entry.last_error,
    }


def outbox(
    config: ConfigOption = Path("dr.toml"),
    state: Annotated[str | None, typer.Option("--state", help="Only this state.")] = None,
    expired: Annotated[bool, typer.Option("--expired", help="Only entries past expiry.")] = False,
    as_json: JsonOption = False,
) -> None:
    """Deferred actions, what justified them, and what they are waiting for."""
    if state is not None and state not in set(OutboxState):
        fail(f"unknown state {state!r}. Known: {', '.join(sorted(OutboxState))}")
    with open_node(config) as node:
        box = Outbox(node.database, node.node_id)
        now_ms = node.clock.last.physical_ms
        entries = box.all(OutboxState(state) if state else None)
        if expired:
            entries = [e for e in entries if e.expired(now_ms)]
        payload = {"entries": [_entry_summary(e, now_ms) for e in entries]}

    def render(p: dict[str, Any]) -> None:
        if not p["entries"]:
            stdout.print("[dim]nothing deferred[/dim]")
            return
        table = Table(box=None, pad_edge=False)
        for column in ("id", "state", "tool", "subject", "consequence", "expires in", "waiting on"):
            table.add_column(column, overflow="fold")
        for entry in p["entries"]:
            approval: dict[str, Any] = entry["approval"] or {}
            waiting = ""
            if entry["state"] == "AWAITING_APPROVAL":
                waiting = str(approval.get("reason") or "approval")
            elif entry["last_error"]:
                waiting = str(entry["last_error"])
            table.add_row(
                entry["id"],
                entry["state"],
                entry["tool"],
                entry["subject"] or "",
                entry["consequence"],
                "expired" if entry["expired"] else f"{entry['expires_in_s']}s",
                waiting,
            )
        stdout.print(table)

    emit(payload, as_json, render)


def approve(
    outbox_id: Annotated[str, typer.Argument(help="Outbox entry id.")],
    by: Annotated[str, typer.Option("--by", help="Who is approving.")],
    config: ConfigOption = Path("dr.toml"),
    note: Annotated[str | None, typer.Option("--note")] = None,
    as_json: JsonOption = False,
) -> None:
    """Approve a held action. The approver and the moment are recorded."""
    _decide(config, outbox_id, by, note, approving=True, as_json=as_json)


def reject(
    outbox_id: Annotated[str, typer.Argument(help="Outbox entry id.")],
    by: Annotated[str, typer.Option("--by", help="Who is rejecting.")],
    config: ConfigOption = Path("dr.toml"),
    note: Annotated[str | None, typer.Option("--note")] = None,
    as_json: JsonOption = False,
) -> None:
    """Reject a held action, which cancels it."""
    _decide(config, outbox_id, by, note, approving=False, as_json=as_json)


def _decide(
    config: Path, outbox_id: str, by: str, note: str | None, approving: bool, as_json: bool
) -> None:
    with open_node(config) as node:
        box = Outbox(node.database, node.node_id)
        entry = box.get(outbox_id)
        if entry is None:
            fail(f"no outbox entry {outbox_id!r}")
            raise AssertionError("unreachable")
        if entry.terminal:
            fail(f"{outbox_id} is already {entry.state} and cannot be decided on")
        transition = box.approve(entry, by, note) if approving else box.reject(entry, by, note)
        node.emit(RecordKind.OUTBOX_STATE, body=transition.as_body(), subject=entry.subject)
        payload = {"id": entry.id, "state": str(entry.state), "by": by, "note": note}

    def render(p: dict[str, Any]) -> None:
        stdout.print(f"{p['id']} is now [bold]{p['state']}[/bold], by {p['by']}")

    emit(payload, as_json, render)
