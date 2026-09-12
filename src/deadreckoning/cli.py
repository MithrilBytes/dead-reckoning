# SPDX-License-Identifier: Apache-2.0
"""The `dr` command line.

Every command takes --json, because the first thing anyone does with a log like
this is feed it to something else. Nothing here opens a network connection: the
commands in this milestone read and write one local database and nothing more.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.table import Table

from deadreckoning.canonical import content_hash
from deadreckoning.cli_chaos import chaos
from deadreckoning.cli_outbox import approve, outbox, reject
from deadreckoning.cli_support import (
    ConfigOption,
    JsonOption,
    emit,
    fail,
    load,
    open_node,
    stderr,
    stdout,
)
from deadreckoning.health import REMEDIATION
from deadreckoning.node import Node
from deadreckoning.records import IdentityState, RecordKind
from deadreckoning.runtime import SCHEMA_VERSION

app = typer.Typer(
    name="dr",
    help="Dead Reckoning: an agent runtime for degraded and disconnected environments.",
    no_args_is_help=True,
    add_completion=False,
)
app.command()(chaos)
app.command()(outbox)
app.command()(approve)
app.command()(reject)


@app.command()
def init(
    config: ConfigOption = Path("dr.toml"),
    node_id: Annotated[
        str | None, typer.Option("--node-id", help="Override the configured id.")
    ] = None,
    as_json: JsonOption = False,
) -> None:
    """Create the data directory, the database, and this node's genesis record."""
    loaded, base = load(config)
    if node_id:
        loaded = loaded.model_copy(
            update={"node": loaded.node.model_copy(update={"node_id": node_id})}
        )
    data_dir = loaded.data_dir(base)

    with Node.open(loaded, base) as node:
        existing = node.store.head(node.node_id)
        if existing is not None:
            fail(
                f"{data_dir} already holds {node.store.count()} record(s) for node"
                f" {node.node_id!r}. Refusing to write a second genesis record."
            )
        genesis = node.emit(
            RecordKind.NODE_INIT,
            body={
                "schema_version": SCHEMA_VERSION,
                "profile": loaded.profile,
                "config_hash": content_hash(loaded.model_dump(mode="json")),
            },
        )
        # Assess before answering. A node that has just been created has a mode
        # like any other, and recording it here means the log opens with what the
        # node believed about itself rather than with a silence.
        node.reassess_mode()
        payload = {
            "node_id": node.node_id,
            "data_dir": str(data_dir),
            "genesis_record_id": genesis.id,
            "hash": genesis.hash,
            "mode": str(node.mode),
        }

    def render(p: dict[str, Any]) -> None:
        stdout.print(f"initialised node [bold]{p['node_id']}[/bold] at {p['data_dir']}")
        stdout.print(f"genesis record {p['genesis_record_id']}, mode {p['mode']}")

    emit(payload, as_json, render)


@app.command()
def status(config: ConfigOption = Path("dr.toml"), as_json: JsonOption = False) -> None:
    """Mode, health, and why. Needs no network and no model."""
    with open_node(config) as node:
        node.reassess_mode()
        health = node.monitor.vector
        payload = {
            "node_id": node.node_id,
            "mode": str(node.mode),
            "pending_mode": str(node.controller.pending) if node.controller.pending else None,
            "time_trust": str(node.time_trust),
            "records": node.store.count(),
            "chaos_enabled": node.injector.enabled,
            "dependencies": [
                {
                    "name": name,
                    "type": node.dependencies[name],
                    "state": str(item.state),
                    "breaker": str(item.breaker),
                    "failure_class": (
                        str(item.last_failure_class) if item.last_failure_class else None
                    ),
                    "latency_ms": item.last_latency_ms,
                    "remediation": (
                        REMEDIATION.get(item.last_failure_class, "")
                        if item.last_failure_class
                        else ""
                    ),
                    "injected": name in node.injector.all_armed(),
                    "needs_reconciliation": item.needs_reconciliation,
                }
                for name, item in sorted(health.items())
            ],
        }

    def render(p: dict[str, Any]) -> None:
        pending = (
            f"  [dim](holding {p['pending_mode']} until the dwell elapses)[/dim]"
            if p["pending_mode"]
            else ""
        )
        stdout.print(f"[bold]{p['node_id']}[/bold]  mode [bold]{p['mode']}[/bold]{pending}")
        stdout.print(f"time trust {p['time_trust']}   records {p['records']}")
        table = Table(box=None, pad_edge=False)
        for column in ("dependency", "type", "state", "breaker", "last class", "note"):
            table.add_column(column, overflow="fold")
        for dep in p["dependencies"]:
            marker = " [yellow](injected)[/yellow]" if dep["injected"] else ""
            table.add_row(
                dep["name"] + marker,
                dep["type"],
                dep["state"],
                dep["breaker"],
                dep["failure_class"] or "",
                dep["remediation"],
            )
        stdout.print(table)

    emit(payload, as_json, render)


@app.command(name="log")
def log_command(
    config: ConfigOption = Path("dr.toml"),
    node: Annotated[str | None, typer.Option("--node", help="Only this node's records.")] = None,
    kind: Annotated[str | None, typer.Option("--kind", help="Only this record kind.")] = None,
    subject: Annotated[str | None, typer.Option("--subject", help="Only this subject.")] = None,
    task: Annotated[str | None, typer.Option("--task", help="Only this task.")] = None,
    limit: Annotated[int | None, typer.Option("--limit", help="Most recent N.")] = None,
    as_json: JsonOption = False,
) -> None:
    """Read the log."""
    if kind is not None and kind not in set(RecordKind):
        fail(f"unknown record kind {kind!r}. Known kinds: {', '.join(sorted(RecordKind))}")
    with open_node(config) as dr:
        rows = list(
            dr.store.iter_records(
                node_id=node,
                kind=RecordKind(kind) if kind else None,
                subject=subject,
                task_id=task,
                limit=limit,
            )
        )
    payload = [r.stored_payload() for r in rows]

    def render(_: Any) -> None:
        if not rows:
            stdout.print("[dim]no records match[/dim]")
            return
        table = Table(box=None, pad_edge=False)
        for column in ("stamp", "kind", "mode", "subject", "id"):
            table.add_column(column, overflow="fold")
        for record in rows:
            table.add_row(
                f"{record.hlc['physical_ms']}-{record.hlc['logical']}",
                str(record.kind),
                str(record.mode),
                record.subject or "",
                record.id,
            )
        stdout.print(table)

    emit(payload, as_json, render)


@app.command()
def show(
    record_id: Annotated[str, typer.Argument(help="Record id.")],
    config: ConfigOption = Path("dr.toml"),
    as_json: JsonOption = False,
) -> None:
    """Show one record in full, with the record before and after it on its chain."""
    with open_node(config) as dr:
        record = dr.store.get(record_id)
        if record is None:
            fail(f"no record with id {record_id!r}")
            raise AssertionError("unreachable")
        chain = list(dr.store.iter_records(node_id=record.node_id))
        position = next(i for i, r in enumerate(chain) if r.id == record.id)
        payload = {
            "record": record.stored_payload(),
            "position": position,
            "previous": chain[position - 1].id if position > 0 else None,
            "next": chain[position + 1].id if position + 1 < len(chain) else None,
            "hash_recomputes": record.compute_hash() == record.hash,
        }

    def render(p: dict[str, Any]) -> None:
        stdout.print_json(json.dumps(p["record"], indent=2, default=str))
        stdout.print(
            f"[dim]position {p['position']} on {record.node_id}"
            f" | previous {p['previous']} | next {p['next']}[/dim]"
        )
        if not p["hash_recomputes"]:
            stderr.print("[red]this record does not match its own hash[/red]")

    emit(payload, as_json, render)


@app.command()
def verify(config: ConfigOption = Path("dr.toml"), as_json: JsonOption = False) -> None:
    """Verify the hash chain of every node present in this store."""
    with open_node(config) as dr:
        results = dr.store.verify_all()
        counts = {n: sum(1 for _ in dr.store.iter_records(node_id=n)) for n in results}
    payload = {
        "ok": all(break_ is None for break_ in results.values()),
        "nodes": [
            {
                "node_id": node_id,
                "records": counts[node_id],
                "ok": break_ is None,
                "break": break_.model_dump() if break_ else None,
            }
            for node_id, break_ in sorted(results.items())
        ],
    }

    def render(p: dict[str, Any]) -> None:
        if not p["nodes"]:
            stdout.print("[dim]no records to verify[/dim]")
            return
        for entry in p["nodes"]:
            if entry["ok"]:
                stdout.print(
                    f"[green]ok[/green] {entry['node_id']}: {entry['records']} record(s) verified"
                )
            else:
                broken: dict[str, Any] = entry["break"]
                stdout.print(
                    f"[red]BREAK[/red] {entry['node_id']} at position {broken['position']},"
                    f" record {broken['record_id']}: {broken['reason']}"
                )
                stdout.print(f"       expected {broken['expected']}")
                stdout.print(f"       found    {broken['found']}")

    emit(payload, as_json, render)
    if not payload["ok"]:
        raise typer.Exit(code=1)


@app.command()
def manifest(config: ConfigOption = Path("dr.toml"), as_json: JsonOption = False) -> None:
    """Print the capability manifest exactly as the model would receive it.

    Needs no network and no model. This is the command for answering "why did it
    decide that", because the manifest is what the model was told it could do, and
    its hash is stamped on the decision.
    """
    from deadreckoning.local_store import LocalStore
    from deadreckoning.manifest import build_manifest, manifest_hash, render_table
    from deadreckoning.tools.registry import ToolRegistry

    with open_node(config) as node:
        node.reassess_mode()
        registry = ToolRegistry()
        built = build_manifest(
            mode=node.mode,
            tier=None,
            identity=IdentityState.NONE,
            identity_ttl_s=None,
            time_trust=node.time_trust,
            registry=registry,
            store=LocalStore(node.database),
            health={name: item.state for name, item in node.monitor.vector.items()},
            now_ms=node.clock.last.physical_ms,
        )
        payload = {"manifest": built, "manifest_hash": manifest_hash(built)}

    def render(p: dict[str, Any]) -> None:
        stdout.print(render_table(p["manifest"]))
        stdout.print(f"\n[dim]manifest_hash {p['manifest_hash']}[/dim]")
        if not p["manifest"]["tools"]:
            stdout.print("[dim]no tools registered yet; they arrive with the demo tool set[/dim]")

    emit(payload, as_json, render)


def main() -> None:
    app()


if __name__ == "__main__":
    sys.exit(app())
