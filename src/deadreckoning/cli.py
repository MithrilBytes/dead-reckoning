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
from rich.console import Console
from rich.table import Table

from deadreckoning.config import Config, ConfigError, load_config
from deadreckoning.node import Node
from deadreckoning.records import RecordKind

app = typer.Typer(
    name="dr",
    help="Dead Reckoning: an agent runtime for degraded and disconnected environments.",
    no_args_is_help=True,
    add_completion=False,
)

_stdout = Console()
_stderr = Console(stderr=True)

ConfigOption = Annotated[Path, typer.Option("--config", "-c", help="Path to dr.toml.")]
JsonOption = Annotated[bool, typer.Option("--json", help="Machine readable output.")]


def _fail(message: str) -> None:
    _stderr.print(f"[red]error[/red] {message}")
    raise typer.Exit(code=2)


def _load(config_path: Path) -> tuple[Config, Path]:
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        _fail(str(exc))
        raise AssertionError("unreachable") from exc
    return config, config_path.resolve().parent


def _open(config_path: Path) -> Node:
    config, base = _load(config_path)
    return Node.open(config, base)


def _emit(payload: Any, as_json: bool, render: Any) -> None:
    if as_json:
        _stdout.print_json(json.dumps(payload, default=str))
    else:
        render(payload)


@app.command()
def init(
    config: ConfigOption = Path("dr.toml"),
    node_id: Annotated[
        str | None, typer.Option("--node-id", help="Override the configured id.")
    ] = None,
    as_json: JsonOption = False,
) -> None:
    """Create the data directory, the database, and this node's genesis record."""
    loaded, base = _load(config)
    if node_id:
        loaded = loaded.model_copy(
            update={"node": loaded.node.model_copy(update={"node_id": node_id})}
        )
    data_dir = loaded.data_dir(base)

    with Node.open(loaded, base) as node:
        existing = node.store.head(node.node_id)
        if existing is not None:
            _fail(
                f"{data_dir} already holds {node.store.count()} record(s) for node"
                f" {node.node_id!r}. Refusing to write a second genesis record."
            )
        genesis = node.emit(
            RecordKind.CHECKPOINT,
            body={"event": "genesis", "schema_version": 1, "profile": loaded.profile},
        )
        payload = {
            "node_id": node.node_id,
            "data_dir": str(data_dir),
            "genesis_record_id": genesis.id,
            "hash": genesis.hash,
        }

    def render(p: dict[str, Any]) -> None:
        _stdout.print(f"initialised node [bold]{p['node_id']}[/bold] at {p['data_dir']}")
        _stdout.print(f"genesis record {p['genesis_record_id']}")

    _emit(payload, as_json, render)


@app.command()
def status(config: ConfigOption = Path("dr.toml"), as_json: JsonOption = False) -> None:
    """Show what this node is and what it holds. Needs no network and no model."""
    with _open(config) as node:
        payload = {
            "node_id": node.node_id,
            "mode": str(node.mode),
            "time_trust": str(node.time_trust),
            "records": node.store.count(),
            "nodes_known": node.store.node_ids(),
            "last_hlc": str(node.clock.last),
            "tiers": [
                {"name": t.name, "rank": t.rank, "kind": str(t.kind)} for t in node.config.tiers
            ],
            "data_dir": str(node.database.path.parent),
        }

    def render(p: dict[str, Any]) -> None:
        table = Table(show_header=False, box=None)
        table.add_row("node", p["node_id"])
        table.add_row("mode", p["mode"])
        table.add_row("time trust", p["time_trust"])
        table.add_row("records", str(p["records"]))
        table.add_row("nodes known", ", ".join(p["nodes_known"]) or "none")
        table.add_row("last stamp", p["last_hlc"])
        table.add_row(
            "tiers", ", ".join(f"{t['name']} (rank {t['rank']}, {t['kind']})" for t in p["tiers"])
        )
        _stdout.print(table)

    _emit(payload, as_json, render)


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
        _fail(f"unknown record kind {kind!r}. Known kinds: {', '.join(sorted(RecordKind))}")
    with _open(config) as dr:
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
            _stdout.print("[dim]no records match[/dim]")
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
        _stdout.print(table)

    _emit(payload, as_json, render)


@app.command()
def show(
    record_id: Annotated[str, typer.Argument(help="Record id.")],
    config: ConfigOption = Path("dr.toml"),
    as_json: JsonOption = False,
) -> None:
    """Show one record in full, with the record before and after it on its chain."""
    with _open(config) as dr:
        record = dr.store.get(record_id)
        if record is None:
            _fail(f"no record with id {record_id!r}")
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
        _stdout.print_json(json.dumps(p["record"], indent=2, default=str))
        _stdout.print(
            f"[dim]position {p['position']} on {record.node_id}"
            f" | previous {p['previous']} | next {p['next']}[/dim]"
        )
        if not p["hash_recomputes"]:
            _stderr.print("[red]this record does not match its own hash[/red]")

    _emit(payload, as_json, render)


@app.command()
def verify(config: ConfigOption = Path("dr.toml"), as_json: JsonOption = False) -> None:
    """Verify the hash chain of every node present in this store."""
    with _open(config) as dr:
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
            _stdout.print("[dim]no records to verify[/dim]")
            return
        for entry in p["nodes"]:
            if entry["ok"]:
                _stdout.print(
                    f"[green]ok[/green] {entry['node_id']}: {entry['records']} record(s) verified"
                )
            else:
                broken: dict[str, Any] = entry["break"]
                _stdout.print(
                    f"[red]BREAK[/red] {entry['node_id']} at position {broken['position']},"
                    f" record {broken['record_id']}: {broken['reason']}"
                )
                _stdout.print(f"       expected {broken['expected']}")
                _stdout.print(f"       found    {broken['found']}")

    _emit(payload, as_json, render)
    if not payload["ok"]:
        raise typer.Exit(code=1)


def main() -> None:
    app()


if __name__ == "__main__":
    sys.exit(app())
