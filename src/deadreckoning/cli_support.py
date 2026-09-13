# SPDX-License-Identifier: Apache-2.0
"""Shared plumbing for the `dr` commands: option types, output, and failure.

Every command takes --json, because the first thing anyone does with a log like
this is feed it to something else, and every command fails the same way, because
an operator reading an error at two in the morning should not have to work out
which subcommand produced it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console

from deadreckoning.config import Config, ConfigError, load_config
from deadreckoning.node import Node

stdout = Console()
stderr = Console(stderr=True)

ConfigOption = Annotated[Path, typer.Option("--config", "-c", help="Path to dr.toml.")]
JsonOption = Annotated[bool, typer.Option("--json", help="Machine readable output.")]


def fail(message: str) -> None:
    stderr.print(f"[red]error[/red] {message}")
    raise typer.Exit(code=2)


def load(config_path: Path) -> tuple[Config, Path]:
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        fail(str(exc))
        raise AssertionError("unreachable") from exc
    return config, config_path.resolve().parent


def open_node(config_path: Path) -> Node:
    config, base = load(config_path)
    return Node.open(config, base)


def emit(payload: Any, as_json: bool, render: Any) -> None:
    if as_json:
        stdout.print_json(json.dumps(payload, default=str))
    else:
        render(payload)
