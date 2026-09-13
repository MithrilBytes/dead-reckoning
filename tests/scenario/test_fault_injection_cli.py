# SPDX-License-Identifier: Apache-2.0
"""A tier failing and recovering, driven through the CLI.

Scripted tiers make this runnable with no network and no model, which is the
whole reason they exist: a mode machine that could only be exercised against live
endpoints could not be tested at all in the environment where it matters.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from typer.testing import CliRunner, Result

from deadreckoning.cli import app

if TYPE_CHECKING:
    from tests.conftest import EgressLog

runner = CliRunner()

CONFIG = """
profile = "demo"

[node]
node_id = "truck-7"
data_dir = "./data"

[[tiers]]
name = "frontier"
rank = 0
kind = "remote"
client = "scripted"
model = "scripted-frontier"
canary = false

[[tiers]]
name = "local-q4"
rank = 2
kind = "local"
client = "scripted"
model = "scripted-local"
canary = false

[task_classes.triage]
min_rank = 2
review_above_rank = 0

[health]
up_dwell_s = 0

[chaos]
enabled = true
"""


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "dr.toml").write_text(CONFIG)
    return tmp_path


def run(workspace: Path, *args: str) -> Result:
    result = runner.invoke(app, [*args, "--config", str(workspace / "dr.toml")])
    assert result.exit_code == 0, result.stdout
    return result


def data(workspace: Path, *args: str) -> Any:
    return json.loads(run(workspace, *args, "--json").stdout)


def test_a_tier_fails_and_recovers_through_the_cli(
    workspace: Path, egress_guard: EgressLog
) -> None:
    run(workspace, "init")

    assert data(workspace, "status")["mode"] == "CONNECTED"

    run(workspace, "chaos", "frontier", "--fail", "CONNECT_TIMEOUT")
    status = data(workspace, "status")
    frontier = next(d for d in status["dependencies"] if d["name"] == "frontier")
    assert status["mode"] == "ISLANDED"
    assert frontier["state"] == "UNREACHABLE"
    assert frontier["failure_class"] == "CONNECT_TIMEOUT"
    assert frontier["remediation"], "an operator needs to be told what this means"
    assert frontier["injected"] is True

    run(workspace, "chaos", "frontier", "--restore")
    assert data(workspace, "status")["mode"] == "CONNECTED"

    assert data(workspace, "verify")["ok"] is True
    assert egress_guard.non_loopback == []


def test_the_log_says_a_fault_was_injected(workspace: Path) -> None:
    """A recording must never be able to pass a staged outage off as a real one."""
    run(workspace, "init")
    run(workspace, "chaos", "frontier", "--fail", "DNS_FAILURE")
    run(workspace, "chaos", "frontier", "--restore")

    records = data(workspace, "log")
    health = [r for r in records if r["kind"] == "HEALTH_CHANGE"]
    assert [r["body"]["source"] for r in health] == ["FAULT_INJECTION", "FAULT_RESTORE"]
    assert health[0]["body"]["failure_class"] == "DNS_FAILURE"
    assert health[0]["body"]["remediation"]

    modes = [r["body"] for r in records if r["kind"] == "MODE_CHANGE"]
    assert [(m["from_mode"], m["to_mode"]) for m in modes] == [
        ("ISLANDED", "CONNECTED"),
        ("CONNECTED", "ISLANDED"),
        ("ISLANDED", "CONNECTED"),
    ], "the node assesses itself at init, then falls and recovers"


def test_the_genesis_record_is_still_first(workspace: Path) -> None:
    """Seeding a scripted tier's health must not write ahead of the node's own start."""
    run(workspace, "init")
    records = data(workspace, "log")
    assert records[0]["kind"] == "NODE_INIT"
    assert records[0]["prev_hash"] == "0" * 64


def test_restore_all_lifts_everything_in_one_derivation(workspace: Path) -> None:
    run(workspace, "init")
    run(workspace, "chaos", "frontier", "--fail", "CONNECT_TIMEOUT")
    run(workspace, "chaos", "local-q4", "--fail", "SERVER_ERROR")
    assert data(workspace, "status")["mode"] == "ISLANDED"

    run(workspace, "chaos", "--restore-all")
    status = data(workspace, "status")
    assert status["mode"] == "CONNECTED"
    assert all(d["state"] == "HEALTHY" for d in status["dependencies"])


def test_chaos_refuses_an_undeclared_dependency(workspace: Path) -> None:
    run(workspace, "init")
    result = runner.invoke(
        app, ["chaos", "nowhere", "--fail", "DNS_FAILURE", "--config", str(workspace / "dr.toml")]
    )
    assert result.exit_code == 2


def test_chaos_refuses_an_unknown_failure_class(workspace: Path) -> None:
    run(workspace, "init")
    result = runner.invoke(
        app, ["chaos", "frontier", "--fail", "MADE_UP", "--config", str(workspace / "dr.toml")]
    )
    assert result.exit_code == 2


def test_health_survives_between_commands(workspace: Path) -> None:
    """Each command is its own process. What one learns, the next has to know."""
    run(workspace, "init")
    run(workspace, "chaos", "frontier", "--fail", "CONNECT_TIMEOUT")
    again = data(workspace, "status")
    frontier = next(d for d in again["dependencies"] if d["name"] == "frontier")
    assert frontier["state"] == "UNREACHABLE"
    assert frontier["needs_reconciliation"] is True
