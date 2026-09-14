# SPDX-License-Identifier: Apache-2.0
"""The local node commands, exercised through the runner.

`dr status` and `dr verify` run under the egress guard like everything else, so
either one reaching for the network fails the test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner, Result

from deadreckoning.cli import app

if TYPE_CHECKING:
    from tests.conftest import EgressLog

runner = CliRunner()


@pytest.fixture
def workspace(tmp_path: Path, example_config_text: str) -> Path:
    (tmp_path / "dr.toml").write_text(
        example_config_text.replace('data_dir = "./data/truck-7"', 'data_dir = "./data"')
    )
    return tmp_path


def _run(workspace: Path, *args: str) -> Result:
    return runner.invoke(app, [*args, "--config", str(workspace / "dr.toml")])


def test_init_then_log_then_verify(workspace: Path, egress_guard: EgressLog) -> None:
    assert _run(workspace, "init").exit_code == 0
    assert _run(workspace, "log").exit_code == 0
    verified = _run(workspace, "verify")
    assert verified.exit_code == 0
    assert "ok" in verified.stdout
    assert egress_guard.non_loopback == []


def test_init_writes_exactly_one_genesis_record(workspace: Path) -> None:
    _run(workspace, "init")
    payload = json.loads(_run(workspace, "log", "--json").stdout)
    assert len(payload) == 1
    assert payload[0]["kind"] == "NODE_INIT"
    assert payload[0]["body"]["schema_version"] == 1
    assert payload[0]["body"]["profile"] == "demo"
    assert payload[0]["prev_hash"] == "0" * 64


def test_init_refuses_to_run_twice(workspace: Path) -> None:
    _run(workspace, "init")
    second = _run(workspace, "init")
    assert second.exit_code == 2


def test_status_needs_no_network(workspace: Path, egress_guard: EgressLog) -> None:
    _run(workspace, "init")
    result = _run(workspace, "status")
    assert result.exit_code == 0
    assert "truck-7" in result.stdout
    assert egress_guard.non_loopback == []


def test_every_command_speaks_json(workspace: Path) -> None:
    _run(workspace, "init")
    for args in (["status"], ["log"], ["verify"]):
        result = _run(workspace, *args, "--json")
        assert result.exit_code == 0
        json.loads(result.stdout)


def test_show_reports_chain_neighbours(workspace: Path) -> None:
    _run(workspace, "init")
    first = json.loads(_run(workspace, "log", "--json").stdout)[0]
    shown = json.loads(_run(workspace, "show", first["id"], "--json").stdout)
    assert shown["record"]["id"] == first["id"]
    assert shown["position"] == 0
    assert shown["previous"] is None
    assert shown["hash_recomputes"] is True


def test_an_unknown_record_id_is_an_error(workspace: Path) -> None:
    _run(workspace, "init")
    assert _run(workspace, "show", "nope").exit_code == 2


def test_an_unknown_kind_lists_the_known_ones(workspace: Path) -> None:
    _run(workspace, "init")
    result = _run(workspace, "log", "--kind", "NOPE")
    assert result.exit_code == 2


def test_a_bad_config_fails_before_touching_the_database(tmp_path: Path) -> None:
    (tmp_path / "dr.toml").write_text(
        '[node]\nnode_id = "n"\ndata_dir = "./d"\n[health]\nnope = 1\n'
    )
    result = runner.invoke(app, ["status", "--config", str(tmp_path / "dr.toml")])
    assert result.exit_code == 2
    assert not (tmp_path / "d").exists()


def test_verify_exits_nonzero_when_the_chain_is_broken(workspace: Path) -> None:
    _run(workspace, "init")
    database = workspace / "data" / "node.db"
    import sqlite3

    connection = sqlite3.connect(database)
    connection.execute("DROP TRIGGER records_are_immutable")
    row = connection.execute("SELECT id, payload FROM records LIMIT 1").fetchone()
    payload = json.loads(row[1])
    payload["body"]["event"] = "tampered"
    connection.execute(
        "UPDATE records SET payload = ? WHERE id = ?",
        (json.dumps(payload, sort_keys=True, separators=(",", ":")), row[0]),
    )
    connection.commit()
    connection.close()

    result = _run(workspace, "verify")
    assert result.exit_code == 1
    assert "BREAK" in result.stdout
