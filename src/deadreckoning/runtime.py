# SPDX-License-Identifier: Apache-2.0
"""Process wiring and ownership of the SQLite connection.

One database per node. Write ahead logging, because a node is read by the CLI
while the agent loop is writing, and a truck loses power without asking.

Tables arrive with the milestone that needs them. Creating a table before there is
code that can fill it would mean guessing at its columns.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import TracebackType

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Append only. There is no UPDATE or DELETE path anywhere in the runtime, and
-- the triggers below make that structural rather than a matter of discipline.
CREATE TABLE IF NOT EXISTS records (
    id           TEXT PRIMARY KEY,
    node_id      TEXT NOT NULL,
    physical_ms  INTEGER NOT NULL,
    logical      INTEGER NOT NULL,
    kind         TEXT NOT NULL,
    subject      TEXT,
    task_id      TEXT,
    prev_hash    TEXT NOT NULL,
    hash         TEXT NOT NULL,
    payload      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS records_by_node  ON records (node_id, physical_ms, logical);
CREATE INDEX IF NOT EXISTS records_by_subj  ON records (subject) WHERE subject IS NOT NULL;
CREATE INDEX IF NOT EXISTS records_by_kind  ON records (kind);
CREATE INDEX IF NOT EXISTS records_by_task  ON records (task_id) WHERE task_id IS NOT NULL;

CREATE TRIGGER IF NOT EXISTS records_are_immutable
BEFORE UPDATE ON records
BEGIN
    SELECT RAISE(ABORT, 'records are append only');
END;

CREATE TRIGGER IF NOT EXISTS records_are_permanent
BEFORE DELETE ON records
BEGIN
    SELECT RAISE(ABORT, 'records are append only');
END;

-- Health survives between commands. `dr chaos` and `dr status` are separate
-- processes, so an assessment that lived only in memory would be forgotten
-- between the fault being armed and anybody asking about it.
CREATE TABLE IF NOT EXISTS health (
    dependency            TEXT PRIMARY KEY,
    state                 TEXT NOT NULL,
    breaker               TEXT NOT NULL,
    last_failure_class    TEXT,
    last_latency_ms       INTEGER,
    consecutive_failures  INTEGER NOT NULL DEFAULT 0,
    consecutive_successes INTEGER NOT NULL DEFAULT 0,
    backoff_s             REAL NOT NULL DEFAULT 0,
    needs_reconciliation  INTEGER NOT NULL DEFAULT 0
);

-- Armed faults, for the same reason. An injected fault that vanished when the
-- command exited could never be observed by the next one.
CREATE TABLE IF NOT EXISTS chaos (
    dependency     TEXT PRIMARY KEY,
    failure_class  TEXT,
    latency_ms     INTEGER,
    drop_pct       REAL,
    until          REAL
);

CREATE TABLE IF NOT EXISTS mode_state (
    id            INTEGER PRIMARY KEY CHECK (id = 1),
    mode          TEXT NOT NULL,
    since_ms      INTEGER NOT NULL,
    pending       TEXT,
    pending_since REAL
);

-- The clock survives restarts here. Losing it would let the node reissue a
-- stamp it has already used, which breaks ordering against every peer.
CREATE TABLE IF NOT EXISTS clock (
    node_id     TEXT PRIMARY KEY,
    physical_ms INTEGER NOT NULL,
    logical     INTEGER NOT NULL
);
"""


class Database:
    """Owns one connection and its lifetime."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.connection = sqlite3.connect(path, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = FULL")
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript(_SCHEMA)
        self.connection.execute(
            "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def open_database(data_dir: Path) -> Database:
    """Open, creating the directory and schema if this is a fresh node."""
    data_dir.mkdir(parents=True, exist_ok=True)
    return Database(data_dir / "node.db")


def load_hlc(database: Database, node_id: str) -> tuple[int, int] | None:
    """The last stamp this node issued, or None if it has never issued one."""
    row = database.connection.execute(
        "SELECT physical_ms, logical FROM clock WHERE node_id = ?", (node_id,)
    ).fetchone()
    return (int(row["physical_ms"]), int(row["logical"])) if row else None


def save_hlc(database: Database, node_id: str, physical_ms: int, logical: int) -> None:
    database.connection.execute(
        "INSERT INTO clock (node_id, physical_ms, logical) VALUES (?, ?, ?)"
        " ON CONFLICT (node_id) DO UPDATE SET physical_ms = excluded.physical_ms,"
        " logical = excluded.logical",
        (node_id, physical_ms, logical),
    )


def load_health(database: Database) -> dict[str, dict[str, object]]:
    rows = database.connection.execute("SELECT * FROM health").fetchall()
    return {row["dependency"]: dict(row) for row in rows}


def save_health(database: Database, dependency: str, fields: dict[str, object]) -> None:
    database.connection.execute(
        "INSERT INTO health (dependency, state, breaker, last_failure_class, last_latency_ms,"
        " consecutive_failures, consecutive_successes, backoff_s, needs_reconciliation)"
        " VALUES (:dependency, :state, :breaker, :last_failure_class, :last_latency_ms,"
        " :consecutive_failures, :consecutive_successes, :backoff_s, :needs_reconciliation)"
        " ON CONFLICT (dependency) DO UPDATE SET"
        " state = excluded.state, breaker = excluded.breaker,"
        " last_failure_class = excluded.last_failure_class,"
        " last_latency_ms = excluded.last_latency_ms,"
        " consecutive_failures = excluded.consecutive_failures,"
        " consecutive_successes = excluded.consecutive_successes,"
        " backoff_s = excluded.backoff_s,"
        " needs_reconciliation = excluded.needs_reconciliation",
        {"dependency": dependency, **fields},
    )


def load_faults(database: Database) -> dict[str, dict[str, object]]:
    rows = database.connection.execute("SELECT * FROM chaos").fetchall()
    return {row["dependency"]: dict(row) for row in rows}


def save_fault(database: Database, dependency: str, fields: dict[str, object]) -> None:
    database.connection.execute(
        "INSERT INTO chaos (dependency, failure_class, latency_ms, drop_pct, until)"
        " VALUES (:dependency, :failure_class, :latency_ms, :drop_pct, :until)"
        " ON CONFLICT (dependency) DO UPDATE SET"
        " failure_class = excluded.failure_class, latency_ms = excluded.latency_ms,"
        " drop_pct = excluded.drop_pct, until = excluded.until",
        {"dependency": dependency, **fields},
    )


def delete_fault(database: Database, dependency: str) -> None:
    database.connection.execute("DELETE FROM chaos WHERE dependency = ?", (dependency,))


def clear_faults(database: Database) -> None:
    database.connection.execute("DELETE FROM chaos")


def load_mode(database: Database) -> dict[str, object] | None:
    row = database.connection.execute("SELECT * FROM mode_state WHERE id = 1").fetchone()
    return dict(row) if row else None


def save_mode(
    database: Database, mode: str, since_ms: int, pending: str | None, pending_since: float | None
) -> None:
    database.connection.execute(
        "INSERT INTO mode_state (id, mode, since_ms, pending, pending_since)"
        " VALUES (1, ?, ?, ?, ?)"
        " ON CONFLICT (id) DO UPDATE SET mode = excluded.mode, since_ms = excluded.since_ms,"
        " pending = excluded.pending, pending_since = excluded.pending_since",
        (mode, since_ms, pending, pending_since),
    )
