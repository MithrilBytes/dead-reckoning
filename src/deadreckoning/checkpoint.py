# SPDX-License-Identifier: Apache-2.0
"""Task state, so a task interrupted by a dead battery can be picked up again.

The hard part is not saving the state, it is resuming without repeating anything
that already happened outside this process. A read-only call can simply be made
again. A call that dispatched a crew cannot, and the difference has to be
recoverable from the log rather than guessed at, which is why a checkpoint records
which calls were issued and the records say which of them came back.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from deadreckoning.runtime import Database

SCHEMA = """
CREATE TABLE IF NOT EXISTS checkpoints (
    task_id         TEXT PRIMARY KEY,
    task_class      TEXT NOT NULL,
    step            INTEGER NOT NULL,
    state           TEXT NOT NULL,
    messages        TEXT NOT NULL,
    pending_calls   TEXT NOT NULL,
    tokens_consumed INTEGER NOT NULL DEFAULT 0,
    started_ms      INTEGER NOT NULL,
    updated_ms      INTEGER NOT NULL,
    outcome_id      TEXT
);
"""


@dataclass(slots=True)
class Checkpoint:
    task_id: str
    task_class: str
    step: int = 0
    state: str = "RUNNING"
    messages: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    pending_calls: list[str] = field(default_factory=list[str])
    tokens_consumed: int = 0
    started_ms: int = 0
    updated_ms: int = 0
    outcome_id: str | None = None

    @property
    def done(self) -> bool:
        return self.state != "RUNNING"


class Checkpointer:
    """Persists task progress. One row per task, rewritten as it advances.

    Records are append only; a checkpoint is not. It is a cursor into a task, and
    keeping every intermediate position would bury the log it sits beside without
    telling anyone anything the records do not already say.
    """

    def __init__(self, database: Database) -> None:
        self._db = database
        self._db.connection.executescript(SCHEMA)

    def save(self, checkpoint: Checkpoint) -> None:
        self._db.connection.execute(
            "INSERT INTO checkpoints (task_id, task_class, step, state, messages, pending_calls,"
            " tokens_consumed, started_ms, updated_ms, outcome_id)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT (task_id) DO UPDATE SET step = excluded.step, state = excluded.state,"
            " messages = excluded.messages, pending_calls = excluded.pending_calls,"
            " tokens_consumed = excluded.tokens_consumed, updated_ms = excluded.updated_ms,"
            " outcome_id = excluded.outcome_id",
            (
                checkpoint.task_id,
                checkpoint.task_class,
                checkpoint.step,
                checkpoint.state,
                json.dumps(checkpoint.messages),
                json.dumps(checkpoint.pending_calls),
                checkpoint.tokens_consumed,
                checkpoint.started_ms,
                checkpoint.updated_ms,
                checkpoint.outcome_id,
            ),
        )

    def load(self, task_id: str) -> Checkpoint | None:
        row = self._db.connection.execute(
            "SELECT * FROM checkpoints WHERE task_id = ?", (task_id,)
        ).fetchone()
        if row is None:
            return None
        return Checkpoint(
            task_id=row["task_id"],
            task_class=row["task_class"],
            step=int(row["step"]),
            state=row["state"],
            messages=json.loads(row["messages"]),
            pending_calls=json.loads(row["pending_calls"]),
            tokens_consumed=int(row["tokens_consumed"]),
            started_ms=int(row["started_ms"]),
            updated_ms=int(row["updated_ms"]),
            outcome_id=row["outcome_id"],
        )

    def unfinished(self) -> list[Checkpoint]:
        """Tasks that were in flight when the process stopped."""
        rows = self._db.connection.execute(
            "SELECT task_id FROM checkpoints WHERE state = 'RUNNING' ORDER BY started_ms"
        ).fetchall()
        return [c for row in rows if (c := self.load(row["task_id"])) is not None]
