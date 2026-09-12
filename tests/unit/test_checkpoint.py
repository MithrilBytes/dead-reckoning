# SPDX-License-Identifier: Apache-2.0
"""Task state across a restart."""

from __future__ import annotations

from deadreckoning.checkpoint import Checkpoint, Checkpointer
from deadreckoning.node import Node


def test_a_task_can_be_picked_up_where_it_stopped(node: Node) -> None:
    checkpointer = Checkpointer(node.database)
    checkpointer.save(
        Checkpoint(
            task_id="t1",
            task_class="triage",
            step=3,
            messages=[{"role": "user", "content": "triage"}],
            pending_calls=["c-1"],
            tokens_consumed=1200,
            started_ms=1,
            updated_ms=2,
        )
    )
    resumed = checkpointer.load("t1")
    assert resumed is not None
    assert resumed.step == 3
    assert resumed.pending_calls == ["c-1"]
    assert resumed.tokens_consumed == 1200


def test_a_checkpoint_is_a_cursor_not_a_history(node: Node) -> None:
    """One row per task, rewritten. The records are the history."""
    checkpointer = Checkpointer(node.database)
    for step in range(1, 5):
        checkpointer.save(Checkpoint("t1", "triage", step=step, started_ms=1, updated_ms=step))
    assert checkpointer.load("t1").step == 4  # pyright: ignore[reportOptionalMemberAccess]
    rows = node.database.connection.execute("SELECT COUNT(*) AS n FROM checkpoints").fetchone()
    assert rows["n"] == 1


def test_unfinished_tasks_are_what_resumption_looks_at(node: Node) -> None:
    checkpointer = Checkpointer(node.database)
    checkpointer.save(Checkpoint("running", "triage", started_ms=1, updated_ms=1))
    checkpointer.save(Checkpoint("done", "triage", state="COMPLETE", started_ms=2, updated_ms=2))
    assert [c.task_id for c in checkpointer.unfinished()] == ["running"]


def test_a_task_that_was_never_started_loads_as_nothing(node: Node) -> None:
    assert Checkpointer(node.database).load("never") is None
