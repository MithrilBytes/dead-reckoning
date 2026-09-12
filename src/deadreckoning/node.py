# SPDX-License-Identifier: Apache-2.0
"""A running node: its identity, clock, store, and the act of emitting a record.

Everything a node does ends in a record, so emitting one is the operation this
module exists to make correct. A record and the clock stamp that produced it are
written in a single transaction. If they could diverge, a crash between them would
either reissue a stamp that is already on a record, or skip one and leave a hole
no peer could explain.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from types import TracebackType
from typing import Any

from deadreckoning.clock import HLC, HybridLogicalClock
from deadreckoning.config import Config
from deadreckoning.records import Mode, Record, RecordKind, RecordStore, TimeTrust, new_record_id
from deadreckoning.runtime import Database, load_hlc, open_database, save_hlc


def _wall_ms() -> int:
    return int(datetime.datetime.now(datetime.UTC).timestamp() * 1000)


class Node:
    """Wires a node's parts together over one database.

    Mode and time trust default honestly rather than optimistically. A node that
    has probed nothing has no healthy remote tier, which is the definition of
    being islanded, and it has not checked its clock against anything, so its time
    is untrusted. Both become real assessments once the health monitor exists.
    """

    def __init__(self, config: Config, database: Database, now_ms: Any = _wall_ms) -> None:
        self.config = config
        self.database = database
        self.node_id = config.node.node_id
        self.store = RecordStore(database, redact_keys=config.redact_keys())
        restored = load_hlc(database, self.node_id)
        last = (
            HLC(physical_ms=restored[0], logical=restored[1], node_id=self.node_id)
            if restored
            else None
        )
        self.clock = HybridLogicalClock(self.node_id, now_ms, last)
        self.mode = Mode.ISLANDED
        self.time_trust = TimeTrust.UNTRUSTED

    @classmethod
    def open(cls, config: Config, base_dir: Path, now_ms: Any = _wall_ms) -> Node:
        return cls(config, open_database(config.data_dir(base_dir)), now_ms)

    def observed_vector(self) -> dict[str, dict[str, int | str]]:
        """The latest stamp this node holds from every node it knows, itself included.

        Two records are concurrent when neither appears in the other's vector, so
        this is what makes a later disagreement detectable rather than invisible.
        """
        vector: dict[str, dict[str, int | str]] = {}
        for node_id in self.store.node_ids():
            head = self.store.head(node_id)
            if head is not None:
                vector[node_id] = head.hlc
        vector[self.node_id] = self.clock.last.as_dict()
        return vector

    def emit(self, kind: RecordKind, body: dict[str, Any] | None = None, **fields: Any) -> Record:
        """Stamp, chain and persist one record, with its clock stamp, atomically."""
        connection = self.database.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            stamp = self.clock.now()
            record = Record(
                id=new_record_id(stamp),
                node_id=self.node_id,
                hlc=stamp.as_dict(),
                wall_time=datetime.datetime.now(datetime.UTC).isoformat(),
                time_trust=self.time_trust,
                kind=kind,
                mode=self.mode,
                observed=self.observed_vector(),
                prev_hash=self.store.prev_hash_for(self.node_id),
                body=body or {},
                **fields,
            )
            stored = self.store.append(record)
            save_hlc(self.database, self.node_id, stamp.physical_ms, stamp.logical)
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        else:
            connection.execute("COMMIT")
            return stored

    def close(self) -> None:
        self.database.close()

    def __enter__(self) -> Node:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
