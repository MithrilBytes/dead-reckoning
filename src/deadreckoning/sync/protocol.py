# SPDX-License-Identifier: Apache-2.0
"""The four endpoints a node and the hub both speak.

Deliberately small, and the same on both sides. A node syncing with the hub and a
node syncing directly with a peer use identical calls, because the hub is not a
special kind of participant: it is a node that stores everything and decides
nothing. Anything it could do to the records, a peer could do, which is why
compromising it yields visibility rather than authority.

Records are exchanged, never merged remotely. The receiver verifies continuity
against what it already holds and refuses the whole batch if it does not join up,
so a node can never be talked into a chain it cannot verify.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, cast

from deadreckoning.records import Record
from deadreckoning.sync.merge import ChainBreakError, Vector, merge, sync_vector

CHAIN_BREAK_STATUS = 409


@dataclass(frozen=True, slots=True)
class Response:
    status: int
    body: dict[str, Any]

    def encode(self) -> bytes:
        return json.dumps(self.body, separators=(",", ":")).encode("utf-8")


class SyncStore:
    """The slice of a record store the protocol needs.

    Narrow on purpose: an endpoint handler that could reach the whole node would
    be a way to make a node do something by talking to it, and the protocol is
    meant only to move records.
    """

    def __init__(self, records: list[Record], append: Any) -> None:
        self._records = records
        self._append = append

    @property
    def records(self) -> list[Record]:
        return list(self._records)

    def accept(self, incoming: list[Record]) -> list[Record]:
        result = merge(self._records, incoming)
        for record in result.accepted:
            self._append(record)
            self._records.append(record)
        return result.accepted


def head(store: SyncStore) -> Response:
    """What this participant holds, per node. The cheapest question in the protocol."""
    vector: Vector = sync_vector(store.records)
    return Response(
        200,
        {
            "heads": {
                node_id: {
                    "hlc": hlc,
                    "hash": next(
                        (
                            r.hash
                            for r in reversed(store.records)
                            if r.node_id == node_id and r.hlc == hlc
                        ),
                        None,
                    ),
                }
                for node_id, hlc in vector.items()
            }
        },
    )


def records_after(
    store: SyncStore, node_id: str, after: dict[str, Any] | None, limit: int = 500
) -> Response:
    floor = (int(after["physical_ms"]), int(after["logical"])) if after else (-1, -1)
    selected = [
        record
        for record in sorted(
            (r for r in store.records if r.node_id == node_id),
            key=lambda r: (int(r.hlc["physical_ms"]), int(r.hlc["logical"])),
        )
        if (int(record.hlc["physical_ms"]), int(record.hlc["logical"])) > floor
    ][:limit]
    return Response(200, {"records": [r.stored_payload() for r in selected]})


def post_records(store: SyncStore, payload: list[dict[str, Any]]) -> Response:
    """Accept a batch, or refuse it whole.

    A 409 rather than a partial accept: half a batch leaves a hole in a chain that
    every later verification reports and nobody can explain.
    """
    incoming = [Record.model_validate(item) for item in payload]
    try:
        accepted = store.accept(incoming)
    except ChainBreakError as exc:
        return Response(
            CHAIN_BREAK_STATUS,
            {
                "error": "CHAIN_BREAK",
                "node_id": exc.node_id,
                "record_id": exc.record_id,
                "expected": exc.expected,
                "found": exc.found,
            },
        )
    return Response(200, {"accepted": len(accepted), "ids": [r.id for r in accepted]})


def handle(store: SyncStore, method: str, path: str, query: dict[str, str], body: Any) -> Response:
    """Route one request. Shared by the hub's server and the in-process test harness."""
    if method == "GET" and path == "/sync/head":
        return head(store)
    if method == "GET" and path == "/sync/records":
        node_id = query.get("node")
        if not node_id:
            return Response(400, {"error": "node is required"})
        after = json.loads(query["after"]) if query.get("after") else None
        return records_after(store, node_id, after, int(query.get("limit", 500)))
    if method == "POST" and path == "/sync/records":
        if not isinstance(body, list):
            return Response(400, {"error": "expected a list of records"})
        return post_records(store, cast("list[dict[str, Any]]", body))
    return Response(404, {"error": f"no route for {method} {path}"})
