# SPDX-License-Identifier: Apache-2.0
"""Talking to a hub or a peer, and knowing what went wrong when it fails.

Every call goes through the same classification as any other dependency, so a hub
that stops answering opens its breaker and moves the mode exactly like a tool
backend would. A sync layer with its own private idea of failure would be a second
opinion about the network, and the mode controller can only act on one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import httpx

from deadreckoning.health import FailureClass
from deadreckoning.records import Record
from deadreckoning.sync.merge import Vector, merge, sync_vector, unsynced
from deadreckoning.transport import classify_exception


@dataclass(slots=True)
class SyncOutcome:
    """One exchange, in the shape the SYNC record body takes."""

    peer: str
    direction: str
    pushed: int = 0
    pulled: int = 0
    conflicts_detected: int = 0
    error: str | None = None
    rejected_node: str | None = None
    failure_class: FailureClass = FailureClass.OK
    accepted: list[Record] = field(default_factory=list[Record])

    def as_body(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "peer": self.peer,
            "direction": self.direction,
            "pushed": self.pushed,
            "pulled": self.pulled,
            "conflicts_detected": self.conflicts_detected,
        }
        if self.error:
            body["error"] = self.error
            body["rejected_node"] = self.rejected_node
        return body


class SyncClient:
    """A thin client over the four endpoints."""

    def __init__(
        self,
        base_url: str,
        peer_name: str,
        token: str | None = None,
        timeout_s: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.peer_name = peer_name
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._timeout = timeout_s

    def _client(self) -> httpx.Client:
        return httpx.Client(timeout=self._timeout, headers=self._headers)

    def head(self) -> Vector:
        with self._client() as client:
            response = client.get(f"{self.base_url}/sync/head")
            response.raise_for_status()
            heads: dict[str, Any] = response.json()["heads"]
        return {node_id: entry["hlc"] for node_id, entry in heads.items()}

    def push(self, records: list[Record], peer_vector: Vector) -> SyncOutcome:
        """Send what the peer has not seen, oldest first.

        Pushed before pulling during reconciliation, so a peer gets the earliest
        chance to detect a conflict before draining its own actions.
        """
        outgoing: list[Record] = []
        for node_id in sorted({r.node_id for r in records}):
            outgoing.extend(unsynced(records, node_id, peer_vector))
        if not outgoing:
            return SyncOutcome(self.peer_name, "PUSH")
        payload = [r.stored_payload() for r in outgoing]
        with self._client() as client:
            response = client.post(f"{self.base_url}/sync/records", json=payload)
        if response.status_code == 409:
            body: dict[str, Any] = response.json()
            return SyncOutcome(
                self.peer_name,
                "PUSH",
                error="CHAIN_BREAK",
                rejected_node=str(body.get("node_id")),
                failure_class=FailureClass.PROTOCOL_ERROR,
            )
        response.raise_for_status()
        return SyncOutcome(self.peer_name, "PUSH", pushed=len(outgoing))

    def pull(self, held: list[Record], known_nodes: list[str]) -> SyncOutcome:
        """Fetch what this node has not seen, and merge it.

        The batch is verified against what is already held before any of it is
        kept, so a peer cannot talk this node into a chain it cannot follow.
        """
        my_vector = sync_vector(held)
        incoming: list[Record] = []
        with self._client() as client:
            for node_id in known_nodes:
                after = my_vector.get(node_id)
                params: dict[str, str] = {"node": node_id}
                if after:
                    params["after"] = json.dumps(after, separators=(",", ":"))
                response = client.get(f"{self.base_url}/sync/records", params=params)
                response.raise_for_status()
                incoming.extend(Record.model_validate(item) for item in response.json()["records"])
        if not incoming:
            return SyncOutcome(self.peer_name, "PULL")
        try:
            result = merge(held, incoming)
        except Exception as exc:  # noqa: BLE001
            return SyncOutcome(
                self.peer_name,
                "PULL",
                error="CHAIN_BREAK",
                rejected_node=getattr(exc, "node_id", None),
                failure_class=FailureClass.PROTOCOL_ERROR,
            )
        return SyncOutcome(self.peer_name, "PULL", pulled=result.count, accepted=result.accepted)


def classify(exc: BaseException, time_trust_is_trusted: bool) -> FailureClass:
    """Sync failures are classified like every other call, not specially."""
    if isinstance(exc, httpx.HTTPStatusError):
        from deadreckoning.health import classify_status

        return classify_status(exc.response.status_code)
    return classify_exception(exc, time_trust_is_trusted)
