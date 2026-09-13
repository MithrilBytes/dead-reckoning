# SPDX-License-Identifier: Apache-2.0
"""The hub: it stores everything and decides nothing.

Deliberately the dullest component here. It never drains an outbox, never resolves
a conflict, never reasons about a record it is holding. Compromising it gets an
attacker a copy of the logs, which is a privacy problem, and no ability to make a
truck do anything, which would be a safety one.

Standard library only. A hub that needed a web framework to forward JSON would be
a heavier thing to audit than the runtime it serves.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from deadreckoning.records import RecordStore
from deadreckoning.runtime import open_database
from deadreckoning.sync.protocol import SyncStore, handle


class Hub:
    """Holds every node's records, in one store, and serves the protocol."""

    def __init__(self, data_dir: Path, token: str | None = None) -> None:
        self.database = open_database(data_dir, threadsafe=True)
        self.store = RecordStore(self.database)
        self.token = token
        # One request per thread, one writer at a time. SQLite in WAL mode reads
        # concurrently without help; writes have to be serialised, and a lock here
        # is cheaper to reason about than a connection pool for a service whose
        # whole job is appending rows it was handed.
        self.lock = threading.Lock()

    def sync_store(self) -> SyncStore:
        return SyncStore(list(self.store.iter_records()), self.store.append)

    def authorised(self, header: str | None) -> bool:
        if self.token is None:
            return True
        return header == f"Bearer {self.token}"

    def close(self) -> None:
        self.database.close()


def make_handler(hub: Hub) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            """Quiet by default: a hub that prints every request buries the demo."""

        def _respond(self, status: int, body: dict[str, Any]) -> None:
            payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _guard(self) -> bool:
            if hub.authorised(self.headers.get("Authorization")):
                return True
            self._respond(401, {"error": "unauthorised"})
            return False

        def do_GET(self) -> None:
            if not self._guard():
                return
            parsed = urlparse(self.path)
            query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            with hub.lock:
                response = handle(hub.sync_store(), "GET", parsed.path, query, None)
            self._respond(response.status, response.body)

        def do_POST(self) -> None:
            if not self._guard():
                return
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"[]"
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                self._respond(400, {"error": "body is not valid JSON"})
                return
            parsed = urlparse(self.path)
            with hub.lock:
                response = handle(hub.sync_store(), "POST", parsed.path, {}, body)
            self._respond(response.status, response.body)

    return Handler


def serve(
    data_dir: Path, port: int = 8700, token: str | None = None
) -> tuple[ThreadingHTTPServer, Hub]:
    """Returns the server and the hub it serves, rather than stashing one on the other.

    A typed pair, because a caller that has to reach through an untyped attribute
    to check what the hub is holding cannot be type checked, and checking what the
    hub is holding is exactly what the tests do.
    """
    hub = Hub(data_dir, token)
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(hub)), hub


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="A passive sync hub for Dead Reckoning nodes.")
    parser.add_argument("--data-dir", type=Path, default=Path("./data/hub"))
    parser.add_argument("--port", type=int, default=8700)
    parser.add_argument("--token", default=None)
    args = parser.parse_args()
    server, _hub = serve(args.data_dir, args.port, args.token)
    print(f"hub listening on 127.0.0.1:{args.port}, storing to {args.data_dir}")
    server.serve_forever()


if __name__ == "__main__":
    main()
