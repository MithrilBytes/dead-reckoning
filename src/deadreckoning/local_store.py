# SPDX-License-Identifier: Apache-2.0
"""Cached tool data, and how old it is.

A LOCAL tool answers from here when its backend is gone, so every row has to
carry enough to say where it came from and when, and the age has to be honest.
The temptation is to treat cached data as simply "the answer"; the whole design
downstream depends on it being "the answer as of a stated moment", because that
is what lets a model lower its confidence and the runtime refuse when the data is
too old to act on.

Rows are keyed by tool and canonical arguments, which is the same lookup a local
substitute performs when answering a call. Keying by content hash alone, as an
earlier draft did, produces a store nothing can read back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from deadreckoning.canonical import canonical_json, content_hash
from deadreckoning.runtime import Database

SCHEMA = """
CREATE TABLE IF NOT EXISTS local_store (
    tool            TEXT NOT NULL,
    args_canonical  TEXT NOT NULL,
    subject         TEXT,
    source          TEXT,
    content         TEXT NOT NULL,
    content_hash    TEXT NOT NULL,
    captured_ms     INTEGER NOT NULL,
    PRIMARY KEY (tool, args_canonical)
);
CREATE INDEX IF NOT EXISTS local_store_by_subject ON local_store (subject)
    WHERE subject IS NOT NULL;
"""


@dataclass(frozen=True, slots=True)
class Row:
    tool: str
    args: dict[str, Any]
    content: Any
    content_hash: str
    captured_ms: int
    subject: str | None = None
    source: str | None = None

    def age_s(self, now_ms: int) -> int:
        """Never negative. A row that claims to be from the future is not fresher
        than one from now; it means a clock moved, and reporting a negative age
        would let a stale row pass a budget check."""
        return max(0, (now_ms - self.captured_ms) // 1000)


class LocalStore:
    """Reads and writes cached tool data. No expiry policy of its own.

    Whether a row is too old is a question about the tool's contract, not about
    the store, so the store reports age and the enforcer decides.
    """

    def __init__(self, database: Database) -> None:
        self._db = database
        self._db.connection.executescript(SCHEMA)

    def put(
        self,
        tool: str,
        args: dict[str, Any],
        content: Any,
        captured_ms: int,
        subject: str | None = None,
        source: str | None = None,
    ) -> Row:
        digest = content_hash(content)
        self._db.connection.execute(
            "INSERT INTO local_store"
            " (tool, args_canonical, subject, source, content, content_hash, captured_ms)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT (tool, args_canonical) DO UPDATE SET"
            " subject = excluded.subject, source = excluded.source, content = excluded.content,"
            " content_hash = excluded.content_hash, captured_ms = excluded.captured_ms",
            (
                tool,
                canonical_json(args).decode("utf-8"),
                subject,
                source,
                canonical_json(content).decode("utf-8"),
                digest,
                captured_ms,
            ),
        )
        return Row(tool, args, content, digest, captured_ms, subject, source)

    def get(self, tool: str, args: dict[str, Any]) -> Row | None:
        row = self._db.connection.execute(
            "SELECT * FROM local_store WHERE tool = ? AND args_canonical = ?",
            (tool, canonical_json(args).decode("utf-8")),
        ).fetchone()
        if row is None:
            return None
        import json

        return Row(
            tool=row["tool"],
            args=args,
            content=json.loads(row["content"]),
            content_hash=row["content_hash"],
            captured_ms=int(row["captured_ms"]),
            subject=row["subject"],
            source=row["source"],
        )

    def freshest_age_s(self, tool: str, now_ms: int) -> int | None:
        """Age of the most recently captured row for a tool, or None if it has none.

        The manifest reports per tool rather than per call, so this is what the
        model is shown: the best case, with the honest caveat that an individual
        answer may be older.
        """
        row = self._db.connection.execute(
            "SELECT captured_ms FROM local_store WHERE tool = ? ORDER BY captured_ms DESC LIMIT 1",
            (tool,),
        ).fetchone()
        return None if row is None else max(0, (now_ms - int(row["captured_ms"])) // 1000)

    def count(self, tool: str | None = None) -> int:
        if tool is None:
            sql, params = "SELECT COUNT(*) AS n FROM local_store", ()
        else:
            sql, params = "SELECT COUNT(*) AS n FROM local_store WHERE tool = ?", (tool,)
        return int(self._db.connection.execute(sql, params).fetchone()["n"])
