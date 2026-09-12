# SPDX-License-Identifier: Apache-2.0
"""Canonical JSON and content hashing.

Two records that say the same thing must hash the same on every node, on every
platform, forever. That is the whole job of this module, and it is why the rules
below are stricter than ordinary JSON serialisation.

The form is: object keys sorted by code point, no insignificant whitespace, UTF-8
output, and floats rendered by Python's repr, which is the shortest string that
round trips to the same double. This is deliberately close to but not the same as
RFC 8785 (JSON Canonicalization Scheme), which serialises numbers by the ECMAScript
Number::toString algorithm. The two agree for every value this runtime actually
stores; they diverge on extremes such as 1e16, where repr gives '1e+16' and
ECMAScript gives '10000000000000000'. Pinning repr keeps a pure standard library
implementation, at the cost of not being able to claim RFC 8785 compliance.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

GENESIS_PREV_HASH = "0" * 64
"""prev_hash of the first record on a node. Sixty four zeros, never a real digest."""

REDACTED = "[REDACTED]"

DEFAULT_REDACT_KEYS: frozenset[str] = frozenset(
    {"api_key", "authorization", "token", "secret", "password"}
)


class CanonicalisationError(ValueError):
    """A value cannot be canonicalised deterministically."""


def _check(value: Any, path: str) -> None:
    """Reject anything whose serialisation would not be stable or valid JSON."""
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalisationError(
                f"{path or '<root>'}: {value!r} is not finite; JSON has no representation for it"
            )
        return
    if isinstance(value, dict):
        for key, item in cast(Mapping[Any, Any], value).items():
            if not isinstance(key, str):
                raise CanonicalisationError(
                    f"{path or '<root>'}: object key {key!r} is not a string"
                )
            _check(item, f"{path}.{key}" if path else key)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(cast(Sequence[Any], value)):
            _check(item, f"{path}[{index}]")
        return
    raise CanonicalisationError(
        f"{path or '<root>'}: {type(value).__name__} has no canonical JSON form"
    )


def canonical_json(value: Any) -> bytes:
    """Serialise to the canonical form. Raises rather than guessing.

    There is no `default` hook on purpose. A type this module does not know how to
    render is a bug in the caller, and silently coercing it to a string would make
    two different values hash alike.
    """
    _check(value, "")
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def redact(value: Any, keys: frozenset[str] | set[str] = DEFAULT_REDACT_KEYS) -> Any:
    """Replace the value of any matching key with a marker, at any depth.

    Matching is case insensitive and exact. Exact rather than substring so that a
    key like `api_key_env`, whose value is the *name* of an environment variable
    and not a secret, survives intact.

    Records travel between nodes and to the hub, so this runs before hashing: the
    digest covers what is stored, not what was collected.
    """
    lowered = {key.lower() for key in keys}

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            return {
                key: (REDACTED if isinstance(key, str) and key.lower() in lowered else walk(item))
                for key, item in cast(Mapping[Any, Any], node).items()
            }
        if isinstance(node, list):
            return [walk(item) for item in cast(Sequence[Any], node)]
        return node

    return walk(value)


def sha256_hex(payload: bytes) -> str:
    """Lowercase hex digest, the form every hash field in this system uses."""
    return hashlib.sha256(payload).hexdigest()


def content_hash(value: Any) -> str:
    """sha256 of the canonical form of a value."""
    return sha256_hex(canonical_json(value))
