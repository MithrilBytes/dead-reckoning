# SPDX-License-Identifier: Apache-2.0
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false
# jsonschema and referencing ship incomplete type information, so strict mode cannot see
# through their public API. The relaxation is scoped to this file and to those three rules.
"""Records this runtime writes must validate against the wire format it ships.

The schemas are not documentation. Peers and the hub accept records by validating
them, so a record the code can write but the schema rejects is a record that will
be refused by every node except the one that made it. Testing the two against each
other is the only thing that keeps them honest as both change.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

from deadreckoning.node import Node
from deadreckoning.records import RecordKind

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schemas"


@pytest.fixture(scope="module")
def record_validator() -> jsonschema.Draft202012Validator:
    """Resolve cross-file refs from the shipped schemas rather than over the network.

    The schemas reference each other by filename, and a validator left to fetch
    those would reach out to the internet during the test run, which the egress
    guard rightly forbids.
    """
    schema: dict[str, Any] = json.loads((SCHEMA_DIR / "decision_record.schema.json").read_text())
    resources = [
        (path.name, Resource.from_contents(json.loads(path.read_text()), DRAFT202012))
        for path in sorted(SCHEMA_DIR.glob("*.json"))
    ]
    registry = Registry().with_resources(resources)
    return jsonschema.Draft202012Validator(schema, registry=registry)


def test_every_shipped_schema_is_itself_valid() -> None:
    for path in sorted(SCHEMA_DIR.glob("*.json")):
        jsonschema.Draft202012Validator.check_schema(json.loads(path.read_text()))


def test_the_genesis_record_validates(
    node: Node, record_validator: jsonschema.Draft202012Validator
) -> None:
    record = node.emit(
        RecordKind.NODE_INIT, body={"schema_version": 1, "profile": "demo", "config_hash": None}
    )
    record_validator.validate(record.stored_payload())


def test_a_power_change_validates(
    node: Node, record_validator: jsonschema.Draft202012Validator
) -> None:
    record = node.emit(
        RecordKind.RESOURCE_CHANGE,
        body={"from_power": "MAINS", "to_power": "BATTERY_LOW", "source": "FAULT_INJECTION"},
    )
    record_validator.validate(record.stored_payload())


def test_the_record_kinds_in_code_and_schema_agree() -> None:
    schema: dict[str, Any] = json.loads((SCHEMA_DIR / "decision_record.schema.json").read_text())
    declared = set(schema["properties"]["kind"]["enum"])
    assert {str(kind) for kind in RecordKind} == declared


def test_a_record_the_schema_forbids_is_rejected(
    node: Node, record_validator: jsonschema.Draft202012Validator
) -> None:
    """The validator is worth nothing unless it refuses something."""
    record = node.emit(RecordKind.NODE_INIT, body={"schema_version": 1, "profile": "demo"})
    payload = record.stored_payload()
    payload["body"]["profile"] = "not-a-profile"
    with pytest.raises(jsonschema.ValidationError):
        record_validator.validate(payload)
