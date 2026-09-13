# SPDX-License-Identifier: Apache-2.0
# pyright: reportUnknownMemberType=false
# jsonschema ships incomplete type information; the relaxation is scoped to this rule.
"""Contract rules, each of which exists because its absence causes a specific bug."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from deadreckoning.tools.contract import (
    Consequence,
    HydrateSpec,
    OfflinePolicy,
    SideEffect,
    ToolContract,
)

SCHEMA = json.loads(
    (Path(__file__).resolve().parents[2] / "schemas" / "tool_contract.schema.json").read_text()
)


def contract(**kwargs: Any) -> ToolContract:
    base: dict[str, Any] = {
        "name": "t",
        "backend": "api",
        "offline_policy": OfflinePolicy.LOCAL,
        "side_effect": SideEffect.NONE,
        "consequence": Consequence.LOW,
        "staleness_budget_s": 60,
        "local_source": "field_input",
    }
    base.update(kwargs)
    return ToolContract(**base)


def test_a_queue_tool_must_declare_an_expiry() -> None:
    """A deferred intent with no expiry could fire against a world that moved on."""
    with pytest.raises(ValueError, match="expiry_s"):
        contract(offline_policy=OfflinePolicy.QUEUE, side_effect=SideEffect.IDEMPOTENT)


def test_a_local_tool_must_say_how_its_store_gets_filled() -> None:
    with pytest.raises(ValueError, match="hydrate or local_source"):
        contract(local_source=None)


def test_a_non_idempotent_tool_must_declare_an_explicit_key() -> None:
    """Hashing every argument cannot express which arguments make two calls the
    same action, which is exactly the judgement this class needs."""
    with pytest.raises(ValueError, match="explicit idempotency_key"):
        contract(
            offline_policy=OfflinePolicy.QUEUE,
            side_effect=SideEffect.NON_IDEMPOTENT,
            expiry_s=7200,
        )


def test_a_tool_with_no_backend_must_be_local() -> None:
    with pytest.raises(ValueError, match="always local"):
        contract(backend=None, offline_policy=OfflinePolicy.FAIL)


def test_queueing_something_with_no_side_effect_is_refused() -> None:
    with pytest.raises(ValueError, match="pointless"):
        contract(offline_policy=OfflinePolicy.QUEUE, side_effect=SideEffect.NONE, expiry_s=60)


def test_a_hydrate_block_must_be_usable() -> None:
    with pytest.raises(ValueError, match="subject_types with arg_from_subject"):
        HydrateSpec(subject_types=["asset"])


def test_hydrate_by_subject_or_by_static_args() -> None:
    assert HydrateSpec(subject_types=["asset"], arg_from_subject="asset_id")
    assert HydrateSpec(static_args=[{}])


def test_the_idempotency_template_names_what_makes_calls_the_same() -> None:
    c = contract(
        offline_policy=OfflinePolicy.QUEUE,
        side_effect=SideEffect.NON_IDEMPOTENT,
        idempotency_key="dispatch:{ticket_id}",
        expiry_s=7200,
    )
    assert c.key_for({"ticket_id": "T-101", "crew_id": "C-1"}) == "dispatch:T-101"
    assert c.key_for({"ticket_id": "T-101", "crew_id": "C-2"}) == "dispatch:T-101", (
        "same ticket is the same dispatch whichever crew is named"
    )


def test_a_template_missing_its_argument_says_which_one() -> None:
    c = contract(
        offline_policy=OfflinePolicy.QUEUE,
        side_effect=SideEffect.NON_IDEMPOTENT,
        idempotency_key="dispatch:{ticket_id}",
        expiry_s=7200,
    )
    with pytest.raises(ValueError, match="ticket_id"):
        c.key_for({"crew_id": "C-1"})


def test_the_default_key_covers_every_argument() -> None:
    c = contract(side_effect=SideEffect.IDEMPOTENT)
    assert c.key_for({"a": 1}) != c.key_for({"a": 2})


def test_the_model_and_the_shipped_schema_agree() -> None:
    """Contracts are validated against the schema at startup, so a contract the
    model accepts and the schema rejects would fail in the field, not here."""
    validator = jsonschema.Draft202012Validator(SCHEMA)
    for built in (
        contract(),
        contract(offline_policy=OfflinePolicy.FAIL, staleness_budget_s=None, local_source=None),
        contract(
            offline_policy=OfflinePolicy.QUEUE,
            side_effect=SideEffect.NON_IDEMPOTENT,
            idempotency_key="k:{x}",
            expiry_s=60,
            staleness_budget_s=None,
            local_source=None,
        ),
    ):
        payload = built.model_dump(mode="json", exclude_none=True)
        validator.validate(payload)
