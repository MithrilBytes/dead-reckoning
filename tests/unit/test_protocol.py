# SPDX-License-Identifier: Apache-2.0
"""Reading a turn out of whatever the model actually emitted.

Forgiving about packaging, unforgiving about content. A missing field is never
guessed at: a decision that silently acquired a subject the model never named is
worse than no decision.
"""

from __future__ import annotations

import pytest

from deadreckoning.llm.protocol import (
    DecisionType,
    ProtocolError,
    TurnType,
    extract_json,
    parse_turn,
)


def test_a_bare_object() -> None:
    turn = parse_turn('{"type":"final","subject":"t:1","outcome":{"key":"p","value":"P2"}}')
    assert turn.turn_type is TurnType.DECISION
    assert turn.decision is not None
    assert turn.decision.outcome_value == "P2"


def test_prose_around_a_fenced_block() -> None:
    """What a quantised local model does when it is feeling chatty."""
    turn = parse_turn(
        "Certainly! Here is the decision:\n```json\n"
        '{"type":"final","subject":"t:1","outcome":{"key":"p","value":"P1"}}\n```\n'
        "Let me know if you need anything else."
    )
    assert turn.decision is not None
    assert turn.decision.outcome_value == "P1"


def test_an_unfenced_object_buried_in_prose() -> None:
    turn = parse_turn('I think: {"type":"escalate","subject":"t:1","reason":"unsafe","to":"human"}')
    assert turn.decision is not None
    assert turn.decision.type is DecisionType.ESCALATE


def test_two_objects_means_the_second_one() -> None:
    """A model that reconsiders mid-answer meant its last word, not its first."""
    turn = parse_turn(
        '{"type":"final","subject":"t:1","outcome":{"key":"p","value":"P3"}}\n'
        "On reflection, I cannot say without the load:\n"
        '{"type":"abstain","subject":"t:1","outcome":{"key":"p"},"reason":"no load",'
        '"depends_on_unavailable":["get_live_load"]}'
    )
    assert turn.decision is not None
    assert turn.decision.type is DecisionType.ABSTAIN


def test_braces_inside_strings_do_not_confuse_it() -> None:
    turn = parse_turn(
        '{"type":"final","subject":"t:1","outcome":{"key":"p","value":"P2"},'
        '"rationale":"the note said {urgent} and \\"quoted\\" text"}'
    )
    assert turn.decision is not None
    assert "{urgent}" in turn.decision.rationale


def test_tool_calls_get_stable_ids() -> None:
    turn = parse_turn(
        '{"type":"tool_call","calls":[{"name":"a","args":{"x":1}},{"name":"b","args":{}}]}',
        call_id_prefix="t1-3",
    )
    assert [c.call_id for c in turn.tool_calls] == ["t1-3-0", "t1-3-1"]
    assert turn.tool_calls[0].args == {"x": 1}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("nothing here", "no JSON object"),
        ('{"type":"wat"}', 'Your JSON object needs a "type"'),
        ('{"type":"final","outcome":{"key":"p","value":1}}', 'must name the "subject"'),
        ('{"type":"final","subject":"t:1"}', 'needs "outcome"'),
        ('{"type":"final","subject":"t:1","outcome":{"key":"p"}}', 'no "value"'),
        (
            '{"type":"abstain","subject":"t:1","outcome":{"key":"p"},"reason":"x"}',
            "must list the tools",
        ),
        ('{"type":"tool_call","calls":[]}', 'non-empty "calls"'),
        ('{"type":"tool_call","calls":[{"args":{}}]}', 'needs a "name"'),
    ],
)
def test_bad_content_is_refused_with_a_usable_correction(text: str, expected: str) -> None:
    with pytest.raises(ProtocolError) as caught:
        parse_turn(text)
    assert expected in caught.value.correction


def test_an_abstention_naming_nothing_is_not_an_abstention() -> None:
    """If nothing was unavailable, the model is deciding, not abstaining."""
    with pytest.raises(ProtocolError, match="abstain naming no unavailable tool"):
        parse_turn('{"type":"abstain","subject":"t:1","outcome":{"key":"p"},"reason":"lazy"}')


def test_bare_string_evidence_is_kept_as_an_untyped_reference() -> None:
    """Losing a citation entirely is worse than recording an untyped one; the
    runtime fills the provenance either way."""
    turn = parse_turn(
        '{"type":"final","subject":"t:1","outcome":{"key":"p","value":"P2"},'
        '"evidence":["tool_result:abc"]}'
    )
    assert turn.decision is not None
    assert turn.decision.evidence == [{"kind": "record", "ref": "tool_result:abc"}]


def test_extract_json_finds_the_object_in_a_fence() -> None:
    assert extract_json('```json\n{"a":1}\n```') == {"a": 1}


def test_an_escalation_needs_no_subject() -> None:
    """A hazard outside the task's remit may not be about the task's subject."""
    turn = parse_turn('{"type":"escalate","reason":"burning smell","to":"human"}')
    assert turn.decision is not None
    assert turn.decision.subject is None
