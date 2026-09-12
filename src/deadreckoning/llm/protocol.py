# SPDX-License-Identifier: Apache-2.0
"""The shape of a model's turn, and getting one out of whatever it actually said.

Two transports produce the same internal turn: native tool calling where the
endpoint supports it, and JSON embedded in text where it does not. The second is
not a fallback for completeness, it is the common case on this project's lowest
tier, where a quantised local model will wrap its JSON in prose, fence it in
backticks, apologise before it, or emit two objects and mean the second.

So the parser is deliberately forgiving about packaging and completely unforgiving
about content. It will dig an object out of a fenced block; it will not guess at a
missing `subject`, because a decision that silently acquired a subject the model
never named is worse than no decision.

When it cannot parse, the model is told precisely what was wrong and asked again,
twice. After that the task escalates rather than looping, because a model that has
failed the same schema three times is not going to pass it on the fourth.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, cast

MAX_CORRECTIONS = 2
"""Attempts after the first. Three tries total, then a human hears about it."""


class TurnType(StrEnum):
    TOOL_CALL = "TOOL_CALL"
    DECISION = "DECISION"
    MALFORMED = "MALFORMED"


class DecisionType(StrEnum):
    FINAL = "final"
    ABSTAIN = "abstain"
    ESCALATE = "escalate"


class ProtocolError(ValueError):
    """The model's output could not be read as a turn. Carries what to tell it."""

    def __init__(self, message: str, correction: str) -> None:
        super().__init__(message)
        self.correction = correction


@dataclass(frozen=True, slots=True)
class ToolCall:
    name: str
    args: dict[str, Any]
    call_id: str


@dataclass(frozen=True, slots=True)
class Decision:
    """A terminal turn: an answer, a refusal to answer, or a hand-off."""

    type: DecisionType
    subject: str | None
    outcome_key: str | None = None
    outcome_value: Any = None
    rationale: str = ""
    reason: str = ""
    evidence: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    depends_on: list[str] = field(default_factory=list[str])
    depends_on_unavailable: list[str] = field(default_factory=list[str])
    partial: dict[str, Any] | None = None
    confidence: float | None = None
    to: str | None = None

    @property
    def abstained(self) -> bool:
        return self.type is DecisionType.ABSTAIN


@dataclass(frozen=True, slots=True)
class ModelTurn:
    turn_type: TurnType
    tool_calls: list[ToolCall] = field(default_factory=list[ToolCall])
    decision: Decision | None = None
    raw: str = ""
    latency_ms: int | None = None
    tokens_prompt: int | None = None
    tokens_completion: int | None = None


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> dict[str, Any]:
    """Find the object in whatever the model wrapped it in.

    Tries the whole string, then fenced blocks, then the last balanced object in
    the text. The last of those matters: a model that reconsiders mid-answer emits
    two objects and means the second one.
    """
    for candidate in _candidates(text):
        try:
            parsed: Any = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return cast("dict[str, Any]", parsed)
    raise ProtocolError(
        "no JSON object found in the response",
        "Your response contained no JSON object. Reply with exactly one JSON object"
        " and nothing else.",
    )


def _candidates(text: str) -> list[str]:
    stripped = text.strip()
    found = [stripped, *(m.strip() for m in _FENCE.findall(text))]
    found.extend(reversed(_balanced_objects(text)))
    return found


def _balanced_objects(text: str) -> list[str]:
    """Every top-level {...} in the text, in order, ignoring braces inside strings."""
    objects: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                objects.append(text[start : index + 1])
                start = -1
    return objects


def parse_turn(text: str, call_id_prefix: str = "c") -> ModelTurn:
    """Read one turn, or raise with the correction to send back."""
    payload = extract_json(text)
    kind = payload.get("type")
    if kind == "tool_call":
        return _parse_tool_call(payload, text, call_id_prefix)
    if kind in set(DecisionType):
        return ModelTurn(TurnType.DECISION, decision=_parse_decision(payload), raw=text)
    raise ProtocolError(
        f"unknown turn type {kind!r}",
        'Your JSON object needs a "type" of "tool_call", "final", "abstain" or "escalate".'
        f" You sent {kind!r}.",
    )


def _parse_tool_call(payload: dict[str, Any], text: str, prefix: str) -> ModelTurn:
    calls: Any = payload.get("calls")
    if not isinstance(calls, list) or not calls:
        raise ProtocolError(
            "tool_call with no calls",
            'A "tool_call" turn needs a non-empty "calls" array, each entry having'
            ' "name" and "args".',
        )
    parsed: list[ToolCall] = []
    for index, raw_call in enumerate(cast("list[Any]", calls)):
        if not isinstance(raw_call, dict) or "name" not in raw_call:
            raise ProtocolError(
                f"call {index} has no name",
                'Every entry in "calls" needs a "name" and an "args" object.',
            )
        call = cast("dict[str, Any]", raw_call)
        args: Any = call.get("args", {})
        parsed.append(
            ToolCall(
                name=str(call["name"]),
                args=_as_dict(args) or {},
                call_id=f"{prefix}-{index}",
            )
        )
    return ModelTurn(TurnType.TOOL_CALL, tool_calls=parsed, raw=text)


def _parse_decision(payload: dict[str, Any]) -> Decision:
    kind = DecisionType(str(payload["type"]))
    subject: Any = payload.get("subject")
    raw_outcome: Any = payload.get("outcome") or {}
    outcome: dict[str, Any] = (
        cast("dict[str, Any]", raw_outcome) if isinstance(raw_outcome, dict) else {}
    )

    if kind is not DecisionType.ESCALATE and not subject:
        raise ProtocolError(
            f"{kind} with no subject",
            f'A "{kind}" must name the "subject" it is about, for example "ticket:T-104".',
        )
    if kind is DecisionType.FINAL:
        if not outcome.get("key"):
            raise ProtocolError(
                "final with no outcome key",
                'A "final" needs "outcome": {"key": ..., "value": ...} naming what you are'
                " asserting about the subject.",
            )
        if "value" not in outcome:
            raise ProtocolError(
                "final with no outcome value",
                'Your "outcome" has a "key" but no "value". A final must state the value.',
            )
    if kind is DecisionType.ABSTAIN:
        unavailable: Any = payload.get("depends_on_unavailable")
        if not isinstance(unavailable, list) or not unavailable:
            raise ProtocolError(
                "abstain naming no unavailable tool",
                'An "abstain" must list the tools you could not use in'
                ' "depends_on_unavailable". If nothing was unavailable, you are not'
                " abstaining, you are deciding.",
            )

    return Decision(
        type=kind,
        subject=str(subject) if subject else None,
        outcome_key=outcome.get("key"),
        outcome_value=outcome.get("value"),
        rationale=str(payload.get("rationale", "")),
        reason=str(payload.get("reason", "")),
        evidence=_as_dict_list(payload.get("evidence")),
        depends_on=_as_str_list(payload.get("depends_on")),
        depends_on_unavailable=_as_str_list(payload.get("depends_on_unavailable")),
        partial=_as_dict(payload.get("partial")),
        confidence=_as_float(payload.get("confidence")),
        to=str(payload["to"]) if payload.get("to") else None,
    )


def _as_dict(value: Any) -> dict[str, Any] | None:
    return cast("dict[str, Any]", value) if isinstance(value, dict) else None


def _as_str_list(value: Any) -> list[str]:
    return [str(v) for v in cast("list[Any]", value)] if isinstance(value, list) else []


def _as_dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for item in cast("list[Any]", value):
        if isinstance(item, dict):
            out.append(cast("dict[str, Any]", item))
        elif isinstance(item, str):
            # A model writing a bare string meant a reference. Keep it rather than
            # discard it; the runtime fills in provenance either way, and losing
            # the citation entirely is worse than recording an untyped one.
            out.append({"kind": "record", "ref": item})
    return out


def _as_float(value: Any) -> float | None:
    try:
        return float(value)  # pyright: ignore[reportArgumentType]
    except (TypeError, ValueError):
        return None
