# SPDX-License-Identifier: Apache-2.0
"""A model that says what it was told to say.

This exists so the runtime's behaviour can be tested without a model's behaviour
being part of the test. The interesting properties of this system, that a missing
input produces an abstention, that a deferred action revalidates, that two nodes
disagreeing surfaces a conflict, are properties of the runtime; asserting them
against a live model would be asserting on prose, which the guardrails rightly
forbid.

It can also be told to misbehave, and that is the more important half. A model
that calls an unavailable tool and then answers confidently anyway is the failure
this whole design is built around, and it cannot be tested by asking a real model
nicely to please do the wrong thing.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from deadreckoning.llm.protocol import ModelTurn, parse_turn


class ScriptExhaustedError(RuntimeError):
    """The loop asked for a turn the script does not have."""


Script = dict[tuple[str, int], str]
"""Keyed by task and step, so a scenario reads as a sequence rather than a queue."""


@dataclass(frozen=True, slots=True)
class ScriptedResponse:
    """One canned response. Raw text, so the parser is exercised too.

    Deliberately not a pre-built turn: half the point of scripting is to feed the
    protocol the malformed, fenced and prose-wrapped output a real local model
    produces, and a pre-built turn would skip the parser entirely.
    """

    text: str
    latency_ms: int = 0
    tokens_prompt: int = 0
    tokens_completion: int = 0


class ScriptedModel:
    """Returns canned turns by (task_id, step)."""

    def __init__(
        self,
        script: Mapping[tuple[str, int], ScriptedResponse | str],
        on_missing: Callable[[str, int], ScriptedResponse] | None = None,
    ) -> None:
        self._script: dict[tuple[str, int], ScriptedResponse] = {
            key: value if isinstance(value, ScriptedResponse) else ScriptedResponse(value)
            for key, value in script.items()
        }
        self._on_missing = on_missing
        self.calls: list[tuple[str, int]] = []

    def chat(self, task_id: str, step: int, **_: Any) -> ModelTurn:
        self.calls.append((task_id, step))
        response = self._script.get((task_id, step))
        if response is None:
            if self._on_missing is None:
                raise ScriptExhaustedError(
                    f"no scripted response for task {task_id!r} step {step}."
                    f" Scripted steps: {sorted(k for k in self._script if k[0] == task_id)}"
                )
            response = self._on_missing(task_id, step)
        turn = parse_turn(response.text, call_id_prefix=f"{task_id}-{step}")
        return ModelTurn(
            turn_type=turn.turn_type,
            tool_calls=turn.tool_calls,
            decision=turn.decision,
            raw=turn.raw,
            latency_ms=response.latency_ms,
            tokens_prompt=response.tokens_prompt,
            tokens_completion=response.tokens_completion,
        )


def tool_call(name: str, **args: Any) -> str:
    return json.dumps({"type": "tool_call", "calls": [{"name": name, "args": args}]})


def final(subject: str, key: str, value: Any, rationale: str = "", **extra: Any) -> str:
    return json.dumps(
        {
            "type": "final",
            "subject": subject,
            "outcome": {"key": key, "value": value},
            "rationale": rationale,
            **extra,
        }
    )


def abstain(subject: str, key: str, reason: str, unavailable: list[str], **extra: Any) -> str:
    return json.dumps(
        {
            "type": "abstain",
            "subject": subject,
            "outcome": {"key": key},
            "reason": reason,
            "depends_on_unavailable": unavailable,
            **extra,
        }
    )


def escalate(subject: str, reason: str, to: str = "human") -> str:
    return json.dumps({"type": "escalate", "subject": subject, "reason": reason, "to": to})
