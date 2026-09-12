# SPDX-License-Identifier: Apache-2.0
"""What a tool promises about itself when its backend is gone.

Most systems describe a tool by what it does. The interesting half here is what it
does when it cannot do that: answer from a local copy, record the intent for
later, or refuse. Declaring it up front is what lets the runtime decide the
dispatch path instead of leaving it to a model that would rather be helpful.

Every rule below is a conditional the schema also enforces, restated here so the
failure arrives at startup with the tool's name in it rather than as a validation
error somewhere downstream.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class OfflinePolicy(StrEnum):
    LOCAL = "LOCAL"
    QUEUE = "QUEUE"
    FAIL = "FAIL"


class SideEffect(StrEnum):
    NONE = "NONE"
    IDEMPOTENT = "IDEMPOTENT"
    NON_IDEMPOTENT = "NON_IDEMPOTENT"


class Consequence(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Approval(StrEnum):
    NEVER = "NEVER"
    WHEN_NOT_CONNECTED = "WHEN_NOT_CONNECTED"
    WHEN_IDENTITY_NOT_FRESH = "WHEN_IDENTITY_NOT_FRESH"
    ALWAYS = "ALWAYS"


class Availability(StrEnum):
    """What the model is told about a tool, and what the enforcer actually did."""

    LIVE = "LIVE"
    LIVE_SLOW = "LIVE_SLOW"
    LOCAL = "LOCAL"
    LOCAL_STALE = "LOCAL_STALE"
    QUEUED = "QUEUED"
    UNAVAILABLE = "UNAVAILABLE"


class PreconditionSpec(BaseModel):
    """A named checker and how to build its arguments from the call's."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    check: str
    args_from: dict[str, str] = {}


class HydrateSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    subject_types: list[str] = []
    arg_from_subject: str | None = None
    static_args: list[dict[str, Any]] = []
    max_rows: int = Field(default=5000, gt=0)

    @model_validator(mode="after")
    def _one_way_or_the_other(self) -> Self:
        keyed = bool(self.subject_types) and self.arg_from_subject is not None
        if not keyed and not self.static_args:
            raise ValueError(
                "hydrate must declare subject_types with arg_from_subject, or static_args, or both"
            )
        return self


class ToolContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str = ""
    backend: str | None = None
    offline_policy: OfflinePolicy
    side_effect: SideEffect
    consequence: Consequence
    approval: Approval = Approval.NEVER
    preconditions: list[PreconditionSpec] = []
    idempotency_key: str | None = None
    expiry_s: int | None = None
    staleness_budget_s: int | None = None
    fail_when_stale: bool = False
    prefer_local_when_slow: bool = False
    live_timeout_s: float = 10.0
    slow_timeout_s: float = 30.0
    args_schema: dict[str, Any] = {}
    verify_supported: bool = False
    hydrate: HydrateSpec | None = None
    local_source: str | None = None

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        if self.backend is None and self.offline_policy is not OfflinePolicy.LOCAL:
            raise ValueError(
                f"{self.name}: a tool with no backend is always local, so its offline_policy"
                f" must be LOCAL, not {self.offline_policy}"
            )
        if self.offline_policy is OfflinePolicy.QUEUE:
            if self.expiry_s is None:
                raise ValueError(
                    f"{self.name}: a QUEUE tool must declare expiry_s. A deferred intent with no"
                    " expiry could fire against a world that has moved on without limit."
                )
            if self.side_effect is SideEffect.NONE:
                raise ValueError(
                    f"{self.name}: queueing a tool with no side effect is pointless; it should be"
                    " LOCAL or FAIL"
                )
        if self.offline_policy is OfflinePolicy.LOCAL and not (self.hydrate or self.local_source):
            raise ValueError(
                f"{self.name}: a LOCAL tool must declare hydrate or local_source. A LOCAL policy"
                " whose store is never filled is a promise to answer offline that cannot be kept."
            )
        if self.side_effect is SideEffect.NON_IDEMPOTENT and not self.idempotency_key:
            raise ValueError(
                f"{self.name}: a NON_IDEMPOTENT tool must declare an explicit idempotency_key"
                " template. Hashing every argument cannot express which arguments make two calls"
                " the same action, which is exactly the judgement this class needs."
            )
        return self

    @property
    def always_local(self) -> bool:
        return self.backend is None

    def key_for(self, args: dict[str, Any]) -> str:
        """The idempotency key for one call.

        A template names the arguments that make two calls the same action. Absent
        one, every argument counts, which is only ever right for a tool whose
        repetition is harmless.
        """
        from deadreckoning.canonical import content_hash

        if self.idempotency_key is None:
            return content_hash({"tool": self.name, "args": args})
        try:
            return self.idempotency_key.format(**args)
        except KeyError as exc:
            raise ValueError(
                f"{self.name}: idempotency_key template needs argument {exc.args[0]!r},"
                f" which this call did not supply"
            ) from exc
