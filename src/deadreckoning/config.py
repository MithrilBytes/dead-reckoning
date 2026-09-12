# SPDX-License-Identifier: Apache-2.0
"""Configuration loading and validation.

Unknown keys are errors, not warnings. A node that silently ignores a misspelled
setting will behave differently from what its operator believes, and the whole
point of this runtime is that its behaviour under failure is predictable. A
typo in a timeout is found at startup, in daylight, not during an outage.

Secrets are never written here. A field names the environment variable that holds
the secret, and the runtime reads it at use time.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

INLINE_SECRET_KEYS = frozenset({"api_key", "token", "secret", "password", "authorization"})
"""Keys that would hold a secret literally. Named so the error can say why."""


class ConfigError(ValueError):
    """Configuration is invalid. The message names section, key, and reason."""


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TierKind(StrEnum):
    REMOTE = "remote"
    ONPREM = "onprem"
    LOCAL = "local"
    SCRIPTED = "scripted"


class DependencyType(StrEnum):
    MODEL_TIER = "MODEL_TIER"
    TOOL_BACKEND = "TOOL_BACKEND"
    IDP = "IDP"
    TIME_SOURCE = "TIME_SOURCE"
    SYNC_HUB = "SYNC_HUB"
    PEER = "PEER"


class NodeConfig(Strict):
    node_id: str
    data_dir: str
    sync_listen: str | None = None


class Canary(Strict):
    method: Literal["HEAD", "GET"] = "HEAD"
    path: str = "/health"


class TierConfig(Strict):
    name: str
    rank: int = Field(ge=0)
    kind: TierKind
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    supports_tools: bool = True
    connect_timeout_s: float = 3.0
    read_timeout_s: float = 60.0
    slow_threshold_ms: int = 8000
    max_tokens: int = Field(default=2048, gt=0)
    seed: int | None = None
    canary: bool = True
    stands_for: TierKind | None = None

    @model_validator(mode="after")
    def _endpoint_required(self) -> Self:
        if self.kind is not TierKind.SCRIPTED and not self.base_url:
            raise ValueError("base_url is required for every tier that is not scripted")
        if self.kind is TierKind.SCRIPTED and self.stands_for is None:
            raise ValueError(
                "a scripted tier must declare stands_for, naming the kind it substitutes."
                " Without it the remote and local partition is incomplete and the node can"
                " never leave ISLANDED."
            )
        if self.stands_for is TierKind.SCRIPTED:
            raise ValueError("stands_for must name a real kind, not 'scripted'")
        return self

    @property
    def effective_kind(self) -> TierKind:
        """What this tier counts as when deriving the mode."""
        return self.stands_for if self.kind is TierKind.SCRIPTED and self.stands_for else self.kind


class DependencyConfig(Strict):
    name: str
    type: DependencyType
    base_url: str | None = None
    slow_threshold_ms: int = 5000
    token_env: str | None = None
    canary: Canary | None = None


class TaskClassConfig(Strict):
    min_rank: int = Field(ge=0)
    review_above_rank: int = Field(ge=0)
    max_tokens_total: int = Field(default=20000, gt=0)
    max_wall_s: int = Field(default=600, gt=0)
    max_steps: int = Field(default=12, gt=0)


class HealthConfig(Strict):
    canary_interval_s: int = 15
    open_after_consecutive_failures: int = Field(default=3, gt=0)
    half_open_backoff_initial_s: int = 5
    half_open_backoff_max_s: int = 120
    close_after_consecutive_successes: int = Field(default=2, gt=0)
    up_dwell_s: int = 10
    rate_limit_window_s: int = 60


class IdentityConfig(Strict):
    fresh_window_s: int = 900
    cached_grant_ttl_s: int = 28800


class TimeConfig(Strict):
    time_fresh_s: int = 3600
    time_stale_s: int = 86400
    skew_threshold_s: int = 300


class SyncConfig(Strict):
    interval_s: int = 30
    page_size: int = Field(default=500, gt=0)
    reconcile_deadline_s: int = Field(default=45, gt=0)


class ProvisioningConfig(Strict):
    auto: bool = False
    interval_s: int = Field(default=300, gt=0)
    horizon_s: int = Field(default=36000, gt=0)
    max_rows_per_tool: int = Field(default=5000, gt=0)
    max_wall_s: int = Field(default=60, gt=0)
    max_calls_per_pass: int = Field(default=500, gt=0)


class ChaosConfig(Strict):
    enabled: bool = False


class RedactionConfig(Strict):
    keys: list[str] = []


class ToolOverride(BaseModel):
    """A partial tool contract layered over the one declared in code."""

    model_config = ConfigDict(extra="allow", frozen=True)
    name: str


class Config(Strict):
    profile: Literal["demo", "production"] = "demo"
    node: NodeConfig
    tiers: list[TierConfig] = []
    dependencies: list[DependencyConfig] = []
    tools: list[ToolOverride] = []
    task_classes: dict[str, TaskClassConfig] = {}
    health: HealthConfig = Field(default_factory=HealthConfig)
    identity: IdentityConfig = Field(default_factory=IdentityConfig)
    time: TimeConfig = Field(default_factory=TimeConfig)
    sync: SyncConfig = Field(default_factory=SyncConfig)
    provisioning: ProvisioningConfig = Field(default_factory=ProvisioningConfig)
    chaos: ChaosConfig = Field(default_factory=ChaosConfig)
    redaction: RedactionConfig = Field(default_factory=RedactionConfig)

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        names = [tier.name for tier in self.tiers]
        duplicate = next((n for n in names if names.count(n) > 1), None)
        if duplicate:
            raise ValueError(f"tiers: duplicate tier name {duplicate!r}")
        ranks = [tier.rank for tier in self.tiers]
        repeated = next((r for r in ranks if ranks.count(r) > 1), None)
        if repeated is not None:
            raise ValueError(f"tiers: rank {repeated} is claimed by more than one tier")
        dependency_names = [dep.name for dep in self.dependencies]
        clash = next((n for n in dependency_names if n in names), None)
        if clash:
            raise ValueError(
                f"dependencies: {clash!r} is already a tier name; every dependency needs a"
                " distinct name because health is tracked by name"
            )
        unprobeable = [
            tier.name
            for tier in self.tiers
            if tier.kind is not TierKind.SCRIPTED and not tier.canary
        ]
        if unprobeable:
            raise ValueError(
                f"tiers: {', '.join(unprobeable)} cannot be probed, so the router can never select"
                " them and the node can never reach CONNECTED. A tier that is not scripted needs"
                " canary = true."
            )
        if self.chaos.enabled and self.profile == "production":
            raise ValueError(
                "chaos: fault injection cannot be enabled while profile is 'production'"
            )
        for class_name, task_class in self.task_classes.items():
            if not any(tier.rank <= task_class.min_rank for tier in self.tiers):
                raise ValueError(
                    f"task_classes.{class_name}: min_rank {task_class.min_rank} cannot be met by"
                    f" any declared tier; declared ranks are {sorted(ranks)}"
                )
        return self

    def tier_by_name(self, name: str) -> TierConfig | None:
        return next((tier for tier in self.tiers if tier.name == name), None)

    def redact_keys(self) -> frozenset[str]:
        from deadreckoning.canonical import DEFAULT_REDACT_KEYS

        return DEFAULT_REDACT_KEYS | {key.lower() for key in self.redaction.keys}

    def data_dir(self, relative_to: Path) -> Path:
        path = Path(self.node.data_dir)
        return path if path.is_absolute() else (relative_to / path).resolve()


def _scan_for_inline_secrets(raw: Any, path: str = "") -> None:
    """Refuse a config that carries a secret literally rather than by variable name."""
    if isinstance(raw, dict):
        for key, value in cast(Mapping[Any, Any], raw).items():
            here = f"{path}.{key}" if path else str(key)
            if isinstance(key, str) and key.lower() in INLINE_SECRET_KEYS:
                raise ConfigError(
                    f"{here}: secrets must never appear in configuration. Name the environment"
                    f" variable instead, for example {key}_env."
                )
            _scan_for_inline_secrets(value, here)
    elif isinstance(raw, list):
        for index, value in enumerate(cast(Sequence[Any], raw)):
            _scan_for_inline_secrets(value, f"{path}[{index}]")


def _explain(error: ValidationError) -> str:
    lines: list[str] = []
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"]) or "<root>"
        lines.append(f"  {location}: {item['msg']}")
    return "\n".join(lines)


def load_config(path: Path) -> Config:
    """Read and validate, or raise ConfigError naming what is wrong and where."""
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"{path}: no such configuration file") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: not valid TOML: {exc}") from exc

    _scan_for_inline_secrets(raw)
    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"{path}: configuration is invalid\n{_explain(exc)}") from exc
