# SPDX-License-Identifier: Apache-2.0
"""The harness the reference scenario drives: one truck, fully wired.

Separated from the narrative so the scenario file reads as the story it is
telling rather than as assembly. Everything here is ordinary runtime wiring, done
once per node.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from deadreckoning.agent import AgentLoop, Task
from deadreckoning.canonical import content_hash
from deadreckoning.clock import TimeTrust
from deadreckoning.config import Config, load_config
from deadreckoning.deferral import DeferralContext, OutboxDeferrer
from deadreckoning.health import HealthState
from deadreckoning.llm.scripted import ScriptedModel
from deadreckoning.local_store import LocalStore
from deadreckoning.manifest import TierView, build_manifest
from deadreckoning.node import Node
from deadreckoning.outbox import Outbox
from deadreckoning.records import IdentityState, Mode, Record, RecordKind
from deadreckoning.sync.client import SyncClient
from deadreckoning.tools.enforcer import ToolEnforcer
from deadreckoning.tools.preconditions import CheckerRegistry
from demo.tools_outage import CHECKERS, Backends, build_registry

HERE = Path(__file__).resolve().parent
PROMPT = (HERE.parent / "docs" / "prompts" / "planner_v1.md").read_text()
BACKENDS: tuple[str, ...] = (
    "gis-api",
    "ticket-api",
    "scada-api",
    "dispatch-api",
    "notify-api",
    "weather-api",
)

CONFIG = """
profile = "demo"

[node]
node_id = "{node_id}"
data_dir = "./data"

[[tiers]]
name = "frontier"
rank = 0
kind = "remote"
client = "scripted"
model = "scripted-frontier"
canary = false

[[tiers]]
name = "local-q4"
rank = 2
kind = "local"
client = "scripted"
model = "scripted-local"
canary = false

[[dependencies]]
name = "gis-api"
type = "TOOL_BACKEND"
base_url = "http://127.0.0.1:8710/gis"

[[dependencies]]
name = "ticket-api"
type = "TOOL_BACKEND"
base_url = "http://127.0.0.1:8710/ticket"

[[dependencies]]
name = "scada-api"
type = "TOOL_BACKEND"
base_url = "http://127.0.0.1:8710/scada"

[[dependencies]]
name = "dispatch-api"
type = "TOOL_BACKEND"
base_url = "http://127.0.0.1:8710/dispatch"

[[dependencies]]
name = "notify-api"
type = "TOOL_BACKEND"
base_url = "http://127.0.0.1:8710/notify"

[[dependencies]]
name = "weather-api"
type = "TOOL_BACKEND"
base_url = "http://127.0.0.1:8710/weather"

[[dependencies]]
name = "hq-hub"
type = "SYNC_HUB"
base_url = "{hub_url}"

[task_classes.triage]
min_rank = 2
review_above_rank = 0

[health]
up_dwell_s = 0

[chaos]
enabled = true
"""


class Truck:
    """One node, with everything wired to it."""

    def __init__(self, node_id: str, root: Path, hub_url: str) -> None:
        root.mkdir(parents=True, exist_ok=True)
        (root / "dr.toml").write_text(CONFIG.format(node_id=node_id, hub_url=hub_url))
        self.config: Config = load_config(root / "dr.toml")
        self.node = Node.open(self.config, root)
        self.backends = Backends()
        self.store = LocalStore(self.node.database)
        self.registry = build_registry(self.backends)
        self.checkers = CheckerRegistry()
        for name, fn in CHECKERS.items():
            self.checkers.register(name, self._bind(fn))
        self.outbox = Outbox(self.node.database, node_id)
        self.counter = 0
        self.deferrer = OutboxDeferrer(
            self.outbox,
            self.checkers,
            self._next_entry_id,
            lambda: self.node.clock.last.physical_ms,
            lambda: self.node.clock.last,
        )
        self.hub_url = hub_url
        self.node.emit(
            RecordKind.NODE_INIT,
            body={"schema_version": 1, "profile": "demo", "config_hash": None},
        )

    def _bind(self, fn: Any) -> Any:
        def checker(**kwargs: Any) -> tuple[bool, Any]:
            return fn(self.backends, **kwargs)

        return checker

    def _next_entry_id(self) -> str:
        self.counter += 1
        return f"{self.node.node_id}-ob-{self.counter}"

    @property
    def node_id(self) -> str:
        return self.node.node_id

    def health(self, **overrides: HealthState) -> dict[str, HealthState]:
        return {**dict.fromkeys(BACKENDS, HealthState.HEALTHY), **overrides}

    def run(
        self,
        task: Task,
        script: dict[tuple[str, int], str],
        health: dict[str, HealthState],
        mode: Mode,
        tier: TierView,
    ) -> list[Record]:
        emitted: list[Record] = []

        def emit(kind: RecordKind, body: dict[str, Any], **fields: Any) -> Record:
            record = self.node.emit(kind, body=body, **fields)
            emitted.append(record)
            if kind is RecordKind.MODEL_TURN:
                self.deferrer.context = DeferralContext(
                    decision_id=record.id,
                    subject=task.subject,
                    identity=IdentityState.CACHED if mode is Mode.ISLANDED else IdentityState.FRESH,
                    connected=mode is Mode.CONNECTED,
                    observed_source="LOCAL" if mode is Mode.ISLANDED else "LIVE",
                )
            return record

        manifest = build_manifest(
            mode=mode,
            tier=tier,
            identity=IdentityState.CACHED if mode is Mode.ISLANDED else IdentityState.FRESH,
            identity_ttl_s=5400,
            time_trust=TimeTrust.DRIFTING if mode is Mode.ISLANDED else TimeTrust.TRUSTED,
            registry=self.registry,
            store=self.store,
            health=health,
            now_ms=self.node.clock.last.physical_ms,
        )
        loop = AgentLoop(
            client=ScriptedModel(script),
            enforcer=ToolEnforcer(
                self.registry,
                self.store,
                health,
                self.node.clock.last.physical_ms,
                self.deferrer,
            ),
            emit=emit,
            prompt_template=PROMPT,
            prompt_template_hash=content_hash(PROMPT),
        )
        loop.run(task, manifest)
        decision = next(
            (r for r in reversed(emitted) if r.kind in (RecordKind.FINAL, RecordKind.ABSTENTION)),
            None,
        )
        if decision is not None:
            self.deferrer.link_to_decision(decision.id)
        return emitted

    def sync(self, rounds: int = 4) -> tuple[int, int]:
        """Push, then pull until the sync vector stops advancing.

        A single pull is a race this scenario loses: two trucks reconnecting
        seconds apart each pull before the other has pushed, so neither sees the
        other's decisions, no conflict is found, and both drain actions that
        contradict each other. Repeating until nothing new arrives makes the
        outcome depend on what the nodes know rather than on which one got signal
        first.
        """
        client = SyncClient(self.hub_url, peer_name="hq-hub")
        held = list(self.node.store.iter_records())
        pushed = client.push(held, client.head())
        self.node.emit(RecordKind.SYNC, body=pushed.as_body())

        total_pulled = 0
        for _ in range(rounds):
            held = list(self.node.store.iter_records())
            pulled = client.pull(held, sorted(set(client.head()) | {self.node_id}))
            for record in pulled.accepted:
                self.node.store.append(record)
            self.node.emit(RecordKind.SYNC, body=pulled.as_body())
            total_pulled += pulled.pulled
            if pulled.pulled == 0:
                break
        return pushed.pushed, total_pulled

    def close(self) -> None:
        self.node.close()
