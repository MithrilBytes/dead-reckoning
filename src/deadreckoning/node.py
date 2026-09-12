# SPDX-License-Identifier: Apache-2.0
"""A running node: its identity, clock, store, and the act of emitting a record.

Everything a node does ends in a record, so emitting one is the operation this
module exists to make correct. A record and the clock stamp that produced it are
written in a single transaction. If they could diverge, a crash between them would
either reissue a stamp that is already on a record, or skip one and leave a hole
no peer could explain.
"""

from __future__ import annotations

import datetime
from dataclasses import replace
from pathlib import Path
from types import TracebackType
from typing import Any

from deadreckoning.chaos import Fault, FaultInjector
from deadreckoning.clock import HLC, HybridLogicalClock, TimeTrust, TimeTrustTracker
from deadreckoning.config import Config, TierKind
from deadreckoning.health import (
    BreakerPolicy,
    BreakerState,
    DependencyHealth,
    FailureClass,
    HealthMonitor,
    HealthState,
    ObservationSource,
    Transition,
)
from deadreckoning.modes import ModeController, ModeInputs, derive_mode
from deadreckoning.records import Mode, Record, RecordKind, RecordStore, new_record_id
from deadreckoning.runtime import (
    Database,
    clear_faults,
    delete_fault,
    load_faults,
    load_health,
    load_hlc,
    load_mode,
    open_database,
    save_fault,
    save_health,
    save_hlc,
    save_mode,
)


def _wall_ms() -> int:
    return int(datetime.datetime.now(datetime.UTC).timestamp() * 1000)


class Node:
    """Wires a node's parts together over one database.

    Mode and time trust default honestly rather than optimistically. A node that
    has probed nothing has no healthy remote tier, which is the definition of
    being islanded, and it has not checked its clock against anything, so its time
    is untrusted. Both become real assessments once the health monitor exists.
    """

    def __init__(self, config: Config, database: Database, now_ms: Any = _wall_ms) -> None:
        self.config = config
        self.database = database
        self.node_id = config.node.node_id
        self.store = RecordStore(database, redact_keys=config.redact_keys())
        restored = load_hlc(database, self.node_id)
        last = (
            HLC(physical_ms=restored[0], logical=restored[1], node_id=self.node_id)
            if restored
            else None
        )
        self.clock = HybridLogicalClock(self.node_id, now_ms, last)
        self._now_ms = now_ms

        self.dependencies = _declared_dependencies(config)
        self.tier_kinds = {tier.name: str(tier.effective_kind) for tier in config.tiers}
        self.monitor = HealthMonitor(
            dependencies=self.dependencies,
            policy=BreakerPolicy(
                open_after_consecutive_failures=config.health.open_after_consecutive_failures,
                close_after_consecutive_successes=config.health.close_after_consecutive_successes,
                half_open_backoff_initial_s=config.health.half_open_backoff_initial_s,
                half_open_backoff_max_s=config.health.half_open_backoff_max_s,
            ),
            slow_thresholds=_slow_thresholds(config),
        )
        self.injector = FaultInjector(
            enabled=config.chaos.enabled,
            profile=config.profile,
            now=lambda: self._now_ms() / 1000,
        )
        self.trust = TimeTrustTracker(
            fresh_s=config.time.time_fresh_s,
            stale_s=config.time.time_stale_s,
            skew_threshold_s=config.time.skew_threshold_s,
        )
        self._restore_health()
        self._restore_faults()
        self.controller = ModeController(
            up_dwell_s=config.health.up_dwell_s, initial=self._restore_mode()
        )
        self._seed_scripted_tiers()

    @property
    def mode(self) -> Mode:
        return self.controller.mode

    @property
    def time_trust(self) -> TimeTrust:
        return self.trust.assess(self._now_ms()).trust

    def _seed_scripted_tiers(self) -> None:
        """A scripted tier is healthy by construction: there is nothing to probe.

        Seeded directly rather than through `observe`, because nothing was
        observed. Emitting a health change here would put health records ahead of
        the node's own genesis record, and would claim a transition that never
        happened.
        """
        for tier in self.config.tiers:
            if tier.kind is not TierKind.SCRIPTED:
                continue
            current = self.monitor.get(tier.name)
            if current.state is HealthState.UNKNOWN:
                self.monitor.restore(
                    replace(current, state=HealthState.HEALTHY, last_failure_class=FailureClass.OK)
                )

    def _restore_health(self) -> None:
        for name, row in load_health(self.database).items():
            if name not in self.dependencies:
                continue
            self.monitor.restore(
                DependencyHealth(
                    name=name,
                    state=HealthState(str(row["state"])),
                    breaker=BreakerState(str(row["breaker"])),
                    last_failure_class=(
                        FailureClass(str(row["last_failure_class"]))
                        if row["last_failure_class"]
                        else None
                    ),
                    last_latency_ms=_as_int(row["last_latency_ms"]),
                    consecutive_failures=int(str(row["consecutive_failures"])),
                    consecutive_successes=int(str(row["consecutive_successes"])),
                    backoff_s=float(str(row["backoff_s"])),
                    needs_reconciliation=bool(row["needs_reconciliation"]),
                )
            )

    def _restore_faults(self) -> None:
        self.injector.load(
            {
                name: Fault(
                    failure_class=(
                        FailureClass(str(row["failure_class"])) if row["failure_class"] else None
                    ),
                    latency_ms=_as_int(row["latency_ms"]),
                    drop_pct=_as_float(row["drop_pct"]),
                    until=_as_float(row["until"]),
                )
                for name, row in load_faults(self.database).items()
            }
        )

    def _restore_mode(self) -> Mode:
        stored = load_mode(self.database)
        return Mode(str(stored["mode"])) if stored else Mode.ISLANDED

    def observe(
        self,
        dependency: str,
        failure_class: FailureClass,
        latency_ms: int | None = None,
        source: ObservationSource = ObservationSource.PASSIVE,
    ) -> Transition:
        """Record one observation, persist it, and emit a record if it changed anything."""
        transition = self.monitor.observe(dependency, failure_class, latency_ms, source)
        after = transition.after
        save_health(
            self.database,
            dependency,
            {
                "state": str(after.state),
                "breaker": str(after.breaker),
                "last_failure_class": (
                    str(after.last_failure_class) if after.last_failure_class else None
                ),
                "last_latency_ms": after.last_latency_ms,
                "consecutive_failures": after.consecutive_failures,
                "consecutive_successes": after.consecutive_successes,
                "backoff_s": after.backoff_s,
                "needs_reconciliation": int(after.needs_reconciliation),
            },
        )
        if transition.state_changed:
            self.emit(
                RecordKind.HEALTH_CHANGE,
                body={
                    "dependency": dependency,
                    "dependency_type": self.dependencies[dependency],
                    "from_state": str(transition.before.state),
                    "to_state": str(after.state),
                    "failure_class": str(failure_class),
                    "breaker_from": str(transition.before.breaker),
                    "breaker_to": str(after.breaker),
                    "source": str(source),
                    "latency_ms": latency_ms,
                    "consecutive_failures": after.consecutive_failures,
                    "remediation": transition.remediation or None,
                },
            )
        return transition

    def reconciliation_work(self) -> frozenset[str]:
        """Dependencies that have work waiting on them coming back.

        Rule 1 needs both a recovered dependency and work that depends on it. A
        dependency merely having been down is not work: with nothing queued there
        is nothing to reconcile, and entering RECONNECTING would start a sequence
        with no steps that could ever finish.

        Empty until there is something that can queue. The outbox arrives in M3,
        the review queue in M4, and unsynced records in M5; each adds its own
        dependencies here.
        """
        return frozenset()

    def reassess_mode(self) -> Mode | None:
        """Re-derive, apply the dwell, and emit a MODE_CHANGE if the mode moved."""
        proposed = derive_mode(
            ModeInputs(
                health=self.monitor.vector,
                types=self.dependencies,
                tier_kinds=self.tier_kinds,
                previous=self.controller.mode,
                reconciliation_work=self.reconciliation_work(),
            )
        )
        previous = self.controller.mode
        now_s = self._now_ms() / 1000
        changed = self.controller.evaluate(proposed, now_s)
        save_mode(
            self.database,
            str(self.controller.mode),
            self._now_ms(),
            str(self.controller.pending) if self.controller.pending else None,
            now_s if self.controller.pending else None,
        )
        if changed is None:
            return None
        self.emit(
            RecordKind.MODE_CHANGE,
            body={
                "from_mode": str(previous),
                "to_mode": str(changed),
                "trigger_dependency": None,
                "health": self.monitor.snapshot(),
                "reconciliation_pending": {
                    "outbox_entries": 0,
                    "review_items": 0,
                    "unsynced_records": 0,
                },
                "dwell_satisfied": True,
            },
        )
        return changed

    def arm_fault(self, dependency: str, **kwargs: Any) -> None:
        """Arm a fault, persist it, and record the observation it causes."""
        if dependency not in self.dependencies:
            raise KeyError(f"{dependency!r} is not a declared dependency")
        fault = self.injector.arm(dependency, **kwargs)
        save_fault(
            self.database,
            dependency,
            {
                "failure_class": str(fault.failure_class) if fault.failure_class else None,
                "latency_ms": fault.latency_ms,
                "drop_pct": fault.drop_pct,
                "until": fault.until,
            },
        )
        if fault.failure_class is not None:
            for _ in range(self.config.health.open_after_consecutive_failures):
                self.observe(
                    dependency, fault.failure_class, source=ObservationSource.FAULT_INJECTION
                )
        self.reassess_mode()

    def lift_faults(self, dependency: str | None = None) -> list[str]:
        """Lift one fault or all of them, then re-derive once."""
        lifted = (
            [dependency]
            if dependency is not None and self.injector.restore(dependency) is not None
            else ([] if dependency is not None else sorted(self.injector.restore_all()))
        )
        for name in lifted:
            delete_fault(self.database, name)
            for _ in range(self.config.health.close_after_consecutive_successes):
                self.observe(
                    name, FailureClass.OK, latency_ms=1, source=ObservationSource.FAULT_RESTORE
                )
        if dependency is None:
            clear_faults(self.database)
        self.reassess_mode()
        return lifted

    @classmethod
    def open(cls, config: Config, base_dir: Path, now_ms: Any = _wall_ms) -> Node:
        return cls(config, open_database(config.data_dir(base_dir)), now_ms)

    def observed_vector(self) -> dict[str, dict[str, int | str]]:
        """The latest stamp this node holds from every node it knows, itself included.

        Two records are concurrent when neither appears in the other's vector, so
        this is what makes a later disagreement detectable rather than invisible.
        """
        vector: dict[str, dict[str, int | str]] = {}
        for node_id in self.store.node_ids():
            head = self.store.head(node_id)
            if head is not None:
                vector[node_id] = head.hlc
        vector[self.node_id] = self.clock.last.as_dict()
        return vector

    def emit(self, kind: RecordKind, body: dict[str, Any] | None = None, **fields: Any) -> Record:
        """Stamp, chain and persist one record, with its clock stamp, atomically."""
        connection = self.database.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            stamp = self.clock.now()
            record = Record(
                id=new_record_id(stamp),
                node_id=self.node_id,
                hlc=stamp.as_dict(),
                wall_time=datetime.datetime.now(datetime.UTC).isoformat(),
                time_trust=self.time_trust,
                kind=kind,
                mode=self.mode,
                observed=self.observed_vector(),
                prev_hash=self.store.prev_hash_for(self.node_id),
                body=body or {},
                **fields,
            )
            stored = self.store.append(record)
            save_hlc(self.database, self.node_id, stamp.physical_ms, stamp.logical)
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        else:
            connection.execute("COMMIT")
            return stored

    def close(self) -> None:
        self.database.close()

    def __enter__(self) -> Node:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def _as_int(value: object) -> int | None:
    return None if value is None else int(str(value))


def _as_float(value: object) -> float | None:
    return None if value is None else float(str(value))


def _declared_dependencies(config: Config) -> dict[str, str]:
    """Tiers are dependencies too. Nothing is contacted that is not declared."""
    declared = {tier.name: "MODEL_TIER" for tier in config.tiers}
    declared.update({dep.name: str(dep.type) for dep in config.dependencies})
    return declared


def _slow_thresholds(config: Config) -> dict[str, int]:
    thresholds = {tier.name: tier.slow_threshold_ms for tier in config.tiers}
    thresholds.update({dep.name: dep.slow_threshold_ms for dep in config.dependencies})
    return thresholds
