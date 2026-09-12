# SPDX-License-Identifier: Apache-2.0
"""Small helpers for assembling a node from its configuration and stored rows.

Separated from `node.py` so that file stays about what a node does rather than
about how its parts are named and coerced.
"""

from __future__ import annotations

from deadreckoning.config import Config


def as_int(value: object) -> int | None:
    return None if value is None else int(str(value))


def as_float(value: object) -> float | None:
    return None if value is None else float(str(value))


def declared_dependencies(config: Config) -> dict[str, str]:
    """Tiers are dependencies too. Nothing is contacted that is not declared."""
    declared = {tier.name: "MODEL_TIER" for tier in config.tiers}
    declared.update({dep.name: str(dep.type) for dep in config.dependencies})
    return declared


def slow_thresholds(config: Config) -> dict[str, int]:
    thresholds = {tier.name: tier.slow_threshold_ms for tier in config.tiers}
    thresholds.update({dep.name: dep.slow_threshold_ms for dep in config.dependencies})
    return thresholds
