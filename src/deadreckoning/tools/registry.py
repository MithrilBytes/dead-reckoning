# SPDX-License-Identifier: Apache-2.0
"""Registered tools, their implementations, and their local substitutes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from deadreckoning.tools.contract import ToolContract

LiveCall = Callable[[dict[str, Any]], Any]
LocalCall = Callable[[dict[str, Any]], Any]


@dataclass(frozen=True, slots=True)
class RegisteredTool:
    contract: ToolContract
    live: LiveCall | None = None
    local: LocalCall | None = None
    verify: Callable[[str], str] | None = None


class ToolRegistry:
    """Everything the agent could call, available or not.

    The registry exposes unavailable tools as deliberately as available ones. A
    model that cannot see a capability exists will work around its absence
    silently; one told the capability exists and is down can say so.
    """

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(
        self,
        contract: ToolContract,
        live: LiveCall | None = None,
        local: LocalCall | None = None,
        verify: Callable[[str], str] | None = None,
    ) -> None:
        if contract.name in self._tools:
            raise ValueError(f"{contract.name} is already registered")
        self._tools[contract.name] = RegisteredTool(contract, live, local, verify)

    def get(self, name: str) -> RegisteredTool:
        if name not in self._tools:
            raise KeyError(f"{name!r} is not a registered tool")
        return self._tools[name]

    def contracts(self) -> list[ToolContract]:
        return [tool.contract for tool in self._tools.values()]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def backends(self) -> set[str]:
        return {t.contract.backend for t in self._tools.values() if t.contract.backend}

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools
