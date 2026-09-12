# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures, and the guard that keeps this package honest about the network.

The runtime claims it makes no network connection unless a declared dependency
asks it to. That claim is worth nothing unless something enforces it, so every
test in this suite runs with outbound sockets to anything but loopback wired to
raise, and the attempt is recorded so a test can assert on it.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from deadreckoning.config import Config, load_config
from deadreckoning.node import Node


class EgressError(RuntimeError):
    """Something tried to reach off this machine."""


class EgressLog:
    def __init__(self) -> None:
        self.attempts: list[tuple[str, int]] = []

    @property
    def non_loopback(self) -> list[tuple[str, int]]:
        return self.attempts


def _is_loopback(host: str) -> bool:
    if host in {"localhost", "", "::1"}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@pytest.fixture(autouse=True)
def egress_guard(monkeypatch: pytest.MonkeyPatch) -> Iterator[EgressLog]:
    log = EgressLog()
    real_connect = socket.socket.connect

    def guarded(self: socket.socket, address: Any) -> Any:
        if isinstance(address, tuple) and len(cast("tuple[Any, ...]", address)) >= 2:
            pair = cast("tuple[Any, ...]", address)
            host = str(pair[0])
            if not _is_loopback(host):
                log.attempts.append((host, int(pair[1])))
                raise EgressError(
                    f"blocked outbound connection to {host}:{pair[1]}."
                    " This package must not reach the network on its own."
                )
        return real_connect(self, cast("Any", address))

    monkeypatch.setattr(socket.socket, "connect", guarded)
    yield log


@pytest.fixture
def example_config_text() -> str:
    path = Path(__file__).resolve().parents[1] / "dr.example.toml"
    return path.read_text(encoding="utf-8")


@pytest.fixture
def config(tmp_path: Path, example_config_text: str) -> Config:
    path = tmp_path / "dr.toml"
    path.write_text(
        example_config_text.replace('data_dir = "./data/truck-7"', 'data_dir = "./data"')
    )
    return load_config(path)


@pytest.fixture
def node(tmp_path: Path, config: Config) -> Iterator[Node]:
    with Node.open(config, tmp_path) as opened:
        yield opened
